class TimedWhisperTokenizer(nn.Module):
    """Transcribe text with word-level timestamps with whisper, and encode using llama tokenizer at a time interval with padding tokens"""

    def __init__(
        self,
        model_name: str,
        hertz: int,
    ):
        super().__init__()
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.language = "en"
        self.tokenizer = self.processor.tokenizer

        self.hertz = hertz

        self.llama_tokenizer = AutoTokenizer.from_pretrained(
            "huggyllama/llama-7b",
            padding_side="right",
            add_prefix_space=False,
        )
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

    def generate_tokens(self, audio):
        input_features = self.processor(
            audio.numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_features
        return self.model.generate(
            input_features, return_timestamps=True, return_token_timestamps=True
        )

    def get_token_probabilities(self, generate_outputs):
        predicted_ids = generate_outputs.sequences[:, 1:]
        scores = torch.cat([x.unsqueeze(0) for x in generate_outputs.scores], dim=0)
        scores = scores.permute([1, 0, 2])
        probabilities = scores.softmax(dim=-1)
        token_probs = torch.gather(
            probabilities, 2, predicted_ids.unsqueeze(2)
        ).squeeze(2)
        ones = torch.ones((predicted_ids.shape[0], 1))
        return torch.cat([ones, token_probs], dim=-1)

    def split_to_word_tokens(self, tokens):
        if self.language in {"zh", "ja", "th", "lo", "my"}:
            return self.split_tokens_on_unicode(tokens)

        return self.split_tokens_on_spaces(tokens)

    def split_tokens_on_unicode(self, tokens: list[int]):
        decoded_full = self.tokenizer.decode(tokens, decode_with_timestamps=True)
        replacement_char = "\ufffd"

        words = []
        word_tokens = []
        current_tokens = []
        unicode_offset = 0

        for token in tokens:
            current_tokens.append(token)
            decoded = self.tokenizer.decode(current_tokens, decode_with_timestamps=True)

            if (
                replacement_char not in decoded
                or decoded_full[unicode_offset + decoded.index(replacement_char)]
                == replacement_char
            ):
                words.append(decoded)
                word_tokens.append(current_tokens)
                current_tokens = []
                unicode_offset += len(decoded)

        return words, word_tokens

    def split_tokens_on_spaces(self, tokens: list[int]):
        subwords, subword_tokens_list = self.split_tokens_on_unicode(tokens)
        words = []
        word_tokens = []

        for subword, subword_tokens in zip(subwords, subword_tokens_list):
            special = subword_tokens[0] >= self.tokenizer.eos_token_id
            with_space = subword.startswith(" ")
            punctuation = subword.strip() in string.punctuation
            if special or with_space or punctuation or len(words) == 0:
                words.append(subword)
                word_tokens.append(subword_tokens)
            else:
                words[-1] = words[-1] + subword
                word_tokens[-1].extend(subword_tokens)

        return words, word_tokens

    def tokens_to_words(self, generate_outputs):
        token_probabilities = self.get_token_probabilities(generate_outputs).numpy()

        timings = []

        for batch_idx in range(token_probabilities.shape[0]):
            predicted_ids = generate_outputs["sequences"][batch_idx].numpy()
            token_timestamps = generate_outputs["token_timestamps"][batch_idx].numpy()

            text_tokens = [token for token in predicted_ids]
            words, word_tokens = self.split_to_word_tokens(text_tokens)
            word_boundaries = np.pad(
                np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0)
            )

            start_times = token_timestamps[word_boundaries[:-1]]
            end_times = token_timestamps[word_boundaries[1:]]

            word_probabilities = [
                np.mean(token_probabilities[batch_idx][i:j])
                for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
            ]

            timings.append(
                [
                    WordTiming(word, tokens, start, end, probability)
                    for word, tokens, start, end, probability in zip(
                        words, word_tokens, start_times, end_times, word_probabilities
                    )
                ]
            )

        return timings

    def merge_punctuations(self, alignment):
        # merge prepended punctuations

        prepend_punctuations = "\"'“¿([{-"
        append_punctuations = "\"'.。,，!！?？:：”)]}、"

        i = len(alignment) - 2
        j = len(alignment) - 1
        while i >= 0:
            previous = alignment[i]
            following = alignment[j]
            if (
                previous.word.startswith(" ")
                and previous.word.strip() in prepend_punctuations
            ):
                # prepend it to the following word
                following.word = previous.word + following.word
                following.tokens = previous.tokens + following.tokens
                previous.word = ""
                previous.tokens = []
            else:
                j = i
            i -= 1

        # merge appended punctuations
        i = 0
        j = 1
        while j < len(alignment):
            previous = alignment[i]
            following = alignment[j]
            if (
                not previous.word.endswith(" ")
                and following.word in append_punctuations
            ):
                # append it to the previous word
                previous.word = previous.word + following.word
                previous.tokens = previous.tokens + following.tokens
                following.word = ""
                following.tokens = []
            else:
                i = j
            j += 1

    def separate_into_buckets(self, data, bucket_size, total_duration):
        buckets = []
        current_time = 0
        # total_duration = 0

        # Finding the maximum duration from the word timings to define how many buckets are needed
        # for entry in data[1:-1]:
        #     if '<|' in entry.word:
        #         continue
        #     total_duration = max(total_duration, entry.end)

        # Creating buckets based on the defined bucket size
        while current_time < total_duration:
            bucket_end_time = current_time + bucket_size
            bucket_words = []

            for entry in data[1:-1]:
                if current_time <= entry.end < bucket_end_time:
                    bucket_words.append(entry.word)

            buckets.append(bucket_words)
            current_time = bucket_end_time

        return buckets

    def forward(self, audio: Tensor, sample_rate: int = 16000):
        assert sample_rate == 16000, "Sample rate must be 16000"
        assert audio.ndim == 2, "Audio must be 2D, batch x time"
        total_duration = audio.shape[1] / sample_rate

        outputs = self.generate_tokens(audio)
        alignment = self.tokens_to_words(outputs)
        [self.merge_punctuations(a) for a in alignment]
        buckets = []
        for a in alignment:
            out = self.separate_into_buckets(
                a, bucket_size=1.0, total_duration=total_duration
            )
            out = ["".join(b) for b in out]
            tokens = self.llama_tokenizer(
                out,
                padding="max_length",
                max_length=self.hertz,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            buckets.append(tokens["input_ids"])
        buckets = torch.stack(buckets).view(len(buckets), -1)
        return buckets
