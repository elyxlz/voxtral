import torch
import re
import string
import transformers as tr
import dataclasses
import typing
import numpy as np


@dataclasses.dataclass
class WordTiming:
    word: str
    tokens: list[int]
    start: float
    end: float


def clean_text(text: str) -> str:
    cleaned = re.sub(r"<\|[\d.]+\|>", "", text)
    cleaned = cleaned.strip()
    cleaned = re.sub(r" +", " ", cleaned)
    return cleaned


def split_tokens_on_unicode(
    tokens: list[int], tokenizer: typing.Any
) -> tuple[list[str], list[list[int]]]:
    decoded_full = tokenizer.decode(tokens, decode_with_timestamps=True)
    replacement_char = "\ufffd"

    words = []
    word_tokens = []
    current_tokens = []
    unicode_offset = 0

    for token in tokens:
        current_tokens.append(token)
        decoded = tokenizer.decode(current_tokens, decode_with_timestamps=True)

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


def split_tokens_on_spaces(
    tokens: list[int], tokenizer: typing.Any
) -> tuple[list[str], list[list[int]]]:
    subwords, subword_tokens_list = split_tokens_on_unicode(tokens, tokenizer)
    words = []
    word_tokens = []

    for subword, subword_tokens in zip(subwords, subword_tokens_list):
        special = subword_tokens[0] >= tokenizer.eos_token_id
        with_space = subword.startswith(" ")
        punctuation = subword.strip() in string.punctuation
        if special or with_space or punctuation or len(words) == 0:
            words.append(subword)
            word_tokens.append(subword_tokens)
        else:
            words[-1] = words[-1] + subword
            word_tokens[-1].extend(subword_tokens)

    return words, word_tokens


def tokens_to_words(
    generate_outputs: dict[str, typing.Any], tokenizer: typing.Any, language: str
) -> list[list[WordTiming]]:
    timings = []

    for batch_idx in range(generate_outputs["sequences"].shape[0]):
        predicted_ids = generate_outputs["sequences"][batch_idx].numpy()
        token_timestamps = generate_outputs["token_timestamps"][batch_idx].numpy()

        text_tokens = [token for token in predicted_ids]
        if language in {"zh", "ja", "th", "lo", "my"}:
            words, word_tokens = split_tokens_on_unicode(text_tokens, tokenizer)
        else:
            words, word_tokens = split_tokens_on_spaces(text_tokens, tokenizer)
        word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))

        start_times = token_timestamps[word_boundaries[:-1]]
        end_times = token_timestamps[word_boundaries[1:]]

        timings.append(
            [
                WordTiming(word, tokens, start, end)
                for word, tokens, start, end in zip(
                    words, word_tokens, start_times, end_times
                )
            ]
        )

    return timings


def merge_punctuations(alignment: list[WordTiming]) -> None:
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
            following.word = previous.word + following.word
            following.tokens = previous.tokens + following.tokens
            previous.word = ""
            previous.tokens = []
        else:
            j = i
        i -= 1

    i = 0
    j = 1
    while j < len(alignment):
        previous = alignment[i]
        following = alignment[j]
        if not previous.word.endswith(" ") and following.word in append_punctuations:
            previous.word = previous.word + following.word
            previous.tokens = previous.tokens + following.tokens
            following.word = ""
            following.tokens = []
        else:
            i = j
        j += 1


def separate_into_buckets(
    data: list[WordTiming], bucket_size: float, total_duration: float
) -> list[list[str]]:
    buckets = []
    current_time = 0

    while current_time < total_duration:
        bucket_end_time = current_time + bucket_size
        bucket_words = []

        for entry in data[1:-1]:
            if current_time <= entry.end < bucket_end_time:
                cleaned_word = clean_text(entry.word)
                bucket_words.append(cleaned_word)

        buckets.append(bucket_words)
        current_time = bucket_end_time

    return buckets


def generate_tokens(
    processor: typing.Any, model: typing.Any, audio: torch.Tensor
) -> dict[str, typing.Any]:
    input_features = processor(
        audio.numpy(), sampling_rate=16000, return_tensors="pt"
    ).input_features.to(audio.device, audio.dtype)
    return model.generate(
        input_features, return_timestamps=True, return_token_timestamps=True
    )


class TimedWhisperTokenizer(torch.nn.Module):
    def __init__(self, model_name: str, hertz: int) -> None:
        super().__init__()
        self.processor = tr.WhisperProcessor.from_pretrained(model_name)
        self.model = tr.WhisperForConditionalGeneration.from_pretrained(model_name)

        self.language: str = "en"
        self.tokenizer: typing.Any = self.processor.tokenizer  # type: ignore
        self.hertz: int = hertz

        self.mistral_tokenizer: tr.AutoTokenizer = tr.AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            padding_side="right",
            add_prefix_space=False,
        )
        self.mistral_tokenizer.pad_token = self.mistral_tokenizer.eos_token

    def forward(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        assert sample_rate == 16000, "Sample rate must be 16000"
        assert audio.ndim == 2, "Audio must be 2D, batch x time"
        total_duration = audio.shape[1] / sample_rate

        outputs = generate_tokens(self.processor, self.model, audio)
        alignment = tokens_to_words(outputs, self.tokenizer, self.language)
        [merge_punctuations(a) for a in alignment]
        buckets = []
        for a in alignment:
            out = separate_into_buckets(
                a, bucket_size=1.0, total_duration=total_duration
            )
            out = [clean_text(" ".join(b)) for b in out]
            tokens = self.mistral_tokenizer(  # type: ignore
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
