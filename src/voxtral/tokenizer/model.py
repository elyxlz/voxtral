import typing

import dotenv
import huggingface_hub as hf_hub
import torch
import torchaudio as ta

from .mimi.models import loaders
from .word_level_whisper import TimedWhisperTokenizer

dotenv.load_dotenv()


class VoxtralTokenizerConfig(typing.NamedTuple):
    mimi_path: str = "kyutai/mimi"
    whisper_path: str = "openai/whisper-tiny.en"
    text_hz: int = 5
    mimi_num_quantizers: int = 4
    text_vocab_size: int = 32768


def interleave(*seqs: list[torch.Tensor], factors: list[int]):
    assert isinstance(seqs[0], torch.Tensor)
    bs = seqs[0].size(0)

    to_cat = []
    for i, seq in enumerate(seqs):
        assert isinstance(seq, torch.Tensor)
        seq = seq.view(bs, -1, factors[i])
        to_cat.append(seq)

    out = torch.cat(to_cat, dim=-1)
    return out.view(bs, -1)


def uninterleave(x: torch.Tensor, factors: list[int]) -> list[torch.Tensor]:
    bs = x.size(0)
    chunks = x.view(bs, -1, sum(factors))
    splits = chunks.split(factors, dim=-1)

    return [split.reshape(bs, -1) for split in splits]


class VoxtralTokenizer(torch.nn.Module):
    def __init__(self, config: VoxtralTokenizerConfig):
        super().__init__()
        self.config = config

        # mimi
        mimi_weight = hf_hub.hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, device="cpu")
        self.mimi.set_num_codebooks(config.mimi_num_quantizers)

        # whisper
        self.whisper = TimedWhisperTokenizer(config.whisper_path, hertz=config.text_hz)

    @property
    def device(self) -> torch.device:
        return next(self.mimi.parameters()).device

    @torch.no_grad()
    def encode(self, x: torch.Tensor, sample_rate: int) -> torch.Tensor:
        assert x.dim() == 3
        assert x.size(1) == 1
        assert sample_rate == 24_000
        x_for_whisper = ta.functional.resample(x, sample_rate, 16_000)[:, 0]

        text_tokens = self.whisper(x_for_whisper, sample_rate=16_000)

        # make sure every quantizer uses different tokens
        mimi_vocab_size = self.mimi.quantizer.cardinality
        token_offset = (
            torch.arange(0, self.config.mimi_num_quantizers) * mimi_vocab_size
        )

        mimi_tokens = self.mimi.encode(x.to(self.device))

        audio_tokens = (
            mimi_tokens
            + token_offset[None, :, None].to(self.device)
            + self.config.text_vocab_size
        )

        interleaved_audio_tokens = interleave(
            *audio_tokens.unbind(1), factors=[1] * self.config.mimi_num_quantizers
        )

        # delay text and audio tokens by a window
        # we do this becuase the text is not perfectly 5 hz alligned, so we avoid
        # corresopnding audio tokens appearing before the text tokens
        #
        # we remove the first two window of the audio tokens
        # and the last two windows of the text tokens
        #
        interleaved_audio_tokens = interleaved_audio_tokens[
            ..., int(self.config.mimi_num_quantizers * self.mimi.frame_rate * 2) :
        ]

        text_tokens = text_tokens[..., : -self.config.text_hz * 2]

        intermediate_tokens = [text_tokens, interleaved_audio_tokens]
        text_to_audio_factor = (
            self.mimi.frame_rate * self.config.mimi_num_quantizers / self.config.text_hz
        )
        tokens = interleave(
            *intermediate_tokens, factors=[1, int(text_to_audio_factor)]
        )

        return tokens

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        assert z.dim() == 2
        # Uninterleave tokens to separate text and audio tokens
        text_to_audio_factor = int(
            self.mimi.frame_rate * self.config.mimi_num_quantizers / self.config.text_hz
        )

        # throw away text tokens
        text_tokens, audio_tokens = uninterleave(z, factors=[1, text_to_audio_factor])

        # Discard text tokens and focus on audio tokens
        audio_tokens = audio_tokens - self.config.text_vocab_size

        # Uninterleave audio tokens
        audio_tokens = uninterleave(
            audio_tokens, factors=[1] * self.config.mimi_num_quantizers
        )

        # Restack audio tokens to shape (batch_size, num_quantizers, sequence_length)
        audio_tokens = torch.stack(audio_tokens, dim=1)

        # Undo the token offset
        mimi_vocab_size = self.mimi.quantizer.cardinality
        token_offset = (
            torch.arange(0, self.config.mimi_num_quantizers, device=audio_tokens.device)
            * mimi_vocab_size
        )
        audio_tokens = audio_tokens - token_offset[None, :, None]
        assert torch.all(
            audio_tokens < self.mimi.quantizer.cardinality
        ), "Some audio tokens exceed the maximum cardinality"
        audio_tokens = torch.clamp(
            audio_tokens, min=0, max=self.mimi.quantizer.cardinality - 1
        )
        # Decode audio tokens using mimi

        decoded_audio = self.mimi.decode(audio_tokens)

        return decoded_audio


if __name__ == "__main__":
    # Initialize the VoxtralTokenizer
    config = VoxtralTokenizerConfig()
    tokenizer = VoxtralTokenizer(config)

    # Create a dummy audio input tensor
    dummy_audio = torch.randn(1, 1, 24000)  # Assuming 1 second of audio at 24kHz

    # Encode the audio
    encoded = tokenizer.encode(dummy_audio, sample_rate=24000)
    print("Encoded shape:", encoded.shape)

    # Decode the encoded tokens
    decoded = tokenizer.decode(encoded)
    print("Decoded shape:", decoded.shape)
