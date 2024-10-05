"""
composite tokenizer mimi + word level whisper into mistral text tokens
coarse to fine [text (5hz) -> semantic tokens (12.5hz) -> acoustic tokens (87.5hz)].
total = 105hz
"""

import torch
import torchaudio as ta
import huggingface_hub as hf_hub
import typing
import dotenv

from .word_level_whisper import TimedWhisperTokenizer
from .mimi.models import loaders

dotenv.load_dotenv()


class VoxtralTokenizerConfig(typing.NamedTuple):
    mimi_path: str = "kyutai/mimi"
    whisper_path: str = "openai/whisper-tiny.en"
    text_hz: int = 5
    mimi_num_quantizers: int = 8


def interleave(*seqs: list[torch.Tensor]):
    assert isinstance(seqs[0], torch.Tensor)
    bs = seqs[0].size(0)
    factors = [2**i for i in range(len(seqs))]

    to_cat = []
    for i, seq in enumerate(seqs):
        assert isinstance(seq, torch.Tensor)
        seq = seq.view(bs, -1, factors[i])
        to_cat.append(seq)

    out = torch.cat(to_cat, dim=-1)
    return out.view(bs, -1)


def uninterleave(x: torch.Tensor, n: int) -> list[torch.Tensor]:
    factors = [2**i for i in range(n)]
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

    def encode(self, x: torch.Tensor, sample_rate: int) -> torch.Tensor:
        assert x.dim() == 3
        assert x.size(1) == 1
        assert sample_rate == 24_000
        x_for_whisper = ta.functional.resample(x, sample_rate, 16_000)[:, 0]

        text_tokens = self.whisper(x_for_whisper, sample_rate=16_000)

        audio_tokens = self.mimi.encode(x)

        breakpoint()
        return audio_tokens

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


if __name__ == "__main__":
    # Initialize the VoxtralTokenizer
    config = VoxtralTokenizerConfig()
    tokenizer = VoxtralTokenizer(config)

    # Create a dummy audio input tensor
    dummy_audio = torch.randn(1, 16000)  # Assuming 1 second of audio at 16kHz

    # Encode the audio
    encoded = tokenizer.encode(dummy_audio)
    print("Encoded shape:", encoded.shape)

    # decoded = tokenizer.decode(encoded)
    #
    #
