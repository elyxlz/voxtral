"""
composite tokenizer mimi + word level whisper into mistral text tokens
coarse to fine [text (5hz) -> semantic tokens (12.5hz) -> acoustic tokens (87.5hz)].
total = 105hz
"""

import torch


class MimistralTokenizer(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randint(0, 512, (x.size(0), 128))
