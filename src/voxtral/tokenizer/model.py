"""
composite tokenizer mimi + word level whisper into mistral text tokens
coarse to fine [text (5hz) -> semantic tokens (12.5hz) -> acoustic tokens (87.5hz)].
total = 105hz
"""

import torch
import huggingface_hub as hf_hub


class VoxtralTokenizer(torch.nn.Module):
    def __init__(self, mimi_path: str = "kyutai/mimi" ** kwargs):
        super().__init__()
        pass

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randint(0, 512, (x.size(0), 128))


mimi_weight = hf_hub.hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
mimi = loaders.get_mimi(mimi_weight, device="cpu")
mimi.set_num_codebooks(8)  # up to 32 for mimi, but limited to 8 for moshi.
