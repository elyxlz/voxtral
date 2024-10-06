import numpy as np
import torchaudio as ta
import torch
from voxtral.tokenizer.model import VoxtralTokenizer, VoxtralTokenizerConfig

arr = np.load("./data/tokens/cb/cb66b29b-b3a7-a5ed-ed54-9d8a4a41f3f6_0.npy")
arr = torch.from_numpy(arr).unsqueeze(0)

tok = VoxtralTokenizer(VoxtralTokenizerConfig())

audio = tok.decode(arr)
ta.save("recon.wav", audio[0], 24_000)

breakpoint()
