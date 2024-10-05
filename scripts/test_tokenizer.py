import torch
from voxtral.tokenizer.model import VoxtralTokenizerConfig, VoxtralTokenizer

config = VoxtralTokenizerConfig()
tokenizer = VoxtralTokenizer(config)

# Create a dummy audio input tensor
dummy_audio = torch.randn(1, 240_000)  # Assuming 1 second of audio at 16kHz

# Encode the audio
encoded = tokenizer.encode(dummy_audio, 24_000)
print("Encoded shape:", encoded.shape)

# decoded = tokenizer.decode(encoded)
