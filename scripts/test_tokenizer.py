import torchaudio
from voxtral.tokenizer.model import VoxtralTokenizerConfig, VoxtralTokenizer

audio_path = "./data/chunks/1e/1ed12ab7-53b6-4d61-9f52-9ddddfb0c32e_169.m4a"
waveform, sample_rate = torchaudio.load(audio_path)
waveform = (
    torchaudio.functional.resample(waveform, sample_rate, 24_000)
    .mean(0, keepdim=True)
    .unsqueeze(0)
)

config = VoxtralTokenizerConfig()
tokenizer = VoxtralTokenizer(config)

# Ensure the audio is mono and has the correct shape
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Encode the audio
print(waveform.shape)
encoded = tokenizer.encode(waveform, 24_000)
print("Encoded shape:", encoded.shape)

decoded = tokenizer.decode(encoded)
torchaudio.save("recon.wav", decoded[0], 24_000)
