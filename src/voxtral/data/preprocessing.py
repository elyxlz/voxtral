import os
import typing
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from voxtral.tokenizer.model import VoxtralTokenizer


class PreprocessingConfig(typing.NamedTuple):
    input_path: str = "data/chunks"
    output_path: str = "data/tokens"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    compile_tokenizer: bool = False
    tokenizer_kwargs: dict = {}
    target_sample_rate: int = 16000
    max_save_workers: int = 4
    use_cuda: bool = torch.cuda.is_available()
    tokenizer_dtype: torch.dtype = torch.float32
    chunk_frames: int = 16000  # New parameter for fixed chunk size
    num_channels: int = 1  # New parameter for number of channels


class AudioChunkDataset(Dataset):
    def __init__(
        self,
        input_path: str,
        target_sample_rate: int,
        chunk_frames: int,
        num_channels: int,
    ):
        self.input_path = input_path
        self.target_sample_rate = target_sample_rate
        self.chunk_frames = chunk_frames
        self.num_channels = num_channels
        self.file_list = []
        self._find_audio_files(input_path)

    def _find_audio_files(self, directory):
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(
                    (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".mp4", ".webm")
                ):
                    self.file_list.append(os.path.join(root, file))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        waveform, sample_rate = torchaudio.load(file_path)

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sample_rate, self.target_sample_rate
            )
            waveform = resampler(waveform)

        # Adjust number of channels
        if waveform.shape[0] < self.num_channels:
            waveform = waveform.repeat(self.num_channels, 1)
        elif waveform.shape[0] > self.num_channels:
            waveform = waveform[: self.num_channels]

        # Pad or cut to chunk_frames
        if waveform.shape[1] < self.chunk_frames:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.chunk_frames - waveform.shape[1])
            )
        elif waveform.shape[1] > self.chunk_frames:
            waveform = waveform[:, : self.chunk_frames]

        return waveform, os.path.basename(self.file_list[idx])


def _create_dataloader(config: PreprocessingConfig) -> DataLoader:
    dataset = AudioChunkDataset(
        config.input_path,
        config.target_sample_rate,
        config.chunk_frames,
        config.num_channels,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )


def _save_tokens(encoded_tokens: np.ndarray, filename: str, output_path: str):
    subdir = filename[:2]
    full_output_path = os.path.join(output_path, subdir)
    os.makedirs(full_output_path, exist_ok=True)
    output_file = os.path.join(full_output_path, f"{os.path.splitext(filename)[0]}.npz")
    np.savez_compressed(output_file, tokens=encoded_tokens)


def preprocess_audio_chunks(config: PreprocessingConfig):
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)

    dataloader = _create_dataloader(config)

    device = torch.device("cuda" if config.use_cuda else "cpu")
    tokenizer = VoxtralTokenizer(**config.tokenizer_kwargs).to(
        device=device, dtype=config.tokenizer_dtype
    )
    if config.compile_tokenizer:
        tokenizer: VoxtralTokenizer = torch.compile(tokenizer)  # type: ignore

    save_executor = ThreadPoolExecutor(max_workers=config.max_save_workers)

    for batch in tqdm(dataloader, desc="Processing audio chunks"):
        waveforms, filenames = batch
        for waveform, filename in zip(waveforms, filenames):
            encoded = tokenizer.encode(waveform)
            save_executor.submit(_save_tokens, encoded, filename, config.output_path)

    save_executor.shutdown(wait=True)


if __name__ == "__main__":
    config = PreprocessingConfig()
    preprocess_audio_chunks(config)