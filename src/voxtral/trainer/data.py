import os
import random
import typing

import numpy as np
import torch
import torch.utils.data as td

from .config import VoxtralTrainConfig


def get_npz_files(path: str) -> list[str]:
    npz_files: list[str] = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".npz"):
                npz_files.append(os.path.join(root, file))
    return npz_files


def get_fake_item() -> dict[str, torch.Tensor]:
    return {"tokens": torch.randint(0, 1000, (100,))}


def get_item(file_path: str) -> dict[str, torch.Tensor]:
    try:
        npz_data = np.load(file_path)
        item: dict[str, torch.Tensor] = {}

        print(item)
        for key, value in npz_data.items():
            item[key] = torch.from_numpy(value)

        if "tokens" not in item:
            raise ValueError(f"NPZ file {file_path} does not contain 'tokens' key")

        if item["tokens"].dim() == 2:
            item["tokens"] = item["tokens"].squeeze()

        return item
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        # Generate a fake item as a fallback
        return get_fake_item()


class VoxtralDataset(td.IterableDataset):
    config: VoxtralTrainConfig
    data_step: int
    rank: int
    world_size: int
    file_paths: list[str]

    def __init__(self, config: VoxtralTrainConfig) -> None:
        super().__init__()
        self.config = config
        self.data_step = 0
        self.rank = config.rank
        self.world_size = config.world_size
        self.fake = config.fake

        if self.fake:
            self.file_paths = []
        else:
            self.file_paths = get_npz_files(config.data_path)

        if not self.fake:
            random.seed(config.seed)
            random.shuffle(self.file_paths)
            print(f"Total number of NPZ files: {len(self.file_paths)}")

    def __len__(self) -> int:
        if self.fake:
            return 100_000
        else:
            assert len(self.file_paths) > 0
            return len(self.file_paths)

    def __iter__(self) -> typing.Iterator[dict[str, torch.Tensor]]:
        worker_info = td.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        stride = num_workers * self.world_size
        offset = self.rank * num_workers + worker_id

        while True:
            self.data_step += stride
            idx = (offset + self.data_step) % len(self)
            if self.fake:
                yield get_fake_item()
            else:
                yield get_item(self.file_paths[idx])