from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle

from data_management import get_observation_nums, get_timeseries_observation_nums

class SpectrogramDataset(Dataset):
    def __init__(self,
                 transform,
                 stmf_data_path: Path,
                 data_dir: Path = None,
                 observation_nums: Iterable = None,
                 csv_delimiter: str = ",") -> None:
        
        if data_dir is None and observation_nums is None:
            raise ValueError("Either `data_dir` or `observation_nums` mus be different from None")
        
        if data_dir is not None:
            observation_nums = get_observation_nums(data_dir)
        self.observation_nums = observation_nums

        self.stmf_data  = pd.read_csv(stmf_data_path, delimiter=csv_delimiter).iloc[observation_nums]
        self.targets = self.stmf_data.BallVr.to_numpy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.targets)
    
    def __getitem__(self, idx: int) -> dict:
        spectrogram, target = self._get_item_helper(idx)
        sample = {"spectrogram": spectrogram, "target": target}

        return sample
    
    def _get_item_helper(self, idx: int) -> tuple:
        stmf_row = self.stmf_data.iloc[idx]

        spectrogram = self.transform(stmf_row)
        target = self.targets[idx]
        target = torch.tensor(target, dtype=torch.float32)

        return spectrogram, target


class TimeSeriesDataset(Dataset):
    def __init__(self, 
                 stmf_data_path: Path,
                 data_dir: Path = None,
                 observation_nums: Iterable = None,
                 csv_delimiter: str = ","):

        if data_dir is None and observation_nums is None:
            raise ValueError("Either `data_dir` or `observation_nums` mus be different from None")

        if data_dir is not None:
            observation_nums = get_timeseries_observation_nums(data_dir)
        self.observation_nums = observation_nums

        self.data_dir = data_dir
        self.stmf_data  = pd.read_csv(stmf_data_path, delimiter=csv_delimiter).iloc[observation_nums]
        self.targets = self.stmf_data.BallVr.to_numpy()

    def __len__(self):
        return len(self.observation_nums)

    def __getitem__(self, idx):
        obs = self.observation_nums[idx]
        pkl_file_name = f"{obs}_timeseries.pkl"
        pkl_file_path = self.data_dir / pkl_file_name

        with open(pkl_file_path, 'rb') as f:
            timeseries_data = pickle.load(f)
        
        samples = timeseries_data["samples"]
        target = self.targets[idx]

        # Separate real and imaginary parts
        real_parts = []
        imag_parts = []
        for sample in samples:
            real_parts.append([s.real for s in sample])
            imag_parts.append([s.imag for s in sample])

        real_part_tensor = torch.tensor(real_parts, dtype=torch.float32)
        imag_part_tensor = torch.tensor(imag_parts, dtype=torch.float32)
        # print(real_part_tensor.shape)
        # print(imag_part_tensor.shape)

        # Stack real and imaginary parts along a new dimension
        samples_tensor = torch.stack((real_part_tensor, imag_part_tensor), dim=-1)
        target_tensor = torch.tensor(target, dtype=torch.float32)

        return samples_tensor, target_tensor
    

def collate_fn(batch):
    samples, targets = zip(*batch)
    
    # Find the maximum sequence length in the batch
    max_length = max(sample.size(1) for sample in samples)
    
    # Pad each sample to the maximum sequence length
    padded_samples = []
    for sample in samples:
        # Pad only along the sequence length dimension (dim=1)
        padding = (0, 0, 0, max_length - sample.size(1))
        padded_sample = torch.nn.functional.pad(sample, padding, "constant", 0)
        padded_samples.append(padded_sample)
    
    # Stack the padded samples and targets
    samples_padded = torch.stack(padded_samples, dim=0)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    
    return samples_padded, targets_tensor
