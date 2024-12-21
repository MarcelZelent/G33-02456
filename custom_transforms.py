#%%
# import datetime
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.functional import resize
from torchvision.transforms.functional import InterpolationMode

class LoadSpectrogram(object):
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    def __call__(self, stmf_row) -> np.ndarray:
        obs_no = stmf_row.name
        spectrogram_filepath = self.root_dir / f"{obs_no}_stacked_spectrograms.npy"

        return np.load(spectrogram_filepath)

class NormalizeSpectrogram(object):
    phase_spectrogram_limits = (-np.pi, np.pi)

    def __call__(self, spectrogram: np.ndarray) -> np.ndarray:

        min_val = np.min(spectrogram[:,:,:4])
        max_val = np.max(spectrogram[:,:,:4])

        np.subtract(spectrogram[:, :, :4], min_val, out=spectrogram[:, :, :4])
        np.divide(
            spectrogram[:, :, :4], max_val - min_val, out=spectrogram[:, :, :4]
        )

        np.subtract(
            spectrogram[:, :, 4:], self.phase_spectrogram_limits[0], out=spectrogram[:, :, 4:]
        )
        np.divide(
            spectrogram[:, :, 4:],
            self.phase_spectrogram_limits[1] - self.phase_spectrogram_limits[0],
            out=spectrogram[:, :, 4:],
        )

        return spectrogram

class InterpolateSpectrogram(object):
    def __init__(self, size: tuple[int, int] = (74, 918)) -> None:
        self.size = size

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return resize(img=spectrogram, size=self.size, interpolation=InterpolationMode.NEAREST)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, spectrogram):
        return _to_tensor(spectrogram=spectrogram)


def _load_spectrogram(spectrogram_filepath: Path) -> np.ndarray:
    return np.load(spectrogram_filepath)    

def _to_tensor(spectrogram: np.ndarray) -> torch.Tensor:
    # swap channel axis because
    # numpy image: H x W x C
    # torch image: C x H x W
    spectrogram = spectrogram.transpose((2, 0, 1))
    return torch.from_numpy(spectrogram.astype(np.float32))
