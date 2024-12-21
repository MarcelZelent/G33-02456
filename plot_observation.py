#%%
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from torch import Tensor
from typing import Union

from custom_transforms import _to_tensor

TS_CROPTWIDTH = (-150, 200)
VR_CROPTWIDTH = (-60, 15)

def plot_spectrogram_with_annotations(spectrogram: Union[Tensor, str, Path, np.ndarray],
                                      target_vr: float,
                                      estimated_vr:float,
                                      spectrogram_channel: int = 0) -> None:
    """Plot spectrogram with true and predicted vr annotation

    Args:
        spectrogram (Tensor | str | Path | np.ndarray): Input spectrogram to plot.
        target_vr (float): True radial ball velocity.
        estimated_vr (float): Predicted radial ball velocity.
        spectrogram_channel (int, optional): Which of the six spectrogram channels
            to plot. Defaults to 0.
    """

    if isinstance(spectrogram, str):
        spectrogram = Path(spectrogram)

    if isinstance(spectrogram, Path):
        spectrogram = np.load(spectrogram)
    
    if isinstance(spectrogram, np.ndarray):
        spectrogram = _to_tensor(spectrogram) 
    
    if isinstance(spectrogram, Tensor):
        spectrogram = spectrogram.squeeze()

    spectrogram = spectrogram[spectrogram_channel, :, :]

    if spectrogram_channel < 4:
        if spectrogram_channel < 0:
            raise IndexError("Channel number must be between 0 and 5")
        vmin = -110
        vmax = -40
    else:
        if spectrogram_channel > 5:
            raise IndexError("Channel number must be between 0 and 5")
        vmin = -np.pi
        vmax = np.pi

    _, ax = plt.subplots(1, 1)
    ax.imshow(spectrogram, aspect="auto", 
        extent=[TS_CROPTWIDTH[0]/1000,TS_CROPTWIDTH[1]/1000,
                VR_CROPTWIDTH[0],VR_CROPTWIDTH[1]],
        vmin=vmin, vmax=vmax,
        origin="lower",
        interpolation='nearest',
        cmap="jet")
    ax.set_ylabel("radial velocity m/s")
    ax.set_xlabel("time [s]")
    ax.plot([TS_CROPTWIDTH[0]/1000, TS_CROPTWIDTH[1]/1000], [target_vr, target_vr], 'w--')
    ax.plot([TS_CROPTWIDTH[0]/1000, TS_CROPTWIDTH[1]/1000], [estimated_vr, estimated_vr], 'w:')
    ax.legend([ r"True $v_{r}$", r"Pred. $\bar{v}_{r}$"])
    plt.show()

#%%
if __name__ == "__main__":
    obs_no = 241518
    targets = pd.read_csv("/dtu-compute/02456-p4-e24/data/stmf_data_3.csv")
    vr = targets.BallVr.iloc[obs_no]

    # Plot by reading file as str
    fname = f"/dtu-compute/02456-p4-e24/data/data_fft-512_tscropwidth-150-200_vrcropwidth-60-15/train/{obs_no}_stacked_spectrograms.npy"
    plot_spectrogram_with_annotations(fname, target_vr=vr, estimated_vr=-22)

    # Plot by reading file as Path
    pathname = Path(fname)
    plot_spectrogram_with_annotations(pathname, target_vr=vr, estimated_vr=-22)

    # Plot from numpy array
    spectrogram = np.load(pathname)
    plot_spectrogram_with_annotations(spectrogram, target_vr=vr, estimated_vr=-22)

    # Plot from tensor
    spectrogram = _to_tensor(spectrogram)
    plot_spectrogram_with_annotations(spectrogram, target_vr=vr, estimated_vr=-22, spectrogram_channel=4)


# %%
