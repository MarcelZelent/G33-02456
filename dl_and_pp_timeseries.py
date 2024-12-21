#%%
from pathlib import Path
import datetime
import pickle 

import pandas as pd

from trackmanlibdata.tsd.tsd_reader import TSDAzure

from data_management import get_observation_nums, make_dataset_name, get_timeseries_observation_nums

#%%
DATA_DIR = Path("/dtu-compute/02456-p4-e24/data")

TS_CROP_WIDTH = (150, 200)  # Duration pre/post impact to keep [ms]
VR_CROP_WIDTH = (-60, 15)  # Radial velocity limits to crop [m/s]
NFFT = 512
N_OBS = 70

datasetname = make_dataset_name(nfft=NFFT, ts_crop_width=TS_CROP_WIDTH, vr_crop_width=VR_CROP_WIDTH)
dest_dat_dir = DATA_DIR / datasetname

#%%

sas_str_path = "/dtu-compute/maalhe/tm/sas_str.txt"
with open(sas_str_path, 'r') as file:
    sas_connection_string = file.read().rstrip()
# %%
print(sas_connection_string)
# %%

data_train_dir = dest_dat_dir / "train"
data_test_dir = dest_dat_dir / "test"


#%%
stmf_data = pd.read_csv("/zhome/c5/7/117975/projects/StudentProject-PINNForVelocityEstimation/data/stmf_data_new 4.csv", sep=";")

train_obs_nos = get_observation_nums(data_train_dir)
train_ts_obs_nos = get_timeseries_observation_nums(data_train_dir)
stmf_train = stmf_data.iloc[train_obs_nos]

test_obs_nos = get_observation_nums(data_test_dir)
test_ts_obs_nos = get_timeseries_observation_nums(data_test_dir)
stmf_test = stmf_data.iloc[test_obs_nos]
# %%
stmfs = (stmf_train, stmf_test)
dirs = (data_train_dir, data_test_dir)
obs_ts_noses = (train_ts_obs_nos, test_ts_obs_nos)
for stmf, dir_, obs_ts_nos in zip(stmfs, dirs, obs_ts_noses):
    k = 0
    for row_no, stmf_row in stmf.iterrows():
        k += 1

        if k >= N_OBS:
            break

        # Ignore existing files
        outpath = dir_ / f"{row_no}_timeseries.pkl"
        if outpath.exists() or row_no in obs_ts_nos:
            continue


        impact_time = datetime.datetime.strptime(
            stmf_row.ImpactTime, "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        tsd = TSDAzure.from_blob_url(stmf_row.TSDFile, sas_connection_string)


        start_time = impact_time + datetime.timedelta(milliseconds=TS_CROP_WIDTH[0])
        end_time = impact_time + datetime.timedelta(milliseconds=TS_CROP_WIDTH[1])

        # t_start and t_end are not the exact times as start_time and end_time as we
        # read the samples in blocks, we get the first block closest to start_time
        # and the last block closest to end_time, the start and end times of those
        # blocks are t_start and t_end.
        # It seems like some indices for temporal cropping are out of range;
        # let's catch, log, and ignore them for now.
        samples, _, _, _ = tsd.get_complex_data(start_time, end_time)
        
        # NOTE: `samples` returned from `tsd` as a list of length 4, corresponding
        # to the four receivers: It is assumed that the order corresponds to RX0, RX1, RX2, RX3.
        # Similarly, `f0_tuple` returned from tsd is a list of length 2, corresponding
        # to the two transmitters: It is assumed that the order corresponds to TX0, TX1:
        # I.e. f0 is the centre frequency of TX0, and f1 is the centre frequency of TX1.
        sample_rate = tsd.get_sample_rate()
        f0_tuple = tsd.get_frequency_list()

        data_dict = {
            "samples": samples,
            "sample_rate": sample_rate,
            "f0_tuple": f0_tuple
        }

        with open(outpath, 'wb') as f:
            pickle.dump(data_dict, f)
# %%
