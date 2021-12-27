import glob
import numpy as np
import os
import sys


def load_d(d_num_layers, d_num_price_levels, d_minutes_per_day):
    npy_data_path = os.path.join('data', 'AAPL*.npy')
    files_to_load = sorted(glob.glob(npy_data_path))

    if not files_to_load:
        sys.exit('Files to load not found')

    d_total_minutes = d_minutes_per_day * len(files_to_load)

    d = np.zeros((d_num_layers, d_num_price_levels, d_total_minutes), np.float32)

    load_pointer = 0
    for file in files_to_load:
        d[:, :, load_pointer:load_pointer + d_minutes_per_day] = np.load(file)
        load_pointer += d_minutes_per_day

    return d
