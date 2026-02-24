from synt_analysis_utils import fcd_variance_excluding_overlap
from paths import Paths
from utils import prepare_fs_default
import os
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def minmaxscale(signal):
    smin = signal.min(axis=0)
    smax = signal.max(axis=0)
    return (signal - smin) / (smax - smin + 1e-12)

type_of_extr = 'traditional'
pids_df = []



for pid in os.listdir(Paths.DERIVATIVES):
    pid_dir = os.path.join(Paths.DERIVATIVES, pid)
    matches = glob.glob(os.path.join(pid_dir, f"*{type_of_extr}_filtered_bold.npz"))

    if not matches:
        # no file → skip
        print(f"Skipping {pid}: no filtered bold file")
        continue

    print(f'Doing {pid}')
    feat_file = f"{Paths.RESULTS}/{pid}/{type_of_extr}_full_emp_results.npz"

    bold_file = matches[0]#f"{Paths.DERIVATIVES}/{pid}/{pid}_run-01_{type_of_extr}_filtered_bold.npz"
    bold = np.load(bold_file)['bold']

    data = np.load(feat_file)
    sim_fc = data['FC']
    sim_fcd = data['FCD']
    sim_alff = data['ALFF']
    regions_names = data['ordered_regions']
    tr = data['time_repetition']
    fs = prepare_fs_default()
    region_labels = [fs[fs['Region']==roi]['Label'].to_list()[0] for roi in regions_names]

    triu_idx = np.triu_indices(sim_fc.shape[0], k=1)
    sim_gbc = np.mean(sim_fc[triu_idx])
    window_length = int(60//tr)
    overlap = window_length - 1
    sim_var_fcd = fcd_variance_excluding_overlap(sim_fcd, window_length=window_length, overlap=overlap) #np.var(sim_fcd,axis=0)
    bold = minmaxscale(bold)

    plt.figure(figsize=(12,6))
    plt.subplot(131)
    plt.plot(range(bold.shape[1]) + 3*bold, linewidth=0.5)
    plt.subplot(132)
    plt.imshow(sim_fc)
    plt.title(f'GBC = {sim_gbc}')
    plt.subplot(133)
    plt.imshow(sim_fcd)
    plt.title(f'VAR_FCD = {sim_var_fcd}')

    plt.savefig(f'{Paths.RESULTS}/{pid}/{type_of_extr}.png')
    plt.close()

    