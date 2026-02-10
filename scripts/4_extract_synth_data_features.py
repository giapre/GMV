from scripts.utils import compact_bold_results
from scripts.analysis_utils import compute_features
import pandas as pd
import numpy as np
import os
from paths import Paths

wes = np.round(np.linspace(0,1,10),2)
means = np.round(np.linspace(0,50,10),2)
stds = np.round(np.linspace(0,5,10),2)

input_dir = Paths.RESULTS

## make loop here 
pid = 'sub-1198'
pid_output_dir = f'{input_dir}/{pid}'
outfile = compact_bold_results(pid_output_dir, means, stds, wes)
data = np.load(outfile)
bold_all = data["bold"]
bold_all = bold_all[:,:84,:] # removing midbrain structures (SN, RF, VTA)
params = data["params"]

fc_ut, fcd_ut, zscored_ALFF, fALFF = compute_features(bold_all, 1000, 60, 59)

## SAVE THE REUSLTS
output_dir = os.path.join(Paths.RESULTS, pid)
os.makedirs(output_dir, exist_ok=True)
output_name = f'{output_dir}/sim_results.npz'

np.savez(output_name, 
            FC=fc_ut,
            FCD=fcd_ut,
            ALFF=zscored_ALFF,
            fALFF=fALFF,
            params=params)

assert os.path.exists(output_name), "Save failed!"
print(f'Data features from simulations saved at {output_name}')