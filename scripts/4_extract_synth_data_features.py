from scripts.utils import compact_bold_results
from scripts.analysis_utils import compute_features
import pandas as pd
import numpy as np
import os
from paths import Paths

wes = np.round(np.linspace(0,1,10),2)
means = np.round(np.linspace(0,50,10),2)
stds = np.round(np.linspace(0,5,10),2)

input_dir = f'{Paths.RESULTS}'

for pid in ['sub-2019052302']:#os.listdir(input_dir):

    if not pid.startswith("sub-"):
        continue

    print(f'Doing patient {pid}')

    pid_path = os.path.join(input_dir, pid)

    outfile = compact_bold_results(pid_path, means, stds, wes)
    data = np.load(outfile)
    bold_all = data["bold"]
    bold_all = bold_all[:,:84,:] # removing midbrain structures (SN, RF, VTA)
    params = data["params"]

    fc_ut, fcd_ut, zscored_ALFF, fALFF = compute_features(bold_all, 1000, 60, 59)

    ## SAVE THE REUSLTS
    output_name = f'{pid_path}/sim_results.npz'

    np.savez(output_name, 
                FC=fc_ut,
                FCD=fcd_ut,
                ALFF=zscored_ALFF,
                fALFF=fALFF,
                params=params)

    assert os.path.exists(output_name), "Save failed!"
    print(f'Data features from simulations saved at {output_name}')