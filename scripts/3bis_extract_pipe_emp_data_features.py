from synt_analysis_utils import make_roi_alff_df, make_roi_fc_mean_df, make_roi_fc_couples_df, fcd_variance_excluding_overlap
from paths import Paths
from utils import prepare_fs_default
import os
import glob

import pandas as pd
import numpy as np

# Scritp for computing the data features based on the review for the selection of the significant features that can distinguish responders to non responders 
# They are the same as in synthetic patient

type_of_extr = 'aCompCor50'
pids_df = []

for pid in os.listdir(Paths.RESULTS):
    pid_dir = os.path.join(Paths.RESULTS, pid)
    matches = glob.glob(os.path.join(pid_dir, f"{type_of_extr}_full_emp_results.npz"))

    if not matches:
        # no file → skip
        print(f"Skipping {pid}: no filtered bold file")
        continue

    print(f'Doing {pid}')
    feat_file = f"{Paths.RESULTS}/{pid}/{type_of_extr}_full_emp_results.npz"

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
    window_length = int(20//tr)
    overlap = window_length - 1
    sim_var_fcd = fcd_variance_excluding_overlap(sim_fcd, window_length=window_length, overlap=overlap) #np.var(sim_fcd,axis=0)

    sim_df = pd.DataFrame({'pid': pid.split('-')[1], 'GBC': sim_gbc, 'VAR_FCD': sim_var_fcd}) #'we': np.round(params[:,0],4), 'sigma': np.round(params[:,1],4)})#'we': np.round(params[:,0],4), 'wd': np.round(params[:,1],4), 'ws': np.round(params[:,2],4)})

    fc_regions = region_labels#['PU', 'CA', 'HI', 'STG', 'CER', 'CACG', 'RACG', 'IN', 'PCG', 'POP', 'POR', 'PTR']
    h_fc_regions = region_labels#['L.'+region for region in fc_regions] + ['R.'+region for region in fc_regions]

    fc_mean_df = make_roi_fc_mean_df(sim_fc, h_fc_regions, region_labels)
    alff_df = make_roi_alff_df(sim_alff, h_fc_regions, region_labels)

    fc_combinations = [['PU', 'RACG'], 
                    ['PU', 'CACG'],
                    ['PU', 'IN'],
                    ['PU', 'CER'],
                    ['CA', 'RACG'],
                    ['CA', 'CACG'],
                    ['CA', 'IN'],
                    ['CA', 'PCG'],
                    ['CA', 'HI'],
                    ['CA', 'CER'],
                    ['HI', 'IN'],]

    h_fc_combinations = []
    for combination in fc_combinations:
        for hemi in ['L.', 'R.']:
            r0 = hemi+combination[0]
            r1 = hemi+combination[1]
            h_fc_combinations.append([r0, r1])

    fc_couples_df = make_roi_fc_couples_df(sim_fc, h_fc_combinations, region_labels)

    final_df = pd.concat([sim_df, fc_mean_df, fc_couples_df, alff_df], axis=1)
    final_df.to_csv(f'{Paths.RESULTS}/{pid}/{type_of_extr}_extracted_features.csv')
    pids_df.append(final_df)
    pd.concat(pids_df).to_csv(f'{Paths.RESULTS}/ALL_{type_of_extr}_extracted_features.csv')
