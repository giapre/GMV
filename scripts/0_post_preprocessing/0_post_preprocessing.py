import os
import sys

import nibabel as nib
from nilearn import image
from nilearn.signal import clean
from nilearn.maskers import NiftiLabelsMasker

import pandas as pd 
import numpy as np 
import json

import matplotlib.pyplot as plt

from postproc_utils import *

# Set-up directories 
DERIVATIVES_DIR = "/data3/VBT_SCZ/derivatives"
pids = [id for id in os.listdir(DERIVATIVES_DIR) if id.startswith('sub-') and len(id.split('-')[1]) == 4]
print(pids)
ses = 'run-01'

for pid in pids:
    print(f"Searching for {pid}_{ses} data")

    json_file = os.path.join(DERIVATIVES_DIR, f"{pid}/{ses}/func/{pid}_task-rest_{ses}_space-T1w_desc-preproc_bold.json") #json file of the raw func image
    if not os.path.exists(json_file):
        print(f'{pid} does not have func processed files')
        continue
    FMRI_DIR = os.path.join(DERIVATIVES_DIR, pid, ses, 'func')
    FS_DIR = os.path.join(DERIVATIVES_DIR, f'freesurfer/{pid}') 
    if not os.path.exists(FS_DIR):
        FS_DIR = os.path.join(DERIVATIVES_DIR, f"freesurfer/{pid}")
    POST_PREPROC_DIR = '/data3/VBT_SCZ/derivatives/postprocessing'

    # Extract relevant info

    tr = get_repetition_time(json_file) # time repetition
    confounds_data = pd.read_csv(f"{FMRI_DIR}/{pid}_task-rest_{ses}_desc-confounds_timeseries.tsv", sep='\t') # confounds
    confounds_json = f"{FMRI_DIR}/{pid}_task-rest_{ses}_desc-confounds_timeseries.json"
    c_comp_cor_top5, w_comp_cor_top5 = get_top5_compcor(confounds_json)
    c_comp_cor_50, w_comp_cor_50 = get_compcor_50pct(confounds_json)
    motion_correction = ['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z']
    conf_dic = {'daniela': motion_correction + ['white_matter', 'csf'],
        'traditional': motion_correction + ['global_signal', 'white_matter', 'csf'],
        'aCompCor': motion_correction + c_comp_cor_top5 + w_comp_cor_top5,
        'aCompCor50': motion_correction + c_comp_cor_50 + w_comp_cor_50
        }

    bold_file = f"{FMRI_DIR}/{pid}_task-rest_{ses}_space-T1w_desc-preproc_bold.nii.gz"
    aparc_file = f"{FS_DIR}/mri/aparc+aseg.nii.gz"

    confound = conf_dic
    # Load images
    bold_img = nib.load(bold_file)
    aparc_img = nib.load(aparc_file)
    masker = NiftiLabelsMasker(
        labels_img=aparc_img,
        standardize=False,
        detrend=True,
        low_pass=0.198,
        high_pass=0.01,
        t_r=tr
    )

    results = []
    emp_bold_path_dict = {}
    for combination in conf_dic:
        selected_confunds_columns = conf_dic[combination]
        selected_confunds = confounds_data[selected_confunds_columns]
        time_series = masker.fit_transform(bold_img, confounds=selected_confunds.values)
        print(f"Bold has shape: {time_series.shape}")

        # print(f"Reshaped BOLD has dimension {time_series[:, idx].shape}")

        filtered_bold = time_series[40:,:] #postproc_utils.bandpass_nilearn(time_series, tr=tr)
        emp_bold_path_dict[combination] = f"{POST_PREPROC_DIR}/{pid}/{pid}_{ses}_{combination}_filtered_bold"
        np.savez(emp_bold_path_dict[combination], bold=filtered_bold, labels=masker.labels_, TimeRepetition=tr)
        fc, gbc, fcd, var_fcd = compute_basic_metrics(filtered_bold, tr)
        #plot_signal_and_matrices(pid, ses, combination, filtered_bold, fcd, var_fcd, fc, gbc, POST_PREPROC_DIR")

    print(f'\n {pid} done! \n')
        #results.append([combination, var_fcd, gbc])

    #df = pd.DataFrame(data=results, columns=['Strategy', 'VAR_FCD', 'GBC',])
    #df.to_csv(f"{pid}_{ses}_basic_metrics.csv", index=False)


