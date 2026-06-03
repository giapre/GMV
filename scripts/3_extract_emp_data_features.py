from paths import Paths
import numpy as np
import pandas as pd
import glob
import os
from analysis_utils import compute_features, fcd_variance_excluding_overlap
from utils import prepare_fs_default
from plot_utils import plot_signal_and_matrices

type_of_filter = 'aCompCor'

DERIVATIVES_DIR = "/data3/VBT_SCZ/derivatives/postprocessing"
pids = [id for id in os.listdir(DERIVATIVES_DIR) if id.startswith('sub-') and len(id.split('-')[1]) == 4]
print(pids)
ses='run-01'

for pid in pids:#in os.listdir(Paths.DERIVATIVES):
    pid_dir = os.path.join(DERIVATIVES_DIR, pid)
    matches = glob.glob(os.path.join(pid_dir, f"*_{type_of_filter}_filtered_bold.npz"))

    if not matches:
        # no file → skip
        print(f"Skipping {pid}: no filtered bold file")
        continue

    empirical_data = np.load(matches[0])
    empirical_bold = empirical_data['bold']
    print(f'Bold has shape {empirical_bold.shape}')
    empirical_labels = list(empirical_data['labels'])
    empirical_tr = empirical_data['TimeRepetition']
    # fMRIPrep includes background label (0) but no signal so I remove it
    if empirical_labels[0] == 0:
        empirical_labels.pop(0)

    assert len(empirical_labels) == empirical_bold.shape[1]
    lut = pd.read_csv(
        f"{Paths.RESOURCES}/FreeSurferColorLUT.txt",
        sep=r"\s+",
        comment="#",
        names=["No", "Region", "R", "G", "B", "A"],
        index_col="No",
    )

    # Load fs_default (target order)
    fs = prepare_fs_default()     
    fs_regions = fs["Region"].tolist()
    # Map fMRIPrep labels to regions
    # LUT rows corresponding to fMRIPrep labels
    empirical_lut = lut.loc[empirical_labels]
    # Build column index mapping, label number to column index in empirical_bold
    label_to_col = {label: i for i, label in enumerate(empirical_labels)}

    ordered_cols = []
    ordered_regions = []

    for region in fs_regions:
        rows = empirical_lut[empirical_lut["Region"] == region]

        if len(rows) == 0:
            continue  # region not present in empirical BOLD

        label_no = rows.index[0]              # FreeSurfer label number
        col_idx = label_to_col[label_no]      # column in empirical_bold

        ordered_cols.append(col_idx)
        ordered_regions.append(region)

    empirical_bold_to_keep = empirical_bold[:, ordered_cols]
    print(f'Now bold has shape {empirical_bold_to_keep.shape}')
    dt = empirical_tr * 1000
    window_size = int(20 // empirical_tr)
    overlap = window_size - 1
    emp_fc_ut, emp_fcd_ut, emp_zscored_ALFF, emp_fALFF = compute_features(empirical_bold_to_keep[:,:,None], dt, window_size, overlap)

    triu_idx = np.triu_indices(emp_fc_ut.shape[0], k=1)
    sim_gbc = np.mean(emp_fc_ut[triu_idx])
    window_length = int(20//empirical_tr)
    overlap = window_length - 1
    sim_var_fcd = fcd_variance_excluding_overlap(emp_fcd_ut, window_length=window_length, overlap=overlap)
    plot_signal_and_matrices(pid, ses, type_of_filter, empirical_bold_to_keep, emp_fcd_ut[:,:,0], sim_var_fcd[0], emp_fc_ut[:,:,0], sim_gbc, DERIVATIVES_DIR)

    output_dir = os.path.join(Paths.RESULTS, pid)
    os.makedirs(output_dir, exist_ok=True)
    output_name = f'{output_dir}/{type_of_filter}_full_emp_results.npz'

    np.savez(output_name, 
            FC=emp_fc_ut,
            FCD=emp_fcd_ut,
            ALFF=emp_zscored_ALFF,
            fALFF=emp_fALFF,
            ordered_regions=ordered_regions,
            time_repetition=empirical_tr)