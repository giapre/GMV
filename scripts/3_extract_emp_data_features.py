from paths import Paths
import numpy as np
import pandas as pd
import glob
import os
from scripts.analysis_utils import compute_features
from scripts.utils import prepare_fs_default

for pid in os.listdir(Paths.DERIVATIVES):
    pid_dir = os.path.join(Paths.DERIVATIVES, pid)
    matches = glob.glob(os.path.join(pid_dir, "*_daniela_filtered_bold.npz"))

    if not matches:
        # no file → skip
        print(f"Skipping {pid}: no filtered bold file")
        continue

    empirical_data = np.load(matches[0])
    empirical_bold = empirical_data['bold']
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
    dt = empirical_tr * 1000
    window_size = int(60 // empirical_tr)
    overlap = window_size - 1
    emp_fc_ut, emp_fcd_ut, emp_zscored_ALFF, emp_fALFF = compute_features(empirical_bold_to_keep[:,:,None], dt, window_size, overlap)

    output_dir = os.path.join(Paths.RESULTS, pid)
    os.makedirs(output_dir, exist_ok=True)
    output_name = f'{output_dir}/emp_results.npz'

    np.savez(output_name, 
            FC=emp_fc_ut,
            FCD=emp_fcd_ut,
            ALFF=emp_zscored_ALFF,
            fALFF=emp_fALFF)