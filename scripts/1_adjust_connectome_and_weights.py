import os
import numpy as np
from utils import adjust_dopamine_connectome, adjust_serotonine_connectome, adjust_serotonin_lengths

base_dir = "/data3/VBT_SCZ/derivatives/freesurfer"
output_base = os.path.abspath("../data/derivatives")
atlas = 'dk'

# Loop over subject folders
for subj in os.listdir(base_dir):

    if not subj.startswith("sub-"):
        continue

    subj_path = os.path.join(base_dir, subj, "dwi")

    if not os.path.isdir(subj_path):
        print(f"Skipping {subj}: no dwi folder")
        continue

    lengths_file = os.path.join(subj_path, "lengths.txt")
    weights_file = os.path.join(subj_path, "dk_weights.txt")

    if not (os.path.exists(lengths_file) and os.path.exists(weights_file)):
        print(f"Skipping {subj}: missing files")
        continue

    print(f"Processing {subj}")

    save_dir = os.path.join(output_base, subj)
    os.makedirs(save_dir, exist_ok=True)
    dopa_weights_df = adjust_dopamine_connectome(subj, weights_file, atlas)
    dopa_weights_df_file = os.path.join(save_dir, f'{atlas}_weights_with_dopa.csv')
    dopa_weights_df.to_csv(dopa_weights_df_file)
    sero_weights_df = adjust_serotonine_connectome(subj, dopa_weights_df_file, atlas)
    sero_weights_df.to_csv(os.path.join(save_dir, f'{atlas}_weights_with_sero_and_dopa.csv'))

    lengths_df = adjust_serotonin_lengths(subj, lengths_file, atlas)
    lengths_df.to_csv(os.path.join(save_dir, f'{atlas}_lengths_with_sero_and_dopa.csv'))

    assert lengths_df.shape == sero_weights_df.shape
