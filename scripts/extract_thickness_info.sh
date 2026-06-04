#!/bin/bash

export SUBJECTS_DIR=/mnt/nasvep/SCZ_CHINA/paula/derivatives/derivatives/freesurfer

OUT_BASE=/mnt/nasvep/SCZ_CHINA/paula/derivatives/derivatives/tct_pipeline

for subj_path in $SUBJECTS_DIR/sub-*; do
    subj=$(basename "$subj_path")

    echo "Processing $subj"

    outdir=/data3/VBT_SCZ/GMV/data/derivatives/$subj
    mkdir -p "$outdir"

    # LEFT hemisphere (LThickness + regions)
    aparcstats2table \
        --subjects "$subj" \
        --hemi lh \
        --meas thickness \
        --tablefile "$outdir/lh_thickness.csv"

    # RIGHT hemisphere (RThickness + regions)
    aparcstats2table \
        --subjects "$subj" \
        --hemi rh \
        --meas thickness \
        --tablefile "$outdir/rh_thickness.csv"

done