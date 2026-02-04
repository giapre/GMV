export SUBJECTS_DIR=/data3/VBT_SCZ/derivatives/freesurfer

aparcstats2table --subjects sub-* \
--hemi rh \
--meas thickness \
--parc aparc \
--tablefile ~/gray_matter_data_right.csv