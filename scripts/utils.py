import numpy as np 
import pandas as pd
import os

PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))
RESOURCES_DIR = os.path.join(PROJECT_DIR, 'resources')

def prepare_lut(resources_dir=RESOURCES_DIR):
    """Read and prepare the freesurfer look up table"""

    fs_default = pd.read_csv(f'{resources_dir}/fs_default.txt', sep='\s+', comment='#')
    fs_default.rename(columns={'Unknown': 'Region', '???': 'Label'}, inplace=True)
    # Remove the two extra thalamic regions
    idx_to_remove = [fs_default[fs_default['Region']=='Left-Thalamus-Proper'].index[0], fs_default[fs_default['Region']=='Right-Thalamus-Proper'].index[0]]
    fs_default.drop(index=idx_to_remove, inplace=True)

    return fs_default

def get_raw_thickness(RAW_DATA_DIR):
    """Read the thickness csv file for the two haemispheres, concatenate them and assures they have the good shape"""
    lgmv = pd.read_csv(f'{RAW_DATA_DIR}/gray_matter_data_left.csv', sep='\t', index_col='lh.aparc.thickness')
    rgmv = pd.read_csv(f'{RAW_DATA_DIR}/gray_matter_data_right.csv', sep='\t', index_col='rh.aparc.thickness')
    gmv = pd.concat([lgmv, rgmv], axis=1)

    assert gmv.shape[1] == 2 * lgmv.shape[1]
    assert gmv.shape[1] == 2 * rgmv.shape[1]

    gmv.rename(columns={'lh_MeanThickness_thickness':'LThickness', 'rh_MeanThickness_thickness':'RThickness'}, inplace=True)
    if gmv.index[0].startswith('sub'):
        gmv.index = gmv.index.str.replace('sub-', '', regex=False)

    return gmv

def adjust_thick_template(demo_and_gmv, template):
    for col in demo_and_gmv.columns:
        for tempcol in template.columns:
            if col.startswith('lh_') & tempcol.startswith('L_'): 
                if col.split('_')[1] == tempcol.split('_')[1]:
                    demo_and_gmv.rename(columns={col: tempcol}, inplace=True)
            elif col.startswith('rh_') & tempcol.startswith('R_'): 
                if col.split('_')[1] == tempcol.split('_')[1]:
                    demo_and_gmv.rename(columns={col: tempcol}, inplace=True)

    demo_and_gmv.reset_index(inplace=True)
    demo_and_gmv.rename(columns={'pid':'SubjectID'}, inplace=True)
    demo_and_gmv.rename(columns={'lh_entorhinal_thickness':'L_entorhil_thickavg', 'lh_supramarginal_thickness':'L_supramargil_thickavg', 'rh_entorhinal_thickness':'R_entorhil_thickavg', 'rh_supramarginal_thickness':'R_supramargil_thickavg'}, inplace=True)
    demo_and_gmv['SITE'] = ['A'] * len(demo_and_gmv)
    demo_and_gmv['Vendor'] = ['na'] * len(demo_and_gmv)
    demo_and_gmv['FreeSurfer_Version'] = ['7.4'] * len(demo_and_gmv)
    
    Males = demo_and_gmv[demo_and_gmv['sex'] == 'Male']
    Males = Males[template.columns]
    Females = demo_and_gmv[demo_and_gmv['sex'] == 'Female']
    Females = Females[template.columns]

    return Males, Females

import pandas as pd

def rename_to_fs_lut_labels(df, lut_df):
    """
    Renames columns from 'L_region_thickavg' format to LUT 'Label' format.
    """
    # 1. Create a helper dictionary from the LUT: { 'ctx-lh-bankssts': 'L.BSTS', ... }
    # We strip the 'ctx-lh-' and 'ctx-rh-' to make matching easier
    lut_map = {}
    for _, row in lut_df.iterrows():
        region_name = str(row['Region'])
        # Standardize FS region names to match your CSV format (L_ or R_)
        clean_reg = region_name.replace('ctx-lh-', 'L_').replace('ctx-rh-', 'R_')
        lut_map[clean_reg] = row['Label']

    new_column_names = {}
    for col in df.columns:
        # Remove the '_thickavg' suffix to get 'L_bankssts'
        base_name = col.replace('_thickavg', '')
        
        # Look up the base name in our LUT map
        if base_name in lut_map:
            new_column_names[col] = lut_map[base_name]
        else:
            # Keep original name if no match is found (e.g., SubID or metadata)
            new_column_names[col] = col

    df.rename(columns=new_column_names, inplace=True)
    df.rename(columns={'L_entorhil_thickavg': 'L.EC', 'R_entorhil_thickavg': 'R.EC',
                       'L_supramargil_thickavg': 'L.SMG', 'R_supramargil_thickavg': 'R.SMG'}, inplace=True)
            
    return df

def rename_to_fs_lut_region(df, lut_df):
    """
    Renames columns from 'L_region_thickavg' format to LUT 'Region' format.
    """
    # 1. Create a helper dictionary from the LUT: { 'ctx-lh-bankssts': 'L.BSTS', ... }
    # We strip the 'ctx-lh-' and 'ctx-rh-' to make matching easier
    lut_map = {}
    for _, row in lut_df.iterrows():
        region_name = str(row['Region'])
        # Standardize FS region names to match your CSV format (L_ or R_)
        clean_reg = region_name.replace('ctx-lh-', 'L_').replace('ctx-rh-', 'R_')
        lut_map[clean_reg] = row['Region']

    new_column_names = {}
    for col in df.columns:
        # Remove the '_thickavg' suffix to get 'L_bankssts'
        base_name = col.replace('_thickavg', '')
        
        # Look up the base name in our LUT map
        if base_name in lut_map:
            new_column_names[col] = lut_map[base_name]
        else:
            # Keep original name if no match is found (e.g., SubID or metadata)
            new_column_names[col] = col

    df.rename(columns=new_column_names, inplace=True)
    df.rename(columns={'L_entorhil_thickavg': 'ctx-lh-entorhinal', 'R_entorhil_thickavg': 'ctx-rh-entorhinal',
                       'L_supramargil_thickavg': 'ctx-lh-supramarginal', 'R_supramargil_thickavg': 'ctx-rh-supramarginal'}, inplace=True)
            
    return df

def dk_extract_gray_matter(regions_lines, stats_folder):
    #regions_lines = np.loadtxt(stats_folder + "fs_default.txt", dtype=str) 
    lh_aparc_lines = np.loadtxt(stats_folder + "lh.aparc.stats", dtype=str) 
    rh_aparc_lines = np.loadtxt(stats_folder + "rh.aparc.stats", dtype=str) 
    aseg_lines = np.loadtxt(stats_folder + "aseg.stats", dtype=str) 
    
    print(f'Number of regions from fs_default: {len(regions_lines)}')
    print(f'Number of regions from aparc: {len(rh_aparc_lines)}')
    print(f'Number of regions from aseg: {len(aseg_lines)}')

    gm_region_volume = []
    for regions_line in regions_lines:
        for aseg_line in aseg_lines:
            if aseg_line[4] in regions_line[2]:
                gm_region_volume.append([regions_line[1], regions_line[2], aseg_line[3]])
                
    for regions_line in regions_lines:
        for lh_aparc_line in lh_aparc_lines:
            if '-lh-' in regions_line[2]:
                if lh_aparc_line[0] == regions_line[2].split('-')[2]:
                    gm_region_volume.append([regions_line[1], regions_line[2], lh_aparc_line[3]])
                    
    for regions_line in regions_lines:            
        for rh_aparc_line in rh_aparc_lines:
            if '-rh-' in regions_line[2]:
                if rh_aparc_line[0] == regions_line[2].split('-')[2]:
                    gm_region_volume.append([regions_line[1], regions_line[2], rh_aparc_line[3], ])
        
    for region in gm_region_volume:
        if "Proper" in region[2]: gm_region_volume.remove(region)

    print(f'Elements in the gray matter volume list: {len(gm_region_volume)}')
    return gm_region_volume