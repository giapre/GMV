import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyvista as pv
import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(PROJECT_DIR)
from scripts.utils import prepare_lut, rename_to_fs_lut_region

def dk_plot_annotation_surface(
    title,
    df_annot_dir,
    pat_name,
    data_dir,
    hemisphere,
    colormap
):
    
    """
    Plot cortical surface annotation data from FreeSurfer using PyVista.

    Parameters:
    - data_dir: Path to subjects folder (e.g. ".../Patients/")
    - pat_name: Name of the patient
    - annt_to_plot: name of the annotation to visualize (e.g. "receptor_number", "receptor_density", "GM Vol.")
    - hemisphere: 'lh' or 'rh'
    - colormap: Matplotlib colormap name (e.g. 'plasma', 'cividis')
    """

    # Prepare the df with the annotation to plot
    lut_df = prepare_lut()
    df_annot = pd.read_csv(df_annot_dir)
    df_annot = rename_to_fs_lut_region(df_annot, lut_df)
    df_annot = df_annot.T

    # Load surface and annotation
    annot_file = f"{data_dir}{pat_name}/label/{hemisphere}.aparc.annot"
    surface_file = f"{data_dir}{pat_name}/surf/{hemisphere}.pial"
    labels, ctab, names = nib.freesurfer.read_annot(annot_file)
    coords, faces = nib.freesurfer.io.read_geometry(surface_file)

    # Decode names and build mapping
    region_names = [name.decode('utf-8') for name in names]
    label_to_name = {i: region_names[i] for i in range(len(region_names))}

    # Map each face to a region label
    def get_face_region(face, labels):
        face_labels = labels[face]
        valid_labels = [label for label in face_labels if label != -1]
        return max(set(valid_labels), key=valid_labels.count) if valid_labels else -1

    face_region_ids = np.array([get_face_region(face, labels) for face in faces])
    face_region_names = np.array([label_to_name.get(fr, 'Unknown') for fr in face_region_ids])

    # Map values to regions
    label_to_value = dict(zip(df_annot.index, df_annot[0]))
    volumes_of_faces = [label_to_value.get(f'ctx-{hemisphere}-{name}', 0) for name in face_region_names]
    weights = np.array(volumes_of_faces)

    # Create mesh for PyVista
    faces_pyvista = np.hstack([
        np.full((faces.shape[0], 1), 3),  # leading 3 for triangle
        faces
    ]).ravel()
    mesh = pv.PolyData(coords, faces_pyvista)
    mesh.cell_data["Annotation"] = weights

    # Plotting
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=False, scalars="Annotation", cmap=colormap, scalar_bar_args={
        "title": title,
        "vertical": True,
        "title_font_size": 12,
        "label_font_size": 10,
    })
    plotter.add_title(title, font_size=13)
    plotter.show()

# === Script execution ===
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot DK annotation surface values")
    parser.add_argument("--title", type=str, default='zscore_CorticalThickness_male',
                        help="Title of the plot")
    parser.add_argument("--df_annot_dir", type=str, default='/Users/giacomopreti/Desktop/VBT/GMV/data/processed/male_chinese_pat_cortical_thick_centile_results/zscore_CorticalThickness_male.csv',
                        help="The path to the annotation df")
    parser.add_argument("--pat_name", type=str, default='sub-001',
                        help="Subject name (e.g., 'sub-001')")
    parser.add_argument("--data_dir", type=str, default='/Users/giacomopreti/Desktop/VBT/sub-001/processed/',
                        help="Path to the base data directory")
    parser.add_argument("--hemisphere", type=str, default='rh',
                        help="Hemisphere: 'lh' or 'rh'")
    parser.add_argument("--colormap", type=str, default='plasma',
                        help="Colormap to use for visualization")

    args = parser.parse_args()

    dk_plot_annotation_surface(
        title=args.title,
        df_annot_dir=args.df_annot_dir,
        pat_name=args.pat_name,
        data_dir=args.data_dir,
        hemisphere=args.hemisphere,
        colormap=args.colormap
    )