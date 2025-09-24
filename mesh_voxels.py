"""
==========================================================
3D Segmentation and Export to GLB
==========================================================

This script processes a 3D volume (TIFF stack) of refractive index values,
segments it into multiple regions based on user-defined thresholds, generates
3D meshes for the selected regions using Marching Cubes, assigns colors and
materials to each region, and exports the result as a GLB 3D model.

Main steps:
1. Load a TIFF volume.
2. Apply a new segmentation using threshold values.
3. Create 3D meshes for selected regions.
4. Build a scene with different materials per region.
5. Export the scene to a GLB file.

Dependencies:
- numpy, tifffile, trimesh, tqdm, matplotlib
"""

import numpy as np
import tifffile as tiff
import trimesh
from trimesh.voxel import ops as voxel_ops
from tqdm import tqdm
import matplotlib.pyplot as plt
from trimesh.visual.material import PBRMaterial

# ===== CONFIGURATION =====
# Input and output paths
INPUT_FILE = "imagenes/alargadas/Segmented_fused_Full_Alagada500_indice.tif"  # Input TIFF stack
OUTPUT_GLb = "imagenes/alargadas/3DSegmented_fused_Full_Alagada500_indice.glb"  # Output GLB file


# Threshold definition, example: 7 values to generate 6 regions 
UMBRAL_VALUES = [1.449, 1.465, 1.492, 1.512, 1.520, 1.540, 1.55]

# Define colors and alpha for each region (RGBA format, 0–255)
REGION_COLORS = {
    1: (0, 0, 0, 255),        # Black region
    2: (102, 102, 0, 255),    # Olive region
    3: (192, 192, 192, 255),  # Dark gray region
    4: (255, 255, 0, 255),    # Yellow region
    5: (255, 0, 0, 255),      # Red region
    6: (0, 0, 255, 255),      # Blue region
}

# ===== REGION SELECTION =====
# Specify which regions to INCLUDE in the GLB file 
# Example: exclude region 3 (dark gray because it is background) by listing only the others
REGIONES_A_INCLUIR = [1, 2, 4, 5, 6]  # Excludes region 3

# Example alternatives:
# REGIONES_A_INCLUIR = [1, 2, 3, 4, 5, 6]  # Include all regions
# REGIONES_A_INCLUIR = [1, 4, 5]           # Only black, yellow, and red regions
# REGIONES_A_INCLUIR = [2, 3, 6]           # Only olive, gray, and blue regions

# ===== FUNCTIONS =====

def load_volume(file_path):
    """
    Load the volume (TIFF stack) and return it as a 3D numpy array.
    """
    try:
        volume = tiff.imread(file_path)
        print(f"Volume loaded: shape {volume.shape}, dtype: {volume.dtype}")
        return np.array(volume)
    except Exception as e:
        raise ValueError(f"Could not load TIFF file: {e}")

def segment_volume(volume, thresholds):

    segmentation = np.zeros(volume.shape, dtype=np.int32)
    # Assume thresholds are already sorted in ascending order.
    for i in range(len(thresholds) - 1):
        lower = thresholds[i]
        upper = thresholds[i+1]
        # For the last region, include the upper limit
        if i == len(thresholds) - 2:
            mask = (volume >= lower) & (volume <= upper)
        else:
            mask = (volume >= lower) & (volume < upper)
        segmentation[mask] = i + 1
    return segmentation

def create_meshes_from_segmentation(segmentation, region_colors, regiones_a_incluir):

    meshes = {}
    print(f"Selected regions to include: {regiones_a_incluir}")
    
    for region in sorted(regiones_a_incluir):
        if region not in region_colors:
            print(f"Warning: Region {region} has no defined color. Skipping.")
            continue
            
        color = region_colors[region]
        mask = (segmentation == region)
        if np.count_nonzero(mask) == 0:
            print(f"No voxels found for region {region}")
            continue
        try:
            # Generate mesh using Marching Cubes
            mesh_region = voxel_ops.matrix_to_marching_cubes(mask)
            # Create and assign PBR material with a unique name
            material = PBRMaterial(
                name=f"region_{region}",
                baseColorFactor=np.array([color[0]/255.0, color[1]/255.0, color[2]/255.0, color[3]/255.0])
            )
            mesh_region.visual.material = material
            meshes[region] = mesh_region
            print(f"✓ Region {region}: {len(mesh_region.vertices)} vertices, {len(mesh_region.faces)} faces")
        except Exception as e:
            print(f"Error generating mesh for region {region}: {e}")
    
    # Show which regions were excluded
    all_regions = set(region_colors.keys())
    excluded_regions = all_regions - set(regiones_a_incluir)
    if excluded_regions:
        print(f"✗ Regions excluded from GLB: {sorted(excluded_regions)}")
    
    return meshes

def create_scene_from_meshes(mesh_dict):

    scene = trimesh.Scene()
    for region, mesh in mesh_dict.items():
        scene.add_geometry(mesh, node_name=f"region_{region}")
    return scene

def export_scene_to_glb(scene, file_path):
    """
    Export the trimesh scene to a GLB file.
    """
    try:
        scene.export(file_path)
        print(f"Scene exported to: {file_path}")
    except Exception as e:
        raise ValueError(f"Error exporting scene: {e}")

# ===== MAIN SCRIPT =====

def main():
    # 1. Load the volume
    volume = load_volume(INPUT_FILE)
    
    # 2. Segment the volume using the defined threshold intervals
    segmentation = segment_volume(volume, UMBRAL_VALUES)
    
    # (Optional) Visualize a middle slice of the segmentation
    slice_idx = volume.shape[0] // 2
    plt.figure(figsize=(6, 6))
    plt.imshow(segmentation[slice_idx], cmap='jet')
    plt.title("Segmentation - Middle Slice")
    plt.colorbar()
    plt.show()
    
    # 3. Generate meshes only for the selected regions
    print("Generating meshes for selected regions...")
    mesh_dict = create_meshes_from_segmentation(segmentation, REGION_COLORS, REGIONES_A_INCLUIR)
    
    if not mesh_dict:
        print("No meshes generated. Check segmentation, thresholds, and selected regions.")
        return
    
    # 4. Create a scene from the meshes, keeping each region as a separate node
    scene = create_scene_from_meshes(mesh_dict)
    
    # 5. Export the scene to a GLB file
    export_scene_to_glb(scene, OUTPUT_GLb)

if __name__ == '__main__':
    main()
