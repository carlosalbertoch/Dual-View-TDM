import numpy as np
import tifffile
from tqdm import tqdm
from utils.back_fusion_methods import fusion_grad_abs_whole_plane_segmented

def main():
    # Refractive index input files
    INPUT_IDX_1 = "images/Full_diatom_T1_indice.tif"
    INPUT_IDX_2 = "images/Full_diatom_T2_indice_moved.tif"
    # Absorption input files
    INPUT_ABS_1 = "images/diatom_T2_absorption.tif"
    INPUT_ABS_2 = "images/Full_diatom_T2_absorption_moved.tif"

    # Output files
    OUTPUT_IDX_FUSED = "images/fused_Full_diatom_indice.tif"
    OUTPUT_ABS_FUSED = "images/fused_Full_diatom_absorption.tif"

    # Fusion parameters
    alpha = 0.05
    threshold = 0.01

    # Read refractive index stacks
    stack_idx1 = tifffile.imread(INPUT_IDX_1)
    stack_idx2 = tifffile.imread(INPUT_IDX_2)

    if stack_idx1.shape != stack_idx2.shape:
        raise ValueError("Incompatible dimensions in refractive index stacks.")

    # Fuse refractive indices (also retrieves plane position map)
    fused_idx, positions_plane = fusion_grad_abs_whole_plane_segmented(
        stack_idx1, stack_idx2,
        alpha=alpha, threshold=threshold
    )

    # Save fused refractive index stack
    tifffile.imwrite(OUTPUT_IDX_FUSED, fused_idx.astype(np.float32))
    print(f"Index fusion completed: {OUTPUT_IDX_FUSED}")

    # Read absorption stacks
    stack_abs1 = tifffile.imread(INPUT_ABS_1)
    stack_abs2 = tifffile.imread(INPUT_ABS_2)

    if stack_abs1.shape != stack_abs2.shape:
        raise ValueError("Incompatible dimensions in absorption stacks.")
    if stack_abs1.shape != positions_plane.shape:
        raise ValueError("positions_plane shape does not match absorption stack shape.")

    # Create fused absorption stack according to positions_plane
    fused_abs = np.zeros_like(stack_abs1, dtype=np.float32)
    Z = positions_plane.shape[0]

    for z in range(Z):
        if positions_plane[z].any() and not (positions_plane[z] == 0).all():
            # If positions = 1, use stack_abs2; if positions = 0, use stack_abs1
            # positions_plane[z] is uniform per plane (0 or 1)
            val = positions_plane[z, 0, 0]
        else:
            # Default: take value from the first position
            val = positions_plane[z, 0, 0]

        fused_abs[z] = stack_abs2[z] if val == 1 else stack_abs1[z]

    # Save fused absorption stack
    tifffile.imwrite(OUTPUT_ABS_FUSED, fused_abs)
    print(f"Absorption fusion completed: {OUTPUT_ABS_FUSED}")

if __name__ == "__main__":
    main()
