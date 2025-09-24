import numpy as np
import tifffile

def corregir_offset_index(
    input_path: str,
    output_path: str,
    offset_index: float
):
    """
    Reads a TIFF stack, adds `offset_index` to each pixel,
    and saves it again in float32 format (ImageJ compatible).
    """
    # 1) Read the stack (e.g., fused_Full_Alagada500_indice.tif)
    stack = tifffile.imread(input_path).astype(np.float32)

    # 2) Apply the correction (convert to float64 for safety during addition)
    corrected = stack.astype(np.float64) + offset_index

    # 3) Save back in float32 (without imagej=True to avoid 'd' error)
    tifffile.imwrite(
        output_path,
        corrected.astype(np.float32)
    )
    print(f"Offset index corrected and saved to: {output_path}")


if __name__ == "__main__":
    INPUT_IDX_FUSED  = "images/fused_Full_diatom_indice.tif"
    OUTPUT_IDX_CORR  = "images/fused_Full_diatom_indice.tif"
    OFFSET_INDEX     = 1.5

    corregir_offset_index(
        INPUT_IDX_FUSED,
        OUTPUT_IDX_CORR,
        OFFSET_INDEX
    )
