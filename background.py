import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from back_mean_filter import apply_mean_filter
from back_median_filter import apply_median_filter
from back_gaussian_mixture_model import apply_gaussian_mixture
from back_bilateral_filter import apply_bilateral_filter
from back_adaptive_subtraction import apply_adaptive_background_subtraction
from tqdm import tqdm

# Configurable variables
INPUT_FILE = "images/fused_Full_diatom_indice.tif"  # Path to the input TIFF file
OUTPUT_FILE = "images/segfused_Full_diatom_indice.tif"  # Path to the output TIFF file
METHOD = "adaptive_background"  # Method to use:'adaptive_background'
SHOW_IMAGE_INDEX = 306  # Index of the frame in the stack to display for debugging (None to disable)

# Function to load TIFF stack
def load_tiff_stack(file_path):
    try:
        stack = tiff.imread(file_path)  # Use tifffile to read the stack
        return np.array(stack)
    except Exception as e:
        raise ValueError(f"Could not load TIFF file: {e}")

# Function to save processed stack as TIFF
def save_tiff_stack(stack, output_path):
    try:
        tiff.imwrite(output_path, stack, dtype=stack.dtype, photometric='minisblack')
        print(f"Segmented stack saved to: {output_path}")
    except Exception as e:
        raise ValueError(f"Could not save TIFF file: {e}")

# Function to display a single image
def show_image(image, title="Image"):
    # Normalize for visualization
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    plt.figure(figsize=(6, 6))
    plt.imshow(normalized_image, cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.show()
 
def main():
    try:
        # Load the image stack
        stack = load_tiff_stack(INPUT_FILE)
        segmented_stack = []

        # Process each frame of the stack with a progress bar
        for frame_index in tqdm(range(len(stack)), desc="Processing stack"):
            if METHOD == 'mean_filter':
                segmented = apply_mean_filter(stack, frame_index)
            elif METHOD == 'median_filter':
                segmented = apply_median_filter(stack, frame_index)
            elif METHOD == 'gaussian_mixture':
                segmented = apply_gaussian_mixture(stack, frame_index)
            elif METHOD == 'adaptive_background':
                # alpha=0.05, note: recommended threshold ≈0.006 for RI and ≈0.005 for Absorption
                segmented = apply_adaptive_background_subtraction(stack, frame_index, alpha=0.05, threshold=0.008)
            elif METHOD == 'bilateral_filter':
                segmented = apply_bilateral_filter(stack, frame_index, d=5, sigma_color=0.1, sigma_space=15)
            else:
                raise ValueError(f"Unknown method: {METHOD}")
            segmented_stack.append(segmented)

            # Show selected frame for debugging
            if SHOW_IMAGE_INDEX is not None and frame_index == SHOW_IMAGE_INDEX:
                show_image(segmented, title=f"Segmented Image - Frame {frame_index}")

        segmented_stack = np.array(segmented_stack, dtype=stack.dtype)
        
        # Remove the first image from the stack before saving
        if len(segmented_stack) > 1:
            segmented_stack_final = segmented_stack[1:]  # Keep from index 1 to the end
            print(f"Original stack size: {len(segmented_stack)} frames")
            print(f"Final stack size after removing first frame: {len(segmented_stack_final)} frames")
        else:
            print("Warning: Stack has only one frame, cannot remove first frame")
            segmented_stack_final = segmented_stack
        
        save_tiff_stack(segmented_stack_final, OUTPUT_FILE)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()