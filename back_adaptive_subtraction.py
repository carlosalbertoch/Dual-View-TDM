import numpy as np
def apply_adaptive_background_subtraction(stack, frame_index, alpha=0.05, threshold=0.01):
 
    if frame_index == 0:

        return stack[frame_index]

    background = (1 - alpha) * stack[:frame_index].mean(axis=0) + alpha * stack[frame_index - 1]


    current_frame = stack[frame_index]
    diff = current_frame - background


    mask = np.abs(diff) > threshold
    result = current_frame * mask.astype(float)

    return result