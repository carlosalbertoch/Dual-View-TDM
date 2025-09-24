"""
fusion_methods.py
============================================================
This module implements several image fusion methods for 2D slices
and 3D stacks, combining two inputs based on arithmetic mean,
median, Gaussian mixture (global/local), local variance/std,
gradient magnitude, and Gaussian of Differences. It also provides
a stack-level fusion based on adaptive background subtraction
and gradient energy.
"""

import numpy as np
import math
from tqdm import tqdm
from scipy.ndimage import uniform_filter, generic_filter
from scipy.ndimage import gaussian_filter
from utils.back_adaptive_sustract import apply_adaptive_background_subtraction


def fusion_mean(im1, im2):
    """Fusion by arithmetic mean."""
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)
    return (im1 + im2) / 2.0


def fusion_median(im1, im2):
    """Fusion by pixel-wise median (similar to mean for 2 images)."""
    stack = np.stack([im1, im2], axis=0)
    return np.median(stack, axis=0).astype(np.float32)


def fusion_gaussian_mixture(im1, im2):
    """Fusion using global Gaussian mixture (variance-based weighting)."""
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    var1, var2 = np.var(im1), np.var(im2)
    w1, w2 = 1/(var1 + 1e-9), 1/(var2 + 1e-9)
    alpha1, alpha2 = w1/(w1 + w2), w2/(w1 + w2)

    return alpha1*im1 + alpha2*im2


def local_stats(img, kernel_size=15):
    """Compute local mean and variance within a neighborhood."""
    img = img.astype(np.float32)
    c1 = uniform_filter(img, size=kernel_size, mode='reflect')
    c2 = uniform_filter(img*img, size=kernel_size, mode='reflect')
    var_map = np.maximum(c2 - c1*c1, 0)
    return c1, var_map


def fusion_local_gaussian_mixture(im1, im2, kernel_size=15):
    """Fusion using local Gaussian mixture with neighborhood statistics."""
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    mean1, var1 = local_stats(im1, kernel_size)
    mean2, var2 = local_stats(im2, kernel_size)

    sigma1, sigma2 = np.sqrt(var1)+1e-9, np.sqrt(var2)+1e-9
    diff1, diff2 = im1 - mean1, im2 - mean2

    pdf1 = (1.0/(math.sqrt(2*math.pi)*sigma1)) * np.exp(-0.5*(diff1**2)/(sigma1**2))
    pdf2 = (1.0/(math.sqrt(2*math.pi)*sigma2)) * np.exp(-0.5*(diff2**2)/(sigma2**2))

    alpha = pdf1 / (pdf1 + pdf2 + 1e-9)
    return alpha*im1 + (1.0 - alpha)*im2


def local_std(image, kernel_size=(3,3)):
    """Compute local standard deviation using a sliding window."""
    def std_function(data): return np.std(data)
    return generic_filter(image.astype(np.float32), std_function, size=kernel_size)


def fusion_std_decision(im1, im2, kernel_size=(3,3)):
    """Fusion choosing pixels from the image with lower local std."""
    std1, std2 = local_std(im1, kernel_size), local_std(im2, kernel_size)
    fused = np.zeros_like(im1, dtype=np.float32)

    mask_im1, mask_im2 = std1 < std2, std1 >= std2
    fused[mask_im1], fused[mask_im2] = im1[mask_im1], im2[mask_im2]

    return fused


def fusion_grad_abs(im1, im2, kernel_size=(3,3)):
    """Fusion choosing pixels from the image with higher gradient magnitude."""
    grad1 = np.sqrt(np.gradient(im1, axis=1)**2 + np.gradient(im1, axis=0)**2)
    grad2 = np.sqrt(np.gradient(im2, axis=1)**2 + np.gradient(im2, axis=0)**2)

    fused = np.zeros_like(im1, dtype=np.float32)
    mask_im1, mask_im2 = grad1 > grad2, grad1 <= grad2
    fused[mask_im1], fused[mask_im2] = im1[mask_im1], im2[mask_im2]

    return fused


def fusion_grad_abs_whole_plane(im1, im2):
    """Fusion selecting the whole plane with larger total gradient."""
    im1, im2 = im1.astype(np.float32), im2.astype(np.float32)

    sumGrad1 = np.sum(np.sqrt(np.gradient(im1, axis=1)**2 + np.gradient(im1, axis=0)**2))
    sumGrad2 = np.sum(np.sqrt(np.gradient(im2, axis=1)**2 + np.gradient(im2, axis=0)**2))

    if sumGrad1 > sumGrad2:
        fused, positions_plane = im1, np.zeros_like(im1, dtype=np.uint8)
    else:
        fused, positions_plane = im2, np.ones_like(im2, dtype=np.uint8)

    return fused


def fusion_gaussian_of_differences(im1, im2, sigma=1.0):
    """Fusion using Gaussian of Differences (edge-based weighting)."""
    im1, im2 = im1.astype(np.float32), im2.astype(np.float32)

    CD1, RD1, CD2, RD2 = np.zeros_like(im1), np.zeros_like(im1), np.zeros_like(im2), np.zeros_like(im2)
    CD1[:, :-1], RD1[:-1, :] = (im1[:, :-1] - im1[:, 1:])**2, (im1[:-1, :] - im1[1:, :])**2
    CD2[:, :-1], RD2[:-1, :] = (im2[:, :-1] - im2[:, 1:])**2, (im2[:-1, :] - im2[1:, :])**2

    D1, D2 = np.sqrt(CD1 + RD1), np.sqrt(CD2 + RD2)
    GD1, GD2 = gaussian_filter(D1, sigma=sigma), gaussian_filter(D2, sigma=sigma)

    denom = GD1 + GD2 + 1e-9
    f1, f2 = GD1/denom, GD2/denom
    return f1*im1 + f2*im2


def fusion_grad_abs_whole_plane_segmented(stack1, stack2, alpha=0.05, threshold=0.01):
    """
    Fusion of two 3D stacks by selecting planes according to total gradient
    after adaptive background subtraction.
    """
    if stack1.shape != stack2.shape:
        raise ValueError("Stacks must have the same shape.")
    
    Z, H, W = stack1.shape
    stack1_seg, stack2_seg = np.zeros_like(stack1, dtype=np.float32), np.zeros_like(stack2, dtype=np.float32)

    for z in tqdm(range(Z), desc="Segmenting stack1", unit="plane"):
        stack1_seg[z] = apply_adaptive_background_subtraction(stack1, z, alpha=alpha, threshold=threshold)
    for z in tqdm(range(Z), desc="Segmenting stack2", unit="plane"):
        stack2_seg[z] = apply_adaptive_background_subtraction(stack2, z, alpha=alpha, threshold=threshold)

    fused_stack, positions_plane = np.zeros_like(stack1, dtype=np.float32), np.zeros((Z, H, W), dtype=np.uint8)

    for z in tqdm(range(Z), desc="Fusing by gradient", unit="plane"):
        grad1 = np.sqrt(np.gradient(stack1_seg[z], axis=1)**2 + np.gradient(stack1_seg[z], axis=0)**2).sum()
        grad2 = np.sqrt(np.gradient(stack2_seg[z], axis=1)**2 + np.gradient(stack2_seg[z], axis=0)**2).sum()

        if grad1 > grad2:
            fused_stack[z], positions_plane[z] = stack1[z], 0
        else:
            fused_stack[z], positions_plane[z] = stack2[z], 1

    return fused_stack, positions_plane
