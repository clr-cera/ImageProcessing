import numpy as np
from scipy.signal import convolve2d

from kernels import new_laplace_kernel, new_sobel_kernels, new_unsharpening_mask_kernel, new_emboss_kernel, new_box_kernel, new_gaussian_kernel, new_shift_kernel

# This file contains functions that apply processes to images using the kernels previously defined

# Sharpness Increase with Laplace
def sharpen_img_laplace(image) -> np.ndarray:
    kernel = new_laplace_kernel()
    return image - convolve2d(image, kernel, mode='same', boundary='symm')

# Sharpness Increase with Unsharpening Mask
def sharpen_img_unsharpening_mask(image, size, sigma) -> np.ndarray:
    kernel = new_unsharpening_mask_kernel(size, sigma)
    return image + convolve2d(image, kernel, mode='same', boundary='symm')

# Embossing
def emboss_image(image, intensity) -> np.ndarray:
    kernel = new_emboss_kernel(intensity)
    return convolve2d(image, kernel, mode='same', boundary='symm')



# Simple Kernel Applications

def apply_box_filter(image, size) -> np.ndarray:
    kernel = new_box_kernel(size)
    return convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)

def apply_gaussian_filter(image, sigma, size) -> np.ndarray:
    kernel = new_gaussian_kernel(size, sigma)
    return convolve2d(image, kernel, mode='same', boundary='symm')

def apply_shift_filter(image, shift_y, shift_x) -> np.ndarray:
    size = max(abs(shift_y), abs(shift_x)) + 1
    kernel = new_shift_kernel(size, shift_y, shift_x)
    return convolve2d(image, kernel, mode='same', boundary='symm')

def apply_laplace_filter(image) -> np.ndarray:
    kernel = new_laplace_kernel()
    return convolve2d(image, kernel, mode='same', boundary='symm')

def apply_sobels_filter(image) -> tuple[np.ndarray, np.ndarray]:
    kernel_x, kernel_y = new_sobel_kernels()
    grad_x = convolve2d(image, kernel_x, mode='same', boundary='symm')
    grad_y = convolve2d(image, kernel_y, mode='same', boundary='symm')
    return grad_x, grad_y