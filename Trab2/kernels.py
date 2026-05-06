import numpy as np
# Shift
def new_shift_kernel(size, shift_y, shift_x) -> np.ndarray:
    kernel = np.zeros((size, size))
    kernel[shift_y, shift_x] = 1
    return kernel

# Box
def new_box_kernel(size) -> np.ndarray:
    kernel = np.ones((size, size)) / (size * size)
    return kernel

# Gaussian
def new_gaussian_kernel(size, sigma) -> np.ndarray:
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel

# Laplace
def new_laplace_kernel() -> np.ndarray:
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    return kernel

# Sobel
def new_sobel_kernels() -> tuple[np.ndarray, np.ndarray]:
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])
    return kernel_x, kernel_y

# Unsharpening Mask using Gaussian
def new_unsharpening_mask_kernel(size, sigma) -> np.ndarray:
    gaussian_kernel = new_gaussian_kernel(size, sigma)
    unsharpening_mask = np.zeros_like(gaussian_kernel)
    unsharpening_mask[size // 2, size // 2] = 1
    unsharpening_mask -= gaussian_kernel
    return unsharpening_mask

def new_emboss_kernel(intensity) -> np.ndarray:
    kernel = np.array([[-2*intensity, -1*intensity, 0],
                       [-1*intensity, 1, 1*intensity],
                       [0, 1*intensity, 2*intensity]])
    return kernel