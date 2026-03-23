import imageio.v3 as imageio
import scipy
import numpy as np

img = imageio.imread('hahaha_bw.png')

kernel_up = np.array([[-1.0, -2.0, -1.0],
                      [0, 0.0, 0],
                      [1.0, 2.0, 1.0]])

kernel_right = np.array([[-1.0, 0, 1.0],
                         [-2.0, 0.0, 2.0],
                         [-1.0, 0, 1.0]])

img = img.astype(np.float64)

result_up = scipy.ndimage.convolve(img, kernel_up)[1:-1, 1:-1]
result_right = scipy.ndimage.convolve(img, kernel_right)[1:-1, 1:-1]

result_up_squared = np.square(result_up)
result_right_squared = np.square(result_right)
result_sum = result_up_squared + result_right_squared

bigger = np.unravel_index(np.argmax(result_sum), result_sum.shape)

print(f"{bigger[0]}, {bigger[1]}")
