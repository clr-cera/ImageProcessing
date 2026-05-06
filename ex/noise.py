import numpy as np
import imageio.v3 as iio

noise_array = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
iio.imwrite("noise.png", noise_array)

#save fft
ffted = np.fft.fft2(noise_array)
iio.imwrite("noise_fft.png", np.abs(ffted).astype(np.uint8))


# do the same with normal distribution
noise_array_normal = np.random.normal(128, 50, (512, 512)).astype(np.uint8)
iio.imwrite("noise_normal.png", noise_array_normal)

ffted_normal = np.fft.fft2(noise_array_normal)
iio.imwrite("noise_normal_fft.png", np.abs(ffted_normal).astype(np.uint8))
