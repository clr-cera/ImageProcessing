import numpy as np
import imageio.v3 as iio
def luminance(pixel):
    return (0.2126*pixel[0] + 0.7152*pixel[1] + 0.0722*pixel[2])

img = iio.imread("pompom.jpeg")

grey_img = np.zeros((img.shape[0], img.shape[1]))

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        grey_img[i][j] = luminance(img[i][j])

ffted = np.fft.fft2(grey_img)



def remove_freqs_below_threshold(ffted, threshold):
    new_ffted = np.copy(ffted)
    new_ffted = np.where(np.abs(new_ffted) < threshold, 0, new_ffted)
    return new_ffted

lowest_freq = np.min(np.abs(ffted))
highest_freq = np.max(np.abs(ffted))

# binary search for the threshold
while lowest_freq < highest_freq - 1e-5:
    threshold = (lowest_freq + highest_freq) / 2
    new_ffted = remove_freqs_below_threshold(ffted, threshold)
    reconstructed_img = np.abs(np.fft.ifft2(new_ffted))
    square_error = np.mean(np.sqrt((grey_img.astype(float) - reconstructed_img.astype(float))**2))
    if square_error < 2:
        lowest_freq = threshold
    else:
        highest_freq = threshold

iio.imwrite("reconstructed.png", reconstructed_img.astype(np.uint8))

count = np.sum(np.abs(ffted)<threshold)
ratio = count / (ffted.shape[0] * ffted.shape[1])
print(f"{ratio:.2f}")