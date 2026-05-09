import numpy as np
import cv2
import imageio.v3 as iio

# FFT2 implementation

def _is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


def _dft_1d(values):
    n = values.shape[0]
    if n == 0:
        return values.copy()
    k = np.arange(n)
    m = k.reshape((n, 1))
    twiddle = np.exp(-2j * np.pi * m * k / n)
    return twiddle @ values


def _fft_1d(values):
    arr = np.asarray(values, dtype=np.complex128).copy()
    n = arr.shape[0]

    if not _is_power_of_two(n):
        return _dft_1d(arr)

    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            arr[i], arr[j] = arr[j], arr[i]

    length = 2
    while length <= n:
        angle = -2j * np.pi / length
        wlen = np.exp(angle)

        for i in range(0, n, length):
            w = 1 + 0j
            half = length // 2
            for j in range(half):
                u = arr[i + j]
                v = arr[i + j + half] * w
                arr[i + j] = u + v
                arr[i + j + half] = u - v
                w *= wlen
        length <<= 1

    return arr


def _ifft_1d(values):
    arr = np.asarray(values, dtype=np.complex128)
    n = arr.shape[0]
    if n == 0:
        return arr.copy()
    return np.conjugate(_fft_1d(np.conjugate(arr))) / n


# Manually made fft2
def fft2(input_img):
    matrix = np.asarray(input_img, dtype=np.complex128)
    if matrix.ndim != 2:
        raise ValueError("fft2 expects a 2D array")

    h, w = matrix.shape
    row_fft = np.empty((h, w), dtype=np.complex128)
    for row in range(h):
        row_fft[row, :] = _fft_1d(matrix[row, :])

    out = np.empty((h, w), dtype=np.complex128)
    for col in range(w):
        out[:, col] = _fft_1d(row_fft[:, col])

    return out


def ifft2(input_fft):
    matrix = np.asarray(input_fft, dtype=np.complex128)
    if matrix.ndim != 2:
        raise ValueError("ifft2 expects a 2D array")

    h, w = matrix.shape
    row_ifft = np.empty((h, w), dtype=np.complex128)
    for row in range(h):
        row_ifft[row, :] = _ifft_1d(matrix[row, :])

    out = np.empty((h, w), dtype=np.complex128)
    for col in range(w):
        out[:, col] = _ifft_1d(row_ifft[:, col])

    return out


# This function creates a video showing frames of a source image with increasing frequencies being allowed in the result frame. 
# It also saves the frames at 20% and 90% of the allowed frequencies
def create_video_fft(input_img, output_path, fps=5):
    h, w = input_img.shape
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    vw = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=True)

    if not vw.isOpened():
        raise RuntimeError("Something in generating the video went wrong")

    centered_fft = np.fft.fftshift(fft2(input_img))
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    dist2 = (y - cy) ** 2 + (x - cx) ** 2

    for keep_radius in range(1, min(h, w) // 2, 5):
        mask = dist2 <= keep_radius**2
        filtered_fft = centered_fft * mask
        filtered_img = ifft2(np.fft.ifftshift(filtered_fft)).real
        filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)
        frame = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
        # Save frames at 20% and 90% of the maximum radius for reference
        if keep_radius in [1 + int(min(h, w) // 10), 1 + int(9 * min(h, w) // 20)]:
            iio.imwrite(output_path.replace('.mp4', f'_frame_{keep_radius}.png'), filtered_img)
        vw.write(frame)

    vw.release()
