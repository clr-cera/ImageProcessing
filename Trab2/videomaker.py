import numpy as np
import cv2


def create_video_fft(input_img, output_path, fps=5):
    h, w = input_img.shape
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    vw = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=True)

    if not vw.isOpened():
        raise RuntimeError("Something in generating the video went wrong")

    centered_fft = np.fft.fftshift(np.fft.fft2(input_img))
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    dist2 = (y - cy) ** 2 + (x - cx) ** 2

    for keep_radius in range(1, min(h, w) // 2, 5):
        mask = dist2 <= keep_radius ** 2
        filtered_fft = centered_fft * mask
        filtered_img = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real
        filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)
        frame = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
        vw.write(frame)

    vw.release()
