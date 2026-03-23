import numpy as np

class Transformation:
    def __init__(self):
        pass
    def _trans_matrix(dx,dy):
        trans_matrix = np.array([[1,0,dx],
                                [0,1,dy],
                                [0,0,1]])
        return trans_matrix
    def _rot_matrix(theta):
        theta_rad = np.radians(theta)
        rot_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0],
                                [np.sin(theta_rad), np.cos(theta_rad), 0],
                                [0, 0, 1]])
        return rot_matrix
    
    def _scale_matrix(sx,sy):
        scale_matrix = np.array([[sx,0,0],
                                [0,sy,0],
                                [0,0,1]])
        return scale_matrix
    
    # I have made the code to trans the image, but as we cannot leave dead pixels, I will just shift the range the canvas use in main.py
    def trans_image(image, dx, dy) -> np.ndarray:
        return Transformation.apply_per_pixel(image, Transformation._trans_matrix(dx, dy))

    def rot_image(image, theta) -> np.ndarray:
        width, height = image.shape[1], image.shape[0]
        final_matrix = (
            Transformation._trans_matrix(width//2, height//2) @
            Transformation._rot_matrix(theta) @ 
            Transformation._trans_matrix(-width//2, -height//2)
        )
        return Transformation.apply_per_pixel(image, final_matrix)
    
    def scale_image(image, sx, sy) -> np.ndarray:
        width, height = image.shape[1], image.shape[0]
        final_width, final_height = int(width * sx), int(height * sy)
        final_matrix = (
            Transformation._trans_matrix(final_width//2, final_height//2) @
            Transformation._scale_matrix(sx, sy) @
            Transformation._trans_matrix(-width//2, -height//2)
        )
        return Transformation.apply_per_pixel(image, final_matrix, new_shape=(final_height, final_width, image.shape[2]) if len(image.shape) == 3 else (final_height, final_width))

    def apply_per_pixel(image, matrix, new_shape = None) -> np.ndarray:
        print("Applying transformation...")
        new_img = np.zeros_like(image) if new_shape is None else np.zeros(new_shape, dtype=image.dtype)
        for i in range(new_img.shape[0]):
            for j in range(new_img.shape[1]):
                pixel_loc = np.array([i, j, 1])
                old_pixel_loc = np.linalg.inv(matrix) @ pixel_loc
                if old_pixel_loc[0] == image.shape[0]:
                    old_pixel_loc[0] = image.shape[0] - 1
                if old_pixel_loc[1] == image.shape[1]:
                    old_pixel_loc[1] = image.shape[1] - 1
                
                if 0 <= old_pixel_loc[0] < image.shape[0] and 0 <= old_pixel_loc[1] < image.shape[1]:
                    new_img[i, j] = image[old_pixel_loc[0].astype(int), old_pixel_loc[1].astype(int)]
        print("Transformation applied.")
        return new_img