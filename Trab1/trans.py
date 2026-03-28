import numpy as np
import copy
class Transformation:
    def __init__(self, img_shape):
        self.rotation_angle = 0
        self.scale_factor = 1
        self.position = (0, 0)
        self.img_shape = img_shape

    def _pos_matrix(self):
        trans_matrix = np.array([[1,0,self.position[0]],
                                [0,1,self.position[1]],
                                [0,0,1]])
        return trans_matrix
    def _rot_matrix(self):
        theta_rad = np.radians(self.rotation_angle)
        rot_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0],
                                [np.sin(theta_rad), np.cos(theta_rad), 0],
                                [0, 0, 1]])
        return rot_matrix
    
    def _scale_matrix(self):
        scale_matrix = np.array([[self.scale_factor,0,0],
                                [0,self.scale_factor,0],
                                [0,0,1]])
        return scale_matrix
    
    def _go_to_origin_matrix(self, reverse=False):
        if reverse:
            go_to_origin_matrix = np.array([[1,0,self.img_shape[1]//2],
                                            [0,1,self.img_shape[0]//2],
                                            [0,0,1]])
        else:
            go_to_origin_matrix = np.array([[1,0,-self.img_shape[1]//2],
                                            [0,1,-self.img_shape[0]//2],
                                            [0,0,1]])
        return go_to_origin_matrix
    
    @property
    def trans_matrix(self):
        origined_matrix = self._go_to_origin_matrix()
        scaled_matrix = self._scale_matrix() @ origined_matrix
        rotated_matrix = self._rot_matrix() @ scaled_matrix
        translated_matrix = self._pos_matrix() @ rotated_matrix
        final_matrix = self._go_to_origin_matrix(reverse=True) @ translated_matrix
        return final_matrix
    
    def translate(self, dx, dy):
        self.position = (self.position[0] + dy, self.position[1] - dx)
        return self

    def rotate(self, theta):
        width, height = self.img_shape[1], self.img_shape[0]

        needed_scaling_factor = 1
        if width > height:
            needed_scaling_factor = (width / height) * np.sin(np.radians(theta)) + np.cos(np.radians(theta))
        else:
            needed_scaling_factor = (height / width) * np.sin(np.radians(theta)) + np.cos(np.radians(theta))

        if needed_scaling_factor > self.scale_factor:
            self.scale_factor = needed_scaling_factor
        self.rotation_angle += theta

        # Adjust position to keep it inside the image
        self.position = (0,0)

        return self

    def scale(self, s):
        self.scale_factor *= s
        return self
    
    def is_translation_valid(self, dx, dy):
        corners = np.array([[0, 0, 1],
                            [0, self.img_shape[0], 1],
                            [self.img_shape[1], 0, 1],
                            [self.img_shape[1], self.img_shape[0], 1]])
        print("Corners before transformation:", corners)
        possible_transformation = copy.deepcopy(self)
        possible_transformation.translate(dx, dy)
        translated_corners = np.array([np.linalg.inv(possible_transformation.trans_matrix) @ corner for corner in corners])
        print("Translated corners:", translated_corners)
        print("Image shape:", self.img_shape)
        if np.all((translated_corners[:, 0] >= 0) & (translated_corners[:, 0] < self.img_shape[1]) &
                  (translated_corners[:, 1] >= 0) & (translated_corners[:, 1] < self.img_shape[0])):
            return True
        return False
        

    def transform_image(self,image) -> np.ndarray:
        matrix = self.trans_matrix
        new_img = np.zeros_like(image)
        for i in range(new_img.shape[0]):
            for j in range(new_img.shape[1]):
                pixel_loc = np.array([i, j, 1])
                old_pixel_loc = np.linalg.inv(matrix) @ pixel_loc
                
                if 0 <= old_pixel_loc[0] < image.shape[0] and 0 <= old_pixel_loc[1] < image.shape[1]:
                    new_img[i, j] = image[old_pixel_loc[0].astype(int), old_pixel_loc[1].astype(int)]
        return new_img