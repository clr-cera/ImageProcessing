# Nome: Clara Ernesto de Carvalho
# NUSP: 14559479
# Sem: 2026.1
# Code: SCC0251
# Título: Trabalho 1 - TransEditor para suas necessidades de trans(formar) imagens!

import numpy as np
# Operations to alter the intensity of an image
# Invert the intensity value
def inverse_intensity(image: np.ndarray):
    return 255 - image

# Logarithmic transformation
def log(image: np.ndarray):
    new_image = image.astype(np.float64)
    c = 255 / np.log(1 + np.max(new_image))
    new_image = c * np.log(1 + new_image)
    return new_image.astype(np.uint8)

# Gamma Correction
def gamma(image: np.ndarray, gamma_value: float):
    new_image = image.astype(np.float64)
    c = 255 / (np.max(image) ** gamma_value)
    new_image = c * (image ** gamma_value)
    return new_image.astype(np.uint8)

# Modulate the intensity with an interval
def modulate_contrast(image:np.ndarray, contrast_lower_limit: float, contrast_upper_limit: float):
    return np.where((image >= contrast_lower_limit) & (image < contrast_upper_limit), 0, 255)

# My function!
# It is a logistic sigmoid function with k = 0.05 and x0 = 128
def sigmoid(image: np.ndarray):
    new_image = image.astype(np.float64)
    new_image = 255 / (1 + np.exp(-0.05 * (new_image - 128)))
    return new_image.astype(np.uint8)