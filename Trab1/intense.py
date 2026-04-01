# Nome: Clara Ernesto de Carvalho
# NUSP: 14559479
# Sem: 2026.1
# Code: SCC0251
# Título: Trabalho 1 - TransEditor para suas necessidades de trans(formar) imagens!

import numpy as np
# Operations to alter the intensity of an image
def inverse_intensity(image: np.ndarray):
    return 255 - image

def log(image: np.ndarray):
    new_image = image.astype(np.float64)
    c = 255 / np.log(1 + np.max(new_image))
    new_image = c * np.log(1 + new_image)
    return new_image.astype(np.uint8)

def gamma(image: np.ndarray, gamma_value: float):
    new_image = image.astype(np.float64)
    c = 255 / (np.max(new_image) ** gamma_value)
    new_image = c * (new_image ** gamma_value)
    return new_image.astype(np.uint8)

def modulate_contrast(image:np.ndarray, contrast_limit: float):
    return np.where(image < contrast_limit, 0, 255)

# My function!
def sigmoid(image: np.ndarray):
    new_image = image.astype(np.float64)
    new_image = 255 / (1 + np.exp(-0.1 * (new_image - 128)))
    return new_image.astype(np.uint8)