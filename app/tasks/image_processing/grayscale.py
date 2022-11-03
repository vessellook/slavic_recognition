import numpy as np


def grayscale(image: np.ndarray, type: str = 'mean'):
    if len(image.shape) < 3 or image.shape[2] == 1:
        return image
    if type == 'min':
        return image.min(axis=2, initial=0)
    elif type == 'max':
        return image.max(axis=2, initial=0)
    else:
        return image.mean(axis=2)
