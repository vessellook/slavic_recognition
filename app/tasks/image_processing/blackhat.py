import numpy as np
import cv2 as cv


# def binarize(image: np.ndarray):
#     src = cv.morphologyEx(image, cv.MORPH_BLACKHAT, np.ones((11, 11)))
#     src = 255 - src
#
#     src = (255 * (src > threshold_otsu(src))).astype(np.uint8)
