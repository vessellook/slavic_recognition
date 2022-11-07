import typing

import numpy as np


def get_vertical_profile(binary: np.ndarray):
    if binary.max(initial=0) > 1:
        binary = binary / binary.max(initial=0)
    return binary.sum(axis=1).astype(np.int8)


def get_point_lists(binary: np.ndarray, step: int):
    h, w = binary.shape
    top_point_lists = []
    bottom_point_lists = []
    min_value = step // 5
    for left in range(0, w, step):
        vertical_bar = binary[:, left:left + step]
        profile = get_vertical_profile(vertical_bar)
        top_points = []
        bottom_points = []
        x = left + step // 2
        for y, (value, next_value) in enumerate(zip(profile[:-1], profile[1:])):
            if value < min_value <= next_value:
                top_points.append((x, y))
            if value >= min_value > next_value:
                bottom_points.append((x, y))
        if len(top_points) > 0:
            top_point_lists.append(top_points)
        if len(bottom_points) > 0:
            bottom_point_lists.append(bottom_points)

    return top_point_lists, bottom_point_lists


def draw_dashes(image: np.ndarray, color: typing.Union[np.ndarray, typing.Tuple[int, int, int], int],
                points: typing.List[typing.Tuple[int, int]], step: int):
    w = image.shape[1]
    for x_middle, y in points:
        left = x_middle - step // 2
        right = x_middle + step // 2
        for x in range(left, min(right + 1, w)):
            image[y, x] = color
    return image


def get_borders(top_point_lists, bottom_point_lists):
    for points in top_point_lists:
        points.reverse()
    for points in bottom_point_lists:
        points.reverse()

    # разворачиваем списки, чтобы удалять элементы из концов списков, а не из начал
    while True:
        max_y = 0
        max_y_list_index = 0
