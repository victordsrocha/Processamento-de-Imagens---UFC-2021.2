import numpy as np


def int3d_to_float3d(array3d):
    return array3d / 255


def float3d_to_int3d(array3d):
    return np.uint8(array3d * 255)


def linear_function_from_two_points(p1, p2):
    a = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p2[1] - a * p2[0]
    return a, b
