import numpy as np
from helper import *


def negative(array3d):
    float3d = int3d_to_float3d(array3d)
    negative_array = 1.0 - float3d
    return float3d_to_int3d(negative_array)


def log_transformation(array3d, c=1):
    float3d = int3d_to_float3d(array3d)
    log_transformation_array = c * np.log2(1.0 + float3d)
    log_transformation_array = np.clip(log_transformation_array, 0.0, 1.0)
    return float3d_to_int3d(log_transformation_array)


def gama_transformation(array3d, gama, c=1):
    float3d = int3d_to_float3d(array3d)
    gama_transformation_array = c * np.power(float3d, gama)
    gama_transformation_array = np.clip(gama_transformation_array, 0.0, 1.0)
    return float3d_to_int3d(gama_transformation_array)


def linear_transformation(int3d, *args):
    float3d = int3d_to_float3d(int3d)

    point_left = args[0]
    point_right = args[-1]
    linear_transformation_array = np.zeros(float3d.shape)
    a, b = 0, 0
    for i in range(1, len(args)):
        point_right = args[i]
        a, b = linear_function_from_two_points(point_left, point_right)
        linear_transformation_array += np.where(np.bitwise_and(point_left[0] <= float3d, float3d < point_right[0]),
                                                a * float3d + b, 0)
        point_left = point_right
    linear_transformation_array += np.where(float3d == point_right[0], a * float3d + b, 0)

    linear_transformation_array = np.clip(linear_transformation_array, 0.0, 1.0)
    return float3d_to_int3d(linear_transformation_array)
