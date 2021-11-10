import numpy as np
from image_processing.helper import *


def gray(int3d):
    float3d = int3d_to_float3d(int3d)
    for line in float3d:
        for pixel in line:
            mean = (pixel[0] + pixel[1] + pixel[2]) / 3
            pixel[0] = mean
            pixel[1] = mean
            pixel[2] = mean
    return float3d_to_int3d(float3d)


def negative(int3d):
    float3d = int3d_to_float3d(int3d)
    negative_array = 1.0 - float3d
    return float3d_to_int3d(negative_array)


def log_transformation(int3d, c=1):
    float3d = int3d_to_float3d(int3d)
    log_transformation_array = c * np.log2(1.0 + float3d)
    log_transformation_array = np.clip(log_transformation_array, 0.0, 1.0)
    return float3d_to_int3d(log_transformation_array)


def gama_transformation(array3d, gama, c=1):
    float3d = int3d_to_float3d(array3d)
    gama_transformation_array = c * np.power(float3d, gama)
    gama_transformation_array = np.clip(gama_transformation_array, 0.0, 1.0)
    return float3d_to_int3d(gama_transformation_array)


def linear_transformation(int3d, points):
    float3d = int3d_to_float3d(int3d)

    point_left = points[0]
    point_right = points[-1]
    linear_transformation_array = np.zeros(float3d.shape)
    a, b = 0, 0
    for i in range(1, len(points)):
        point_right = points[i]
        a, b = linear_function_from_two_points(point_left, point_right)
        linear_transformation_array += np.where(np.bitwise_and(point_left[0] <= float3d, float3d < point_right[0]),
                                                a * float3d + b, 0)
        point_left = point_right
    linear_transformation_array += np.where(float3d == point_right[0], a * float3d + b, 0)

    linear_transformation_array = np.clip(linear_transformation_array, 0.0, 1.0)
    return float3d_to_int3d(linear_transformation_array)
