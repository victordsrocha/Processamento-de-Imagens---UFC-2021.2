import numpy as np


def int3d_to_float3d(array3d):
    return array3d / 255


def float3d_to_int3d(array3d):
    return np.uint8(array3d * 255)


def int1d_to_int3d(int1d):
    lines = int1d.shape[0]
    columns = int1d.shape[1]
    int3d = np.zeros(shape=(lines, columns, 3), dtype='uint8')
    for i in range(lines):
        for j in range(columns):
            for k in range(3):
                int3d[i][j][0] = int1d[i][j]
                int3d[i][j][1] = int1d[i][j]
                int3d[i][j][2] = int1d[i][j]
    return int3d


def linear_function_from_two_points(p1, p2):
    a = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p2[1] - a * p2[0]
    return a, b
