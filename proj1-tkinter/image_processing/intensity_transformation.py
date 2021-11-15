import numpy as np
import skimage.color
import matplotlib.pyplot as plt
import image_processing.helper
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


def negative(int1d):
    negative_array = 255 - int1d
    return negative_array


def log_transformation(int1d, c=1):
    float1d = int1d_to_float1d(int1d)
    log_transformation_array = c * np.log2(1.0 + float1d)
    log_transformation_array = np.clip(log_transformation_array, 0.0, 1.0)
    return float1d_to_int1d(log_transformation_array)


def gama_transformation(int1d, gama, c=1):
    float1d = int1d_to_float1d(int1d)
    gama_transformation_array = c * np.power(float1d, gama)
    gama_transformation_array = np.clip(gama_transformation_array, 0.0, 1.0)
    return float1d_to_int1d(gama_transformation_array)


def linear_transformation(int1d, points):
    float1d = int1d_to_float1d(int1d)

    point_left = points[0]
    point_right = points[-1]
    linear_transformation_array = np.zeros(float1d.shape)
    a, b = 0, 0
    for i in range(1, len(points)):
        point_right = points[i]
        a, b = linear_function_from_two_points(point_left, point_right)
        linear_transformation_array += np.where(np.bitwise_and(point_left[0] <= float1d, float1d < point_right[0]),
                                                a * float1d + b, 0)
        point_left = point_right
    linear_transformation_array += np.where(float1d == point_right[0], a * float1d + b, 0)

    linear_transformation_array = np.clip(linear_transformation_array, 0.0, 1.0)
    return float1d_to_int1d(linear_transformation_array)


def binary(int1d, threshold):
    int1d = np.vectorize(lambda x: 255 if x >= threshold else 0)(int1d)
    int3d = np.uint8(int1d)
    return int3d


def bit_plane(int1d, pos_bit):
    new_int1d = int1d.copy()
    for y in range(len(new_int1d)):
        for x in range(len(new_int1d[0])):
            if new_int1d[y][x] < pow(2, (pos_bit - 1)):
                new_int1d[y][x] = 0
            else:
                bin_color = bin(new_int1d[y][x])
                positive_bit = bin_color[-pos_bit]
                bin_color_list = [char for char in bin_color]
                for i in range(2, len(bin_color_list)):
                    bin_color_list[i] = '0'
                bin_color_list[-pos_bit] = positive_bit
                new_int1d[y][x] = int(''.join(bin_color_list), 2)
    return new_int1d


def record_message(int1d, message):
    recorded_int3d = int1d.copy()
    bin_message = string_to_bin(message)
    for i in range(len(bin_message)):
        color_value = recorded_int3d[0][i]
        color_value_bin = list(np.binary_repr(int(color_value)))
        color_value_bin[-1] = bin_message[i]
        new_value = int(''.join(color_value_bin), 2)
        recorded_int3d[0][i] = np.uint8(new_value)
    return recorded_int3d


def read_message(int1d):
    bits = ''
    for i in range(len(int1d[0])):
        color_value = int1d[0][i]
        lsb = np.binary_repr(int(color_value))[-1]
        bits += lsb
    message = bin_to_string(bits)
    return message


def string_to_bin(string):
    ints = [ord(c) for c in string]
    bin_string = ''
    for i in ints:
        bin_string += np.binary_repr(i, 8)
    return bin_string


def bin_to_string(bits):
    string = ''
    for pos in range(0, len(bits), 8):
        character = bits[pos:pos + 8]
        string += chr(int(character, 2))
    return string


def gray_histogram(int1d):
    # https://datacarpentry.org/image-processing/05-creating-histograms/

    plt.hist(int1d.flatten(), bins=256, range=(0, 255))
    plt.show()


def gray_eq(int1d):
    image = int1d.copy()
    hist = np.histogram(image, bins=256, range=(0, 255))[0]
    prob = hist / (int1d.shape[0] * int1d.shape[1])
    prob_acc = np.zeros(256)
    prob_acc[0] = prob[0]
    for i in range(1, 256):
        prob_acc[i] = prob_acc[i - 1] + prob[i]
    prob_acc = 255 * prob_acc
    image_map = np.round(prob_acc)
    image_map = np.uint8(image_map)

    new_image = np.vectorize(lambda x: image_map[x])(image)

    return new_image
