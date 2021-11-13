import numpy as np
import skimage.color
import matplotlib.pyplot as plt
import image_processing.helper


def binary(int3d, threshold):
    int3d = np.vectorize(lambda x: 255 if x >= threshold else 0)(int3d)
    int3d = np.uint8(int3d)
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


def record_message(int3d, message):
    recorded_int3d = int3d.copy()
    bin_message = string_to_bin(message)
    for i in range(len(bin_message)):
        color_value = recorded_int3d[0][i][0]
        color_value_bin = list(np.binary_repr(int(color_value)))
        color_value_bin[-1] = bin_message[i]
        new_value = int(''.join(color_value_bin), 2)
        recorded_int3d[0][i][0] = np.uint8(new_value)
    return recorded_int3d


def read_message(int3d):
    bits = ''
    for i in range(len(int3d[0])):
        color_value = int3d[0][i][0]
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


def gray_histogram(int3d):
    # https://datacarpentry.org/image-processing/05-creating-histograms/

    image = skimage.color.rgb2gray(int3d)
    image = skimage.util.img_as_ubyte(image)

    plt.hist(image.flatten(), bins=256, range=(0, 255))
    plt.show()


def gray_eq(int3d):
    image = skimage.color.rgb2gray(int3d)
    image = skimage.util.img_as_ubyte(image)

    hist = np.histogram(image, bins=256, range=(0, 255))[0]
    prob = hist / (int3d.shape[0] * int3d.shape[1])
    prob_acc = np.zeros(256)
    prob_acc[0] = prob[0]
    for i in range(1, 256):
        prob_acc[i] = prob_acc[i - 1] + prob[i]
    prob_acc = 255 * prob_acc
    image_map = np.round(prob_acc)
    image_map = np.uint8(image_map)

    new_image = np.vectorize(lambda x: image_map[x])(image)

    return image_processing.helper.int1d_to_int3d(new_image)


if __name__ == '__main__':
    b = string_to_bin('Victor de Sousa Rocha')
    print(b)
    s = bin_to_string(b)
    print(s)
