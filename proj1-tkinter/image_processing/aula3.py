import numpy as np


def binary(int3d):
    int3d = np.vectorize(lambda x: 0 if x == 0 else 255)(int3d)
    int3d = np.uint8(int3d)
    return int3d


def bit_plane(int3d, pos_bit):
    new_int3d = int3d.copy()
    for line in new_int3d:
        for pixel in line:
            for channel in range(len(pixel)):
                if pixel[channel] < pow(2, (pos_bit - 1)):
                    pixel[channel] = 0
                else:
                    bin_color = bin(pixel[channel])
                    positive_bit = bin_color[-pos_bit]
                    bin_color_list = [char for char in bin_color]
                    for i in range(2, len(bin_color_list)):
                        bin_color_list[i] = '0'
                    bin_color_list[-pos_bit] = positive_bit
                    pixel[channel] = int(''.join(bin_color_list), 2)
    return new_int3d


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


if __name__ == '__main__':
    x = string_to_bin('Victor de Sousa Rocha')
    print(x)
    s = bin_to_string(x)
    print(s)
