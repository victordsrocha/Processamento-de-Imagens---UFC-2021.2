import numpy as np


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
