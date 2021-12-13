import numpy as np
from image_processing import helper
from image_processing.spatial_transformation import region_of_interest


def media_geometrica(int1d, kernel_size):
    new_image = helper.int1d_to_float1d(int1d)

    expanded_image = np.pad(new_image, kernel_size - 1, mode='constant')
    new_expanded_image = expanded_image.copy()

    delta = int(np.ceil(kernel_size / 2) - 1)

    for lin in range(delta, new_expanded_image.shape[0] - delta):
        for col in range(delta, new_expanded_image.shape[1] - delta):
            roi = region_of_interest((lin, col), kernel_size, delta, expanded_image)

            prod = 1
            for roi_lin in range(roi.shape[0]):
                for roi_col in range(roi.shape[1]):
                    prod *= roi[roi_lin][roi_col]

            media_g = np.power(prod, (1 / (roi.shape[0] * roi.shape[1])))
            new_pixel_value = media_g
            new_expanded_image[lin][col] = new_pixel_value

    new_expanded_image = new_expanded_image[
                         kernel_size - 1: new_expanded_image.shape[0] - kernel_size + 1,
                         kernel_size - 1: new_expanded_image.shape[1] - kernel_size + 1
                         ]

    # normalize
    new_expanded_image = helper.normalize_data(new_expanded_image)

    return helper.float1d_to_int1d(new_expanded_image)


def media_harmonica(int1d, kernel_size):
    new_image = helper.int1d_to_float1d(int1d)

    expanded_image = np.pad(new_image, kernel_size - 1, mode='constant')
    new_expanded_image = expanded_image.copy()

    delta = int(np.ceil(kernel_size / 2) - 1)

    for lin in range(delta, new_expanded_image.shape[0] - delta):
        for col in range(delta, new_expanded_image.shape[1] - delta):
            roi = region_of_interest((lin, col), kernel_size, delta, expanded_image)

            sum_ = 0
            for roi_lin in range(roi.shape[0]):
                for roi_col in range(roi.shape[1]):
                    if roi[roi_lin][roi_col] == 0:
                        sum_ += 0
                    else:
                        sum_ += 1 / (roi[roi_lin][roi_col])

            if sum_ == 0:
                media_h = 0
            else:
                media_h = (roi.shape[0] * roi.shape[1]) / sum_
            new_pixel_value = media_h
            new_expanded_image[lin][col] = new_pixel_value

    new_expanded_image = new_expanded_image[
                         kernel_size - 1: new_expanded_image.shape[0] - kernel_size + 1,
                         kernel_size - 1: new_expanded_image.shape[1] - kernel_size + 1
                         ]

    # normalize
    new_expanded_image = helper.normalize_data(new_expanded_image)

    return helper.float1d_to_int1d(new_expanded_image)


def media_contra_harmonica(int1d, kernel_size, Q):
    new_image = helper.int1d_to_float1d(int1d)

    expanded_image = np.pad(new_image, kernel_size - 1, mode='constant')
    new_expanded_image = expanded_image.copy()

    delta = int(np.ceil(kernel_size / 2) - 1)

    for lin in range(delta, new_expanded_image.shape[0] - delta):
        for col in range(delta, new_expanded_image.shape[1] - delta):
            roi = region_of_interest((lin, col), kernel_size, delta, expanded_image)

            sum1 = 0
            sum2 = 0
            for roi_lin in range(roi.shape[0]):
                for roi_col in range(roi.shape[1]):
                    sum1 += np.power(roi[roi_lin][roi_col], Q + 1)
                    sum2 += np.power(roi[roi_lin][roi_col], Q)

            if sum2 == 0:
                media_ch = 0
            else:
                media_ch = sum1 / sum2
            new_pixel_value = media_ch
            new_expanded_image[lin][col] = new_pixel_value

    new_expanded_image = new_expanded_image[
                         kernel_size - 1: new_expanded_image.shape[0] - kernel_size + 1,
                         kernel_size - 1: new_expanded_image.shape[1] - kernel_size + 1
                         ]

    # normalize
    new_expanded_image = helper.normalize_data(new_expanded_image)

    return helper.float1d_to_int1d(new_expanded_image)
