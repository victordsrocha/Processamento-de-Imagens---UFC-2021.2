import numpy as np
import skimage.color
import matplotlib.pyplot as plt
import image_processing.helper as helper


def region_of_interest(center, kernel_size, delta, expanded_image):
    new_roi = np.zeros((kernel_size, kernel_size))
    x, y = 0, 0
    for i in range(center[0] - delta, center[0] + delta + 1):
        for j in range(center[1] - delta, center[1] + delta + 1):
            new_roi[y][x] = expanded_image[i][j]
            x += 1
        x = 0
        y += 1
    return new_roi


def generic_filter(int1d, kernel):
    image = helper.int1d_to_float1d(int1d)
    filtered_image = apply_kernel(image, kernel, normalize='normalize')
    return helper.float1d_to_int1d(filtered_image)


def box_filter(int1d, box_size):
    kernel = np.ones(shape=(box_size, box_size))
    kernel = kernel / (box_size * box_size)
    image = helper.int1d_to_float1d(int1d)
    filtered_image = apply_kernel(image, kernel, normalize='normalize')
    return helper.float1d_to_int1d(filtered_image)


def gaussian_kernel(_kernel_size, sigma=1):
    # https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(_kernel_size - 1) / 2., (_kernel_size - 1) / 2., _kernel_size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    _kernel = np.outer(gauss, gauss)
    return _kernel / np.sum(_kernel)


def gaussian_smoothing_filter(int1d, kernel_size):
    kernel = gaussian_kernel(_kernel_size=kernel_size)
    image = helper.int1d_to_float1d(int1d)
    filtered_image = apply_kernel(image, kernel, normalize='normalize')
    return helper.float1d_to_int1d(filtered_image)


def apply_kernel(image, kernel, normalize='normalize'):
    kernel_size = kernel.shape[0]
    expanded_image = np.pad(image, kernel_size - 1, mode='constant')
    new_expanded_image = expanded_image.copy()
    delta = int(np.ceil(kernel_size / 2) - 1)

    for lin in range(delta, new_expanded_image.shape[0] - delta):
        for col in range(delta, new_expanded_image.shape[1] - delta):
            roi = region_of_interest((lin, col), kernel_size, delta, expanded_image)
            filtered_values = roi * kernel
            new_pixel_value = filtered_values.sum()
            new_expanded_image[lin][col] = new_pixel_value

        # normalize

    new_expanded_image = new_expanded_image[
                         kernel_size - 1: new_expanded_image.shape[0] - kernel_size + 1,
                         kernel_size - 1: new_expanded_image.shape[1] - kernel_size + 1
                         ]

    if normalize == 'normalize':
        new_expanded_image = helper.normalize_data(new_expanded_image)
    elif normalize == 'clip':
        new_expanded_image = np.clip(new_expanded_image, 0, 1)

    return new_expanded_image


def median_filter(int1d, kernel_size):
    new_image = helper.int1d_to_float1d(int1d)

    expanded_image = np.pad(new_image, kernel_size - 1, mode='constant')
    new_expanded_image = expanded_image.copy()

    delta = int(np.ceil(kernel_size / 2) - 1)

    for lin in range(delta, new_expanded_image.shape[0] - delta):
        for col in range(delta, new_expanded_image.shape[1] - delta):
            roi = region_of_interest((lin, col), kernel_size, delta, expanded_image)
            median = np.median(roi)
            new_pixel_value = median
            new_expanded_image[lin][col] = new_pixel_value

    new_expanded_image = new_expanded_image[
                         kernel_size - 1: new_expanded_image.shape[0] - kernel_size + 1,
                         kernel_size - 1: new_expanded_image.shape[1] - kernel_size + 1
                         ]

    # normalize
    new_expanded_image = helper.normalize_data(new_expanded_image)

    return helper.float1d_to_int1d(new_expanded_image)


def laplacian_filter(int1d):
    kernel = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ])
    image = helper.int1d_to_float1d(int1d)
    # como o objetivo é somente visualizar o filtro, realizamos a normalização
    laplace_image = apply_kernel(image, kernel, normalize='normalize')
    return helper.float1d_to_int1d(laplace_image)


def blend(int1d, alpha):
    kernel = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ])
    image = helper.int1d_to_float1d(int1d)
    filtered_image = apply_kernel(image, kernel, normalize='none')
    blend_image = image * (1 - alpha * filtered_image)
    blend_image = np.clip(blend_image, 0, 1)
    return helper.float1d_to_int1d(blend_image)


def high_boost(int1d, alpha, kernel_size):
    kernel = gaussian_kernel(_kernel_size=kernel_size)
    image = helper.int1d_to_float1d(int1d)
    filtered_image = apply_kernel(image, kernel, normalize='normalize')
    new_image = image + alpha * (image - filtered_image)
    new_image = np.clip(new_image, 0, 1)
    return helper.float1d_to_int1d(new_image)
