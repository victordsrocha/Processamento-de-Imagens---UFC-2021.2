import math
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


def sobel_x(int1d):
    kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    image = helper.int1d_to_float1d(int1d)
    filtered_image = apply_kernel(image, kernel, normalize='normalize')
    return helper.float1d_to_int1d(filtered_image)


def sobel_y(int1d):
    kernel = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    image = helper.int1d_to_float1d(int1d)
    filtered_image = apply_kernel(image, kernel, normalize='normalize')
    return helper.float1d_to_int1d(filtered_image)


def sobel_magnitude(int1d):
    sx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    sy = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    image = helper.int1d_to_float1d(int1d)
    sobel_x_image = apply_kernel(image, sx, normalize='clip')
    sobel_y_image = apply_kernel(image, sy, normalize='clip')
    magnitude_image = np.abs(sobel_x_image) + np.abs(sobel_y_image)
    magnitude_image = np.clip(magnitude_image, 0, 1)
    return helper.float1d_to_int1d(magnitude_image)


def int_in_range(value, min_, max_):
    f = np.floor(value)
    c = np.ceil(value)

    if f > max_:
        f = max_
    elif f < min_:
        f = min_

    if c > max_:
        c = max_
    elif c < min_:
        c = min_

    return int(f), int(c)


def linear_scale(int3d, horizontal_scale, vertical_scale):
    original_image = helper.int3d_to_float3d(int3d)

    new_image_height = int(np.round(int3d.shape[0] * vertical_scale))
    new_image_width = int(np.round(int3d.shape[1] * horizontal_scale))

    new_image = np.zeros(shape=(new_image_height, new_image_width, 3))

    for y in range(new_image_height):
        for x in range(new_image_width):
            real_x = (x + 1) / horizontal_scale
            real_y = (y + 1) / vertical_scale

            x_floor, x_ceil = int_in_range(real_x, 0, original_image.shape[1] - 1)
            y_floor, y_ceil = int_in_range(real_y, 0, original_image.shape[0] - 1)

            pixel11 = original_image[y_floor, x_floor, :]
            pixel12 = original_image[y_floor, x_ceil, :]
            pixel21 = original_image[y_ceil, x_floor, :]
            pixel22 = original_image[y_ceil, x_ceil, :]

            h1 = pixel11 * math.modf(real_x)[0] + pixel12 * (1 - math.modf(real_x)[0])
            h2 = pixel21 * math.modf(real_x)[0] + pixel22 * (1 - math.modf(real_x)[0])

            v = h1 * math.modf(real_y)[0] + h2 * (1 - math.modf(real_y)[0])

            new_image[y, x, :] = v

    return helper.float3d_to_int3d(new_image)


def repetition_scale(int3d, horizontal_scale, vertical_scale):
    original_image = helper.int3d_to_float3d(int3d)

    new_image_height = int(np.round(int3d.shape[0] * vertical_scale))
    new_image_width = int(np.round(int3d.shape[1] * horizontal_scale))

    new_image = np.zeros(shape=(new_image_height, new_image_width, 3))

    for y in range(new_image_height):
        for x in range(new_image_width):
            real_x = (x + 1) / horizontal_scale
            real_y = (y + 1) / vertical_scale

            x_floor, x_ceil = int_in_range(real_x, 0, original_image.shape[1] - 1)
            y_floor, y_ceil = int_in_range(real_y, 0, original_image.shape[0] - 1)

            pixel11 = original_image[y_floor, x_floor, :]

            new_image[y, x, :] = pixel11

    return helper.float3d_to_int3d(new_image)


def rotate(int3d, angle):
    # https://gautamnagrawal.medium.com/rotating-image-by-any-angle-shear-transformation-using-only-numpy-d28d16eb5076
    image = helper.int3d_to_float3d(int3d)
    height, width = image.shape[0], image.shape[1]

    angle = math.radians(angle)
    angle_cos = math.cos(angle)
    angle_sin = math.sin(angle)

    # calculo do novo tamanho da imagem!
    new_height = round(abs(image.shape[0] * angle_cos) + abs(image.shape[1] * angle_sin)) + 1
    new_width = round(abs(image.shape[1] * angle_cos) + abs(image.shape[0] * angle_sin)) + 1

    output = np.zeros((new_height, new_width, image.shape[2]))

    original_centre_height = round(((image.shape[0] + 1) / 2) - 1)
    original_centre_width = round(((image.shape[1] + 1) / 2) - 1)

    new_centre_height = round(((new_height + 1) / 2) - 1)
    new_centre_width = round(((new_width + 1) / 2) - 1)

    for i in range(height):
        for j in range(width):
            # coordenadas com relação ao centro
            y = image.shape[0] - 1 - i - original_centre_height
            x = image.shape[1] - 1 - j - original_centre_width

            # rotação
            new_y = round(-x * angle_sin + y * angle_cos)
            new_x = round(x * angle_cos + y * angle_sin)

            # ajuste com relação ao novo centro
            new_y = new_centre_height - new_y
            new_x = new_centre_width - new_x

            output[new_y, new_x, :] = image[i, j, :]

    return helper.float3d_to_int3d(output)


if __name__ == '__main__':
    test_img = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ]
    )
    print(test_img)
    print(repetition_scale(test_img, horizontal_scale=2, vertical_scale=2))
    print(linear_scale(test_img, horizontal_scale=2, vertical_scale=2))
    print(rotate(test_img, 90))
