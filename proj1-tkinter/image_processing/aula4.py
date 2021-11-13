import numpy as np
import skimage.color
import matplotlib.pyplot as plt
import image_processing.helper as helper


def region_of_interest(matrix: np.ndarray, roi_center: tuple[int, int], roi_shape: tuple[int, int]) -> np.ndarray:
    """
    Creates a region of interest from a 2D numpy.ndarray (ROI)
    Args:
        matrix:
            A numpy.ndarray with two dimensions.
        roi_center:
            A tuple of integers containing the coordinates of the center of the region of interest.
            For example, if the center is in row 3 and column 4, roi_center should get (3,4)
        roi_shape:
            A tuple of integers containing the dimensions of the region of interest. The dimensions must be odd.
            For example, if the region of interest has height 3 and width 5, roi_shape should get (3,5)
    Returns:
        An ROI matrix (a cutout of the original matrix).
    """
    expanded_matrix = matrix.copy()
    expanded_matrix = np.pad(expanded_matrix, 5, mode='constant')

    roi_center_x, roi_center_y = roi_center
    roi_center_x += 5
    roi_center_y += 5
    roi_height, roi_width = roi_shape
    roi = expanded_matrix[
          roi_center_x - (roi_width - 1) // 2: 1 + roi_center_x + (roi_width - 1) // 2,
          roi_center_y - (roi_height - 1) // 2: 1 + roi_center_y + (roi_height - 1) // 2]
    return roi


def generic_filter2(int3d, kernel):
    image = skimage.color.rgb2gray(int3d)
    image_height, image_width = image.shape

    for lin in range(image_height):
        for col in range(image_width):
            roi = region_of_interest(image, (lin, col), kernel.shape)
            filtered_values = roi * kernel
            new_pixel_value = filtered_values.sum()
            image[lin][col] = new_pixel_value

    # normalize
    image = helper.normalize_data(image)

    return helper.float1d_to_int1d(image)


def generic_filter(int1d, kernel):
    def calc_roi(center):
        new_roi = np.zeros((kernel_size, kernel_size))
        x, y = 0, 0
        for i in range(center[0] - delta, center[0] + delta + 1):
            for j in range(center[1] - delta, center[1] + delta + 1):
                new_roi[y][x] = expanded_image[i][j]
                x += 1
            x = 0
            y += 1
        return new_roi

    new_image = helper.int1d_to_float1d(int1d)
    image_height, image_width = new_image.shape

    kernel_size = kernel.shape[0]

    expanded_image = np.pad(new_image, kernel_size - 1, mode='constant')
    new_expanded_image = expanded_image.copy()

    delta = int(np.ceil(kernel_size / 2) - 1)

    for lin in range(delta, new_expanded_image.shape[0] - delta):
        for col in range(delta, new_expanded_image.shape[1] - delta):
            roi = calc_roi((lin, col))
            filtered_values = roi * kernel
            new_pixel_value = filtered_values.sum()
            new_expanded_image[lin][col] = new_pixel_value

    new_expanded_image = new_expanded_image[
                         kernel_size - 1: new_expanded_image.shape[0] - kernel_size + 1,
                         kernel_size - 1: new_expanded_image.shape[1] - kernel_size + 1
                         ]

    # normalize
    new_expanded_image = helper.normalize_data(new_expanded_image)

    return helper.float1d_to_int1d(new_expanded_image)
