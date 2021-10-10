import cv2 as cv
import numpy
import numpy as np


def normalize_as_float(img_mat):
    height, width, colors = img_mat.shape

    for y in range(0, height):
        for x in range(0, width):
            azul = img_mat.item(y, x, 0)
            verde = img_mat.item(y, x, 1)
            vermelho = img_mat.item(y, x, 2)

            img_mat.itemset((y, x, 0), 0)
            img_mat.itemset((y, x, 1), 0)

    return img_mat


def negative(img_matrix):
    height, width = img_matrix.shape

    for y in range(0, height):
        for x in range(0, width):
            img_matrix[y][x] = 1 - img_matrix[y][x]

    return img_matrix


def show_image(img, colors):
    from matplotlib import pyplot as plt
    if colors:
        # TODO convert bgr to rgb before
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.close()


def int_to_float_img(img_matrix):
    height, width = img_matrix.shape

    new_img_matrix = np.zeros(img_matrix.shape, dtype=float)

    for y in range(0, height):
        for x in range(0, width):
            new_img_matrix[y][x] = img_matrix[y][x] / 255

    return new_img_matrix


def float_to_int_img(img_matrix):
    height, width = img_matrix.shape

    new_img_matrix = np.zeros(img_matrix.shape, dtype='uint8')

    for y in range(0, height):
        for x in range(0, width):
            new_img_matrix[y][x] = numpy.uint8(img_matrix[y][x] * 255)

    return new_img_matrix


mat = np.array(([255, 255, 255, 255, 255],
                [255, 128, 128, 128, 255],
                [255, 128, 0, 128, 255],
                [255, 128, 128, 128, 255],
                [255, 255, 255, 255, 255]), dtype='uint8')

img_cat = cv.imread('images/grumpy-cat.jpg', cv.IMREAD_GRAYSCALE)

mat_float = int_to_float_img(img_cat)
neg = negative(mat_float)
neg = negative(neg)
mat_int = float_to_int_img(neg)

show_image(mat_int, False)

