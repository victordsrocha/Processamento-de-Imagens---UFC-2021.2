import numpy as np

import skimage.io
import skimage.color


def start():
    image = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype='uint8')
    skimage.io.imsave(fname='data/test_images/img_test_1.jpg', arr=image)


if __name__ == '__main__':
    start()
