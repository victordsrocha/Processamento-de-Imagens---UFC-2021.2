import math
import numpy as np

from image_processing import helper


def rgb_to_gray(rgb_int3d, weighted=None):
    height, width, colors = rgb_int3d.shape
    rgb_float3d = helper.int3d_to_float3d(rgb_int3d)
    gray_image = np.zeros(shape=(height, width))

    for y in range(height):
        for x in range(width):
            r, g, b = rgb_float3d[y][x]
            if weighted is None:
                gray_image[y][x] = (r + g + b) / 3
            elif weighted == 'octave':
                # https://octave.org/doc/v4.4.1/Color-Conversion.html
                gray_image[y][x] = 0.298936 * r + 0.587043 * g + 0.114021 * b

    return helper.float1d_to_int1d(gray_image)


def rgb_to_hsv(rgb_image_int3d):
    hsv_image = np.zeros(rgb_image_int3d.shape)
    float3d = helper.int3d_to_float3d(rgb_image_int3d)

    height, width, colors = float3d.shape

    for y in range(height):
        for x in range(width):
            r, g, b = float3d[y][x]
            h, s, v = pixel_rgb_to_hsv(r, g, b)
            hsv_image[y][x] = [h, s, v]

    return hsv_image


def hsv_to_rgb(hsv_image):
    height, width = hsv_image.shape[0], hsv_image.shape[1]
    rgb_image = np.zeros(hsv_image.shape)

    for y in range(height):
        for x in range(width):
            h, s, v = hsv_image[y][x]
            r, g, b = pixel_hsv_to_rgb(h, s, v)
            rgb_image[y][x] = [r, g, b]

    return rgb_image


def pixel_rgb_to_hsv(r, g, b):
    # https://gist.github.com/mathebox/e0805f72e7db3269ec22
    r = float(r)
    g = float(g)
    b = float(b)
    high = max(r, g, b)
    low = min(r, g, b)
    h, s, v = high, high, high

    d = high - low
    s = 0 if high == 0 else d / high

    if high == low:
        h = 0.0
    else:
        h = {
            r: (g - b) / d + (6 if g < b else 0),
            g: (b - r) / d + 2,
            b: (r - g) / d + 4,
        }[high]
        h /= 6

    return h, s, v


def pixel_hsv_to_rgb(h, s, v):
    # https://gist.github.com/mathebox/e0805f72e7db3269ec22
    i = math.floor(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    r, g, b = [
        (v, t, p),
        (q, v, p),
        (p, v, t),
        (p, q, v),
        (t, p, v),
        (v, p, q),
    ][int(i % 6)]

    return r, g, b


if __name__ == '__main__':
    print(pixel_rgb_to_hsv(1, 0, 0))
