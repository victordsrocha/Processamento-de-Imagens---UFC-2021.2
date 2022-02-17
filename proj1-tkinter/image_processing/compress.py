import math

import numpy as np


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


def rgb_to_hsv(rgb_image_int3d):
    hsv_image = np.zeros(rgb_image_int3d.shape)
    float3d = rgb_image_int3d / 255

    height, width, colors = float3d.shape

    for y in range(height):
        for x in range(width):
            r, g, b = float3d[y][x]
            h, s, v = pixel_rgb_to_hsv(r, g, b)
            hsv_image[y][x] = [h, s, v]

    return hsv_image


def avg_scale(float3d, delta):
    original_image = float3d

    new_image_height = int(np.round(float3d.shape[0] / delta))
    new_image_width = int(np.round(float3d.shape[1] / delta))

    new_image = np.zeros(shape=(new_image_height, new_image_width, 3))

    for y in range(new_image_height):
        for x in range(new_image_width):

            pixels = []
            for i in range(delta):
                for j in range(delta):
                    pixels.append(original_image[y * delta + i, x * delta + j, :])

            h, s, v = 0, 0, 0
            for p in pixels:
                h += p[0]
                s += p[1]
                v += p[2]

            h = h / len(pixels)
            s = s / len(pixels)
            v = v / len(pixels)

            new_image[y][x] = [h, s, v]

    return new_image


def compactar_hsv(rgb_int3d, delta=2):
    hsv_image = rgb_to_hsv(rgb_int3d)

    v_image = hsv_image[:, :, 2]

    scale_hsv_image = avg_scale(hsv_image, delta)
    h_image = scale_hsv_image[:, :, 0]
    s_image = scale_hsv_image[:, :, 1]

    h_image = (h_image * 255).astype('uint8')
    s_image = (s_image * 255).astype('uint8')
    v_image = (v_image * 255).astype('uint8')

    return h_image, s_image, v_image


def descompactar_hsv(fname):
    h_values = np.load(fname + '/h.npy')
    s_values = np.load(fname + '/s.npy')
    v_values = np.load(fname + '/v.npy')
    hsv_image = np.zeros(shape=(v_values.shape[0], v_values.shape[1], 3))
    delta = int(v_values.shape[0] / h_values.shape[0])

    for i in range(hsv_image.shape[0]):
        for j in range(hsv_image.shape[1]):
            hsv_image[i][j][2] = v_values[i][j] / 255
            hsv_image[i][j][0] = h_values[int(i / delta)][int(j / delta)] / 255
            hsv_image[i][j][1] = s_values[int(i / delta)][int(j / delta)] / 255

    rgb_image = hsv_to_rgb(hsv_image)
    rgb_image = (rgb_image * 255).astype('uint8')
    return rgb_image


def hsv_to_rgb(hsv_image):
    height, width = hsv_image.shape[0], hsv_image.shape[1]
    rgb_image = np.zeros(hsv_image.shape)

    for y in range(height):
        for x in range(width):
            h, s, v = hsv_image[y][x]
            r, g, b = pixel_hsv_to_rgb(h, s, v)
            rgb_image[y][x] = [r, g, b]

    return rgb_image


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


def compactacao_completa(rgb_int3d, delta=2):
    hsv_image = rgb_to_hsv(rgb_int3d)

    v_values = hsv_image[:, :, 2]

    scale_hsv_image = avg_scale(hsv_image, delta)
    h_values = scale_hsv_image[:, :, 0]
    s_values = scale_hsv_image[:, :, 1]

    h_values = (h_values * 255).astype('uint8')
    s_values = (s_values * 255).astype('uint8')
    v_values = (v_values * 255).astype('uint8')

    h_values = lzw_compactacao(h_values)
    s_values = lzw_compactacao(s_values)
    v_values = lzw_compactacao(v_values)

    h_values = np.array(h_values, dtype=np.uint16)
    s_values = np.array(s_values, dtype=np.uint16)
    v_values = np.array(v_values, dtype=np.uint32)

    return h_values, s_values, v_values


def descompactacao_completa(fname):
    h_values = np.load(fname + '/h.npy')
    s_values = np.load(fname + '/s.npy')
    v_values = np.load(fname + '/v.npy')

    h_height, h_width, h_values = lzw_descompactacao(list(h_values))
    s_height, s_width, s_values = lzw_descompactacao(list(s_values))
    v_height, v_width, v_values = lzw_descompactacao(list(v_values))

    h_image = unflatten(h_values, h_height, h_width)
    s_image = unflatten(s_values, s_height, s_width)
    v_image = unflatten(v_values, v_height, v_width)

    hsv_image = np.zeros(shape=(v_height, v_width, 3))
    delta = 2

    for i in range(hsv_image.shape[0]):
        for j in range(hsv_image.shape[1]):
            hsv_image[i][j][2] = v_image[i][j] / 255
            hsv_image[i][j][0] = h_image[int(i / delta)][int(j / delta)] / 255
            hsv_image[i][j][1] = s_image[int(i / delta)][int(j / delta)] / 255

    rgb_image = hsv_to_rgb(hsv_image)
    rgb_image = (rgb_image * 255).astype('uint8')
    return rgb_image


def unflatten(int1d, height, width):
    image = np.zeros(shape=(height, width))
    i = 0
    for y in range(height):
        for x in range(width):
            if i < len(int1d):
                image[y][x] = int1d[i]
                i += 1
            else:
                image[y][x] = 0

    return image


def compactacao_lzw_rgb(int3d):
    red = int3d[:, :, 0]
    green = int3d[:, :, 1]
    blue = int3d[:, :, 2]

    pass


def lzw_compactacao(int1d):
    image = int1d
    dict_size = 256
    result = [int1d.shape[0], int1d.shape[1]]
    idict = {str(i): i for i in range(dict_size)}
    img_flatten = image.flatten()
    w = ''
    for i in range(len(img_flatten)):
        c = str(img_flatten[i])

        if w != '':
            wc = w + ',' + c
        else:
            wc = c

        if wc in idict:
            w = wc
        else:
            result.append(idict[w])
            idict[wc] = dict_size
            dict_size += 1
            w = c
    if w:
        result.append(idict[w])
    return result


def lzw_descompactacao(compressed):
    dict_size = 256
    idict = {i: str(i) for i in range(dict_size)}
    saida = []

    height = compressed.pop(0)
    width = compressed.pop(0)

    s_proc = compressed.pop(0)
    s_rec = 0
    while (len(compressed)) > 0:
        s_rec = s_proc
        s_proc = compressed.pop(0)

        idict[dict_size] = str(s_rec) + ',' + str(s_proc)
        min_s_proc = s_proc
        while min_s_proc > 255:
            previous_s_proc = min_s_proc
            min_s_proc = int(idict[previous_s_proc].split(',')[0])
            # n2 = int(idict[previous_s_proc].split(',')[1])
            # compressed.insert(0, n2)

        if ',' in idict[s_rec]:
            saidas = idict[s_rec].split(',')
            for i in range(len(saidas)):
                saidas[i] = int(saidas[i])
            while max(saidas) > 255:
                new_saidas = []
                for s in saidas:
                    if s > 255:
                        new_saidas.extend(idict[s].split(','))
                    else:
                        new_saidas.append(s)
                saidas = new_saidas
                for i in range(len(saidas)):
                    saidas[i] = int(saidas[i])
            for s in saidas:
                saida.append(int(s))
        else:
            saida.append(int(s_rec))

        idict[dict_size] = str(s_rec) + ',' + str(min_s_proc)
        dict_size += 1
    saida.append(int(s_proc))
    return height, width, saida


if __name__ == '__main__':
    v_values = np.load(
        '/home/victor/code/Processamento-de-Imagens---UFC-2021.2/proj1-tkinter/data/compress/teste/v.npy')
    # v_values = np.array([[39, 39, 126, 126, 39, 39, 126, 126], [39, 39, 126, 126, 39, 39, 126, 126]])
    compressed = lzw_compactacao(v_values)
    result = lzw_descompactacao(compressed)
    print(result)
