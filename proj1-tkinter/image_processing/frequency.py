"""
1: fazer todas as tarefas usando biblioteca
2. somente depois tentar substituir aos poucos por funções criadas por mim
3. na implementação não otimizada, utilizar o matplotlib para exibir as imagens
pois serão imagens 10x10 no máximo

implementação da versão otimizada:
https://towardsdatascience.com/fast-fourier-transform-937926e591cb
"""
import math

import image_processing.spatial_transformation
from image_processing import helper, spatial_transformation
import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift
import matplotlib.pyplot as plt


def fast_fourier(int1d, threshold):
    float1d = helper.int1d_to_float1d(int1d)
    itp = fft2(float1d)
    itp = fftshift(itp)
    itp_image = np.abs(itp)
    itp_image = np.clip(itp_image, 0, threshold)
    itp_image = helper.normalize_data(itp_image)

    # plt.imshow(itp_image, cmap='Greys')
    # plt.show()

    return helper.float1d_to_int1d(itp_image), itp
    # plt.imshow(image_f, cmap='Greys')
    # plt.show()


def fast_fourier_inverse(int1d, itp):
    float1d = helper.int1d_to_float1d(int1d)
    itp = itp * float1d

    # itp_image = np.abs(itp)
    # itp_image = np.clip(itp_image, 0, 500)
    # itp_image = helper.normalize_data(itp_image)
    # plt.imshow(itp_image, cmap='Greys')
    # plt.show()

    itp = ifftshift(itp)
    ip = np.real(ifft2(itp))
    ip = np.clip(ip, 0, 1)
    return helper.float1d_to_int1d(ip)


def DFT_slow(int1d, threshold):
    """Compute the discrete Fourier Transform of the 1D array x"""
    int1d = np.asarray(int1d, dtype=float)
    N = int1d.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, int1d)


def my_slow_fdt(float1d):
    # https://www.corsi.univr.it/documenti/OccorrenzaIns/matdid/matdid027832.pdf (pag 26)
    M, N = float1d.shape
    fdt = np.zeros(shape=(M, N), dtype=complex)
    for k in range(M):
        for l in range(N):
            value = 0j
            for m in range(M):
                for n in range(N):
                    value += float1d[m][n] * np.exp(-2j * np.pi * (((k * m) / M) + ((l * n) / N)))
            fdt[k][l] = value
    return fdt


def my_slow_ifdt(fdt):
    # https://www.corsi.univr.it/documenti/OccorrenzaIns/matdid/matdid027832.pdf (pag 26)
    M, N = fdt.shape
    ifdt = np.zeros(shape=(M, N), dtype=complex)
    for m in range(M):
        for n in range(N):
            value = 0j
            for k in range(M):
                for l in range(N):
                    value += fdt[k][l] * np.exp(2j * np.pi * (((k * m) / M) + ((l * n) / N)))
            ifdt[m][n] = value * (1 / (M * N))
    return ifdt


def fourier_discrete_transform(int1d, threshold):
    float1d = helper.int1d_to_float1d(int1d)

    fdt = my_slow_fdt(float1d)
    fdt_shifted = fftshift(fdt)
    fdt_shifted_image = np.abs(fdt_shifted)
    fdt_shifted_image = np.clip(fdt_shifted_image, 0, threshold)
    fdt_shifted_image = helper.normalize_data(fdt_shifted_image)
    # fdt_shifted_image = 1 - fdt_shifted_image

    ifdt = my_slow_ifdt(fdt)
    ifdt = np.real(ifdt)
    ifdt = np.clip(ifdt, 0, 1)

    ffdt = fft2(float1d)
    ffdt_shifted = fftshift(ffdt)
    ffdt_shifted_image = np.abs(ffdt_shifted)
    ffdt_shifted_image = np.clip(ffdt_shifted_image, 0, threshold)
    ffdt_shifted_image = helper.normalize_data(ffdt_shifted_image)
    # ffdt_shifted_image = 1 - ffdt_shifted_image

    iffdt = ifft2(ffdt)
    iffdt = np.real(iffdt)
    iffdt = np.clip(iffdt, 0, 1)

    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.suptitle('...')
    ax[0][0].imshow(fdt_shifted_image, cmap='gray', vmin=0, vmax=1)
    ax[0][1].imshow(ffdt_shifted_image, cmap='gray', vmin=0, vmax=1)
    ax[1][0].imshow(ifdt, cmap='gray', vmin=0, vmax=1)
    ax[1][1].imshow(iffdt, cmap='gray', vmin=0, vmax=1)
    plt.show()


def filtro_passa_alta(int1d, threshold, ratio, tipo='alta', gaussian=False):
    float1d = helper.int1d_to_float1d(int1d)
    itp = fft2(float1d)
    itp = fftshift(itp)

    itp_image = np.abs(itp)
    itp_image = np.clip(itp_image, 0, threshold)
    itp_image = helper.normalize_data(itp_image)

    filter_mask = draw_circle(np.zeros(float1d.shape), ratio)  # fundo preto, circulo branco

    if gaussian:
        filter_mask = draw_gaussian_circle(filter_mask.shape, ratio)

    if tipo == 'alta':
        filter_mask = 1 - filter_mask

    plt.imshow(filter_mask, cmap='gray')
    plt.show()

    filtered_itp = itp * filter_mask

    filtered_itp_image = np.abs(filtered_itp)
    filtered_itp_image = np.clip(filtered_itp_image, 0, threshold)
    filtered_itp_image = helper.normalize_data(filtered_itp_image)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.suptitle('...')
    ax[0].imshow(itp_image, cmap='gray', vmin=0, vmax=1)
    ax[1].imshow(filtered_itp_image, cmap='gray', vmin=0, vmax=1)
    plt.show()

    itp = ifftshift(filtered_itp)
    ip = np.real(ifft2(itp))
    ip = np.clip(ip, 0, 1)
    return helper.float1d_to_int1d(ip)


def draw_circle(fourier_image, ratio):
    image = fourier_image.copy()
    height, width = image.shape
    circle_diameter = np.round(np.min([height, width]) * ratio)
    circle_ratio = circle_diameter / 2
    for x in range(width):
        for y in range(height):
            if (x - width / 2) ** 2 + ((-y) + height / 2) ** 2 <= (circle_ratio ** 2):
                image[y][x] = 1.0
    return image


def draw_gaussian_circle(image_shape, ratio):
    def fspecial_gauss(size, sigma):
        # https://newbedev.com/how-to-obtain-a-gaussian-filter-in-python
        """Function to mimic the 'fspecial' gaussian MATLAB function
        """

        x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        return g / g.sum()

    height, width = image_shape
    circle_diameter = np.round(np.min([height, width]))
    circle_gaussian_mask = fspecial_gauss(circle_diameter, sigma=ratio ** 3 * circle_diameter)
    circle_gaussian_mask_correct_shape = circle_gaussian_mask

    if height > width:
        pad_value = height - circle_gaussian_mask.shape[0]
        if pad_value % 2 == 0:
            parcial_pad = int(pad_value / 2)
            circle_gaussian_mask_correct_shape = np.pad(circle_gaussian_mask,
                                                        ((parcial_pad, parcial_pad), (0, 0)),
                                                        constant_values=0)
        else:
            parcial_pad = int(pad_value / 2)
            circle_gaussian_mask_correct_shape = np.pad(circle_gaussian_mask,
                                                        ((parcial_pad, parcial_pad + 1), (0, 0)),
                                                        constant_values=0)
    elif width > height:
        pad_value = width - circle_gaussian_mask.shape[1]
        if pad_value % 2 == 0:
            parcial_pad = int(pad_value / 2)
            circle_gaussian_mask_correct_shape = np.pad(circle_gaussian_mask,
                                                        ((0, 0), (parcial_pad, parcial_pad)),
                                                        constant_values=0)
        else:
            parcial_pad = int(pad_value / 2)
            circle_gaussian_mask_correct_shape = np.pad(circle_gaussian_mask,
                                                        ((0, 0), (parcial_pad, parcial_pad + 1)),
                                                        constant_values=0)

    # plt.imshow(expanded_circle_filter, cmap='gray')
    # plt.show()

    normalized_mask = helper.normalize_data(circle_gaussian_mask_correct_shape)

    return normalized_mask


def draw_gaussian_circle2(circle_image):
    # pesquisar como fazer um circulo gaussiano
    # por enquanto estou somente suavizado de acordo com a distancia ao centro

    def dist_center(point):
        return math.dist(point, (0, 0))

    image = circle_image.copy()
    height, width = image.shape

    alpha = 0.1

    for x in range(width):
        for y in range(height):
            if image[y][x] != 0:
                dist = dist_center((x - width / 2, (-y) + height / 2))
                dist = dist * alpha
                image[y][x] = image[y][x] / (1 + dist)
                if image[y][x] > 1:
                    image[y][x] = 1.0

    plt.imshow(circle_image, cmap='gray')
    plt.show()
    plt.imshow(image, cmap='gray')
    plt.show()
    return image


if __name__ == '__main__':
    draw_gaussian_circle((501, 1000), 0.5)
