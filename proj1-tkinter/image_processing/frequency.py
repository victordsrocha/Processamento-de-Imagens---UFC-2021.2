"""
1: fazer todas as tarefas usando biblioteca
2. somente depois tentar substituir aos poucos por funções criadas por mim
3. na implementação não otimizada, utilizar o matplotlib para exibir as imagens
pois serão imagens 10x10 no máximo

implementação da versão otimizada:
https://towardsdatascience.com/fast-fourier-transform-937926e591cb
"""

from image_processing import helper
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
