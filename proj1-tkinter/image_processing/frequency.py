"""
1: fazer todas as tarefas usando biblioteca
2. somente depois tentar substituir aos poucos por funções criadas por mim
3. na implementação não otimizada, utilizar o matplotlib para exibir as imagens
pois serão imagens 10x10 no máximo

* é melhor não tentar fazer a tarefa de "pincel"
provavelmente já estou bem encaminhado para tirar uma boa nota nesta disciplina e
preciso priorizar "passar em tudo"

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
