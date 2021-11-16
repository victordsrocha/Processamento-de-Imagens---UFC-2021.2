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
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt


def fast_fourier(int1d):
    float1d = helper.int1d_to_float1d(int1d)
    image_f = np.real(fftshift(fft2(float1d)))
    image_f = np.clip(image_f, 0, 400)
    color_map = plt.cm.get_cmap('Greys')
    reversed_color_map = color_map.reversed()
    plt.imshow(image_f, cmap=reversed_color_map)
    plt.show()
