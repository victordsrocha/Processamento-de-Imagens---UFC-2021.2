import skimage.transform
import numpy as np
import skimage.io
import skimage.color
import image_processing.helper as helper
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from dialog import CustomDialog
from edit_image_dialog import CanvasDialog
from image_processing import intensity_transformation, spatial_transformation, frequency, image_restoration, \
    color_image_processing, compress


class GUI(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        w, h = 1325, 680
        master.minsize(width=w, height=h)
        master.maxsize(width=w, height=h)
        self.pack()

        self.kernel = None
        self.img_path = None
        self.img_array = None
        self.previous_img_array = None
        self.img_tk = None

        self.container0 = Frame(master)
        self.container0.pack(side=LEFT, fill='y')

        self.container1 = Frame(master)
        self.container1.pack(side=TOP, fill='y')
        lbl1 = Label(self.container1, text='aula 2: ')
        lbl1.pack(side=LEFT)

        self.button_browse = Button(self.container0, text='Browse', command=self.choose)
        self.button_save = Button(self.container0, text='Save', command=self.save)
        self.button_pyplot = Button(self.container0, text='pyplot', command=self.show_pyplot)
        self.button_restore = Button(self.container0, text='Restore', command=self.restore)
        self.button_undo = Button(self.container0, text='undo', command=self.undo)

        self.button_negative = Button(self.container1, text='Negative', command=self.negative)
        self.button_log = Button(self.container1, text='log', command=self.log_transformation)
        self.button_gama = Button(self.container1, text='gama', command=self.gama_transformation)
        self.button_linear = Button(self.container1, text='linear', command=self.linear_transformation)
        self.button_binary = Button(self.container1, text='binary', command=self.binary)
        self.button_bit_plane = Button(self.container1, text='bit plane', command=self.bit_plane)
        self.button_record_message = Button(self.container1, text='record message', command=self.record_message)
        self.button_read_message = Button(self.container1, text='read message', command=self.read_message)
        self.button_hist_gray = Button(self.container1, text='gray histogram', command=self.gray_hist)
        self.button_eq_gray = Button(self.container1, text='gray eq', command=self.gray_eq)

        self.button_browse.pack(side=TOP, fill='x')
        self.button_save.pack(side=TOP, fill='x')
        self.button_pyplot.pack(side=TOP, fill='x')
        self.button_restore.pack(side=TOP, fill='x')
        self.button_undo.pack(side=TOP, fill='x')
        self.button_negative.pack(side=LEFT)
        self.button_log.pack(side=LEFT)
        self.button_gama.pack(side=LEFT)
        self.button_linear.pack(side=LEFT)
        lbl2 = Label(self.container1, text='aula 3: ')
        lbl2.pack(side=LEFT)
        self.button_binary.pack(side=LEFT)
        self.button_bit_plane.pack(side=LEFT)
        self.button_record_message.pack(side=LEFT)
        self.button_read_message.pack(side=LEFT)
        self.button_hist_gray.pack(side=LEFT)
        self.button_eq_gray.pack(side=LEFT)

        self.container2 = Frame(master)
        self.container2.pack(side=TOP, fill='y')
        lbl4 = Label(self.container2, text='aula 4: ')
        lbl4.pack(side=LEFT)
        self.button_generic_filter = Button(self.container2, text='generic filter', command=self.generic_filter)
        self.button_generic_filter.pack(side=LEFT)
        self.button_box_filter = Button(self.container2, text='box filter', command=self.box_filter)
        self.button_box_filter.pack(side=LEFT)
        self.button_gaussian_filter = Button(self.container2, text='gaussian filter', command=self.gaussian_filter)
        self.button_gaussian_filter.pack(side=LEFT)
        self.button_median = Button(self.container2, text='median filter', command=self.median_filter)
        self.button_median.pack(side=LEFT)
        lbl5 = Label(self.container2, text='aula 5: ')
        lbl5.pack(side=LEFT)
        self.button_laplace = Button(self.container2, text='laplacian filter', command=self.laplacian_filter)
        self.button_laplace.pack(side=LEFT)
        self.button_laplace_blend = Button(self.container2, text='laplacian filter blend',
                                           command=self.laplacian_filter_blend)
        self.button_laplace_blend.pack(side=LEFT)
        self.button_high_boost = Button(self.container2, text='high boost', command=self.high_boost)
        self.button_high_boost.pack(side=LEFT)
        lbl6 = Label(self.container2, text='aula 6: ')
        lbl6.pack(side=LEFT)
        self.button_sobel_x = Button(self.container2, text='sobel x', command=self.sobel_x)
        self.button_sobel_x.pack(side=LEFT)
        self.button_sobel_y = Button(self.container2, text='sobel y', command=self.sobel_y)
        self.button_sobel_y.pack(side=LEFT)
        self.button_sobel_mag = Button(self.container2, text='magnitude', command=self.sobel_magnitude)
        self.button_sobel_mag.pack(side=LEFT)

        self.container3 = Frame(master)
        self.container3.pack(side=TOP, fill='y')
        lbl7 = Label(self.container3, text='aula 7-9: ')
        lbl7.pack(side=LEFT)
        self.button_fourier = Button(self.container3, text='fourier', command=self.fourier)
        self.button_fourier.pack(side=LEFT)
        self.button_fast_fourier = Button(self.container3, text='fast fourier', command=self.fast_fourier)
        self.button_fast_fourier.pack(side=LEFT)
        self.button_filtro_passa_rejeita = Button(self.container3, text='filtro passa/rejeita',
                                                  command=self.filtro_passa_rejeita)
        self.button_filtro_passa_rejeita.pack(side=LEFT)
        lbl10 = Label(self.container3, text='aula 10: ')
        lbl10.pack(side=LEFT)
        self.button_media_geometrica = Button(self.container3, text='media geometrica', command=self.media_geometrica)
        self.button_media_geometrica.pack(side=LEFT)
        self.button_media_harmonica = Button(self.container3, text='media harmonica', command=self.media_harmonica)
        self.button_media_harmonica.pack(side=LEFT)
        self.button_media_contra_harmonica = Button(self.container3, text='media contra harmonica',
                                                    command=self.media_contra_harmonica)
        self.button_media_contra_harmonica.pack(side=LEFT)

        self.container4 = Frame(master)
        self.container4.pack(side=TOP, fill='y')
        lbl11 = Label(self.container4, text='aula 11: ')
        lbl11.pack(side=LEFT)
        self.button_cinza = Button(self.container4, text='cinza', command=self.rgb_to_gray)
        self.button_cinza.pack(side=LEFT)
        self.button_cinza_octave = Button(self.container4, text='cinza ponderado', command=self.rgb_to_gray_weighted)
        self.button_cinza_octave.pack(side=LEFT)
        self.button_show_hsv = Button(self.container4, text='hsv', command=self.show_hsv)
        self.button_show_hsv.pack(side=LEFT)
        self.button_matiz = Button(self.container4, text='matiz', command=self.matiz)
        self.button_matiz.pack(side=LEFT)
        self.button_saturacao = Button(self.container4, text='saturacao', command=self.saturacao)
        self.button_saturacao.pack(side=LEFT)
        self.button_brilho = Button(self.container4, text='brilho', command=self.brilho)
        self.button_brilho.pack(side=LEFT)
        lbl12 = Label(self.container4, text='aula 12: ')
        lbl12.pack(side=LEFT)
        self.button_chroma_key = Button(self.container4, text='chroma key', command=self.chroma_key)
        self.button_chroma_key.pack(side=LEFT)
        self.button_color_hist = Button(self.container4, text='histograma', command=self.color_hist)
        self.button_color_hist.pack(side=LEFT)
        self.button_color_eq = Button(self.container4, text='equaliza????o (hsv)', command=self.color_eq)
        self.button_color_eq.pack(side=LEFT)
        self.button_suave_color = Button(self.container4, text='box filter hsv', command=self.suave_color)
        self.button_suave_color.pack(side=LEFT)
        self.button_laplace_color = Button(self.container4, text='laplace filter hsv', command=self.laplace_color)
        self.button_laplace_color.pack(side=LEFT)

        self.container5 = Frame(master)
        self.container5.pack(side=TOP, fill='y')
        lbl13 = Label(self.container5, text='aula 13: ')
        lbl13.pack(side=LEFT)
        self.button_scale_rep = Button(self.container5, text='escala repeti????o', command=self.scale_repetition)
        self.button_scale_rep.pack(side=LEFT)
        self.button_scale = Button(self.container5, text='escala bilinear', command=self.linear_scale)
        self.button_scale.pack(side=LEFT)
        self.button_rotate = Button(self.container5, text='rotate', command=self.rotate)
        self.button_rotate.pack(side=LEFT)

        lbl14 = Label(self.container5, text=' / compress??o: ')
        lbl14.pack(side=LEFT)
        self.button_compress1s = Button(self.container5, text='compactar simples', command=self.compress1)
        self.button_compress1s.pack(side=LEFT)
        self.button_descompactars = Button(self.container5, text='descompactar simples', command=self.descompactar)
        self.button_descompactars.pack(side=LEFT)
        self.button_compress1 = Button(self.container5, text='compactar', command=self.compress_completo)
        self.button_compress1.pack(side=LEFT)
        self.button_descompactar = Button(self.container5, text='descompactar', command=self.descompactar_completo)
        self.button_descompactar.pack(side=LEFT)

        self.container_panel = Frame(master)
        self.container_panel.pack(side=LEFT, fill='y')
        self.panel = Label(self.container_panel)
        self.panel.pack(side=TOP, fill='y', expand=True)

        self.kernel_area = Frame(master)
        self.kernel_area.pack(side=RIGHT, fill='y', expand=True)
        self.button_update_kernel = Button(self.kernel_area, text='update kernel', command=self.update_kernel)
        self.button_update_kernel.pack()
        self.kernel_area_0 = Frame(self.kernel_area)
        self.kernel_area_0.pack(side=TOP)
        self.kernel_entry00 = Entry(self.kernel_area_0, width=3)
        self.kernel_entry00.insert(END, '1')
        self.kernel_entry00.pack(side=LEFT)
        self.kernel_entry01 = Entry(self.kernel_area_0, width=3)
        self.kernel_entry01.pack(side=LEFT)
        self.kernel_entry01.insert(END, '1')
        self.kernel_entry02 = Entry(self.kernel_area_0, width=3)
        self.kernel_entry02.pack(side=LEFT)
        self.kernel_entry02.insert(END, '1')
        self.kernel_area_1 = Frame(self.kernel_area)
        self.kernel_area_1.pack(side=TOP)
        self.kernel_entry10 = Entry(self.kernel_area_1, width=3)
        self.kernel_entry10.pack(side=LEFT)
        self.kernel_entry10.insert(END, '1')
        self.kernel_entry11 = Entry(self.kernel_area_1, width=3)
        self.kernel_entry11.pack(side=LEFT)
        self.kernel_entry11.insert(END, '1')
        self.kernel_entry12 = Entry(self.kernel_area_1, width=3)
        self.kernel_entry12.pack(side=LEFT)
        self.kernel_entry12.insert(END, '1')
        self.kernel_area_2 = Frame(self.kernel_area)
        self.kernel_area_2.pack(side=TOP)
        self.kernel_entry20 = Entry(self.kernel_area_2, width=3)
        self.kernel_entry20.pack(side=LEFT)
        self.kernel_entry20.insert(END, '1')
        self.kernel_entry21 = Entry(self.kernel_area_2, width=3)
        self.kernel_entry21.pack(side=LEFT)
        self.kernel_entry21.insert(END, '1')
        self.kernel_entry22 = Entry(self.kernel_area_2, width=3)
        self.kernel_entry22.pack(side=LEFT)
        self.kernel_entry22.insert(END, '1')

        self.start()

    def start(self):
        self.img_path = 'data/compress/benchmark.bmp'
        self.img_array = skimage.io.imread(fname=self.img_path)
        self.previous_img_array = self.img_array
        self.show_image()
        self.update_kernel()

    def choose(self):
        img_path = filedialog.askopenfilename(filetypes=[
            ('image', '.jpg'),
            ('image', '.bmp')
        ])
        if len(img_path) > 0:
            self.img_path = img_path
            self.img_array = skimage.io.imread(fname=self.img_path)
            self.previous_img_array = self.img_array
            self.show_image()
        else:
            pass

    def save(self):
        img_path = filedialog.asksaveasfile(filetypes=[
            ('image', '.jpg'),
            ('image', '.bmp')
        ])
        skimage.io.imsave(fname=img_path.name, arr=self.img_array)

    def show_pyplot(self):
        plt.imshow(self.img_array, cmap='gray', vmin=0, vmax=255)
        plt.show()

    def restore(self):
        self.previous_img_array = self.img_array
        self.img_array = skimage.io.imread(fname=self.img_path)
        self.show_image()

    def undo(self):
        aux = self.img_array
        self.img_array = self.previous_img_array
        self.previous_img_array = aux
        self.show_image()

    def negative(self):
        self.previous_img_array = self.img_array
        self.img_array = intensity_transformation.negative(self.img_array)
        self.show_image()

    def log_transformation(self):
        self.previous_img_array = self.img_array
        value = CustomDialog(self, "c").show()
        value = float(value)
        self.img_array = intensity_transformation.log_transformation(self.img_array, value)
        self.show_image()

    def gama_transformation(self):
        self.previous_img_array = self.img_array
        dialog_return = CustomDialog(self, "gama;c").show()
        gama, c = dialog_return.split(';')
        gama = float(gama)
        c = float(c)
        self.img_array = intensity_transformation.gama_transformation(
            self.img_array, gama=gama, c=c)
        self.show_image()

    def linear_transformation(self):
        self.previous_img_array = self.img_array
        dialog_return = CustomDialog(self, "point;point;...").show()
        points_str_list = dialog_return.split(';')
        points = []
        for point_str in points_str_list:
            x, y = point_str.split(',')
            points.append((float(x), float(y)))
        self.img_array = intensity_transformation.linear_transformation(
            self.img_array, points)
        self.show_image()

    def binary(self):
        self.previous_img_array = self.img_array
        value = CustomDialog(self, "threshold").show()
        value = int(value)
        self.img_array = intensity_transformation.binary(self.img_array, value)
        self.show_image()

    def bit_plane(self):
        self.previous_img_array = self.img_array
        value = CustomDialog(self, "bit").show()
        value = int(value)
        self.img_array = intensity_transformation.bit_plane(self.img_array, value)
        self.show_image()

    def record_message(self):
        self.previous_img_array = self.img_array
        value = CustomDialog(self, "message").show()
        self.img_array = intensity_transformation.record_message(self.img_array, value)
        self.show_image()

    def read_message(self):
        message = intensity_transformation.read_message(self.img_array)
        print(r'{}'.format(message))
        messagebox.showinfo(title=r'message', message=message)

    def gray_hist(self):
        intensity_transformation.gray_histogram(self.img_array)

    def color_hist(self):
        color_image_processing.color_histogram(self.img_array)

    def show_hsv(self):
        color_image_processing.show_hsv(self.img_array)

    def gray_eq(self):
        self.previous_img_array = self.img_array
        self.img_array = intensity_transformation.gray_eq(self.img_array)
        self.show_image()

    def color_eq(self):
        # Equaliza????o
        self.previous_img_array = self.img_array
        self.img_array = color_image_processing.hsv_equalization(self.img_array)
        self.show_image()

    def update_kernel(self):
        self.kernel = np.zeros(shape=(3, 3))
        self.kernel[0][0] = (float(self.kernel_entry00.get()))
        self.kernel[0][1] = (float(self.kernel_entry01.get()))
        self.kernel[0][2] = (float(self.kernel_entry02.get()))
        self.kernel[1][0] = (float(self.kernel_entry10.get()))
        self.kernel[1][1] = (float(self.kernel_entry11.get()))
        self.kernel[1][2] = (float(self.kernel_entry12.get()))
        self.kernel[2][0] = (float(self.kernel_entry20.get()))
        self.kernel[2][1] = (float(self.kernel_entry21.get()))
        self.kernel[2][2] = (float(self.kernel_entry22.get()))

    def generic_filter(self):
        self.previous_img_array = self.img_array
        self.img_array = spatial_transformation.generic_filter(self.img_array, self.kernel)
        self.show_image()

    def box_filter(self):
        self.previous_img_array = self.img_array
        value = CustomDialog(self, "kernel size").show()
        value = int(value)
        self.img_array = spatial_transformation.box_filter(self.img_array, value)
        self.show_image()

    def matiz(self):
        self.previous_img_array = self.img_array
        value = CustomDialog(self, "delta").show()
        value = float(value)
        self.img_array = color_image_processing.altera_matiz(self.img_array, value)
        self.show_image()

    def saturacao(self):
        self.previous_img_array = self.img_array
        value = CustomDialog(self, "delta").show()
        value = float(value)
        self.img_array = color_image_processing.altera_saturacao(self.img_array, value)
        self.show_image()

    def brilho(self):
        self.previous_img_array = self.img_array
        value = CustomDialog(self, "delta").show()
        value = float(value)
        self.img_array = color_image_processing.altera_brilho(self.img_array, value)
        self.show_image()

    def gaussian_filter(self):
        self.previous_img_array = self.img_array
        value = CustomDialog(self, "kernel size").show()
        value = int(value)
        self.img_array = spatial_transformation.gaussian_smoothing_filter(self.img_array, value)
        self.show_image()

    def median_filter(self):
        self.previous_img_array = self.img_array
        value = CustomDialog(self, "kernel size").show()
        value = int(value)
        self.img_array = spatial_transformation.median_filter(self.img_array, value)
        self.show_image()

    def laplacian_filter(self):
        self.previous_img_array = self.img_array
        self.img_array = spatial_transformation.laplacian_filter(self.img_array)
        self.show_image()

    def laplacian_filter_blend(self):
        self.previous_img_array = self.img_array
        alpha = CustomDialog(self, "alpha").show()
        alpha = int(alpha)
        self.img_array = spatial_transformation.blend(self.img_array, alpha)
        self.show_image()

    def high_boost(self):
        self.previous_img_array = self.img_array
        alpha = CustomDialog(self, "alpha").show()
        alpha = int(alpha)
        kernel_size = CustomDialog(self, "kernel size").show()
        kernel_size = int(kernel_size)
        self.img_array = spatial_transformation.high_boost(self.img_array, alpha=alpha, kernel_size=kernel_size)
        self.show_image()

    def sobel_x(self):
        self.previous_img_array = self.img_array
        self.img_array = spatial_transformation.sobel_x(self.img_array)
        self.show_image()

    def sobel_y(self):
        self.previous_img_array = self.img_array
        self.img_array = spatial_transformation.sobel_y(self.img_array)
        self.show_image()

    def sobel_magnitude(self):
        self.previous_img_array = self.img_array
        self.img_array = spatial_transformation.sobel_magnitude(self.img_array)
        self.show_image()

    def fourier(self):
        threshold = CustomDialog(self, "threshold").show()
        threshold = int(threshold)
        frequency.fourier_discrete_transform(self.img_array, threshold)

    def fast_fourier(self):
        self.previous_img_array = self.img_array
        threshold = CustomDialog(self, "threshold").show()
        threshold = int(threshold)
        fourier_image, itp = frequency.fast_fourier(self.img_array, threshold)
        edited_fourier_mask = CanvasDialog(self, width=50, int1d=fourier_image).show()
        self.img_array = frequency.fast_fourier_inverse(edited_fourier_mask, itp)
        self.show_image()

    def filtro_passa_rejeita(self):
        self.previous_img_array = self.img_array
        threshold = CustomDialog(self, "threshold").show()
        threshold = int(threshold)
        ratio1 = CustomDialog(self, "tamanho circulo").show()
        ratio1 = float(ratio1)
        tipo = CustomDialog(self, "tipo: alta, baixa, faixa alta, faixa baixa").show()
        tipo = str(tipo)

        ratio2 = None
        gaussian = False
        if tipo == 'alta' or tipo == 'baixa':
            gaussian = CustomDialog(self, "gaussian").show()
            gaussian = bool(int(gaussian))
        else:
            ratio2 = CustomDialog(self, "tamanho circulo interno").show()
            ratio2 = float(ratio2)

        self.img_array = frequency.filtro_passa_alta(self.img_array,
                                                     threshold, ratio1,
                                                     ratio2=ratio2,
                                                     tipo=tipo,
                                                     gaussian=gaussian)
        self.show_image()

    def media_geometrica(self):
        self.previous_img_array = self.img_array
        value = CustomDialog(self, "kernel size").show()
        value = int(value)
        self.img_array = image_restoration.media_geometrica(self.img_array, value)
        self.show_image()

    def media_harmonica(self):
        self.previous_img_array = self.img_array
        value = CustomDialog(self, "kernel size").show()
        value = int(value)
        self.img_array = image_restoration.media_harmonica(self.img_array, value)
        self.show_image()

    def rgb_to_gray(self):
        self.previous_img_array = self.img_array
        self.img_array = color_image_processing.rgb_to_gray(self.img_array)
        self.show_image()

    def rgb_to_gray_weighted(self):
        self.previous_img_array = self.img_array
        self.img_array = color_image_processing.rgb_to_gray(self.img_array, weighted='octave')
        self.show_image()

    def suave_color(self):
        self.previous_img_array = self.img_array
        value = CustomDialog(self, "kernel size").show()
        value = int(value)
        self.img_array = color_image_processing.suavizacao_hsv(self.img_array, kernel_size=value)
        self.show_image()

    def laplace_color(self):
        self.previous_img_array = self.img_array
        value = CustomDialog(self, "alpha").show()
        value = float(value)
        self.img_array = color_image_processing.laplace_hsv(self.img_array, alpha=value)
        self.show_image()

    def chroma_key(self):
        self.previous_img_array = self.img_array

        chroma_img_path = filedialog.askopenfilename(filetypes=[
            ('image', '.jpg'),
            ('image', '.bmp')
        ])

        chroma_img_array = None
        if len(chroma_img_path) > 0:
            chroma_img_array = skimage.io.imread(fname=chroma_img_path)
            chroma_img_array = skimage.transform.resize(
                chroma_img_array, (self.img_array.shape[0], self.img_array.shape[1]), anti_aliasing=True)
        else:
            pass

        threshold = CustomDialog(self, "threshold [0,1.73]").show()
        threshold = float(threshold)
        self.img_array = color_image_processing.chroma_key(self.img_array, threshold, chroma_img_array)
        self.show_image()

    def media_contra_harmonica(self):
        self.previous_img_array = self.img_array
        value = CustomDialog(self, "kernel size").show()
        value = int(value)
        q_value = CustomDialog(self, "Q value").show()
        q_value = int(value)
        self.img_array = image_restoration.media_contra_harmonica(self.img_array, kernel_size=value, Q=q_value)
        self.show_image()

    def linear_scale(self):
        self.previous_img_array = self.img_array
        horizontal_scale = CustomDialog(self, "horizontal_scale").show()
        horizontal_scale = float(horizontal_scale)
        vertical_scale = CustomDialog(self, "vertical_scale").show()
        vertical_scale = float(vertical_scale)
        self.img_array = spatial_transformation.linear_scale(self.img_array,
                                                             horizontal_scale=horizontal_scale,
                                                             vertical_scale=vertical_scale)
        self.show_image()

    def rotate(self):
        self.previous_img_array = self.img_array
        angle = CustomDialog(self, "angle").show()
        angle = float(angle)
        self.img_array = spatial_transformation.rotate(self.img_array, angle)
        self.show_image()

    def scale_repetition(self):
        self.previous_img_array = self.img_array
        horizontal_scale = CustomDialog(self, "horizontal_scale").show()
        horizontal_scale = float(horizontal_scale)
        vertical_scale = CustomDialog(self, "vertical_scale").show()
        vertical_scale = float(vertical_scale)
        self.img_array = spatial_transformation.repetition_scale(self.img_array,
                                                                 horizontal_scale=horizontal_scale,
                                                                 vertical_scale=vertical_scale)
        self.show_image()

    def compress1(self):
        delta = CustomDialog(self, "delta").show()
        delta = int(delta)
        h_values, s_values, v_values = compress.compactar_hsv(self.img_array, delta=delta)
        np.save('/home/victor/code/Processamento-de-Imagens---UFC-2021.2/proj1-tkinter/data/compress/teste/h.npy',
                h_values)
        np.save('/home/victor/code/Processamento-de-Imagens---UFC-2021.2/proj1-tkinter/data/compress/teste/s.npy',
                s_values)
        np.save('/home/victor/code/Processamento-de-Imagens---UFC-2021.2/proj1-tkinter/data/compress/teste/v.npy',
                v_values)

    def compress_completo(self):
        h_values, s_values, v_values = compress.compactacao_completa(self.img_array)
        np.save('/home/victor/code/Processamento-de-Imagens---UFC-2021.2/proj1-tkinter/data/compress/teste/h.npy',
                h_values)
        np.save('/home/victor/code/Processamento-de-Imagens---UFC-2021.2/proj1-tkinter/data/compress/teste/s.npy',
                s_values)
        np.save('/home/victor/code/Processamento-de-Imagens---UFC-2021.2/proj1-tkinter/data/compress/teste/v.npy',
                v_values)

    def descompactar(self):
        self.previous_img_array = self.img_array
        self.img_array = compress.descompactar_hsv(
            fname='/home/victor/code/Processamento-de-Imagens---UFC-2021.2/proj1-tkinter/data/compress/teste',
        )
        self.show_image()

    def descompactar_completo(self):
        self.previous_img_array = self.img_array
        self.img_array = compress.descompactacao_completa(
            fname='/home/victor/code/Processamento-de-Imagens---UFC-2021.2/proj1-tkinter/data/compress/teste',
        )
        self.show_image()

    def show_image(self):
        max_size = 500
        height, width = self.img_array.shape[0], self.img_array.shape[1]
        if height > max_size or width > max_size:
            if height > width:
                ratio = height / max_size
            else:
                ratio = width / max_size
            new_height = height // ratio
            new_width = width // ratio
            image_rescaled = skimage.transform.resize(self.img_array, (new_height, new_width), anti_aliasing=True)
            image_rescaled = helper.float1d_to_int1d(image_rescaled)
            image_to_show = image_rescaled
        else:
            image_to_show = self.img_array
        self.img_tk = ImageTk.PhotoImage(image=Image.fromarray(image_to_show))
        self.panel.configure(image=self.img_tk)
        self.panel.image = self.img_tk


root = Tk()
app = GUI(master=root)
app.mainloop()
# root.destroy()
