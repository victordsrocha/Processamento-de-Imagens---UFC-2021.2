from tkinter import *
from tkinter import filedialog, messagebox
import skimage.transform
import numpy as np
from PIL import Image, ImageTk
from dialog import CustomDialog
from edit_image_dialog import CanvasDialog
from image_processing import intensity_transformation, spatial_transformation, frequency
import skimage.io
import skimage.color
import image_processing.helper as helper
import matplotlib.pyplot as plt


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

        self.button_gray = Button(self.container1, text='gray', command=self.gray)
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
        self.button_gray.pack(side=LEFT)
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
        self.button_filtro_passa_alta = Button(self.container3, text='filtro passa alta',
                                               command=self.filtro_passa_alta)
        self.button_filtro_passa_alta.pack(side=LEFT)
        self.button_filtro_passa_baixa = Button(self.container3, text='filtro passa baixa',
                                                command=self.filtro_passa_baixa)
        self.button_filtro_passa_baixa.pack(side=LEFT)

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
        self.img_path = 'data/cap3/breast_digital_Xray.jpg'
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

    def gray(self):
        self.previous_img_array = self.img_array
        self.img_array = intensity_transformation.gray(self.img_array)
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

    def gray_eq(self):
        self.previous_img_array = self.img_array
        self.img_array = intensity_transformation.gray_eq(self.img_array)
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

    def filtro_passa_alta(self):
        self.previous_img_array = self.img_array
        threshold = CustomDialog(self, "threshold").show()
        threshold = int(threshold)
        ratio = CustomDialog(self, "ratio").show()
        ratio = float(ratio)
        gaussian = CustomDialog(self, "gaussian").show()
        gaussian = bool(int(gaussian))
        self.img_array = frequency.filtro_passa_alta(self.img_array, threshold, ratio, tipo='alta', gaussian=gaussian)
        self.show_image()

    def filtro_passa_baixa(self):
        self.previous_img_array = self.img_array
        threshold = CustomDialog(self, "threshold").show()
        threshold = int(threshold)
        ratio = CustomDialog(self, "ratio").show()
        ratio = float(ratio)
        gaussian = CustomDialog(self, "gaussian").show()
        gaussian = bool(int(gaussian))
        self.img_array = frequency.filtro_passa_alta(self.img_array, threshold, ratio, tipo='baixa', gaussian=gaussian)
        self.show_image()

    def show_image(self):
        max_size = 500
        height, width = self.img_array.shape
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
