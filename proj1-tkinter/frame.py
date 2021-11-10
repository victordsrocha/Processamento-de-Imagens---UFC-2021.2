from tkinter import *
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import image_processing.intensity_transformation
from dialog import CustomDialog
import image_processing.aula3
import skimage.io


class GUI(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        w, h = 950, 650
        master.minsize(width=w, height=h)
        master.maxsize(width=w, height=h)
        self.pack()

        self.img_path = None
        self.img_array = None
        self.previous_img_array = None
        self.img_tk = None

        self.container0 = Frame(master)
        self.container0.pack(side=LEFT, fill='y')

        self.container1 = Frame(master)
        self.container1.pack(side=TOP, fill='y')
        lbl2 = Label(self.container1, text='aula 2: ')
        lbl2.pack(side=LEFT)

        self.container_panel = Frame(master)
        self.container_panel.pack(side=TOP, fill='y', expand=True)

        self.panel = Label(self.container_panel)

        self.button_browse = Button(self.container0, text='Browse', command=self.choose)
        self.button_save = Button(self.container0, text='Save', command=self.save)
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

        self.button_browse.pack(side=TOP, fill='x')
        self.button_save.pack(side=TOP, fill='x')
        self.button_restore.pack(side=TOP, fill='x')
        self.button_undo.pack(side=TOP, fill='x')
        self.button_gray.pack(side=LEFT)
        self.button_negative.pack(side=LEFT)
        self.button_log.pack(side=LEFT)
        self.button_gama.pack(side=LEFT)
        self.button_linear.pack(side=LEFT)
        lbl3 = Label(self.container1, text='     aula 3: ')
        lbl3.pack(side=LEFT)
        self.button_binary.pack(side=LEFT)
        self.button_bit_plane.pack(side=LEFT)
        self.button_record_message.pack(side=LEFT)
        self.button_read_message.pack(side=LEFT)

        self.panel.pack(side=TOP, fill='y', expand=True)
        self.start()

    def start(self):
        self.img_path = 'data/images/stinkbug.jpg'
        self.img_array = skimage.io.imread(fname=self.img_path)
        self.previous_img_array = self.img_array
        self.img_tk = ImageTk.PhotoImage(image=Image.fromarray(self.img_array))
        self.panel.configure(image=self.img_tk)
        self.panel.image = self.img_tk

    def choose(self):
        img_path = filedialog.askopenfilename(filetypes=[
            ('image', '.jpg'),
            ('image', '.bmp')
        ])
        if len(img_path) > 0:
            self.img_path = img_path
            self.img_array = skimage.io.imread(fname=self.img_path)
            self.previous_img_array = self.img_array
            self.img_tk = ImageTk.PhotoImage(image=Image.fromarray(self.img_array))
            self.panel.configure(image=self.img_tk)
            self.panel.image = self.img_tk
        else:
            pass

    def save(self):
        img_path = filedialog.asksaveasfile(filetypes=[
            ('image', '.jpg'),
            ('image', '.bmp')
        ])
        skimage.io.imsave(fname=img_path.name, arr=self.img_array)

    def restore(self):
        self.img_array = skimage.io.imread(fname=self.img_path)
        self.previous_img_array = self.img_array
        self.img_tk = ImageTk.PhotoImage(image=Image.fromarray(self.img_array))
        self.panel.configure(image=self.img_tk)
        self.panel.image = self.img_tk

    def undo(self):
        aux = self.img_array
        self.img_array = self.previous_img_array
        self.previous_img_array = aux
        self.img_tk = ImageTk.PhotoImage(image=Image.fromarray(self.img_array))
        self.panel.configure(image=self.img_tk)
        self.panel.image = self.img_tk

    def gray(self):
        self.previous_img_array = self.img_array
        self.img_array = image_processing.intensity_transformation.gray(self.img_array)
        self.img_tk = ImageTk.PhotoImage(image=Image.fromarray(self.img_array))
        self.panel.configure(image=self.img_tk)
        self.panel.image = self.img_tk

    def negative(self):
        self.previous_img_array = self.img_array
        self.img_array = image_processing.intensity_transformation.negative(self.img_array)
        self.img_tk = ImageTk.PhotoImage(image=Image.fromarray(self.img_array))
        self.panel.configure(image=self.img_tk)
        self.panel.image = self.img_tk

    def log_transformation(self):
        self.previous_img_array = self.img_array
        value = CustomDialog(self, "c").show()
        value = float(value)
        self.img_array = image_processing.intensity_transformation.log_transformation(self.img_array, value)
        self.img_tk = ImageTk.PhotoImage(image=Image.fromarray(self.img_array))
        self.panel.configure(image=self.img_tk)
        self.panel.image = self.img_tk

    def gama_transformation(self):
        self.previous_img_array = self.img_array
        dialog_return = CustomDialog(self, "gama;c").show()
        gama, c = dialog_return.split(';')
        gama = float(gama)
        c = float(c)
        self.img_array = image_processing.intensity_transformation.gama_transformation(
            self.img_array, gama=gama, c=c)
        self.img_tk = ImageTk.PhotoImage(image=Image.fromarray(self.img_array))
        self.panel.configure(image=self.img_tk)
        self.panel.image = self.img_tk

    def linear_transformation(self):
        self.previous_img_array = self.img_array
        dialog_return = CustomDialog(self, "point;point;...").show()
        points_str_list = dialog_return.split(';')
        points = []
        for point_str in points_str_list:
            x, y = point_str.split(',')
            points.append((float(x), float(y)))
        self.img_array = image_processing.intensity_transformation.linear_transformation(
            self.img_array, points)
        self.img_tk = ImageTk.PhotoImage(image=Image.fromarray(self.img_array))
        self.panel.configure(image=self.img_tk)
        self.panel.image = self.img_tk

    def binary(self):
        self.previous_img_array = self.img_array
        self.img_array = image_processing.aula3.binary(self.img_array)
        self.img_tk = ImageTk.PhotoImage(image=Image.fromarray(self.img_array))
        self.panel.configure(image=self.img_tk)
        self.panel.image = self.img_tk

    def bit_plane(self):
        self.previous_img_array = self.img_array
        value = CustomDialog(self, "bit").show()
        value = int(value)
        self.img_array = image_processing.aula3.bit_plane(self.img_array, value)
        self.img_tk = ImageTk.PhotoImage(image=Image.fromarray(self.img_array))
        self.panel.configure(image=self.img_tk)
        self.panel.image = self.img_tk

    def record_message(self):
        self.previous_img_array = self.img_array
        value = CustomDialog(self, "message").show()
        self.img_array = image_processing.aula3.record_message(self.img_array, value)
        self.img_tk = ImageTk.PhotoImage(image=Image.fromarray(self.img_array))
        self.panel.configure(image=self.img_tk)
        self.panel.image = self.img_tk

    def read_message(self):
        message = image_processing.aula3.read_message(self.img_array)
        print(r'{}'.format(message))
        messagebox.showinfo(title=r'message', message=message)


root = Tk()
app = GUI(master=root)
app.mainloop()
# root.destroy()
