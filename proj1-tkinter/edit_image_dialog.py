import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk, ImageDraw
import numpy as np


class CanvasDialog(tk.Toplevel):
    def __init__(self, parent, width, int1d):
        tk.Toplevel.__init__(self, parent)

        size = '{}x{}'.format(int1d.shape[1], int1d.shape[0])

        # self.app = Tk()
        self.geometry(size)

        # self.img_tk = ImageTk.PhotoImage(image=Image.fromarray(image_to_show))
        self.photo_image = ImageTk.PhotoImage(image=Image.fromarray(int1d))

        self.last_x, self.last_y = 0, 0
        self.width = width

        self.canvas = Canvas(self, bg='black')
        self.canvas.pack(anchor='nw', fill='both', expand=1)

        self.canvas.create_image(0, 0, image=self.photo_image, anchor='nw')

        self.canvas.bind("<Button-1>", self.get_x_and_y)
        self.canvas.bind("<B1-Motion>", self.draw_something)

        # memory only, not visible
        # self.edited_image = Image.fromarray(int1d)
        self.edited_image = Image.new("L", (int1d.shape[1], int1d.shape[0]), 255)
        self.draw_edit = ImageDraw.Draw(self.edited_image)

    def get_x_and_y(self, event):
        self.last_x, self.last_y = event.x, event.y

    def draw_something(self, event):
        self.canvas.create_line((self.last_x, self.last_y, event.x, event.y),
                                fill='black',
                                width=self.width)

        center1 = (self.last_x, self.last_y)
        center2 = (event.x, event.y)

        self.last_x, self.last_y = event.x, event.y

        # do the PIL image/draw (in memory) drawings
        self.draw_edit.line([center1, center2], fill=0, width=10)

    def show(self):
        self.wm_deiconify()
        self.wait_window()
        return np.array(self.edited_image)
