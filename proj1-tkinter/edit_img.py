from tkinter import *
from PIL import Image, ImageTk, ImageDraw

app = Tk()
# app.geometry('500x500')


def get_x_and_y(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y


def draw_smth(event):
    width = 10

    global last_x, last_y
    canvas.create_line((last_x, last_y, event.x, event.y),
                       fill='white',
                       width=width)

    center1 = (last_x, last_y)
    center2 = (event.x, event.y)

    last_x, last_y = event.x, event.y

    # do the PIL image/draw (in memory) drawings
    draw.line([center1, center2], fill=255, width=10)
    # PIL image can be saved as .png .jpg .gif or .bmp file (among others)

    filename = "my_drawing.jpg"
    image1.save(filename)


canvas = Canvas(app, bg='black')
canvas.pack(anchor='nw', fill='both', expand=1)

canvas.bind("<Button-1>", get_x_and_y)
canvas.bind("<B1-Motion>", draw_smth)

# PIL create an empty image and draw object to draw on
# memory only, not visible
image1 = Image.new("L", (500, 500), 0)
draw = ImageDraw.Draw(image1)

# PIL image can be saved as .png .jpg .gif or .bmp file (among others)
filename = "my_drawing.jpg"
image1.save(filename)

"""
image = Image.open('data/cap3/lua.bmp')
image = image.resize((500, 500), Image.ANTIALIAS)
image = ImageTk.PhotoImage(image)
canvas.create_image(0, 0, image=image, anchor='nw')
"""

app.mainloop()
