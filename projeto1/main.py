import pygame
import numpy as np
from helper import *
from aula3.aula3 import *
from transformations import *

# activate the pygame library .
# initiate pygame and give permission
# to use pygame's functionality.
pygame.init()

# define the RGB value
# for white colour
white = (255, 255, 255)

# assigning values to X and Y variable
X = 1200
Y = 650

# create the display surface object
# of specific dimension..e(X, Y).
display_surface = pygame.display.set_mode((X, Y))

# set the pygame window name
pygame.display.set_caption('Image')

# create a surface object, image is drawn on it.
# gray_img = pygame.image.load(r'./data/images/stinkbug.png')
original_surface = pygame.image.load(r'./data/images/cat.jpg')
original_surface = pygame.transform.scale(original_surface, (500, 500))
color_img_array = surface_to_array3d(original_surface)
color_img_array = bit_plane(color_img_array, 8)
img_surface = array3d_to_surface(color_img_array)
# color_img_array_negative = linear_transformation(color_img_array, (0, 1.0), (63 / 255, 192 / 255), (106 / 255, 74 / 255),
#                                                 (172 / 255, 217 / 255), (218 / 255, 59 / 255), (1.0, 0.0))
# color_surface_negative = array3d_to_surface(color_img_array_negative)

# new_gray = surface_to_matrix(gray_img, True)
# new_color = surface_to_matrix(color_img, False)

# new_color = negative(new_color)
# new_surf = matrix_to_surface(new_color)

# plt.imshow(new_gray, cmap='gray', vmin=0, vmax=255)
# plt.imshow(new_color)



# infinite loop
while True:

    # completely fill the surface object
    # with white colour
    display_surface.fill(white)

    # copying the image surface object
    # to the display surface object at
    # (0, 0) coordinate.

    display_surface.blit(original_surface, (10, 10))
    display_surface.blit(img_surface, (img_surface.get_width() + 20, 10))

    # iterate over the list of Event objects
    # that was returned by pygame.event.get() method.
    for event in pygame.event.get():

        # if event object type is QUIT
        # then quitting the pygame
        # and program both.
        if event.type == pygame.QUIT:
            # deactivates the pygame library
            pygame.quit()

            # quit the program.
            quit()

        # Draws the surface object to the screen.
        pygame.display.update()
