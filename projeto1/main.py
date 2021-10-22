import pygame
import numpy as np
from helper import *
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
Y = 720

# create the display surface object
# of specific dimension..e(X, Y).
display_surface = pygame.display.set_mode((X, Y))

# set the pygame window name
pygame.display.set_caption('Image')

# create a surface object, image is drawn on it.
# gray_img = pygame.image.load(r'./data/images/stinkbug.png')
color_img = pygame.image.load(r'./data/images/cat.jpg')
color_img_array = surface_to_array3d(color_img)
color_img_array_negative = linear_transformation(color_img_array, (0, 1.0), (63 / 255, 192 / 255), (106 / 255, 74 / 255),
                                                 (172 / 255, 217 / 255), (218 / 255, 59 / 255), (1.0, 0.0))
color_surface_negative = array3d_to_surface(color_img_array_negative)

# new_gray = surface_to_matrix(gray_img, True)
# new_color = surface_to_matrix(color_img, False)

# new_color = negative(new_color)
# new_surf = matrix_to_surface(new_color)

# plt.imshow(new_gray, cmap='gray', vmin=0, vmax=255)
# plt.imshow(new_color)

color_img = pygame.transform.scale(color_img, (500, 500))
color_surface_negative = pygame.transform.scale(color_surface_negative, (500, 500))

# infinite loop
while True:

    # completely fill the surface object
    # with white colour
    display_surface.fill(white)

    # copying the image surface object
    # to the display surface object at
    # (0, 0) coordinate.

    display_surface.blit(color_img, (10, 10))
    display_surface.blit(color_surface_negative, (color_img.get_width() + 20, 10))

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
