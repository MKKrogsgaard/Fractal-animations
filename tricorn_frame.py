import numpy as np

from PIL import Image

import os
from tqdm import tqdm

# Modules
import colors
from colors import hexToRGB

def tricorn(pixel: list[float], offset_real: float, offset_img: float, scale: float, width: int, height: int, max_iterations: int):
    '''
    Computes wether a given pixel is in the tricorn and colors the pixel according to the number of iterations it takes before it is out of bounds.

    args:
        pixel (list[float]): Pixel coordinates
        offset_real, offset_img (float): The offset applied to c. The computation is done on c + offset
        scale: (float): Determines the scale of the resulting plot
        width, height (float): The resolution of the resulting plot.
        max_iterations (int): The max number of iterations to apply the tricorn function for.
    '''
    x, y = pixel[0], pixel[1]
    # Only perform calculations if x and y are actually within the bounds of the image
    if x <= width and y <= height:
        # Scales the pixels so that the image is centered on (0, 0) in the complex plane and such that more pixels actually fall within the tricorn set
        c_real = offset_real + (x - width / 2) * scale
        c_img = offset_img + (- y + height / 2) * scale

        # tricorn function loop
        z_real = 0.0
        z_img = 0.0
        iteration = 0
        
        # Escape condition, the complex number in the tricorn set with the largest magnitude is 2, so any number with a magnitude higher than that has left the tricorn set
        while z_real**2 + z_img**2 <= 2*2 and iteration < max_iterations:
            # Implements the tricorn function f_c(z) = (z*)^2 + c
            z_real_temp = z_real**2 - z_img**2 + c_real
            z_img_temp = -2*z_real*z_img + c_img

            z_real = z_real_temp
            z_img = z_img_temp

            iteration += 1
        
        if iteration == max_iterations:
            # The point is in the tricorn set, return 0
            return 0
        else:
            # Return the number of iterations it took before the point left the bounds
            return iteration
        
    else:
        return 0 # If the current pixel is out of bounds
    

def generateFrame(offset_real: float, offset_img: float, scale: float, width: int, height: int, max_iterations: int, file_path: str, colorlist: list, k: int):
    '''
    Calculates whether each pixel in the image is in the tricorn set for a given number of iterations. Saves the resulting frame.

    args:
        x, y (float): Pixel coordinates.
        offset_real (float): The real offset applied to c. The computation is done on c + offset.
        offset_img (float): The imaginary offset applied to c. The computation is done on c + offset.
        scale: (float): Determines the scale of the resulting plot.
        width (float): The width of the resulting plot.
        height (float): The height of the resulting plot.
        max_iterations (int): The max number of iterations to apply the tricorn function for.
        save_path (str): Location to save the image at.
        colorlist (list): A list of RGB color values to use for the color gradient.
        k (int): The degree of the B-splines used to compute the color gradient
    '''
    # Initialize image and pixel coordinates
    # Notice that heigth and width are swapped because PIL expects that orientation when converting from an array to an image!
    img = np.zeros((height, width, 3), dtype=np.uint8)
    pixels = np.array([[x, y] for x in range(width) for y in range(height)])

    # Generate color palette of RGB colors
    colorpalette = colors.BSplinePalette(colors_RGB=colorlist, k=k, n_colors=max_iterations)

    print('Generating tricorn frame with the following settings:')
    print(f'Resolution: {width}x{height}, scale: {scale}, max iterations: {max_iterations}')
    # Color pixels according to the iteration reached
    for i in tqdm(range(len(pixels))):
        pixel = pixels[i]

        iteration = tricorn(pixel=pixel, offset_real=offset_real, offset_img=offset_img, scale=scale, width=width, height=height, max_iterations=max_iterations)

        x, y = pixel[0], pixel[1]
        img[y, x, :] = colorpalette[iteration]
    
    # Convert to uint8 and save the image
    Image.fromarray(img.astype("uint8"), mode="RGB").save(file_path)

# For uniform scaling accross different resolutions
resolution = [2000, 2000]
# We want to scale the Mandelbrot set to just barely fit inside the plotted area
# This means we need (min(width, height) / 2) * scale = 2
# Hence
scale = 4 / min(resolution)

# For the color gradient
COLORLIST = ['#000000', "#FFFFFF", '#000000']
COLORLIST = [hexToRGB(hex) for hex in COLORLIST]

generateFrame(offset_real=0, offset_img=0, scale=scale, width=resolution[0], height=resolution[0], max_iterations=35, file_path='img/test.jpg', colorlist=COLORLIST, k=2)
