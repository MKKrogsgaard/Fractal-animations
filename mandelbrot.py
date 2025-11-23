import numpy as np

from PIL import Image

import os
from tqdm import tqdm

# Modules
import colors

def mandelbrot(pixel: list[float], offset_real: float, offset_img: float, scale: float, width: int, height: int, max_iterations: int):
    '''
    Computes wether a given pixel is in the Mandelbrot set. For plotting.

    args:
        pixel (list[float]): Pixel coordinates
        offset_real, offset_img (float): The offset applied to c. The computation is done on c + offset
        scale: (float): Determines the scale of the resulting plot
        width, height (float): The resolution of the resulting plot.
        max_iterations (int): The max number of iterations to apply the Mandelbrot function for.
    '''
    x, y = pixel[0], pixel[1]
    # Only perform calculations if x and y are actually within the bounds of the image
    if x <= width and y <= height:
        # Scales the pixels so that the image is centered on (0, 0) in the complex plane and such that more pixels actually fall within the Mandelbrot set
        c_real = offset_real + (x - width / 2) * scale
        c_img = offset_img + (- y + height / 2) * scale

        # Mandelbrot function loop
        z_real = 0.0
        z_img = 0.0
        iteration = 0
        
        # Escape condition, the complex number in the Mandelbrot set with the largest magnitude is 2, so any number with a magnitude higher than that has left the Mandelbrot set
        while z_real**2 + z_img**2 <= 2*2 and iteration < max_iterations:
            # Implements the Mandelbrot function
            z_real_temp = z_real**2 - z_img**2 + c_real
            z_img_temp = 2*z_real*z_img + c_img

            z_real = z_real_temp
            z_img = z_img_temp

            iteration += 1
        
        if iteration == max_iterations:
            # The point is in the Mandelbrot set, return 0
            return 0
        else:
            # Return the number of iterations it took before the point left the bounds
            return iteration
        
    else:
        return 0 # If the current pixel is out of bounds
    

def generateFrame(offset_real: float, offset_img: float, scale: float, width: int, height: int, max_iterations: int, save_path: str, colorlist: list, k: int):
    '''
    Calculates wether the pixels in the image are in the Mandelbrot set for a given number of iterations. Saves the resulting frame.

    args:
        x, y (float): Pixel coordinates.

        offset_real (float): The real offset applied to c. The computation is done on c + offset.

        offset_img (float): The imaginary offset applied to c. The computation is done on c + offset.

        scale: (float): Determines the scale of the resulting plot.

        width (float): The width of the resulting plot.

        height (float): The height of the resulting plot.

        max_iterations (int): The max number of iterations to apply the Mandelbrot function for.

        save_path (str): Location to save the image at.

        colorlist (list): A list of RGB color values to use for the color gradient.

        k (int): The degree of the B-splines used to compute the color gradient
    '''
    # Initialize image and pixel coordinates
    img = np.zeros((height, width, 3), dtype=np.uint8)
    pixels = np.array([[x, y] for x in range(width) for y in range(height)])

    # Generate color palette of RGB colors
    colorpalette = colors.BSplinePalette(colors_RGB=colorlist, k=k, n_colors=max_iterations)

    # Color pixels according to the iteration reached
    for i in tqdm(range(len(pixels))):
        pixel = pixels[i]

        iteration = mandelbrot(pixel=pixel, offset_real=offset_real, offset_img=offset_img, scale=scale, width=width, height=height, max_iterations=max_iterations)

        x, y = pixel[0], pixel[1]
        img[y, x, :] = colorpalette[iteration]
    
    # Convert to uint8 and save the image
    Image.fromarray(img.astype("uint8"), mode="RGB").save(save_path)


    
    
# For the color gradient, RGB values
COLORLIST = [
    [0,0,0],
    [255, 255, 255],
    [255, 0, 0]
]

generateFrame(offset_real=0, offset_img=0, scale=0.005, width=1080, height=1080, max_iterations=100, save_path="img/test.png", colorlist=COLORLIST, k=2)
