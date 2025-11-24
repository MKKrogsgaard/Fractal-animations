import numpy as np
import math

from PIL import Image

import os
from tqdm import tqdm

# Modules
import colors
from colors import hexToRGB

from numba import cuda, config


# Threads per block, should be a multiple of 32
threads_per_block = 64
@cuda.jit
def mandelbrotKernel(offset_real, offset_img, scale, width, height, max_iterations, img, palette):
    '''
    Numba CUDA kernel for the Mandelbrot function. Assigns a single pixel to each thread and colors the pixel according to the number of iterations it takes before it is out of bounds.

    args:
        pixel (list[float]): Pixel coordinates
        offset_real (float): The real offset applied to c. The computation is done on c + offset.
        offset_img (float): The imaginary offset applied to c. The computation is done on c + offset.
        scale: (float): Determines the scale of the resulting plot.
        width (float): The width of the resulting plot.
        height (float): The height of the resulting plot.
        max_iterations (int): The max number of iterations to apply the Mandelbrot function for.
        img (ndarray): The image to apply the coloring to.
        palette (ndarray): The color palette with which to color the pixels.
    '''
    
    # Get the pixel coordinates corresponding to the current thread
    x, y = cuda.grid(2)

    # Only perform calculations if x and y are actually within the bounds of the image
    if 0 <= x < width and 0 <= y < height:
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
            # The point is in the Mandelbrot set, paint it with the first color in the palette
            img[y, x, 0] = palette[0][0]
            img[y, x, 1] = palette[0][1]
            img[y, x, 2] = palette[0][2]
        else:
            # Paint the point according to the number of iterations it took before |z| > 2
            img[y, x, 0] = palette[iteration][0]
            img[y, x, 1] = palette[iteration][1]
            img[y, x, 2] = palette[iteration][2]
    else:
        # The point is out of bounds, paint it with the first color in the palette
        img[y, x, 0] = palette[0][0]
        img[y, x, 1] = palette[0][1]
        img[y, x, 2] = palette[0][2]


def generateFrame(offset_real: float, offset_img: float, scale: float, width: int, height: int, max_iterations: int, file_path: str, colorlist: list, k: int):
    '''
    Calculates whether each pixel in the image is in the Mandelbrot set for a given number of iterations. Saves the resulting frame.

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
    # Debugging, check if GPU is available
    print("cuda available:", cuda.is_available())

    # Initialize image and pixel coordinates on host (CPU)
    # Notice that heigth and width are swapped because PIL expects that orientation when converting from an array to an image!
    host_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Generate color palette of RGB colors
    host_colorpalette = colors.BSplinePalette(colors_RGB=colorlist, k=k, n_colors=max_iterations)

    # Copy arrays to the GPU
    device_img = cuda.to_device(host_img)

    device_colorpalette = cuda.to_device(host_colorpalette)

    # Dimension of each CUDA block
    block_dim = (int(np.sqrt(threads_per_block)), int(np.sqrt(threads_per_block)))

    # Dimension of each grid (in blocks), using ceil to ensure there are enough blocks for all of the threads (one thread per pixel)
    grid_dim = (math.ceil(width / block_dim[0]), math.ceil(height / block_dim[1]))
    
    # Info for debugging
    print(f'Number of pixels: {width}x{height}={width*height}')
    print(f'Dimension of blocks: {block_dim}')
    print(f'Dimension of grids: {grid_dim}')
    print(f'Total threads: {block_dim[0]*block_dim[1] * grid_dim[0]*grid_dim[1] * threads_per_block}')

    mandelbrotKernel[grid_dim, block_dim](offset_real, offset_img, scale, width, height, max_iterations, device_img, device_colorpalette)

    # Synch to ensure the kernel is done with all threads before attempting to copy the results
    result_host_img = device_img.copy_to_host()

    # Convert to uint8 and save the image
    Image.fromarray(result_host_img, mode="RGB").save(file_path)


# For the color gradient
# COLORLIST = ['#000000', "#390000", "#F01010"]
COLORLIST = ['#000000', "#FFFFFF", '#000000']
COLORLIST = [hexToRGB(hex) for hex in COLORLIST]

# For uniform scaling accross different resolutions
resolution = [6000, 6000]
# We want to scale the Mandelbrot set to just barely fit inside the plotted area
# This means we need (min(width, height) / 2) * scale = 2
# Hence
scale = 4 / min(resolution)

generateFrame(offset_real=0, offset_img=0, scale=scale, width=resolution[0], height=resolution[1], max_iterations=100, file_path="img/test.png", colorlist=COLORLIST, k=2)
