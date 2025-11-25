import numpy as np
import math

from PIL import Image

from numba import cuda, config

import moviepy

import os
import shutil

from tqdm import tqdm

# Modules
from colors import hexToRGB, BSplinePalette


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
            # The point is in the Mandelbrot set, paint it black
            img[y, x, 0] = 0
            img[y, x, 1] = 0
            img[y, x, 2] = 0
        else:
            # Paint the point according to the number of iterations it took before |z| > 2
            img[y, x, 0] = palette[iteration][0]
            img[y, x, 1] = palette[iteration][1]
            img[y, x, 2] = palette[iteration][2]
    else:
        # The point is out of bounds, do nothing
        return


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

    # Initialize image and pixel coordinates on host (CPU)
    # Notice that heigth and width are swapped because PIL expects that orientation when converting from an array to an image!
    host_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Generate color palette of RGB colors
    host_colorpalette = BSplinePalette(colors_RGB=colorlist, k=k, n_colors=max_iterations)
    # Convert to uint8 to save GPU memory
    host_colorpalette = host_colorpalette.astype(np.uint8)

    # Copy arrays to the GPU
    device_img = cuda.to_device(host_img)

    device_colorpalette = cuda.to_device(host_colorpalette)

    # Dimension of each CUDA block
    block_dim = (int(np.sqrt(threads_per_block)), int(np.sqrt(threads_per_block)))

    # Dimension of each grid (in blocks), using ceil to ensure there are enough blocks for all of the threads (one thread per pixel)
    grid_dim = (math.ceil(width / block_dim[0]), math.ceil(height / block_dim[1]))
    
    # Info for debugging
    # print(f'Number of pixels: {width}x{height}={width*height}')
    # print(f'Dimension of blocks: {block_dim}')
    # print(f'Dimension of grids: {grid_dim}')
    # print(f'Total threads: {block_dim[0]*block_dim[1] * grid_dim[0]*grid_dim[1] * threads_per_block}')

    mandelbrotKernel[grid_dim, block_dim](offset_real, offset_img, scale, width, height, max_iterations, device_img, device_colorpalette)

    # Synch to ensure the kernel is done with all threads before attempting to copy the results
    cuda.synchronize()
    result_host_img = device_img.copy_to_host()

    # Delete GPU arrays from memory
    del device_img
    del device_colorpalette
    import gc
    gc.collect()

    # Convert to uint8 and save the image
    Image.fromarray(result_host_img, mode='RGB').save(file_path)

def generateVideo(offset_real: float, offset_img: float, scale: float, width: int, height: int, colorlist: list, k: int, n_frames: int, fps: int):

    # Check if GPU is available
    if cuda.is_available():
        print('cuda available:', cuda.is_available())
    else:
        raise(RuntimeError('No CUDA GPU available!'))

    frames_folder = 'img/frames'
    video_path = 'img/mandelbrot_animation.mp4'

    # Prepare frames folder if it doesn't exist already
    if not os.path.isdir(frames_folder):
        os.mkdir(frames_folder)

    # Generate the frames
    print(f'Number of frames: {n_frames}')
    print(f'Framerate: {fps} frames per second')
    print('Generating frames...')

    frame_paths = []

    for i in tqdm(range(n_frames)):
        frame_path = f'{frames_folder}/mandelbrot-{i}.jpg'
        frame_paths.append(frame_path)

        generateFrame(offset_real=0, offset_img=0, scale=scale, width=resolution[0], height=resolution[1], max_iterations=i + 1, file_path=frame_path, colorlist=COLORLIST, k=2)
    
    # Generate the video
    print(f'Saving video as: {video_path}')
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frame_paths, fps=fps)
    clip.write_videofile(video_path)

    # Delete the frames
    for filename in os.listdir(frames_folder):
        file_path = os.path.join(frames_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')



# Example color palettes
# Black and white: ['#000000', "#FFFFFF", '#000000']
# Fire: ['#ff0000', '#ff9400', '#ffda00']
# Water: ['#1A2A80', '#3B38A0', "#7A85C1", '#B2B0E8']

COLORLIST = ['#1A2A80', '#3B38A0', "#7A85C1", '#B2B0E8']
COLORLIST = [hexToRGB(hex) for hex in COLORLIST]

# For uniform scaling accross different resolutions
resolution = [2000, 2000]
# We want to scale the Mandelbrot set to just barely fit inside the plotted area
# This means we need (min(width, height) / 2) * scale = 2
# Hence
scale = 4 / min(resolution)

# For easy access to color interpolation degree
degree = 1

# Test frame
generateFrame(offset_real=0, offset_img=0, scale=scale, width=resolution[0], height=resolution[1], max_iterations=100, file_path='img/test.jpg', colorlist=COLORLIST, k=degree)

# Video generation
generateVideo(offset_real=0, offset_img=0, scale=scale, width=resolution[0], height=resolution[1], colorlist=COLORLIST, k=degree, n_frames=100, fps=10)