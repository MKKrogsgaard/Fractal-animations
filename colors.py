# DESCRIPTION: This module generates color gradients by interpolating between some specified colors

import numpy as np
from scipy import interpolate

# Helper functions for converting between RGB and hex color codes

def hexToRGB(hex: str):
    '''
    Converts a hex triplet string to a RGB value "#FFFFFF" -> "[0-255, 0-255, 0-255]"
    '''
    # Skip the first character, pass the hex values to the int() function two characters at a time
    return [int(hex[i:i + 2], base=16) for i in range(1, 6, 2)]

def RBGToHex(RGB: list[int]):
    '''
    Converts an RGB triplet to a hex triplet string "[0-255, 0-255, 0-255]" -> "#FFFFFF"
    '''
    RGB = [int(x) for x in RGB] # Ensures that components are integers, otherwise transforming to hex won't work
    return "#{:02x}{:02x}{:02x}".format(RGB[0], RGB[1], RGB[2]) # Format each color in the RGB triplet to a hexadecimal value and string them together





def BSplinePalette(colors_RGB, k=3, n_colors=2):
    '''
    Creates a color palette by interpolating between colors using B-splines of degree (k).
    
    Args:
        colors_RGB (list): An list of RGB colors, e.g.[[0-255, 0-255, 0-255], ...]
        k (int): The degree of the B=splines
        n_colors (int): The desired number of colors in the final palette
    '''
    # Make sure the degree is not greater than n_datapoints - 1
    n_points = len(colors_RGB)

    if k > n_points - 1:
        k = n_points - 1

    # Convert to float ndarray and ensure the shape is (n_points, 3)
    colors_RGB = np.array(colors_RGB, dtype=float)
    if colors_RGB.shape[0] != n_points or colors_RGB.shape[1] != 3:
        raise ValueError("colors_RGB must be a list of RGB triplets!")
    
    # make_splprep expects one array for each coordinate, i.e. [[x1, x2, ...], [y1, y2, ...], [z1, z2, ...]]
    coords = [colors_RGB[:, i] for i in range(3)]

    # Fit B-spline
    spl, u_spl = interpolate.make_splprep(x=coords, k=k)

    # Evaluate B-spline at evenly spaced intervals to get a color palette
    u_eval = np.linspace(0.0, 1.0, n_colors)
    palette = np.array([spl(u) for u in u_eval])
    
    return palette




    
