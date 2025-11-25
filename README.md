# Fractal animations
 Animations of the Mandelbrot set and (possibly) other fractals. Files that end in `_video_cuda.py` generate animations. They use numba cuda to parallelize the computation of whether a given pixel is in the fractal, and hence only works on NVIDIA GPUs.

 FIles that end in `_frame.py` generate a single frame. They only use the CPU and do not require a NVIDIA GPU, but they are quite slow, especially for large resolutions.

 # Example animation

https://github.com/user-attachments/assets/4781f597-473d-4c03-ad27-de84bdac3805

