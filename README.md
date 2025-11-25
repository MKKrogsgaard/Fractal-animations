# Fractal animations
 Animations of the Mandelbrot set and other related fractals. Files that end in `_video_cuda.py` generate animations. They use numba cuda to parallelize the computation of whether a given pixel is in the fractal, and hence only works on NVIDIA GPUs.

 FIles that end in `_frame.py` generate a single frame. They only use the CPU and do not require a NVIDIA GPU, but they are quite slow, especially for large resolutions.

 # Example animations

https://github.com/user-attachments/assets/78036b27-ece4-4082-a6d1-8bb8a215c237

https://github.com/user-attachments/assets/2eb1645a-a02d-4614-b866-8b6630ba744d


