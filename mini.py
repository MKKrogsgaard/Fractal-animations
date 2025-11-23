from numba import cuda
import numpy as np
print("cuda available:", cuda.is_available())
a = np.zeros((4,4), dtype=np.uint8)
try:
    d = cuda.to_device(a)
    print("to_device OK")
except Exception as e:
    print("to_device exception:", repr(e))