import numpy as np
import timeit

### 2D Conv using 4 For loops
def conv2D_0(x, w, b, pad = 0, stride=1):
    N, C, H, W  = x.shape
    F, _, HH, WW= w.shape

    # Zero-pad Input
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    # Initialize the output matrix
    Hy = 1 + (H + 2*pad - HH)//stride
    Wy = 1 + (W + 2*pad - WW)//stride

    out = np.zeros((N, F, Hy, Wy))

    for exp in range(N):
        for filt in range(F):
            for hh in range(Hy):
                h_start = hh*stride
                h_end   = h_start + HH

                for ww in range(Wy):
                    w_start = ww*stride
                    w_end   = w_start + WW

                    out[exp, filt, hh, ww] = np.sum(x_padded[exp, :, h_start:h_end, w_start:w_end] * w[filt]) + b[filt]

    return out

### 2D Conv using 4 For loops - other implementation (re-arranging for loops)
def conv2D_1(x, w, b, pad = 0, stride=1):
    N, C, H, W  = x.shape
    F, _, HH, WW= w.shape

    # Zero-pad Input
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    # Initialize the output matrix
    Hy = 1 + (H + 2*pad - HH)//stride
    Wy = 1 + (W + 2*pad - WW)//stride

    out = np.zeros((N, F, Hy, Wy))

    for exp in range(N):
        for hh in range(Hy):
            h_start = hh*stride
            h_end   = h_start + HH

            for ww in range(Wy):
                w_start = ww*stride
                w_end   = w_start + WW

                x_patch = x_padded[exp, :, h_start:h_end, w_start:w_end]

                for filt in range(F):
                    out[exp, filt, hh, ww] = np.sum(x_patch * w[filt]) + b[filt]

    return out

### 2D Conv using 3 For loops
def conv2D_2(x, w, b, pad = 0, stride=1):
    N, C, H, W  = x.shape
    F, _, HH, WW= w.shape

    # Zero-pad Input
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    # Initialize the output matrix
    Hy = 1 + (H + 2*pad - HH)//stride
    Wy = 1 + (W + 2*pad - WW)//stride

    out = np.zeros((N, F, Hy, Wy))

    # Reshape weight matrix
    wts = w.reshape((F, -1))

    for exp in range(N):
        for hh in range(Hy):
            h_start = hh*stride
            h_end   = h_start + HH
            for ww in range(Wy):
                w_start = ww*stride
                w_end   = w_start + WW

                x_patch = x_padded[exp, :, h_start:h_end, w_start:w_end]
                out[exp, :, hh, ww] = wts @ x_patch.flatten() + b

    return out

### 2D Conv using 2 For loops
def conv2D_3(x, w, b, pad = 0, stride=1):
    N, C, H, W  = x.shape
    F, _, HH, WW= w.shape

    # Zero-pad Input
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    # Initialize the output matrix
    Hy = 1 + (H + 2*pad - HH)//stride
    Wy = 1 + (W + 2*pad - WW)//stride

    out = np.zeros((N, F, Hy, Wy))

    # Reshape weight matrix
    wts = w.reshape((F, -1)).transpose()

    for hh in range(Hy):
        h_start = hh*stride
        h_end   = h_start + HH
        for ww in range(Wy):
            w_start = ww*stride
            w_end   = w_start + WW

            x_patch = x_padded[:, :, h_start:h_end,  w_start:w_end]
            x_patch = x_patch.reshape((N, -1))
            out[:, :, hh, ww] = x_patch @ wts + b

    return out

##################TEST ################################
'''
x = np.arange(2*3*4*5).reshape(2, 3, 4, 5).astype(np.float32)
w = np.arange(2*3*2*3).reshape(2, 3, 2, 3).astype(np.float32)
b = np.zeros(shape=(2,), dtype=np.float32)

print(conv2D_0(x, w, b, pad = 0, stride=1))
print(conv2D_1(x, w, b, pad = 0, stride=1))
print(conv2D_2(x, w, b, pad = 0, stride=1))
print(conv2D_3(x, w, b, pad = 0, stride=1))
'''

setup_code = """
import numpy as np
from __main__ import conv2D_0, conv2D_1, conv2D_2, conv2D_3, conv2D_3

x = np.random.randn(32, 3, 28, 28).astype(np.float32)
w = np.random.randn(16, 3, 3, 3).astype(np.float32)
b = np.random.randn(16, ).astype(np.float32)
"""

Niter = 2**6
print('Avg Time taken by **conv2D_0**: {:.2f}s'.format(timeit.timeit(stmt = 'conv2D_0(x, w, b, pad = 0, stride = 1)', setup = setup_code, number=Niter)/Niter))
print('Avg Time taken by **conv2D_1**: {:.2f}s'.format(timeit.timeit(stmt = 'conv2D_1(x, w, b, pad = 0, stride = 1)', setup = setup_code, number=Niter)/Niter))
print('Avg Time taken by **conv2D_2**: {:.2f}s'.format(timeit.timeit(stmt = 'conv2D_2(x, w, b, pad = 0, stride = 1)', setup = setup_code, number=Niter)/Niter))
print('Avg Time taken by **conv2D_3**: {:.2f}s'.format(timeit.timeit(stmt = 'conv2D_3(x, w, b, pad = 0, stride = 1)', setup = setup_code, number=Niter)/Niter))
