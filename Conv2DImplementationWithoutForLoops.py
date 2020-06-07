import numpy as np
import timeit

### CS231n Implementation
def conv_forward_strides(x, w, b, pad = 0, stride = 1):
    
    N, C, H, W      = x.shape
    F, _, HH, WW    = w.shape

    # Pad the input
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    # Figure out output dimensions
    H += 2 * pad
    W += 2 * pad
    out_h = (H - HH) // stride + 1
    out_w = (W - WW) // stride + 1

    # Perform an im2col operation by picking clever strides
    shape   = (C,     HH, WW, N,         out_h,     out_w)
    strides = (H * W, W,   1, C * H * W, stride * W, stride)
    strides = x.itemsize * np.array(strides)
    x_stride= np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)

    x_cols          = np.ascontiguousarray(x_stride)
    x_cols.shape    = (C * HH * WW, N * out_h * out_w)
    
    # Now all our convolutions are a big matrix multiply
    res = w.reshape(F, -1).dot(x_cols) + b.reshape(-1, 1)

    # Reshape the output
    res.shape = (F, N, out_h, out_w)
    
    out = res.transpose(1, 0, 2, 3)

    # Be nice and return a contiguous array
    # The old version of conv_forward_fast doesn't do this, so for a fair
    # comparison we won't either
    out = np.ascontiguousarray(out)
    
    return out
    
    
### CS231n Implementation
def get_im2col_indices(x_shape, field_height, field_width, pad=0, stride=1):
    # First figure out what the size of the output should be
    C, H, W = x_shape[1:]
   
    out_height  = (H + 2*pad - field_height) // stride + 1
    out_width   = (W + 2*pad - field_width ) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
   
    i1 = stride * np.repeat(np.arange(out_height), out_width)
   
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
   
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)

def im2col_indices(x, field_height, field_width, pad=0, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, pad, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]

    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
   
    return cols

def conv_forward_im2col(x, w, b, pad = 0, stride = 1):
    """
    A fast implementation of the forward pass for a convolutional layer
    based on im2col and col2im.
    """
    N, _, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape

    # Create output
    out_height  = (H + 2 * pad - filter_height) // stride + 1
    out_width   = (W + 2 * pad - filter_width ) // stride + 1
    
    out         = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

    x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
    res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
    out = out.transpose(3, 0, 1, 2)

    return out


### Implentation from: https://sgugger.github.io/convolution-in-depth.html
def convForward(x, w, b, pad=0, stride=1):
    N, _, H, W = x.shape
    F, _, HH, WW = w.shape

    weights = w.reshape((F, -1)).transpose(1, 0)

    y = arr2vec(x, (HH, WW), stride, pad) @ weights + b
    y = np.transpose(y, (0, 2, 1))
    
    Hy = (H - HH + 2*pad)//stride + 1
    Wy = (W - WW + 2*pad)//stride + 1

    return y.reshape(N, F, Hy, Wy)

def arr2vec(x, kernel_size, stride=1, pad=0):
    
    x_padded    = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    N, C, H, W  = x_padded.shape
    HH, WW      = kernel_size
    
    grid        = np.array([j + W*i + H*W*k for k in range(C) for i in range(HH) for j in range(WW)])
    start_idx   = np.array([j + W*i         for i in range(0, H-HH+1, stride) for j in range(0, W-WW+1, stride)])
    batch       = np.array(range(N)) * C * H * W
    
    return x_padded.take(batch[:, None, None] + start_idx[None, :, None] + grid[None, None, :])


##################################################
'''
x = np.arange(2*3*4*5).reshape(2, 3, 4, 5).astype(np.float32)
w = np.arange(2*3*2*3).reshape(2, 3, 2, 3).astype(np.float32)
b = np.zeros(shape=(2,), dtype=np.float32)

print(conv_forward_strides(x, w, b, pad = 0, stride=1))
print(conv_forward_im2col(x, w, b, pad = 0, stride=1))
print(convForward(x, w, b, pad = 0, stride=1))
'''

setup_code = """
import numpy as np
from __main__ import conv_forward_strides, conv_forward_im2col, convForward 

x = np.random.randn(32, 3, 28, 28).astype(np.float32)
w = np.random.randn(16, 3, 3, 3).astype(np.float32)
b = np.random.randn(16, ).astype(np.float32)
"""
Niter = 2**12
print('Avg Time taken by **conv_forward_strides**: {:.4f}s'.format(timeit.timeit(stmt = 'conv_forward_strides(x, w, b, pad = 0, stride = 1)', setup = setup_code, number=Niter)/Niter))
print('Avg Time taken by **conv_forward_im2col** : {:.4f}s'.format(timeit.timeit(stmt = 'conv_forward_im2col(x, w, b, pad = 0, stride = 1)',  setup = setup_code, number=Niter)/Niter))
print('Avg Time taken by **convForward**         : {:.4f}s'.format(timeit.timeit(stmt = 'convForward(x, w, b, pad = 0, stride = 1)',          setup = setup_code, number=Niter)/Niter))
