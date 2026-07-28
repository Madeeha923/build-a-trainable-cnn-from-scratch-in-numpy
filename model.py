"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
def argmax_rows(matrix):
    # TODO: return the index of the largest element in each row of a 2D array
    return np.argmax(matrix, axis = 1)

# Step 2 - row_max
import numpy as np

def row_max(matrix):
    # TODO: return the maximum value of each row of `matrix` with keepdims True for broadcasting.
    return np.max(matrix, axis=1, keepdims=True)

# Step 3 - row_sum
import numpy as np

def row_sum(matrix):
    """Return per-row sums of a 2D array with shape (N, 1)."""
    # TODO: return the sum along axis 1 keeping the reduced dimension
    return np.sum(matrix, axis=1, keepdims=True)

# Step 4 - exp_shifted
import numpy as np

def row_max(matrix):
    return np.max(matrix, axis=1, keepdims=True)

def exp_shifted(logits):
    shifted = logits - row_max(logits)

    return np.exp(shifted)

# Step 5 - stable_softmax
def exp_shifted(logits):
    # Subtract max value from each row for numerical stability
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    return np.exp(shifted)

def row_sum(matrix):
    # Sum each row
    return np.sum(matrix, axis=1, keepdims=True)



def stable_softmax(logits):



    exp_values = exp_shifted(logits)

    sums = row_sum(exp_values)
    return exp_values / sums

# Step 6 - one_hot
import numpy as np
def one_hot(labels, num_classes):
    Y = np.zeros((labels.shape[0], num_classes), dtype=float)

    Y[np.arange(labels.shape[0]), labels] = 1.0

    return Y

# Step 7 - gather_true_class_probs
def gather_true_class_probs(probs, labels):
    return probs[np.arange(labels.shape[0]), labels]

# Step 8 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, labels, eps=1e-12):
    # TODO: return the mean negative log-likelihood of the true-class probabilities
    true_class_probs = gather_true_class_probs(probs, labels)
    true_class_probs = np.clip(true_class_probs, eps, 1.0)

    # losses = -np.log(true_class_probs)
    return -np.mean(np.log(true_class_probs))

# Step 9 - accuracy
def accuracy(logits_or_probs, labels):
    # TODO: return the fraction of rows whose argmax matches the integer label.
    preds = argmax_rows(logits_or_probs)
    return np.mean(preds == labels)

# Step 10 - he_std
def he_std(fan_in):
    return float(np.sqrt(2.0/fan_in))

# Step 11 - he_init
def he_init(shape, fan_in, seed):
    # TODO: sample a weight tensor from a normal distribution scaled by He std using the seed.
    np.random.seed(seed)
    std = he_std(fan_in)
    return np.random.randn(*shape) * std

# Step 12 - init_zero_bias
import numpy as np

def init_zero_bias(length):
    # TODO: return a 1D float array of zeros with the given length.
    return np.zeros(length, dtype = np.float64)

# Step 13 - pad_2d
def pad_2d(images, pad):
    # TODO: zero-pad the spatial (H, W) dims of a 4D (N, C, H, W) tensor by `pad` on each side.
    return np.pad(
        images, pad_width = ((0, 0),
        (0,0),
        (pad, pad),
        (pad, pad)

    ),
    mode = 'constant',
    constant_values = 0
    )

# Step 14 - output_spatial_size
def output_spatial_size(input_size, kernel, stride, padding):
    return ((input_size - kernel + 2 * padding) // stride) + 1

# Step 15 - im2col
import numpy as np

def im2col(images, kernel_h, kernel_w, stride, padding):
    N, C, H, W = images.shape
    padded = pad_2d(images, padding)

    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)

    cols = np.zeros(
        (N * out_h * out_w, C * kernel_h * kernel_w),
        dtype=images.dtype
    )

    row_idx = 0

    for n in range(N):
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                h_end = h_start + kernel_h
                w_start = j * stride
                w_end = w_start + kernel_w

                patch = padded[n, :, h_start:h_end, w_start:w_end]
                cols[row_idx] = patch.reshape(-1)

                row_idx += 1

    return cols

# Step 16 - col2im
def col2im(cols, input_shape, kernel_h, kernel_w, stride, padding):
    N, C, H, W = input_shape

    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)

    padded_h = H + 2 * padding
    padded_w = W + 2* padding

    images = np.zeros((N, C, padded_h, padded_w), dtype=cols.dtype)

    row_idx = 0

    for n in range(N):
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                h_end = h_start + kernel_h
                w_start = j * stride
                w_end = w_start + kernel_w

                # Convert row back to patch
                patch = cols[row_idx].reshape(C, kernel_h, kernel_w)

                # Add patch back (important for overlaps)
                images[n, :, h_start:h_end, w_start:w_end] += patch

                row_idx += 1

    # Remove padding if applied
    if padding > 0:
        return images[:, :, padding:-padding, padding:-padding]

    return images

# Step 17 - conv2d_forward
import numpy as np

def conv2d_forward(x, weights, bias, stride, padding):
    N, C_in, H, W = x.shape
    C_out, _, kernel_h, kernel_w = weights.shape

    # Output spatial dimensions
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)

    # Convert input into columns
    cols = im2col(x, kernel_h, kernel_w, stride, padding)

    # Flatten weights
    w_col = weights.reshape(C_out, -1)

    # Matrix multiplication + bias
    out = cols @ w_col.T + bias

    # Reshape to (N, C_out, out_h, out_w)
    out = out.reshape(N, out_h, out_w, C_out).transpose(0, 3, 1, 2)

    # Cache for backward pass
    cache = {
        "x_shape": x.shape,
        "weights": weights,
        "cols": cols,
        "stride": stride,
        "padding": padding,
        "kernel_h": kernel_h,
        "kernel_w": kernel_w
    }

    return out, cache

# Step 18 - conv2d_grad_input
def conv2d_grad_input(d_out, cache):
    """
    Compute gradient with respect to the convolution input.
    """

    x_shape = cache["x_shape"]
    weights = cache["weights"]
    stride = cache["stride"]
    padding = cache["padding"]
    kernel_h = cache["kernel_h"]
    kernel_w = cache["kernel_w"]

    C_out = weights.shape[0]

    # Flatten filters
    W_col = weights.reshape(C_out, -1)

    # (N,C_out,H_out,W_out) -> (N*H_out*W_out, C_out)
    d_out_col = d_out.transpose(0, 2, 3, 1).reshape(-1, C_out)

    # Gradient wrt im2col matrix
    d_cols = d_out_col @ W_col

    # Fold back into image
    dx = col2im(
        d_cols,
        x_shape,
        kernel_h,
        kernel_w,
        stride,
        padding
    )

    return dx

# Step 19 - conv2d_grad_weights
def conv2d_grad_weights(d_out, cache):
    """
    Compute gradient with respect to convolution weights.

    Inputs:
    - d_out: (N, C_out, out_h, out_w)
    - cache: contains
        cols
        weights
        kernel_h
        kernel_w

    Returns:
    - dW: (C_out, C_in, kernel_h, kernel_w)
    """

    cols = cache["cols"]
    weights = cache["weights"]
    kernel_h = cache["kernel_h"]
    kernel_w = cache["kernel_w"]

    C_out, C_in, _, _ = weights.shape

    # (N, C_out, out_h, out_w)
    # -> (N*out_h*out_w, C_out)
    d_out_col = d_out.transpose(0, 2, 3, 1).reshape(-1, C_out)

    # (C_out, N*out_h*out_w) @ (N*out_h*out_w, C_in*kH*kW)
    dW_col = d_out_col.T @ cols

    # Restore original filter shape
    dW = dW_col.reshape(C_out, C_in, kernel_h, kernel_w)

    return dW

# Step 20 - conv2d_grad_bias
def conv2d_grad_bias(d_out):
    # TODO: return a length C_out gradient by reducing d_out over batch and spatial axes
    db = np.sum(d_out, axis=(0,2,3))
    return db

# Step 21 - conv2d_backward
def conv2d_backward(d_out, cache):
    # TODO: return (dx, dW, db) using the conv2d gradient helpers and the forward cache
    dx = conv2d_grad_input(d_out, cache)
    dw = conv2d_grad_weights(d_out, cache)
    db = conv2d_grad_bias(d_out)

    return dx, dw, db

# Step 22 - maxpool2d_forward
def maxpool2d_forward(x, kernel, stride):
    """
    Forward pass for 2D max pooling.

    Inputs:
    - x: Input of shape (N, C, H, W)
    - kernel: Pooling kernel size
    - stride: Stride of pooling

    Returns:
    - out: Output after max pooling
    - cache: Dictionary containing:
        x_shape
        argmax
        kernel
        stride
    """

    N, C, H, W = x.shape

    out_h = output_spatial_size(H, kernel, stride, 0)
    out_w = output_spatial_size(W, kernel, stride, 0)

    out = np.zeros((N, C, out_h, out_w), dtype=x.dtype)
    argmax = np.zeros((N, C, out_h, out_w), dtype=np.int64)

    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):

                    h_start = i * stride
                    h_end = h_start + kernel

                    w_start = j * stride
                    w_end = w_start + kernel

                    # Extract pooling window
                    window = x[n, c, h_start:h_end, w_start:w_end]

                    # Store maximum value
                    out[n, c, i, j] = np.max(window)

                    # Store flattened index of maximum
                    argmax[n, c, i, j] = np.argmax(window)

    cache = {
        "x_shape": x.shape,
        "argmax": argmax,
        "kernel": kernel,
        "stride": stride
    }

    return out, cache

# Step 23 - scatter_grad_window
import numpy as np

def scatter_grad_window(grad_value, argmax_index, kernel):
    window_grad = np.zeros((kernel, kernel), dtype=float)

    row = argmax_index // kernel
    col = argmax_index % kernel

    window_grad[row, col] = grad_value

    return window_grad

# Step 24 - maxpool2d_backward
def maxpool2d_backward(d_out, cache):
    """
    Backward pass for 2D max pooling.

    Inputs:
    - d_out: Upstream gradient of shape (N, C, out_h, out_w)
    - cache: Dictionary containing:
        x_shape
        argmax
        kernel
        stride

    Returns:
    - dx: Gradient with respect to input, shape (N, C, H, W)
    """

    x_shape = cache["x_shape"]
    argmax = cache["argmax"]
    kernel = cache["kernel"]
    stride = cache["stride"]

    N, C, H, W = x_shape
    _, _, out_h, out_w = d_out.shape

    dx = np.zeros(x_shape, dtype=d_out.dtype)

    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):

                    h_start = i * stride
                    h_end = h_start + kernel
                    w_start = j * stride
                    w_end = w_start + kernel

                    # Scatter gradient into pooling window
                    grad_window = scatter_grad_window(
                        d_out[n, c, i, j],
                        argmax[n, c, i, j],
                        kernel
                    )

                    # Accumulate (handles overlapping windows)
                    dx[n, c, h_start:h_end, w_start:w_end] += grad_window

    return dx

# Step 25 - relu_forward
def relu_forward(x):
    out = np.maximum(0, x)
    cache = {
        "x":  x
    }
    return out, cache

# Step 26 - relu_backward
def relu_backward(d_out, cache):
    # TODO: mask the upstream gradient by the positive entries of the cached input.
    x = cache["x"]
    dx = d_out * (x > 0)

    return dx

# Step 27 - flatten_forward
def flatten_forward(x):
    N = x.shape[0]
    
    out = x.reshape(N, -1)
    cache = {
        "x_shape": x.shape

    }

    return out, cache

# Step 28 - flatten_backward
import numpy as np

def flatten_backward(d_out, cache):
    x_shape = cache["x_shape"]

    dx = d_out.reshape(x_shape)

    return dx

# Step 29 - linear_forward
def linear_forward(x, weights, bias):
    out = x @ weights + bias

    cache = {
        "x": x,
        "weights": weights

    }

    return out, cache

# Step 30 - linear_grad_input
import numpy as np

def linear_grad_input(d_out, cache):
    """Gradient of a linear layer w.r.t. its input X."""
    # TODO: return dL/dX given d_out (N, D_out) and cache['weights'] (D_in, D_out)
    weights = cache["weights"]

    dx = d_out @ weights.T

    return dx

# Step 31 - linear_grad_weights
import numpy as np

def linear_grad_weights(x, dout):
    """Gradient of loss wrt linear-layer weights W of shape (D_in, D_out)."""
    # TODO: Compute the gradient of a linear layer's loss wrt its weight matrix W.
  #  x = cache["x"]
    dw = x.T @ dout
    return dw

# Step 32 - linear_grad_bias
import numpy as np

def linear_grad_bias(dout):
    # TODO: Compute the bias gradient of a linear layer given upstream gradient dout.
    db = np.sum(dout, axis=0)

    return db

# Step 33 - linear_backward
def linear_backward(dout, cache):
    # TODO: combine input, weight, and bias gradients for a linear layer using the cache
    dx = linear_grad_input(dout, cache)
    x = cache["x"]
    dw = linear_grad_weights(x, dout)
    db = linear_grad_bias(dout)

    return dx, dw, db

# Step 34 - softmax_cross_entropy_forward
def softmax_cross_entropy_forward(logits, y):
    # TODO: return the mean cross-entropy loss for logits (N, C) and integer labels y (N,).
    probs = stable_softmax(logits)
    loss = cross_entropy_loss(probs, y)
    if np.isclose(loss, 0.0):
        loss = 0.0
    return float(loss)

# Step 35 - softmax_cross_entropy_backward
def softmax_cross_entropy_backward(logits, y):
    # TODO: return the fused softmax-cross-entropy gradient of shape (N, C).
    N = logits.shape[0]

    probs = stable_softmax(logits)
    y_one_hot = one_hot(y, logits.shape[1])

    dlogits = (probs - y_one_hot) / N 

    return dlogits

# Step 36 - sgd_step
import numpy as np

def sgd_step(param, grad, lr):
    # TODO: return the SGD-updated parameter array (param - lr * grad).
    updated_param = param - lr * grad
    return updated_param

# Step 37 - adam_update_m
import numpy as np

def adam_update_m(m, grad, beta_one):
    m = beta_one * m + (1 - beta_one) * grad
    return m

# Step 38 - adam_update_v
import numpy as np

def adam_update_v(v, grad, beta_two):
    # TODO: return the updated Adam second moment estimate as an EMA of squared gradients.
    v = beta_two * v + (1-beta_two) * (grad ** 2)
    return v

# Step 39 - adam_bias_correct
def adam_bias_correct(moment, beta, t):
    # TODO: return moment divided by (1 - beta**t) to undo Adam's zero-init bias.
    cm = moment / (1-beta ** t)
    return cm

# Step 40 - adam_param_step
import numpy as np

def adam_param_step(param, m_hat, v_hat, lr, eps):
    # TODO: apply one Adam parameter update using bias-corrected moments
    updated_param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return updated_param

# Step 41 - adam_step
import numpy as np

def adam_step(param, grad, m, v, t, lr, beta_one, beta_two, eps):
    # TODO: chain the four Adam helpers and return (new_param, new_m, new_v)
    m = adam_update_m(m, grad, beta_one)
    v = adam_update_v(v, grad, beta_two)

    m_hat = adam_bias_correct(m, beta_one, t)
    v_hat = adam_bias_correct(v, beta_two, t)

    updated_param = adam_param_step(param, m_hat, v_hat, lr, eps)

    return updated_param, m, v

# Step 42 - init_conv_layer
def init_conv_layer(out_channels, in_channels, kernel_size, seed=0):
    # TODO: Build He-initialized weights and a zero bias for a single conv layer.
    fan_in = in_channels * kernel_size * kernel_size

    W = he_init(
        (out_channels, in_channels, kernel_size, kernel_size),
        fan_in,
        seed
    )

    b = init_zero_bias(out_channels)

    params = {
        "W": W,
        "b": b
    }

    return params

# Step 43 - init_linear_layer
def init_linear_layer(in_features, out_features, seed=0):
    # TODO: return {'W': He-init matrix (in_features, out_features), 'b': zero bias (out_features,)}
    W = he_init(
        (in_features, out_features),
        in_features,
        seed
    )

    b = init_zero_bias(out_features)

    return {
        "W": W,
        "b": b
    }

# Step 44 - init_lenet
def init_lenet(in_channels, num_classes, seed=0):
    params = {
        "conv1": init_conv_layer(
            in_channels=in_channels,
            out_channels=6,
            kernel_size=5,
            seed=seed
        ),

        "conv2": init_conv_layer(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            seed=seed + 1
        ),

        "fc1": init_linear_layer(
            in_features=16 * 4 * 4,
            out_features=120,
            seed=seed + 2
        ),

        "fc2": init_linear_layer(
            in_features=120,
            out_features=num_classes,
            seed=seed + 3
        )
    }

    return params

# Step 45 - forward_conv_block
def forward_conv_block(x, W, b, pool_size, stride, pad):
    # Convolution
    conv_out, conv_cache = conv2d_forward(
        x,
        W,
        b,
        stride,
        pad
    )

    # ReLU
    relu_out, relu_cache = relu_forward(conv_out)

    # Max Pool (non-overlapping)
    out, pool_cache = maxpool2d_forward(
        relu_out,
        kernel=pool_size,
        stride=pool_size
    )

    cache = {
        "conv_cache": conv_cache,
        "relu_cache": relu_cache,
        "pool_cache": pool_cache
    }

    return out, cache

# Step 46 - forward_classifier_block
def forward_classifier_block(x, fc1, fc2):
    # TODO: run flatten -> linear -> relu -> linear and return logits plus a cache dict.
     # Flatten
    flat, flatten_cache = flatten_forward(x)

    # First fully connected layer
    hidden, fc1_cache = linear_forward(
        flat,
        fc1["W"],
        fc1["b"]
    )

    # ReLU
    hidden_relu, relu_cache = relu_forward(hidden)

    # Second fully connected layer
    logits, fc2_cache = linear_forward(
        hidden_relu,
        fc2["W"],
        fc2["b"]
    )

    cache = {
        "flatten_cache": flatten_cache,
        "fc1_cache": fc1_cache,
        "relu_cache": relu_cache,
        "fc2_cache": fc2_cache
    }

    return logits, cache

# Step 47 - lenet_forward
def lenet_forward(x, params):
    # TODO: run two conv blocks then the classifier block and return (logits, caches).
    # First convolution block
    out1, block1_cache = forward_conv_block(
        x,
        params["conv1"]["W"],
        params["conv1"]["b"],
        pool_size=2,
        stride=1,
        pad=0
    )

    # Second convolution block
    out2, block2_cache = forward_conv_block(
        out1,
        params["conv2"]["W"],
        params["conv2"]["b"],
        pool_size=2,
        stride=1,
        pad=0
    )

    # Classifier block
    logits, classifier_cache = forward_classifier_block(
        out2,
        params["fc1"],
        params["fc2"]
    )

    cache = {
        "block1": block1_cache,
        "block2": block2_cache,
        "classifier": classifier_cache
    }

    return logits, cache

# Step 48 - backward_conv_block
def backward_conv_block(dout, cache):
    """
    Backward pass for a Conv -> ReLU -> MaxPool block.

    Inputs:
    - dout: Upstream gradient from the block output
    - cache: Dictionary containing:
        'conv_cache'
        'relu_cache'
        'pool_cache'

    Returns:
    - dx: Gradient w.r.t. block input
    - dW: Gradient w.r.t. convolution weights
    - db: Gradient w.r.t. convolution bias
    """

    # MaxPool backward
    dpool = maxpool2d_backward(
        dout,
        cache["pool_cache"]
    )

    # ReLU backward
    drelu = relu_backward(
        dpool,
        cache["relu_cache"]
    )

    # Convolution backward
    dx, dW, db = conv2d_backward(
        drelu,
        cache["conv_cache"]
    )

    return dx, dW, db

# Step 49 - backward_classifier_block
def backward_classifier_block(dlogits, cache):
    # FC2 backward
    d_relu, dW2, db2 = linear_backward(
        dlogits,
        cache["fc2_cache"]
    )

    # ReLU backward
    d_fc1 = relu_backward(
        d_relu,
        cache["relu_cache"]
    )

    # FC1 backward
    d_flat, dW1, db1 = linear_backward(
        d_fc1,
        cache["fc1_cache"]
    )

    # Flatten backward
    dx = flatten_backward(
        d_flat,
        cache["flatten_cache"]
    )

    return {
        "dx": dx,
        "fc1": {
            "dW": dW1,
            "db": db1
        },
        "fc2": {
            "dW": dW2,
            "db": db2
        }
    }

# Step 50 - lenet_backward
def lenet_backward(dlogits, caches):
    # Backward through classifier
    classifier_grads = backward_classifier_block(
        dlogits,
        caches["classifier"]
    )

    # Backward through second convolution block
    dx, dW2, db2 = backward_conv_block(
        classifier_grads["dx"],
        caches["block2"]
    )

    # Backward through first convolution block
    _, dW1, db1 = backward_conv_block(
        dx,
        caches["block1"]
    )

    grads = {
        "conv1": {
            "dW": dW1,
            "db": db1
        },
        "conv2": {
            "dW": dW2,
            "db": db2
        },
        "fc1": classifier_grads["fc1"],
        "fc2": classifier_grads["fc2"]
    }

    return grads

# Step 51 - lenet_predict (not yet solved)
# TODO: implement

# Step 52 - build_synthetic_image_dataset (not yet solved)
# TODO: implement

# Step 53 - shuffle_indices (not yet solved)
# TODO: implement

# Step 54 - train_test_split (not yet solved)
# TODO: implement

# Step 55 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 56 - train_step (not yet solved)
# TODO: implement

# Step 57 - train_one_epoch (not yet solved)
# TODO: implement

# Step 58 - train_loop (not yet solved)
# TODO: implement

# Step 59 - evaluate (not yet solved)
# TODO: implement

