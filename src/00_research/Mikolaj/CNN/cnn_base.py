import numpy as np

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - np.tanh(x)**2

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def cross_entropy_loss(y_pred, y_true):
    """
    Computes cross-entropy loss.
    y_pred: predicted probabilities (from softmax), shape (num_classes,)
    y_true: true label (integer)
    Returns loss and the gradient with respect to y_pred pre-softmax.
    """
    loss = -np.log(y_pred[y_true] + 1e-8)
    grad = y_pred.copy()
    grad[y_true] -= 1
    return loss, grad

def conv2d_forward(input, weight, bias, stride=1):
    """
    Forward pass for a 2D convolution.
    input: shape (C, H, W)
    weight: shape (num_filters, C, kH, kW)
    bias: shape (num_filters,)
    Returns:
      output: shape (num_filters, H_out, W_out)
      cache: dictionary with necessary variables for backward pass
    """
    C, H, W = input.shape
    num_filters, C_w, kH, kW = weight.shape
    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1
    output = np.zeros((num_filters, H_out, W_out))
    
    for f in range(num_filters):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                region = input[:, h_start:h_start+kH, w_start:w_start+kW]
                output[f, i, j] = np.sum(region * weight[f]) + bias[f]
    
    cache = {
        'input': input,
        'weight': weight,
        'bias': bias,
        'stride': stride,
        'output_shape': output.shape
    }
    return output, cache

def conv2d_backward(dout, cache):
    """
    Backward pass for a 2D convolution.
    dout: gradient of loss with respect to the output, shape (num_filters, H_out, W_out)
    Returns:
      dinput: gradient with respect to input, same shape as input
      dweight: gradient with respect to weight, same shape as weight
      dbias: gradient with respect to bias, same shape as bias
    """
    input = cache['input']
    weight = cache['weight']
    bias = cache['bias']
    stride = cache['stride']
    
    C, H, W = input.shape
    num_filters, H_out, W_out = dout.shape
    kH, kW = weight.shape[2], weight.shape[3]
    
    dinput = np.zeros_like(input)
    dweight = np.zeros_like(weight)
    dbias = np.zeros_like(bias)
    
    # Compute dbias (sum gradients over each filter's output)
    for f in range(num_filters):
        dbias[f] = np.sum(dout[f])
    
    # Compute gradients for weight and input
    for f in range(num_filters):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                region = input[:, h_start:h_start+kH, w_start:w_start+kW]
                dweight[f] += dout[f, i, j] * region
                dinput[:, h_start:h_start+kH, w_start:w_start+kW] += dout[f, i, j] * weight[f]
                
    return dinput, dweight, dbias

def avg_pool_forward(input, size=2, stride=2):
    """
    Forward pass for average pooling.
    input: 2D array of shape (H, W)
    Returns:
      output: pooled output, shape (H_out, W_out)
      cache: dictionary for backward pass
    """
    H, W = input.shape
    H_out = (H - size) // stride + 1
    W_out = (W - size) // stride + 1
    output = np.zeros((H_out, W_out))
    
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            region = input[h_start:h_start+size, w_start:w_start+size]
            output[i, j] = np.mean(region)
    
    cache = {
        'input': input,
        'size': size,
        'stride': stride,
        'output_shape': output.shape
    }
    return output, cache

def avg_pool_backward(dout, cache):
    """
    Backward pass for average pooling.
    dout: gradient with respect to the pooled output, shape (H_out, W_out)
    Returns:
      dinput: gradient with respect to the input, same shape as input.
    """
    input = cache['input']
    size = cache['size']
    stride = cache['stride']
    H, W = input.shape
    H_out, W_out = cache['output_shape']
    dinput = np.zeros_like(input)
    
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            # Distribute the gradient evenly over the pooling window
            dinput[h_start:h_start+size, w_start:w_start+size] += dout[i, j] / (size*size)
    return dinput

def fc_forward(x, weight, bias):
    """
    Fully connected forward pass.
    x: input vector of shape (n,)
    weight: weight matrix of shape (m, n)
    bias: bias vector of shape (m,)
    Returns:
      out: output vector of shape (m,)
      cache: dictionary with x, weight, and bias
    """
    out = np.dot(weight, x) + bias
    cache = {'x': x, 'weight': weight, 'bias': bias}
    return out, cache

def fc_backward(dout, cache):
    """
    Fully connected backward pass.
    dout: gradient with respect to the output, shape (m,)
    Returns:
      dx: gradient with respect to input x, shape (n,)
      dweight: gradient with respect to weight, shape (m, n)
      dbias: gradient with respect to bias, shape (m,)
    """
    x = cache['x']
    weight = cache['weight']
    
    dx = np.dot(weight.T, dout)
    dweight = np.outer(dout, x)
    dbias = dout
    return dx, dweight, dbias

class LeNet5:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.conv1_weight = np.random.randn(6, 1, 5, 5) * 0.1
        self.conv1_bias = np.random.randn(6) * 0.1
        
        self.conv2_weight = np.random.randn(16, 6, 5, 5) * 0.1
        self.conv2_bias = np.random.randn(16) * 0.1
        
        self.conv3_weight = np.random.randn(120, 16, 5, 5) * 0.1
        self.conv3_bias = np.random.randn(120) * 0.1
        
        self.fc1_weight = np.random.randn(84, 120) * 0.1
        self.fc1_bias = np.random.randn(84) * 0.1
        
        self.fc2_weight = np.random.randn(10, 84) * 0.1
        self.fc2_bias = np.random.randn(10) * 0.1

    def forward(self, x):
        """
        Forward pass through the network.
        x: input image of shape (1, 32, 32) (grayscale)
        Returns:
          out_sm: softmax probabilities (output layer)
          caches: dictionary of intermediate variables for backpropagation
        """
        caches = {}
        
        # ---- C1: Convolution + Tanh ----
        out, cache_conv1 = conv2d_forward(x, self.conv1_weight, self.conv1_bias, stride=1)
        out = tanh(out)
        caches['conv1'] = (cache_conv1, out)
        
        # ---- S2: Average Pooling (apply per channel) ----
        pool1 = []
        pool1_caches = []
        for i in range(out.shape[0]):
            pooled, cache_pool = avg_pool_forward(out[i], size=2, stride=2)
            pool1.append(pooled)
            pool1_caches.append(cache_pool)
        pool1 = np.array(pool1)
        caches['pool1'] = pool1_caches
        caches['pool1_out'] = pool1
        
        # ---- C3: Convolution + Tanh ----
        out, cache_conv2 = conv2d_forward(pool1, self.conv2_weight, self.conv2_bias, stride=1)
        out = tanh(out)
        caches['conv2'] = (cache_conv2, out)
        
        # ---- S4: Average Pooling (apply per channel) ----
        pool2 = []
        pool2_caches = []
        for i in range(out.shape[0]):
            pooled, cache_pool = avg_pool_forward(out[i], size=2, stride=2)
            pool2.append(pooled)
            pool2_caches.append(cache_pool)
        pool2 = np.array(pool2)
        caches['pool2'] = pool2_caches
        caches['pool2_out'] = pool2
        
        # ---- C5: Convolution (fully connected convolution) + Tanh ----
        out, cache_conv3 = conv2d_forward(pool2, self.conv3_weight, self.conv3_bias, stride=1)
        out = tanh(out)
        caches['conv3'] = (cache_conv3, out)
        
        # ---- Flatten ----
        flat = out.flatten()
        caches['flat_shape'] = out.shape
        caches['flat'] = flat
        
        # ---- F6: Fully Connected + Tanh ----
        out, cache_fc1 = fc_forward(flat, self.fc1_weight, self.fc1_bias)
        out = tanh(out)
        caches['fc1'] = (cache_fc1, out)
        
        # ---- Output Layer: Fully Connected + Softmax ----
        out, cache_fc2 = fc_forward(out, self.fc2_weight, self.fc2_bias)
        out_sm = softmax(out)
        caches['fc2'] = (cache_fc2, out, out_sm)
        
        return out_sm, caches

    def backward(self, caches, y_true):
        """
        Backward pass: computes gradients of the loss with respect to all parameters.
        y_true: true label (integer)
        Returns:
          loss: cross-entropy loss value
          grads: dictionary of gradients for each parameter
        """
        grads = {}
        
        # ---- Start with Cross-Entropy Loss (combined with softmax) ----
        cache_fc2, fc2_pre, out_sm = caches['fc2']
        loss, dloss = cross_entropy_loss(out_sm, y_true)
        
        # ---- Backprop through Output FC Layer ----
        dx_fc2, dfc2_weight, dfc2_bias = fc_backward(dloss, cache_fc2)
        grads['fc2_weight'] = dfc2_weight
        grads['fc2_bias'] = dfc2_bias
        
        # ---- Backprop through F6 (tanh) ----
        cache_fc1, fc1_out = caches['fc1']
        dtanh_fc1 = dtanh(fc1_out) * dx_fc2
        dx_fc1, dfc1_weight, dfc1_bias = fc_backward(dtanh_fc1, cache_fc1)
        grads['fc1_weight'] = dfc1_weight
        grads['fc1_bias'] = dfc1_bias
        
        # ---- Reshape gradient to match conv3 output ----
        dflat = dx_fc1
        conv3_shape = caches['flat_shape']
        dconv3 = dflat.reshape(conv3_shape)
        
        # ---- Backprop through C5 (tanh) ----
        cache_conv3, conv3_out = caches['conv3']
        dtanh_conv3 = dtanh(conv3_out) * dconv3
        dpool2, dconv3_weight, dconv3_bias = conv2d_backward(dtanh_conv3, cache_conv3)
        grads['conv3_weight'] = dconv3_weight
        grads['conv3_bias'] = dconv3_bias
        
        # ---- Backprop through S4: Average Pooling ----
        # Use the shape of conv2 output (stored in caches['conv2'][1]) which is (16, 10, 10)
        conv2_out = caches['conv2'][1]
        dconv2_out = np.zeros_like(conv2_out)
        for i in range(len(caches['pool2'])):
            dconv2_out[i] = avg_pool_backward(dpool2[i], caches['pool2'][i])

        
        # ---- Backprop through C3 (tanh) ----
        cache_conv2, conv2_out = caches['conv2']
        dtanh_conv2 = dtanh(conv2_out) * dconv2_out
        dpool1, dconv2_weight, dconv2_bias = conv2d_backward(dtanh_conv2, cache_conv2)
        grads['conv2_weight'] = dconv2_weight
        grads['conv2_bias'] = dconv2_bias
        
        # ---- Backprop through S2: Average Pooling ----
        dconv1_out = np.zeros_like(caches['conv1'][1])
        for i in range(len(caches['pool1'])):
            dconv1_out[i] = avg_pool_backward(dpool1[i], caches['pool1'][i])
        
        # ---- Backprop through C1 (tanh) ----
        cache_conv1, conv1_out = caches['conv1']
        dtanh_conv1 = dtanh(conv1_out) * dconv1_out
        dinput, dconv1_weight, dconv1_bias = conv2d_backward(dtanh_conv1, cache_conv1)
        grads['conv1_weight'] = dconv1_weight
        grads['conv1_bias'] = dconv1_bias
        
        return loss, grads

    def update_params(self, grads):
        """
        Updates network parameters using gradient descent.
        """
        lr = self.learning_rate
        self.fc2_weight -= lr * grads['fc2_weight']
        self.fc2_bias   -= lr * grads['fc2_bias']
        self.fc1_weight -= lr * grads['fc1_weight']
        self.fc1_bias   -= lr * grads['fc1_bias']
        self.conv3_weight -= lr * grads['conv3_weight']
        self.conv3_bias   -= lr * grads['conv3_bias']
        self.conv2_weight -= lr * grads['conv2_weight']
        self.conv2_bias   -= lr * grads['conv2_bias']
        self.conv1_weight -= lr * grads['conv1_weight']
        self.conv1_bias   -= lr * grads['conv1_bias']

    def train_step(self, x, y_true):
        """
        Performs one training step: forward pass, backpropagation, and parameter update.
        x: input image (shape (1, 32, 32))
        y_true: true label (integer)
        Returns:
          loss: scalar loss value
          y_pred: predicted probability distribution
        """
        y_pred, caches = self.forward(x)
        loss, grads = self.backward(caches, y_true)
        self.update_params(grads)
        return loss, y_pred

