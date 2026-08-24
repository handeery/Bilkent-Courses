import numpy as np
import struct
import time
import matplotlib.pyplot as plt
import os

# Suppress NumPy warnings (e.g., overflow)
np.seterr(all='ignore')

# ----------- 1. DATA LOADING AND PREPARATION FUNCTIONS ----------- #

def load_mnist_images(filename):
    """Reads MNIST image file and returns (N, 784) float32 array in [0,1]."""
    with open(filename, 'rb') as f:
        # Read magic number, number of images, rows, and columns
        _magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        # Flatten and normalize to [0, 1]
        data = data.reshape(num, rows * cols).astype(np.float32) / 255.0
    return data

def load_mnist_labels(filename):
    """Reads MNIST label file and returns (N,) uint8 array (0–9)."""
    with open(filename, 'rb') as f:
        _magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def labels_to_targets_cnn(labels):
    """
    Converts digit labels (0–9) to 10-length output vectors (one-hot 0/1).
    Output neuron order: [1, 2, 3, 4, 5, 6, 7, 8, 9, 0].
    """
    N = labels.shape[0]
    targets = np.zeros((N, 10), dtype=np.float32)
    # Map digit -> neuron index: 1->0, 2->1, ..., 9->8, 0->9
    indices = np.where(labels == 0, 9, labels - 1)
    targets[np.arange(N), indices] = 1.0
    return targets

# ----------- 2. ACTIVATION AND LOSS FUNCTIONS ----------- #

# Sigmoid
def sigmoid(x):
    x = np.clip(x, -50, 50) 
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv_from_output(y):
    # dsigma/dv = y * (1 - y)
    return y * (1.0 - y)

# Softmax
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# Loss and Backward Derivatives

# Case 1: MSE
def mse_loss(output, target):
    return 0.5 * np.sum((target - output) ** 2, axis=-1)

# Case 1: Sigmoid + MSE backward (dE/dv)
def d_mse_sigmoid(output, target):
    # dE/do * do/dv = (o-d) * o(1-o)
    return (output - target) * sigmoid_deriv_from_output(output)

# Case 2: Cross-Entropy (CE)
def cross_entropy_loss(output_y, target_d):
    # Prevent log(0) error
    return -np.sum(target_d * np.log(output_y + 1e-9), axis=-1)

# Case 2: Softmax + Cross-Entropy backward (dE/dv)
def d_ce_softmax(output_y, target_d):
    # Derivative: y_i - d_i
    return output_y - target_d

# ----------- 3. AUXILIARY LAYER OPERATIONS ----------- #

def im2col(input_data, filter_h, filter_w, stride=1):
    """Transforms image patches into columns for faster convolution."""
    N, C, H, W = input_data.shape
    out_h = (H - filter_h) // stride + 1
    out_w = (W - filter_w) // stride + 1

    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype=input_data.dtype)

    for r in range(filter_h):
        r_max = r + stride * out_h
        for c in range(filter_w):
            c_max = c + stride * out_w
            col[:, :, r, c, :, :] = input_data[:, :, r:r_max:stride, c:c_max:stride]

    # Reshape to (N*OH*OW, C*FH*FW)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1):
    """Inverse transformation of im2col for backpropagation."""
    N, C, H, W = input_shape
    out_h = (H - filter_h) // stride + 1
    out_w = (W - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros(input_shape, dtype=col.dtype)
    
    for r in range(filter_h):
        r_max = r + stride * out_h
        for c in range(filter_w):
            c_max = c + stride * out_w
            img[:, :, r:r_max:stride, c:c_max:stride] += col[:, :, r, c, :, :]
            
    return img

# ----------- 4. CNN LAYER CLASSES ----------- #

class ConvolutionLayer:
    """H1 and H3 Convolutional Layer (Stride=1, Padding=0, Linear Activation)"""
    def __init__(self, input_depth, num_filters, kernel_size, with_bias, eta, init_range=0.01): 
        self.num_filters = num_filters
        self.input_depth = input_depth
        self.kernel_size = kernel_size
        self.with_bias = with_bias
        self.eta = eta
        
        # Weight initialization: Uniform [-0.01, 0.01] (Assignment constraint)
        self.W = np.random.uniform(-init_range, init_range, 
                                   (num_filters, input_depth, kernel_size, kernel_size)).astype(np.float32)
        
        if with_bias:
            self.B = np.random.uniform(-init_range, init_range, num_filters).astype(np.float32)
            self.dB = np.zeros_like(self.B) 
        else:
            self.B = None
            self.dB = None # Initialized for safety

        # Gradients (to be accumulated over the batch/online step)
        self.dW = np.zeros_like(self.W)
        
        # Saved for backpropagation
        self.X = None
        self.col = None
        
    def forward(self, X):
        """X: (N, C, H, W)"""
        N, C, H, W = X.shape
        FH, FW = self.kernel_size, self.kernel_size
        
        self.X = X
        
        # Output dimensions calculation
        OH = (H - FH) + 1
        OW = (W - FW) + 1
        
        # im2col transformation
        col = im2col(X, FH, FW)
        self.col = col
        
        # Vectorize filters: (num_filters, C*FH*FW)
        col_W = self.W.reshape(self.num_filters, -1).T
        
        # Matrix multiplication
        out = col @ col_W
        
        # Reshape to output shape: (N, num_filters, OH, OW)
        out = out.reshape(N, OH, OW, self.num_filters).transpose(0, 3, 1, 2)
        
        # Add bias (if present)
        if self.with_bias:
            out += self.B.reshape(1, self.num_filters, 1, 1)
            
        return out

    def backward(self, dOut, batch_size=1):
        """dOut: dE/dZ (Delta from the next layer)"""
        
        # Vectorize dE/dZ for matrix multiplication
        dOut = dOut.transpose(0, 2, 3, 1).reshape(-1, self.num_filters)
        
        # 1. Calculate dE/dB (Bias gradient)
        if self.with_bias:
            self.dB += np.sum(dOut, axis=0)
            
        # 2. Calculate dE/dW (Weight gradient, shared across positions)
        self.dW += (self.col.T @ dOut).T.reshape(self.W.shape)
        
        # 3. Calculate dE/dX (Delta to be propagated back)
        col_W = self.W.reshape(self.num_filters, -1)
        dcol = dOut @ col_W
        
        # Transform dcol back to dX (input shape)
        dX = col2im(dcol, self.X.shape, self.kernel_size, self.kernel_size)
        
        return dX
    
    def update(self, batch_size):
        """Weight update based on accumulated gradients (dW, dB)"""
        self.W -= self.eta * (self.dW / batch_size)
        
        # ATTRIBUTE ERROR FIX: Only update B if bias is present
        if self.with_bias: 
            self.B -= self.eta * (self.dB / batch_size)
        
        # Reset gradients
        self.dW.fill(0)
        if self.with_bias:
            self.dB.fill(0)

class AveragePoolingLayer:
    """H2 and H4: 2x2 Average Pooling Layer"""
    def __init__(self, pool_h=2, pool_w=2):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.X = None
        
    def forward(self, X):
        """X: (N, C, H, W)"""
        N, C, H, W = X.shape
        
        # Use im2col to get pooling blocks
        col = im2col(X, self.pool_h, self.pool_w, stride=self.pool_h)
        col = col.reshape(-1, self.pool_h * self.pool_w)
        
        # Calculate mean (average pooling)
        out = np.mean(col, axis=1)
        
        # Reshape to output shape: (N, C, OH, OW)
        OH = H // self.pool_h
        OW = W // self.pool_w
        out = out.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
        
        self.X = X
        return out
    
    def backward(self, dOut):
        """Propagate delta back by distributing the gradient equally (1/4)"""
        N, C, H, W = self.X.shape
        PH, PW = self.pool_h, self.pool_w
        
        # Reshape dE/dH_out to be distributed
        dOut = dOut.transpose(0, 2, 3, 1).repeat(PH, axis=1).repeat(PW, axis=2)
        dOut = dOut.reshape(-1, C * PH * PW)
        
        # Delta distribution: distribute the gradient equally (1/PH*PW)
        dcol = dOut * (1.0 / (PH * PW))
        
        # Transform dcol back to dX (input shape)
        dX = col2im(dcol, self.X.shape, PH, PW, stride=PH)
        return dX


class FullyConnectedLayer:
    """H5: Fully Connected Layer + Bias + Output Activation"""
    def __init__(self, input_dim, output_dim, eta, init_range=0.01): 
        self.eta = eta
        
        # Weight initialization: Uniform [-0.01, 0.01]
        self.W = np.random.uniform(-init_range, init_range, (input_dim, output_dim)).astype(np.float32)
        self.B = np.random.uniform(-init_range, init_range, output_dim).astype(np.float32)
        
        self.dW = np.zeros_like(self.W)
        self.dB = np.zeros_like(self.B)
        
        self.H_in = None

    def forward(self, H_in, activation_fn):
        """H_in: (N, input_dim) -> O: (N, 10)"""
        self.H_in = H_in
        
        # Linear sum: V = H_in @ W + B
        V = H_in @ self.W + self.B
        
        # Activation
        O = activation_fn(V)
        return V, O

    def backward(self, dE_dV, batch_size=1):
        """dE_dV: Derivative of Error w.r.t. the linear input V"""
        
        # 1. Calculate dE/dB
        self.dB += np.sum(dE_dV, axis=0)
        
        # 2. Calculate dE/dW
        self.dW += self.H_in.T @ dE_dV
        
        # 3. Calculate dE/dH_in (Delta to be propagated back to H4)
        dE_dH_in = dE_dV @ self.W.T
        return dE_dH_in

    def update(self, batch_size):
        """Weight update"""
        self.W -= self.eta * (self.dW / batch_size)
        self.B -= self.eta * (self.dB / batch_size)
        
        # Reset gradients
        self.dW.fill(0)
        self.dB.fill(0)


class CNN:
    def __init__(self, with_bias, case, eta=0.01): 
        self.with_bias = with_bias
        self.case = case
        
        # Define layers (LeNet-inspired architecture)
        self.H1 = ConvolutionLayer(1, 4, 5, with_bias, eta) # 1x28x28 -> 4x24x24
        self.H2 = AveragePoolingLayer()                     # 4x24x24 -> 4x12x12
        self.H3 = ConvolutionLayer(4, 3, 5, with_bias, eta) # 4x12x12 -> 3x8x8
        self.H4 = AveragePoolingLayer()                     # 3x8x8 -> 3x4x4
        
        # H5 Input dimension: 3*4*4=48 (Corrected)
        self.H5 = FullyConnectedLayer(input_dim=48, output_dim=10, eta=eta)
        
        # Set activation/loss functions based on the case
        if case == 1: # Sigmoid + MSE
            self.activation_fn = sigmoid
            self.loss_fn = mse_loss
            self.dE_dV_fn = d_mse_sigmoid
        elif case == 2: # Softmax + Cross-Entropy
            self.activation_fn = softmax
            self.loss_fn = cross_entropy_loss
            self.dE_dV_fn = d_ce_softmax
        
    def forward(self, X):
        """Forward pass"""
        N = X.shape[0]
        X_reshaped = X.reshape(N, 1, 28, 28)
        
        # H1: Convolution
        H1_out = self.H1.forward(X_reshaped)
        
        # H2: Avg Pooling
        H2_out = self.H2.forward(H1_out)
        
        # H3: Convolution
        H3_out = self.H3.forward(H2_out)
        
        # H4: Avg Pooling
        H4_out = self.H4.forward(H3_out)
        
        # H5 Input: Flatten (N, 3, 4, 4) -> (N, 48)
        H5_in_flatten = H4_out.reshape(N, -1)
        
        # H5: Fully Connected
        V, O = self.H5.forward(H5_in_flatten, self.activation_fn)
        
        return [H1_out, H2_out, H3_out, H4_out, H5_in_flatten, V, O]
    
    def predict(self, X):
        return self.forward(X)[-1]

    def backward(self, target, O, V, H5_in_flatten, H3_out, H1_out, batch_size=1):
        """Backward pass"""
        
        # 1. H5 Backprop
        dE_dV = self.dE_dV_fn(O, target) 
        dE_dH5_in = self.H5.backward(dE_dV, batch_size)
        
        # 2. H4 Backprop (Pooling)
        dE_dH4_out = dE_dH5_in.reshape(H3_out.shape[0], 3, 4, 4)
        dE_dH3_out = self.H4.backward(dE_dH4_out)
        
        # 3. H3 Backprop (Convolution)
        dE_dH2_out = self.H3.backward(dE_dH3_out, batch_size)
        
        # 4. H2 Backprop (Pooling)
        dE_dH1_out = self.H2.backward(dE_dH2_out)
        
        # 5. H1 Backprop (Convolution)
        self.H1.backward(dE_dH1_out, batch_size)
    
    def update_weights(self, batch_size):
        """Updates weights for trainable layers (H1, H3, H5)."""
        self.H1.update(batch_size)
        self.H3.update(batch_size)
        self.H5.update(batch_size)
        
    # ----------- 5. TRAINING AND EVALUATION METHODS ----------- #

    def train(self, X, D, y_labels, epochs=60, batch_size=1, verbose=True):
        """Online (B=1) or Mini-Batch (B>1) training."""
        N = X.shape[0]
        epoch_times = []
        loss_history = []
        
        for epoch in range(epochs):
            t0 = time.time()
            
            # Shuffle data
            idx = np.random.permutation(N)
            X_sh, D_sh, y_sh = X[idx], D[idx], y_labels[idx]

            # Batch/Online iteration loop
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                X_batch, D_batch = X_sh[start:end], D_sh[start:end]
                B = X_batch.shape[0]
                
                # Forward Pass
                H1_out, H2_out, H3_out, H4_out, H5_in_flatten, V, O = self.forward(X_batch)
                
                # Backward Pass (Accumulate gradients)
                self.backward(D_batch, O, V, H5_in_flatten, H3_out, H1_out, batch_size=B)
                
                # Weight Update
                if batch_size > 1 or B == 1:
                    self.update_weights(B)
                
            t1 = time.time()
            epoch_times.append(t1 - t0)
            
            # End-of-epoch evaluation
            avg_loss, train_acc, _ = self.evaluate(X, y_labels)
            loss_history.append(avg_loss)
            
            if verbose:
                loss_name = "MSE" if self.case == 1 else "CE"
                print(f"[Epoch {epoch + 1}/{epochs}, B={batch_size}] "
                      f"Avg. Train {loss_name}={avg_loss:.4f}, Acc={train_acc:.4f}, "
                      f"Time={epoch_times[-1]:.3f}s")
                      
        return epoch_times, loss_history

    def evaluate(self, X, labels):
        """Calculates accuracy, average loss, and class-wise misclassification."""
        
        # Forward Pass
        _, _, _, _, _, _, O = self.forward(X)
        
        # Loss calculation
        losses = self.loss_fn(O, labels_to_targets_cnn(labels))
        avg_loss = np.mean(losses)
        
        # Classification
        pred_idx = np.argmax(O, axis=1)
        # Convert index (0-9) to digit (1-9, 0)
        pred_digits = np.where(pred_idx == 9, 0, pred_idx + 1)

        correct = (pred_digits == labels)
        accuracy = correct.mean()

        misclassified_per_digit = {}
        for d in range(10):
            mask = (labels == d)
            mis = np.logical_and(mask, ~correct).sum()
            misclassified_per_digit[d] = int(mis)

        return avg_loss, accuracy, misclassified_per_digit

def inference_time_per_sample(model, X, repeats=5):
    """Measures the average time to classify a single pattern."""
    N = X.shape[0]
    t0 = time.time()
    for _ in range(repeats):
        model.predict(X)
    t1 = time.time()
    return (t1 - t0) / (repeats * N)

def plot_and_save_loss(loss_history, title, filename):
    """Plots and saves the loss history graph."""
    plt.figure()
    plt.plot(np.arange(1, len(loss_history) + 1), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss") # Changed to English
    plt.title(title)
    plt.grid(True)
    os.makedirs("results", exist_ok=True)
    plt.savefig(os.path.join("results", filename))
    plt.close()

def visualize_feature_maps(model, X_sample, title_prefix, filename_prefix):
    """Visualizes feature maps from intermediate layers."""
    # Forward pass for a single sample
    H1_out, H2_out, H3_out, H4_out, _, _, _ = model.forward(X_sample.reshape(1, -1))

    # Layer names and outputs
    layers = {
        "H1_Conv (4x24x24)": H1_out[0],
        "H2_AvgPool (4x12x12)": H2_out[0],
        "H3_Conv (3x8x8)": H3_out[0],
        "H4_AvgPool (3x4x4)": H4_out[0],
    }

    fig, axes = plt.subplots(4, 12, figsize=(15, 6)) 
    fig.suptitle(f"{title_prefix} - Feature Maps")

    row_idx = 0
    for layer_name, maps in layers.items():
        depth = maps.shape[0]
        
        map_avg = np.mean(maps)
        map_std = np.std(maps)
        
        for i in range(12): 
            ax = axes[row_idx, i]
            ax.set_xticks([])
            ax.set_yticks([])
            
            if i < depth:
                img = maps[i]
                # Normalize contrast
                ax.imshow(img, cmap='gray', vmin=map_avg - 2*map_std, vmax=map_avg + 2*map_std) 
            else:
                ax.axis('off')

            if i == 0:
                # Set y-label using the layer name
                ax.set_ylabel(layer_name.split('_')[0], rotation=90, size='large')
        
        row_idx += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs("feature_maps", exist_ok=True)
    plt.savefig(os.path.join("feature_maps", f"{filename_prefix}_maps.png"))
    plt.close()

# ----------- MAIN EXPERIMENT RUN ----------- #

def main():
    # ASSIGNMENT CONSTRAINT: ETA = 0.01
    ETA = 0.01 
    EPOCHS = 60 

    # 1) Load Data
    train_images_file = "train-images-idx3-ubyte"
    train_labels_file = "train-labels-idx1-ubyte"
    test_images_file = "t10k-images-idx3-ubyte"
    test_labels_file = "t10k-labels-idx1-ubyte"

    print("Loading MNIST files...") # Changed to English
    X_train = load_mnist_images(train_images_file)
    y_train = load_mnist_labels(train_labels_file)
    X_test = load_mnist_images(test_images_file)
    y_test = load_mnist_labels(test_labels_file)

    D_train = labels_to_targets_cnn(y_train)
    D_test = labels_to_targets_cnn(y_test)
    
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}") # Changed to English

    
    # Define Configurations
    configurations = [
        {"case": 1, "bias": False, "desc": "Sigmoid+MSE, No Bias", "tag": "S_MSE_NoBias"},
        {"case": 1, "bias": True, "desc": "Sigmoid+MSE, With Bias", "tag": "S_MSE_Bias"},
        {"case": 2, "bias": False, "desc": "Softmax+CE, No Bias", "tag": "SM_CE_NoBias"},
        {"case": 2, "bias": True, "desc": "Softmax+CE, With Bias", "tag": "SM_CE_Bias"},
    ]
    
    # 4 Online Training Runs (B=1)
    online_results = []
    
    print("\n\n=== 6. ONLINE TRAINING (B=1) - 4 CONFIGURATIONS ===") # Changed to English
    
    for i, config in enumerate(configurations):
        
        print(f"\n--- Configuration {i+1}: {config['desc']} (ETA={ETA}) ---") # Changed to English
        
        # Restart model
        model = CNN(with_bias=config['bias'], case=config['case'], eta=ETA)
        
        t_start = time.time()
        epoch_times, loss_hist = model.train(
            X_train, D_train, y_train, epochs=EPOCHS, batch_size=1, verbose=True
        )
        t_total_train = time.time() - t_start

        # Metrics
        train_loss, train_acc, train_mis = model.evaluate(X_train, y_train)
        test_loss, test_acc, test_mis = model.evaluate(X_test, y_test)
        inf_time = inference_time_per_sample(model, X_test)
        
        # Save results
        online_results.append({
            "config": config["desc"],
            "tag": config["tag"],
            "case": config["case"],
            "bias": config["bias"],
            "B": 1,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_mis": train_mis,
            "test_mis": test_mis,
            "avg_epoch_time": np.mean(epoch_times),
            "total_train_time": t_total_train,
            "inf_time_per_sample": inf_time,
            "loss_history": loss_hist,
            "model": model 
        })
        
        print(f"Test Accuracy: {test_acc:.4f}, Total Duration: {t_total_train:.2f}s") # Changed to English
        
        # Plot Loss
        loss_name = "MSE" if config['case'] == 1 else "CE"
        plot_and_save_loss(loss_hist, f"Training Loss ({loss_name}) - {config['desc']} (B=1)", f"{config['tag']}_B1_loss.png")
        
        # Feature Map Visualization
        digit_map = {0: 3, 1: 7, 2: 0, 3: 5} 
        target_digit = digit_map.get(i, 1)
        
        sample_idx_list = np.where(y_test == target_digit)[0]
        if len(sample_idx_list) > 0:
            sample_idx = sample_idx_list[0]
            X_sample = X_test[sample_idx]
            visualize_feature_maps(model, X_sample, f"Online {config['desc']} - Digit {target_digit}", f"{config['tag']}_digit{target_digit}") # Changed to English
        else:
            print(f"Warning: Digit {target_digit} not found in the test set.") # Changed to English


    # 7. SELECT BEST CONFIGURATION
    best_result = max(online_results, key=lambda r: r["test_acc"])
    best_config = {
        "case": best_result["case"], 
        "bias": best_result["bias"], 
        "desc": best_result["config"], 
        "tag": best_result["tag"]
    }
    
    print("\n\n=== 7. BEST CONFIGURATION SELECTED ===") # Changed to English
    print(f"Selected: {best_config['desc']}, Test Accuracy (Online): {best_result['test_acc']:.4f}") # Changed to English

    # 8. BATCH LEARNING (B=10, B=100)
    batch_sizes = [10, 100]
    batch_results = []
    
    print("\n\n=== 8. BATCH TRAINING (B=10, B=100) ===") # Changed to English
    
    for B in batch_sizes:
        
        print(f"\n--- Batch Size B={B} - Configuration: {best_config['desc']} (ETA={ETA}) ---") # Changed to English
        
        # Restart model
        model = CNN(with_bias=best_config['bias'], case=best_config['case'], eta=ETA)
        
        t_start = time.time()
        epoch_times, loss_hist = model.train(
            X_train, D_train, y_train, epochs=EPOCHS, batch_size=B, verbose=True
        )
        t_total_train = time.time() - t_start
        
        # Metrics
        train_loss, train_acc, train_mis = model.evaluate(X_train, y_train)
        test_loss, test_acc, test_mis = model.evaluate(X_test, y_test)
        inf_time = inference_time_per_sample(model, X_test)
        
        # Save results
        batch_results.append({
            "config": best_config["desc"],
            "tag": best_config["tag"],
            "case": best_config["case"],
            "bias": best_config["bias"],
            "B": B,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_mis": train_mis,
            "test_mis": test_mis,
            "avg_epoch_time": np.mean(epoch_times),
            "total_train_time": t_total_train,
            "inf_time_per_sample": inf_time,
            "loss_history": loss_hist,
            "model": model
        })
        
        print(f"Test Accuracy: {test_acc:.4f}, Total Duration: {t_total_train:.2f}s") # Changed to English
        
        # Plot Loss
        loss_name = "MSE" if best_config['case'] == 1 else "CE"
        plot_and_save_loss(loss_hist, f"Training Loss ({loss_name}) - {best_config['desc']} (B={B})", f"{best_config['tag']}_B{B}_loss.png")

        # Feature Map Visualization
        target_digit = 9 if B == 10 else 2
        
        sample_idx_list = np.where(y_test == target_digit)[0]
        if len(sample_idx_list) > 0:
            sample_idx = sample_idx_list[0]
            X_sample = X_test[sample_idx]
            visualize_feature_maps(model, X_sample, f"Batch B={B} {best_config['desc']} - Digit {target_digit}", f"{best_config['tag']}_B{B}_digit{target_digit}") # Changed to English
        else:
            print(f"Warning: Digit {target_digit} not found in the test set.") # Changed to English
        
    
    # 9. REPORT SUMMARY (Console Output)
    print("\n\n=== 9. SUMMARY TABLE OF ALL EXPERIMENTAL RESULTS ===") # Changed to English
    
    all_results = online_results + batch_results
    
    # Table headers
    header = ["Config (Tag)", "B", "Train Acc", "Test Acc", "Avg Epoch Time (s)", "Total Time (s)", "Inf Time (s)"]
    print("-" * 110)
    print(f"{header[0]:<20} | {header[1]:<3} | {header[2]:<10} | {header[3]:<10} | {header[4]:<20} | {header[5]:<15} | {header[6]:<15}")
    print("-" * 110)
    
    for res in all_results:
        print(f"{res['tag']:<20} | {res['B']:<3} | {res['train_acc']:.4f} | {res['test_acc']:.4f} | {res['avg_epoch_time']:.4f} | {res['total_train_time']:.2f} | {res['inf_time_per_sample']:.2e}")
    print("-" * 110)

if __name__ == "__main__":
    main()