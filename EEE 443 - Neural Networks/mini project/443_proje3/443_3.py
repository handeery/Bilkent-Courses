import numpy as np
import struct
import time
import matplotlib

# NumPy uyarılarını sustur (overflow vs.)
np.seterr(all='ignore')


# ----------------- MNIST LOADING HELPERS ----------------- #

def load_mnist_images(filename):
    """Reads MNIST image file and returns (N, 784) float32 array in [0,1]."""
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows * cols).astype(np.float32) / 255.0
    return data


def load_mnist_labels(filename):
    """Reads MNIST label file and returns (N,) uint8 array with digits 0–9."""
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def labels_to_targets(labels):
    """
    Converts digit labels (0–9) to desired output vectors of length 10.

    Output neuron order: 1,2,3,4,5,6,7,8,9,0
    For digit i: only corresponding neuron is +1, others are -1.
    """
    N = labels.shape[0]
    targets = -np.ones((N, 10), dtype=np.float32)
    # digit -> neuron index: 1->0, 2->1, ..., 9->8, 0->9
    indices = np.where(labels == 0, 9, labels - 1)
    targets[np.arange(N), indices] = 1.0
    return targets


# ----------------- ACTIVATION FUNCTIONS ----------------- #

def tanh(x):
    return np.tanh(x)


def tanh_deriv_from_output(y):
    # derivative of tanh wrt input, expressed via output y = tanh(v)
    return 1.0 - y ** 2


def relu(x):
    return np.maximum(0.0, x)


def relu_deriv_from_input(x):
    return (x > 0).astype(np.float32)


# ----------------- MLP IMPLEMENTATION ----------------- #

class MLP:
    """
    2-layer MLP: input -> hidden -> output
    Bias is handled by adding an extra input of 1 at each layer.
    """

    def __init__(self, input_dim, hidden_dim, output_dim,
                 activation_hidden='tanh', activation_output='tanh',
                 eta=0.01, seed=None):

        if seed is not None:
            np.random.seed(seed)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.eta = eta

        # choose activations
        if activation_hidden == 'tanh':
            self.act_h = tanh
            self.d_act_h_input = None      # use output-based deriv
        elif activation_hidden == 'relu':
            self.act_h = relu
            self.d_act_h_input = relu_deriv_from_input
        else:
            raise ValueError("Unsupported hidden activation.")

        if activation_output == 'tanh':
            self.act_o = tanh
            self.d_act_o_output = tanh_deriv_from_output
        else:
            raise ValueError("Unsupported output activation.")

        # Small Gaussian initialization to avoid overflow
        self.W1 = (0.01 * np.random.randn(input_dim + 1, hidden_dim)).astype(np.float32)
        self.W2 = (0.01 * np.random.randn(hidden_dim + 1, output_dim)).astype(np.float32)

    # ---- forward ---- #

    def _forward(self, X):
        """
        Forward pass.
        X: (batch, input_dim)
        returns: Xb, Z1, H, Hb, Z2, O
        """
        batch_size = X.shape[0]

        # add bias to input
        Xb = np.concatenate([X, np.ones((batch_size, 1), dtype=np.float32)],
                            axis=1)
        Z1 = Xb @ self.W1
        Z1 = np.clip(Z1, -50, 50)
        H = self.act_h(Z1)

        # add bias to hidden activations
        Hb = np.concatenate([H, np.ones((batch_size, 1), dtype=np.float32)],
                            axis=1)
        Z2 = Hb @ self.W2
        Z2 = np.clip(Z2, -50, 50)
        O = self.act_o(Z2)

        return Xb, Z1, H, Hb, Z2, O

    def predict(self, X):
        """Returns network output for given inputs X."""
        _, _, _, _, _, O = self._forward(X)
        return O

    # ---- training (batch) ---- #

    def _update_batch(self, X_batch, D_batch):
        """
        Performs one gradient step using a batch.
        X_batch: (B, input_dim)
        D_batch: (B, output_dim)
        """
        Xb, Z1, H, Hb, Z2, O = self._forward(X_batch)
        B = X_batch.shape[0]

        # output layer delta
        dE_dO = (O - D_batch)                      # (B, out)
        delta_o = dE_dO * self.d_act_o_output(O)   # (B, out)

        # hidden layer delta (exclude bias column from W2)
        d_hidden = (delta_o @ self.W2.T)[:, :-1]   # (B, hidden)
        if self.d_act_h_input is None:
            # tanh: derivative via output H
            d_act_h = tanh_deriv_from_output(H)
        else:
            # ReLU: derivative via input Z1
            d_act_h = self.d_act_h_input(Z1)

        delta_h = d_hidden * d_act_h               # (B, hidden)

        # gradients
        grad_W2 = Hb.T @ delta_o / B
        grad_W1 = Xb.T @ delta_h / B

        # gradient descent step
        self.W2 -= self.eta * grad_W2
        self.W1 -= self.eta * grad_W1

    def train_batch(self, X, D, batch_size=20, epochs=60, verbose=True):
        """
        Batch learning with given batch size.
        Returns (epoch_times, loss_history).
        """
        N = X.shape[0]
        epoch_times = []
        loss_history = []

        for epoch in range(epochs):
            t0 = time.time()

            idx = np.random.permutation(N)
            X_sh = X[idx]
            D_sh = D[idx]

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                xb = X_sh[start:end]
                db = D_sh[start:end]
                self._update_batch(xb, db)

            t1 = time.time()
            epoch_times.append(t1 - t0)

            mse = self.compute_mse(X, D)
            loss_history.append(mse)

            if verbose:
                print(f"[batch B={batch_size}] epoch {epoch + 1}/{epochs}, "
                      f"MSE={mse:.4f}, time={epoch_times[-1]:.3f}s")

        return epoch_times, loss_history

    # ---- metrics ---- #

    def compute_mse(self, X, D):
        O = self.predict(X)
        return 0.5 * np.mean(np.sum((D - O) ** 2, axis=1))

    def evaluate_classification(self, X, labels):
        """
        Returns: accuracy, mse, misclassified_per_digit (dict digit->count)
        """
        O = self.predict(X)
        # index 0..9 -> digits 1..9,0
        pred_idx = np.argmax(O, axis=1)
        pred_digits = np.where(pred_idx == 9, 0, pred_idx + 1)

        correct = (pred_digits == labels)
        accuracy = correct.mean()

        misclassified_per_digit = {}
        for d in range(10):
            mask = (labels == d)
            mis = np.logical_and(mask, ~correct).sum()
            misclassified_per_digit[d] = int(mis)

        D = labels_to_targets(labels)
        mse = self.compute_mse(X, D)

        return accuracy, mse, misclassified_per_digit


def inference_time_per_sample(model, X, repeats=5):
    """Roughly measures average time to classify a single pattern."""
    N = X.shape[0]
    t0 = time.time()
    for _ in range(repeats):
        model.predict(X)
    t1 = time.time()
    return (t1 - t0) / (repeats * N)


# ----------------- MAIN EXPERIMENT ----------------- #

def main():
    # 1) Load data
    train_images_file = "train-images-idx3-ubyte"
    train_labels_file = "train-labels-idx1-ubyte"
    test_images_file = "t10k-images-idx3-ubyte"
    test_labels_file = "t10k-labels-idx1-ubyte"

    print("Loading MNIST files...")
    X_train = load_mnist_images(train_images_file)
    y_train = load_mnist_labels(train_labels_file)
    X_test = load_mnist_images(test_images_file)
    y_test = load_mnist_labels(test_labels_file)

    print(f"Train samples: {X_train.shape[0]}, dim: {X_train.shape[1]}")
    print(f"Test samples : {X_test.shape[0]}, dim: {X_test.shape[1]}")

    input_dim = X_train.shape[1]   # should be 784
    output_dim = 10

    hidden_list = [400, 800]
    etas = [0.01, 0.03]
    cases = [1, 2]   # 1: tanh-tanh, 2: ReLU-tanh
    epochs_batch = 60
    batch_main = 20

    D_train = labels_to_targets(y_train)

    results = []
    loss_curves = {}  # (case, hidden, eta) -> loss_history

    # --------- TRAIN ALL CONFIGS WITH BATCH (B=20) --------- #
    for case in cases:
        for N_hid in hidden_list:
            for eta in etas:
                if case == 1:
                    act_h, act_o = 'tanh', 'tanh'
                else:
                    act_h, act_o = 'relu', 'tanh'

                print("\n=== CONFIG: case={}, hidden={}, eta={} (B=20) ==="
                      .format(case, N_hid, eta))

                model = MLP(input_dim, N_hid, output_dim,
                            activation_hidden=act_h,
                            activation_output=act_o,
                            eta=eta)

                epoch_times, loss_hist = model.train_batch(
                    X_train, D_train,
                    batch_size=batch_main,
                    epochs=epochs_batch,
                    verbose=True
                )

                avg_epoch_time = float(np.mean(epoch_times))
                total_train_time = float(np.sum(epoch_times))

                # loss eğrilerini kaydet
                loss_curves[(case, N_hid, eta)] = loss_hist

                # Train metrics
                train_acc, train_mse, train_mis = \
                    model.evaluate_classification(X_train, y_train)

                # Test metrics
                test_acc, test_mse, test_mis = \
                    model.evaluate_classification(X_test, y_test)

                inf_time = inference_time_per_sample(model, X_test)

                print(f"Train acc: {train_acc:.4f}, "
                      f"Test acc: {test_acc:.4f}")
                print(f"Train MSE: {train_mse:.4f}, "
                      f"Test MSE: {test_mse:.4f}")
                print(f"Avg epoch time: {avg_epoch_time:.4f}s, "
                      f"Total train time: {total_train_time:.2f}s")
                print(f"Inference time per sample: {inf_time:.6e}s")
                print("Misclassified per digit (test):", test_mis)

                results.append({
                    "case": case,
                    "hidden": N_hid,
                    "eta": eta,
                    "train_acc": train_acc,
                    "train_mse": train_mse,
                    "test_acc": test_acc,
                    "test_mse": test_mse,
                    "train_mis": train_mis,
                    "test_mis": test_mis,
                    "avg_epoch_time": avg_epoch_time,
                    "total_train_time": total_train_time,
                    "inf_time_per_sample": inf_time,
                })

    # --------- PLOT TRAINING LOSS CURVES FOR ALL CONFIGS (B=20) --------- #
    for key, losses in loss_curves.items():
        case, N_hid, eta = key
        epochs = np.arange(1, len(losses) + 1)

        plt.figure()
        plt.plot(epochs, losses)
        plt.xlabel("Epoch")
        plt.ylabel("Training MSE")
        plt.title(f"Case {case}, hidden={N_hid}, eta={eta}, B=20")
        import os
        out_dir = "results_3"
        os.makedirs(out_dir, exist_ok=True)
        filename = f"case{case}_hidden{N_hid}_eta{eta}_B20.png"
        plt.savefig(os.path.join(out_dir, filename))
        plt.grid(True)

    # plt.show()

    # --------- SELECT BEST CONFIG BY TEST ACCURACY --------- #
    best = max(results, key=lambda r: r["test_acc"])
    print("\n=== BEST CONFIGURATION (BY TEST ACC) ===")
    print("case = {}, hidden = {}, eta = {}, test_acc = {:.4f}"
          .format(best["case"], best["hidden"],
                  best["eta"], best["test_acc"]))

    # --------- BATCH LEARNING FOR BEST CONFIG WITH B=20 AND B=200 --------- #
    B_values = [20, 200]

    for B in B_values:
        print("\n=== BATCH LEARNING: B={}, case={}, hidden={}, eta={} ==="
              .format(B, best["case"], best["hidden"], best["eta"]))

        if best["case"] == 1:
            act_h, act_o = 'tanh', 'tanh'
        else:
            act_h, act_o = 'relu', 'tanh'

        model = MLP(input_dim, best["hidden"], output_dim,
                    activation_hidden=act_h,
                    activation_output=act_o,
                    eta=best["eta"])

        epoch_times, loss_hist_B = model.train_batch(
            X_train, D_train,
            batch_size=B,
            epochs=epochs_batch,
            verbose=True
        )

        avg_epoch_time = float(np.mean(epoch_times))
        total_train_time = float(np.sum(epoch_times))

        train_acc, train_mse, train_mis = \
            model.evaluate_classification(X_train, y_train)
        test_acc, test_mse, test_mis = \
            model.evaluate_classification(X_test, y_test)
        inf_time = inference_time_per_sample(model, X_test)

        print(f"[B={B}] Train acc: {train_acc:.4f}, "
              f"Test acc: {test_acc:.4f}")
        print(f"[B={B}] Train MSE: {train_mse:.4f}, "
              f"Test MSE: {test_mse:.4f}")
        print(f"[B={B}] Avg epoch time: {avg_epoch_time:.4f}s, "
              f"Total train time: {total_train_time:.2f}s")
        print(f"[B={B}] Inference time per sample: {inf_time:.6e}s")
        print(f"[B={B}] Misclassified per digit (test): {test_mis}")

        # B için loss grafiği çiz
        epochs_arr = np.arange(1, len(loss_hist_B) + 1)
        plt.figure()
        plt.plot(epochs_arr, loss_hist_B)
        plt.xlabel("Epoch")
        plt.ylabel("Training MSE")
        plt.title(f"Best config - Batch size {B}")
        import os
        out_dir = "results_3"
        os.makedirs(out_dir, exist_ok=True)
        filename = f"bestconfig_batch{B}.png"
        plt.savefig(os.path.join(out_dir, filename))
        plt.grid(True)

    # plt.show()


if __name__ == "__main__":
    main()