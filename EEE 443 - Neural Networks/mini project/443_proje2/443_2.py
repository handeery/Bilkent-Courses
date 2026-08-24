import numpy as np
import matplotlib.pyplot as plt
import math
import os

# Parameter ranges for experimentation
eta_values = np.array([0.1, 0.3, 0.5])
sample_counts = np.array([20, 30, 40])
hidden_neuron_options = [10, 15, 20]

# Directory to store generated figures
os.makedirs("results", exist_ok=True)

# ------------------------------------------------------------
# TRAINING FUNCTION
# ------------------------------------------------------------
def train_network(x_vals, x_augmented, target_vals, eta_initial, hidden_count):
    """
    Trains the neural network using online gradient descent.
    Each sample triggers an immediate weight update (stochastic learning).
    """

    iteration = 0
    max_iterations = 5000
    stopping_threshold = 1e-4
    eta = eta_initial
    error_history = []
    gradient_clip = 100   # prevents exploding updates

    # Weight initialization (small, random)
    w_hidden = np.random.uniform(-0.01, 0.01, size=(hidden_count, 2))
    w_output = np.random.uniform(-0.1, 0.1, size=(hidden_count + 1,))

    avg_error = 1

    while iteration <= max_iterations and avg_error > stopping_threshold:

        # Learning rate decay for stability
        if (iteration + 1) % 50 == 0:
            eta *= 0.9

        predictions = []

        for k in range(len(x_augmented)):
            inp = x_augmented[k]

            # Forward pass
            hidden_raw = np.dot(w_hidden, inp)
            hidden_act = np.tanh(hidden_raw)
            hidden_act_augmented = np.concatenate([hidden_act, [-1.0]])  # add bias to hidden output
            y_hat = np.dot(hidden_act_augmented, w_output)
            predictions.append(y_hat)

            # Error signal
            error = target_vals[k] - y_hat

            # Output layer update
            w_output = update_output_layer(eta, w_output, hidden_act_augmented, error, gradient_clip)

            # Hidden layer update
            w_hidden = update_hidden_layer(eta, hidden_raw, w_output[:-1], w_hidden, inp, error, gradient_clip)

        # Mean squared error for this epoch
        avg_error = mean_squared_error(target_vals, predictions)
        error_history.append(avg_error)
        iteration += 1

    return w_hidden, w_output, error_history, np.array(predictions)


# ------------------------------------------------------------
# TEST PHASE
# ------------------------------------------------------------
def evaluate_network(test_inputs, test_targets, w_hidden, w_output):
    """
    Evaluates the trained network on previously unseen test data.
    """
    test_predictions = []
    for val in test_inputs:
        val_ext = np.array([val, -1.0])
        h_raw = np.dot(w_hidden, val_ext)
        h_act = np.tanh(h_raw)
        h_aug = np.concatenate([h_act, [-1.0]])
        output = np.dot(h_aug, w_output)
        test_predictions.append(output)

    test_predictions = np.array(test_predictions)
    test_error = mean_squared_error(test_targets, test_predictions)
    return test_predictions, test_error


# ------------------------------------------------------------
# SUPPORT FUNCTIONS
# ------------------------------------------------------------
def mean_squared_error(true_vals, pred_vals):
    """
    Computes the mean squared error (MSE) between true and predicted outputs.
    """
    return np.mean((np.array(true_vals) - np.array(pred_vals)) ** 2)

def generate_training_points(N_val):
    """
    Generates training input samples and their bias-augmented versions.
    """
    xs, xs_aug = [], []
    for idx in range(1, 8*N_val + 2):
        x = -math.pi + ((idx-1) / (4*N_val)) * math.pi
        xs.append(x)
        xs_aug.append([x, -1.0])  # bias included
    return np.array(xs), np.array(xs_aug)

def midpoint_test_set(train_x):
    """
    Generates test inputs at midpoints between training samples.
    """
    return np.array([(train_x[i] + train_x[i+1]) / 2 for i in range(len(train_x)-1)])

def target_function(x_values):
    """
    Defines the function to be approximated by the network.
    """
    return np.array([0.25 * (v ** 2) * math.sin(v) for v in x_values])

def update_hidden_layer(eta, net_vals, w_out, w_hidden, input_vec, error, clip):
    """
    Adjusts hidden layer weights using backpropagated error and tanh derivative.
    """
    new_weights = []
    for i in range(len(w_hidden)):
        derivative = 1 - np.tanh(net_vals[i])**2
        delta = error * w_out[i] * derivative
        gradient = delta * np.array(input_vec)

        # Gradient clipping for numerical stability
        norm = np.linalg.norm(gradient)
        if norm > clip:
            gradient *= (clip / norm)

        new_weights.append(w_hidden[i] + eta * gradient)
    return np.array(new_weights)

def update_output_layer(eta, w_out, hidden_act_augmented, error, clip):
    """
    Updates the output layer weights using gradient descent.
    """
    updated_weights = []
    for j in range(len(w_out)):
        grad = error * hidden_act_augmented[j]
        norm = np.linalg.norm(grad)
        if norm > clip:
            grad *= (clip / norm)
        updated_weights.append(w_out[j] + eta * grad)
    return np.array(updated_weights)


# ------------------------------------------------------------
# MAIN EXPERIMENT DRIVER
# ------------------------------------------------------------
def run_experiments(N_list, eta_list, hidden_list):
    for N_val in N_list:
        train_x, train_x_aug = generate_training_points(N_val)
        train_targets = target_function(train_x)

        test_x = midpoint_test_set(train_x)
        test_targets = target_function(test_x)

        for eta in eta_list:
            for h in hidden_list:

                print(f"\n>> Training Model: N={N_val}, η={eta}, Hidden={h}")

                w_h, w_o, err_curve, train_pred = train_network(train_x, train_x_aug, train_targets, eta, h)
                test_pred, test_err = evaluate_network(test_x, test_targets, w_h, w_o)

                print(f"Test Error: {test_err:.6f}")

                # Store 3-in-1 figure result
                fig, axes = plt.subplots(1, 3, figsize=(15,4))
                fig.suptitle(f"N={N_val}, η={eta}, Hidden={h}")

                # Training fit
                axes[0].plot(train_x, train_targets, 'r')
                axes[0].plot(train_x, train_pred, 'b')
                axes[0].grid(True)
                axes[0].set_title("Training Approximation")

                # Error curve
                axes[1].plot(err_curve)
                axes[1].set_title("Training Error Convergence")
                axes[1].grid(True)

                # Test fit
                axes[2].plot(test_x, test_targets, 'r')
                axes[2].plot(test_x, test_pred, 'b--')
                axes[2].grid(True)
                axes[2].set_title(f"Test Approximation (Err={test_err:.4f})")

                filename = f"results/N{N_val}_eta{eta}_hidden{h}.png"
                fig.savefig(filename, dpi=200)
                plt.close(fig)

run_experiments(sample_counts, eta_values, hidden_neuron_options)