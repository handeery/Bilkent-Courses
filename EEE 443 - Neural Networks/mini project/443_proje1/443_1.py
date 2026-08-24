# ===== Minimal imports (perceptron-only) =====
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["savefig.transparent"] = False

# ------------------------------------------------------------
# Steps 1–2: Uniform data in [0,1]^2 and labeling by line (0,a)–(1,b)
# ------------------------------------------------------------
# ----- Step 1 helpers: generate points only & plot them (no labels, no line)

def generate_uniform_points(n: int = 100, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 2), dtype=np.float64)
    return X


def plot_points_only(X: np.ndarray, title: str, save_path: str):
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], s=20, c="gray", label="Points (unlabeled)")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel("x1"); plt.ylabel("x2"); plt.title(title)
    plt.legend(); plt.grid(True, linewidth=0.3, alpha=0.6); plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.show()

# ------------------------------------------------------------
# Steps 1–2: Uniform data in [0,1]^2 and labeling by line (0,a)–(1,b)
# ------------------------------------------------------------
def generate_uniform_with_line(n: int = 100, a: float = None, b: float = None, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 2), dtype=np.float64)  # uniform in [0,1]^2
    if a is None:
        a = float(rng.random())
    if b is None:
        b = float(rng.random())
    # line: x2 = a + (b - a) * x1
    x1 = X[:, 0]
    line_vals = a + (b - a) * x1
    # Step 2 labeling rule: on/above -> Class 1, else -> Class 2
    y = (X[:, 1] >= line_vals).astype(np.int64)
    return X, y, a, b

# ----- Step 2 helper: given X and (a,b), return labels; if a/b None, sample them
def label_by_line(X: np.ndarray, a: float = None, b: float = None, seed: int = 42):
    rng = np.random.default_rng(seed)
    if a is None:
        a = float(rng.random())
    if b is None:
        b = float(rng.random())
    x1 = X[:, 0]
    line_vals = a + (b - a) * x1
    y = (X[:, 1] >= line_vals).astype(np.int64)
    return y, a, b

# ------------------------------------------------------------
# Step 2: Plot points with colors (red=Class 1, blue=Class 2) + the separating line
# ------------------------------------------------------------
def plot_points_with_true_line(X: np.ndarray, y: np.ndarray, a: float, b: float, title: str, save_path: str):
    m0, m1 = (y == 0), (y == 1)
    plt.figure(figsize=(6, 6))
    # Colors requested: Class 1 -> red, Class 2 -> blue
    plt.scatter(X[m1, 0], X[m1, 1], s=20, c="red", label="Class 1 (y=1)")
    plt.scatter(X[m0, 0], X[m0, 1], s=20, c="blue", label="Class 2 (y=0)")
    xs = np.linspace(0.0, 1.0, 201)
    ys = a + (b - a) * xs
    plt.plot(xs, ys, "--", lw=2, c="black", label="True separator (Step 2)")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel("x1"); plt.ylabel("x2"); plt.title(title)
    plt.legend(); plt.grid(True, linewidth=0.3, alpha=0.6); plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.show()

# ------------------------------------------------------------
# Steps 4–5: Single-layer perceptron (2 inputs, 1 threshold) + training until convergence
# ------------------------------------------------------------
class Perceptron:
    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        # Initialize w1, w2, theta in [-1,1]
        self.w = (2.0 * rng.random(2) - 1.0).astype(np.float64)  # w1, w2
        self.b = (2.0 * rng.random() - 1.0)                      # bias = -theta  (z = w·x + b)

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = self.predict_logits(X)
        return (z >= 0.0).astype(np.int64)  # step → {0,1}

    def fit(self, X: np.ndarray, y: np.ndarray, eta: float = 0.1, max_epochs: int = 10000):
        """Train until convergence. Return list of misclassified counts per epoch."""
        mis_hist: list[int] = []
        for _ in range(max_epochs):
            preds = self.predict(X)
            mis = (preds != y)
            miscount = int(mis.sum())
            mis_hist.append(miscount)
            if miscount == 0:
                break  # converged (linearly separable here)
            # Perceptron update (sequential over misclassified samples)
            for i in np.where(mis)[0]:
                xi = X[i]
                ei = int(y[i]) - int(preds[i])  # ∈ {-1, +1}
                self.w = self.w + eta * ei * xi
                self.b = self.b + eta * ei      # theta <- theta - eta*ei
        return mis_hist

# ------------------------------------------------------------
# Steps 5–7 (per-eta combined): true vs learned line + misclass curve
# ------------------------------------------------------------

def plot_steps5to7_combined_for_eta(X: np.ndarray, y: np.ndarray, a: float, b: float,
                                    model: 'Perceptron', mis_hist: list,
                                    eta: float, n: int, save_path: str):
    xs = np.linspace(0.0, 1.0, 201)
    y_true = a + (b - a) * xs
    m0, m1 = (y == 0), (y == 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Step 6 — true vs learned line
    ax = axes[0]
    # learned line: w1*x1 + w2*x2 + b = 0  =>  x2 = -(w1/w2)*x1 - b/w2
    if abs(model.w[1]) < 1e-12:
        x_cut = -model.b / (model.w[0] + 1e-12)
        ax.plot([x_cut, x_cut], [0, 1], '-', lw=2, c='green', label='Learned (perceptron)')
    else:
        y_learned = -(model.w[0] / model.w[1]) * xs - (model.b / model.w[1])
        ax.plot(xs, y_learned, '-', lw=2, c='green', label='Learned (perceptron)')
    ax.scatter(X[m1, 0], X[m1, 1], s=15, c='red', label='Class 1 (y=1)')
    ax.scatter(X[m0, 0], X[m0, 1], s=15, c='blue', label='Class 2 (y=0)')
    ax.plot(xs, y_true, '--', lw=2, c='black', label='True separator')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"Step 6: True vs Learned (η={eta}, n={n})")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.legend(); ax.grid(True, linewidth=0.3, alpha=0.6)

    # Right panel: Step 7 — misclassified vs epoch
    ax = axes[1]
    ax.plot(np.arange(1, len(mis_hist) + 1), mis_hist, linewidth=2)
    ax.set_title(f"Step 7: Misclassified vs Epoch (η={eta}, n={n})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("# Misclassified")
    ax.grid(True, linewidth=0.3, alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# ------------------------------------------------------------
# Step 6: Plot, on the same figure, learned separator (Step 5) and original line (Step 2)
# ------------------------------------------------------------
def plot_true_vs_learned_lines_combined(X: np.ndarray, y: np.ndarray, a: float, b: float, models_by_eta: dict, save_path: str):
    xs = np.linspace(0.0, 1.0, 201)
    y_true = a + (b - a) * xs
    m0, m1 = (y == 0), (y == 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, eta in zip(axes, (0.1, 1.0, 10.0)):
        model = models_by_eta[eta]
        # learned line: w1*x1 + w2*x2 + b = 0  =>  x2 = -(w1/w2)*x1 - b/w2
        if abs(model.w[1]) < 1e-12:
            x_cut = -model.b / (model.w[0] + 1e-12)
            ax.plot([x_cut, x_cut], [0, 1], '-', lw=2, c='green', label='Learned (perceptron)')
        else:
            y_learned = -(model.w[0] / model.w[1]) * xs - (model.b / model.w[1])
            ax.plot(xs, y_learned, '-', lw=2, c='green', label='Learned (perceptron)')
        # scatter and true line
        ax.scatter(X[m1, 0], X[m1, 1], s=15, c='red', label='Class 1 (y=1)')
        ax.scatter(X[m0, 0], X[m0, 1], s=15, c='blue', label='Class 2 (y=0)')
        ax.plot(xs, y_true, '--', lw=2, c='black', label='True separator')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"η={eta}")
        ax.set_xlabel("x1"); ax.set_ylabel("x2")
        ax.legend(); ax.grid(True, linewidth=0.3, alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

# ------------------------------------------------------------
# Step 7: Plot number of misclassified points vs. epoch
# ------------------------------------------------------------
def plot_misclass_curves(histories: dict, title: str, save_path: str):
    plt.figure(figsize=(7, 5))
    for name, hist in histories.items():
        plt.plot(np.arange(1, len(hist) + 1), hist, linewidth=2, label=str(name))
    plt.xlabel("Epoch"); plt.ylabel("# Misclassified")
    plt.title(title); plt.grid(True, linewidth=0.3, alpha=0.6)
    plt.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.show()

# ------------------------------------------------------------
# Steps 8–9 wrapper: compare η ∈ {0.1, 1, 10} and repeat for n=500
# ------------------------------------------------------------
def run_perceptron_suite_for_n(n: int, seed: int = 7):
    # Step 1: generate points only
    X = generate_uniform_points(n=n, seed=seed)
    plot_points_only(
        X,
        title=f"Random points (Step 1, n={n})",
        save_path=f"/Users/handeeryilmaz/Downloads/perc_points_random_n{n}.png",
    )

    # Step 2: choose a,b and label by the line (0,a)-(1,b)
    y, a, b = label_by_line(X, a=None, b=None, seed=seed)

    # Step 2: plot points with colors + true line
    plot_points_with_true_line(
        X, y, a, b,
        title=f"Labeled points + true separator (Step 2, n={n})",
        save_path=f"/Users/handeeryilmaz/Downloads/perc_points_true_n{n}.png",
    )

    # Step 4: Single-layer perceptron setup (2 inputs, 1 threshold, 1 output)
    perc_step4 = Perceptron(seed=seed)
    theta_step4 = -perc_step4.b  # since we store bias b = -theta
    print("[Step 4] Initialized single-layer perceptron:")
    print(f"  w1={perc_step4.w[0]:.4f}, w2={perc_step4.w[1]:.4f}, theta={theta_step4:.4f} (bias b={perc_step4.b:.4f})")
    print("  Desired outputs: Class 1 -> 1, Class 2 -> 0")

    # Steps 4–5 (train) + Step 8 (η comparison)
    histories = {}
    models_by_eta = {}
    for eta in (0.1, 1.0, 10.0):
        p = Perceptron(seed=seed + int(eta * 100))
        mis_hist = p.fit(X, y, eta=eta, max_epochs=10000)
        histories[f"η={eta}"] = mis_hist
        models_by_eta[eta] = p

    # ---- Single PNG with 6 panels: Step 6 & Step 7 for η=0.1,1,10 ----
    etas = (0.1, 1.0, 10.0)
    xs = np.linspace(0.0, 1.0, 201)
    y_true = a + (b - a) * xs
    m0, m1 = (y == 0), (y == 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Top row: Step 6 (True vs Learned) for each eta
    for j, eta in enumerate(etas):
        ax = axes[0, j]
        model = models_by_eta[eta]
        if abs(model.w[1]) < 1e-12:
            x_cut = -model.b / (model.w[0] + 1e-12)
            ax.plot([x_cut, x_cut], [0, 1], '-', lw=2, c='green', label=f'Learned (η={eta})')
        else:
            y_learned = -(model.w[0] / model.w[1]) * xs - (model.b / model.w[1])
            ax.plot(xs, y_learned, '-', lw=2, c='green', label=f'Learned (η={eta})')
        ax.scatter(X[m1, 0], X[m1, 1], s=12, c='red', label='Class 1')
        ax.scatter(X[m0, 0], X[m0, 1], s=12, c='blue', label='Class 2')
        ax.plot(xs, y_true, '--', lw=2, c='black', label='True line')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f'Step 6: True vs Learned (η={eta})')
        ax.set_xlabel('x1'); ax.set_ylabel('x2')
        ax.grid(True, linewidth=0.3, alpha=0.6)
        if j == 0:
            ax.legend()

    # Bottom row: Step 7 (Misclassified vs Epoch) for each eta
    for j, eta in enumerate(etas):
        ax = axes[1, j]
        hist = histories[f'η={eta}']
        ax.plot(np.arange(1, len(hist) + 1), hist, linewidth=2)
        ax.set_title(f'Step 7: Misclassified vs Epoch (η={eta})')
        ax.set_xlabel('Epoch'); ax.set_ylabel('# Misclassified')
        ax.grid(True, linewidth=0.3, alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"/Users/handeeryilmaz/Downloads/perc_steps6_and_7_all_etas_n{n}.png", dpi=150, bbox_inches='tight')
    plt.show()

    # ---- Two-in-one figure: Misclassified vs Epoch (overlay) + True vs Learned (overlay) ----
    etas = (0.1, 1.0, 10.0)
    xs = np.linspace(0.0, 1.0, 201)
    y_true = a + (b - a) * xs
    m0, m1 = (y == 0), (y == 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Misclassified vs Epoch Comparison (overlay all η)
    for eta in etas:
        hist = histories[f'η={eta}']
        axes[0].plot(np.arange(1, len(hist) + 1), hist, linewidth=2, label=f'η={eta}')
    axes[0].set_title('Misclassified vs Epoch Comparison')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('# Misclassified')
    axes[0].grid(True, linewidth=0.3, alpha=0.6)
    axes[0].legend()

    # Right: True vs Learned Comparison (overlay all η)
    for eta, color in zip(etas, ('green', 'orange', 'purple')):
        model = models_by_eta[eta]
        if abs(model.w[1]) < 1e-12:
            x_cut = -model.b / (model.w[0] + 1e-12)
            axes[1].plot([x_cut, x_cut], [0, 1], '-', lw=2, c=color, label=f'Learned (η={eta})')
        else:
            y_learned = -(model.w[0] / model.w[1]) * xs - (model.b / model.w[1])
            axes[1].plot(xs, y_learned, '-', lw=2, c=color, label=f'Learned (η={eta})')
    axes[1].plot(xs, y_true, '--', lw=2, c='black', label='True line')
    axes[1].scatter(X[m1, 0], X[m1, 1], s=10, c='red', label='Class 1')
    axes[1].scatter(X[m0, 0], X[m0, 1], s=10, c='blue', label='Class 2')
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
    axes[1].set_title('True vs Learned Comparison')
    axes[1].set_xlabel('x1'); axes[1].set_ylabel('x2')
    axes[1].grid(True, linewidth=0.3, alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"/Users/handeeryilmaz/Downloads/perc_steps6_7_overlay_comparison_n{n}.png", dpi=150, bbox_inches='tight')
    plt.show()

# ------------------------------------------------------------
# DRIVER (Step 9: repeat for n=500)
# ------------------------------------------------------------
if __name__ == "__main__":
    run_perceptron_suite_for_n(n=100, seed=7)
    run_perceptron_suite_for_n(n=500, seed=7)