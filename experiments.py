"""
Experiments for the paper:
    "An Empirical Study of Double Descent in Polynomial Regression"

Generates four figures into ./figures/:
    1. fig_classical_ushape.pdf       -- the classical bias-variance U-curve
    2. fig_double_descent.pdf         -- double descent as p crosses n
    3. fig_bias_variance_decomp.pdf   -- Monte Carlo bias^2 / variance / total
    4. fig_ridge_regularization.pdf   -- ridge regularisation removes the spike

Run with:
    python3 experiments.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "lines.linewidth": 1.2,
})

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# Data and model
#
# Truth:  y = sin(2*pi*x) + 0.3*x + epsilon,    epsilon ~ N(0, sigma^2)
# Inputs: x ~ Uniform(0, 1)
# Basis:  Legendre polynomials on [-1, 1] (orthonormal version)
# ---------------------------------------------------------------------------
def f_star(x):
    return np.sin(2 * np.pi * x) + 0.3 * x


def sample_data(n, sigma, rng):
    x = rng.uniform(0.0, 1.0, size=n)
    y = f_star(x) + sigma * rng.standard_normal(n)
    return x, y


def design_matrix(x, p):
    """Orthonormal Legendre design matrix with p columns (degrees 0..p-1).

    We rescale x from [0, 1] to [-1, 1] and use NumPy's built-in
    legvander to get the columns P_0(z), P_1(z), ..., P_{p-1}(z).
    Multiplying by sqrt(2k+1) gives the L2-orthonormal Legendre basis.
    A well-conditioned design matrix is essential here -- raw monomials
    would blow up numerically long before p reaches n.
    """
    z = 2.0 * x - 1.0
    X = np.polynomial.legendre.legvander(z, p - 1)
    X = X * np.sqrt(2.0 * np.arange(p) + 1.0)
    return X


def fit_predict(Xtr, ytr, Xte, ridge=0.0):
    """Least-squares fit on (Xtr, ytr), then predict on Xte.

    For ridge == 0 we use the Moore-Penrose pseudo-inverse via
    numpy.linalg.lstsq, which is the minimum-norm solution when p > n
    -- the canonical estimator for double descent.
    """
    n, p = Xtr.shape
    if ridge > 0.0:
        A = Xtr.T @ Xtr + ridge * np.eye(p)
        beta = np.linalg.solve(A, Xtr.T @ ytr)
    else:
        beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return Xte @ beta, beta


# ---------------------------------------------------------------------------
# Experiment 1: classical U-curve (n >> p)
# ---------------------------------------------------------------------------
def experiment_classical_ushape():
    n_train, n_test, sigma = 200, 2000, 0.3
    degrees = np.arange(1, 25)
    n_repeats = 30

    train_err = np.zeros((n_repeats, len(degrees)))
    test_err = np.zeros((n_repeats, len(degrees)))

    rng = np.random.default_rng(0)
    for r in range(n_repeats):
        xtr, ytr = sample_data(n_train, sigma, rng)
        xte, yte = sample_data(n_test, sigma, rng)
        for j, d in enumerate(degrees):
            p = d + 1
            Xtr = design_matrix(xtr, p)
            Xte = design_matrix(xte, p)
            yhat_tr, _ = fit_predict(Xtr, ytr, Xtr)
            yhat_te, _ = fit_predict(Xtr, ytr, Xte)
            train_err[r, j] = np.mean((yhat_tr - ytr) ** 2)
            test_err[r, j] = np.mean((yhat_te - yte) ** 2)

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(degrees, train_err.mean(0), "o-", label="Train MSE",
            color="#1f77b4", markersize=3)
    ax.plot(degrees, test_err.mean(0), "s-", label="Test MSE",
            color="#d62728", markersize=3)
    ax.fill_between(degrees,
                    test_err.mean(0) - test_err.std(0),
                    test_err.mean(0) + test_err.std(0),
                    color="#d62728", alpha=0.15)
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Mean squared error")
    ax.set_yscale("log")
    ax.legend(frameon=False, loc="best")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.set_title("Classical regime ($n=200$, $p \\ll n$)")
    fig.savefig(os.path.join(OUT, "fig_classical_ushape.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 2: double descent as p crosses n
# ---------------------------------------------------------------------------
def experiment_double_descent():
    n_train, n_test, sigma = 30, 2000, 0.3
    p_values = np.unique(np.concatenate([
        np.arange(1, n_train),               # 1..29
        np.arange(n_train, n_train + 6),     # 30..35  (around the spike)
        np.arange(n_train + 6, 200, 4),      # sparser further out
    ]))
    n_repeats = 80

    train_err = np.zeros((n_repeats, len(p_values)))
    test_err = np.zeros((n_repeats, len(p_values)))

    rng = np.random.default_rng(1)
    for r in range(n_repeats):
        xtr, ytr = sample_data(n_train, sigma, rng)
        xte, yte = sample_data(n_test, sigma, rng)
        for j, p in enumerate(p_values):
            Xtr = design_matrix(xtr, int(p))
            Xte = design_matrix(xte, int(p))
            yhat_tr, _ = fit_predict(Xtr, ytr, Xtr)
            yhat_te, _ = fit_predict(Xtr, ytr, Xte)
            train_err[r, j] = np.mean((yhat_tr - ytr) ** 2)
            test_err[r, j] = np.mean((yhat_te - yte) ** 2)

    test_med = np.median(test_err, axis=0)
    test_q25 = np.percentile(test_err, 25, axis=0)
    test_q75 = np.percentile(test_err, 75, axis=0)
    train_mean = np.maximum(train_err.mean(0), 1e-12)  # clip for log plot

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(p_values, train_mean, "o-", label="Train MSE",
            color="#1f77b4", markersize=2.5)
    ax.plot(p_values, test_med, "s-", label="Test MSE (median)",
            color="#d62728", markersize=2.5)
    ax.fill_between(p_values, test_q25, test_q75, color="#d62728", alpha=0.15)
    ax.axvline(n_train, color="black", linestyle="--", linewidth=0.8,
               label=f"$p = n = {n_train}$")
    ax.set_xlabel("Number of features $p$")
    ax.set_ylabel("Mean squared error")
    ax.set_yscale("log")
    ax.set_ylim(1e-12, 1e10)
    ax.legend(frameon=False, loc="best", fontsize=7)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.set_title("Double descent ($n=30$, varying $p$)")
    fig.savefig(os.path.join(OUT, "fig_double_descent.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 3: Monte Carlo bias-variance decomposition
# ---------------------------------------------------------------------------
def experiment_bias_variance():
    n_train, sigma = 30, 0.3
    p_values = np.arange(2, 120, 2)
    n_repeats = 400

    x_grid = np.linspace(0.02, 0.98, 200)
    f_grid = f_star(x_grid)
    preds = np.zeros((n_repeats, len(p_values), len(x_grid)))

    rng = np.random.default_rng(2)
    for r in range(n_repeats):
        xtr, ytr = sample_data(n_train, sigma, rng)
        for j, p in enumerate(p_values):
            Xtr = design_matrix(xtr, int(p))
            Xgr = design_matrix(x_grid, int(p))
            yhat, _ = fit_predict(Xtr, ytr, Xgr)
            preds[r, j, :] = yhat

    mean_pred = preds.mean(axis=0)
    bias_sq = np.mean((mean_pred - f_grid) ** 2, axis=1)
    variance = np.mean(preds.var(axis=0), axis=1)
    total = bias_sq + variance + sigma ** 2

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(p_values, np.maximum(bias_sq, 1e-4), "-", label="Bias$^2$",
            color="#2ca02c")
    ax.plot(p_values, np.maximum(variance, 1e-4), "--", label="Variance",
            color="#ff7f0e")
    ax.plot(p_values, np.maximum(total, 1e-4), "-",
            label="Total + $\\sigma^2$", color="#d62728", linewidth=1.6)
    ax.axvline(n_train, color="black", linestyle=":", linewidth=0.8,
               label=f"$p = n = {n_train}$")
    ax.set_xlabel("Number of features $p$")
    ax.set_ylabel("Error")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1e10)
    ax.legend(frameon=False, loc="best", fontsize=7, ncol=2)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.set_title("Bias-variance decomposition")
    fig.savefig(os.path.join(OUT, "fig_bias_variance_decomp.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 4: ridge regularisation removes the spike
# ---------------------------------------------------------------------------
def experiment_ridge():
    n_train, n_test, sigma = 30, 1000, 0.3
    p_values = np.unique(np.concatenate([
        np.arange(1, n_train, 2),
        np.arange(n_train, n_train + 6),
        np.arange(n_train + 6, 130, 6),
    ]))
    n_repeats = 30
    ridges = [0.0, 1e-4, 1e-2, 1.0]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

    rng = np.random.default_rng(3)
    data = [
        (*sample_data(n_train, sigma, rng), *sample_data(n_test, sigma, rng))
        for _ in range(n_repeats)
    ]

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for c, lam in zip(colors, ridges):
        test_err = np.zeros((n_repeats, len(p_values)))
        for r, (xtr, ytr, xte, yte) in enumerate(data):
            for j, p in enumerate(p_values):
                Xtr = design_matrix(xtr, int(p))
                Xte = design_matrix(xte, int(p))
                yhat_te, _ = fit_predict(Xtr, ytr, Xte, ridge=lam)
                test_err[r, j] = np.mean((yhat_te - yte) ** 2)
        med = np.median(test_err, axis=0)
        label = "$\\lambda=0$ (min-norm)" if lam == 0 else f"$\\lambda={lam:g}$"
        ax.plot(p_values, med, "-", label=label, color=c)
    ax.axvline(n_train, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Number of features $p$")
    ax.set_ylabel("Test MSE (median)")
    ax.set_yscale("log")
    ax.legend(frameon=False, loc="best", ncol=2)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.set_title("Effect of ridge regularization")
    fig.savefig(os.path.join(OUT, "fig_ridge_regularization.pdf"))
    plt.close(fig)


def main():
    print("Running experiment 1 (classical U-shape)...")
    experiment_classical_ushape()
    print("Running experiment 2 (double descent)...")
    experiment_double_descent()
    print("Running experiment 3 (bias-variance decomposition)...")
    experiment_bias_variance()
    print("Running experiment 4 (ridge regularization)...")
    experiment_ridge()
    print("All figures saved to", OUT)


if __name__ == "__main__":
    main()
