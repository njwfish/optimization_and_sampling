from __future__ import annotations

from pathlib import Path
import os
import sys

import numpy as np


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    """Vectorized approximation to the standard normal CDF Φ(z).

    Uses a classic rational approximation (Abramowitz--Stegun style) that is
    accurate enough for visualization and avoids a SciPy dependency.
    """

    z = np.asarray(z, dtype=float)
    abs_z = np.abs(z)

    # Abramowitz & Stegun 7.1.26-inspired polynomial/rational approximation.
    p = 0.2316419
    b1, b2, b3, b4, b5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    t = 1.0 / (1.0 + p * abs_z)
    poly = t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
    approx = 1.0 - _norm_pdf(abs_z) * poly
    return np.where(z >= 0.0, approx, 1.0 - approx)


def _norm_logcdf(z: np.ndarray) -> np.ndarray:
    cdf = _norm_cdf(z)
    # Clip away from 0 to avoid -inf in plots; this is only for visualization.
    return np.log(np.clip(cdf, 1e-300, 1.0))


# --- Logistic sigmoid helpers -------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""

    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def _log_sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute log(sigmoid(z)) stably."""

    z = np.asarray(z, dtype=float)
    return -np.logaddexp(0.0, -z)


def _repo_root() -> Path:
    # This file lives at: Lecture Tex/figures/handouts/convexity_optimization/<this_file>.py
    return Path(__file__).resolve().parents[4]


def _import_stat221_utils() -> None:
    repo = _repo_root()
    util_dir = repo / 'Corrected and Solved with util functions'
    sys.path.insert(0, str(util_dir))


def _configure_writable_caches() -> None:
    # Codex runs in a sandbox where the home directory may not be writable.
    # Matplotlib and fontconfig can fail if they cannot write caches.
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/stat221_mplconfig')
    os.environ.setdefault('XDG_CACHE_HOME', '/tmp/stat221_cache')

def _local_minima_2d(z: np.ndarray) -> list[tuple[int, int]]:
    """Return indices (iy, ix) of strict local minima in a 2D array."""

    mins: list[tuple[int, int]] = []
    for iy in range(1, z.shape[0] - 1):
        for ix in range(1, z.shape[1] - 1):
            val = float(z[iy, ix])
            nbrs = z[iy - 1 : iy + 2, ix - 1 : ix + 2]
            if val == float(np.min(nbrs)) and np.count_nonzero(nbrs == val) == 1:
                mins.append((iy, ix))
    return mins


def _logistic_risks_on_grid(
    beta0_grid: np.ndarray,
    beta1_grid: np.ndarray,
    X: np.ndarray,
    p_star: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (NLL, MSE) risks on a (beta0, beta1) grid."""

    B0, B1 = np.meshgrid(beta0_grid, beta1_grid)
    eta = X[:, 0, None, None] * B0[None, :, :] + X[:, 1, None, None] * B1[None, :, :]

    nll = -np.mean(
        p_star[:, None, None] * _log_sigmoid(eta)
        + (1.0 - p_star[:, None, None]) * _log_sigmoid(-eta),
        axis=0,
    )
    mse = np.mean((_sigmoid(eta) - p_star[:, None, None]) ** 2, axis=0)
    return nll, mse


def _logistic_mse_grad(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta = np.asarray(beta, dtype=float).reshape(-1)
    z = X @ beta
    p = _sigmoid(z)
    dp = p * (1.0 - p)
    w = (p - y) * dp
    return (2.0 / float(len(y))) * (X.T @ w)


def _logistic_mse_hessian(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta = np.asarray(beta, dtype=float).reshape(-1)
    z = X @ beta
    p = _sigmoid(z)
    dp = p * (1.0 - p)
    ddp = dp * (1.0 - 2.0 * p)
    coeff = dp**2 + (p - y) * ddp
    return (2.0 / float(len(y))) * ((X.T * coeff) @ X)


def make_logistic_landscape_contourf_figure(out_path: Path) -> None:
    _import_stat221_utils()
    _configure_writable_caches()
    import matplotlib.pyplot as plt  # noqa: E402

    from stat221_viz import STAT221_COLORS, loss_landscape_contourf, savefig, set_stat221_style  # noqa: E402

    set_stat221_style()

    # A small discrete "population" over feature vectors in R^2.
    # Interpret beta = (beta0, beta1) as an intercept + slope in a 1D logistic model,
    # so eta = beta0 + beta1 x.
    x = np.array([-2.0, -1.0, 1.0, 2.0], dtype=float)
    X = np.column_stack([np.ones_like(x), x])

    beta0_grid = np.linspace(-6.0, 6.0, 241)
    beta1_grid = np.linspace(-6.0, 6.0, 241)

    beta_star = np.array([0.0, 1.0])
    p_star_real = _sigmoid(X @ beta_star)

    # Misspecified p*: chosen to be non-monotone in x, hence not representable as sigmoid(beta0 + beta1 x).
    # (E.g., this could arise after marginalizing over a latent class / omitted variable.)
    # This choice yields a visually clear second basin for the population MSE risk.
    p_star_miss = np.array([0.49, 0.87, 0.92, 0.36], dtype=float)

    nll_real, mse_real = _logistic_risks_on_grid(beta0_grid, beta1_grid, X, p_star_real)
    nll_miss, mse_miss = _logistic_risks_on_grid(beta0_grid, beta1_grid, X, p_star_miss)

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.0), sharex=True, sharey=True)

    panels = [
        (axes[0, 0], nll_real, 'Realizable: NLL (convex)', False),
        (axes[0, 1], nll_miss, 'Misspecified: NLL (still convex)', True),
        (axes[1, 0], mse_real, 'Realizable: MSE logistic', False),
        (axes[1, 1], mse_miss, 'Misspecified: MSE logistic', True),
    ]

    for ax, z, title, show_cbar in panels:
        loss_landscape_contourf(
            ax,
            beta0_grid,
            beta1_grid,
            z,
            levels=44,
            log10_excess=True,
            eps=1e-6,
            add_colorbar=show_cbar,
        )

        j = int(np.argmin(np.min(z, axis=0)))
        i = int(np.argmin(z[:, j]))
        b0_hat = float(beta0_grid[j])
        b1_hat = float(beta1_grid[i])
        ax.scatter(
            [b0_hat],
            [b1_hat],
            s=75,
            marker='X',
            color=STAT221_COLORS['red'],
            edgecolor='black',
            linewidth=0.7,
            zorder=6,
        )

        ax.set_title(title)

    # Annotate a spurious local minimum on the misspecified MSE landscape (if present).
    spurious_idx = None
    mins = _local_minima_2d(mse_miss)
    if len(mins) >= 2:
        mins_sorted = sorted(mins, key=lambda ij: float(mse_miss[ij[0], ij[1]]))
        global_ij = mins_sorted[0]
        for ij in mins_sorted[1:]:
            # Ignore near-duplicate grid minima in the same basin; we want a visibly
            # distinct spurious basin.
            if float(mse_miss[ij[0], ij[1]] - mse_miss[global_ij[0], global_ij[1]]) > 1e-2:
                spurious_idx = ij
                break
    if spurious_idx is not None:
        iy, ix = spurious_idx
        x_spur = float(beta0_grid[ix])
        y_spur = float(beta1_grid[iy])
        x_global = float(beta0_grid[global_ij[1]])
        y_global = float(beta1_grid[global_ij[0]])
        dx = x_spur - x_global
        dy = y_spur - y_global
        norm_xy = float(np.hypot(dx, dy))
        if norm_xy < 1e-12:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = dx / norm_xy, dy / norm_xy

        # Place label away from the global basin (to avoid clutter) while keeping it in view.
        offset = 1.8
        x_text = x_spur + offset * float(ux)
        y_text = y_spur + offset * float(uy)
        x_min = float(beta0_grid[0])
        x_max = float(beta0_grid[-1])
        y_min = float(beta1_grid[0])
        y_max = float(beta1_grid[-1])
        pad_x = 1.05  # leave room for the label box so it doesn't get clipped
        pad_y = 0.9
        x_text = float(np.clip(x_text, x_min + pad_x, x_max - pad_x))
        y_text = float(np.clip(y_text, y_min + pad_y, y_max - pad_y))

        # Choose label alignment so the text box tends to extend toward the plot interior.
        ha = 'right' if float(ux) > 0.0 else 'left'
        va = 'top' if float(uy) > 0.0 else 'bottom'
        if x_text >= x_max - pad_x - 1e-12:
            ha = 'right'
        if x_text <= x_min + pad_x + 1e-12:
            ha = 'left'
        if y_text >= y_max - pad_y - 1e-12:
            va = 'top'
        if y_text <= y_min + pad_y + 1e-12:
            va = 'bottom'
        # Two-layer ring for contrast (white underlay + purple outline).
        axes[1, 1].scatter(
            [x_spur],
            [y_spur],
            s=180,
            facecolor='none',
            edgecolor='white',
            linewidth=4.2,
            marker='o',
            zorder=7,
        )
        axes[1, 1].scatter(
            [x_spur],
            [y_spur],
            s=180,
            facecolor='none',
            edgecolor=STAT221_COLORS['purple'],
            linewidth=2.4,
            marker='o',
            zorder=8,
        )
        axes[1, 1].annotate(
            'spurious local min',
            xy=(x_spur, y_spur),
            xytext=(x_text, y_text),
            arrowprops={
                'arrowstyle': '->',
                'color': STAT221_COLORS['gray'],
                'alpha': 0.85,
                'lw': 1.4,
            },
            fontsize=11,
            color='black',
            ha=ha,
            va=va,
            bbox={
                'boxstyle': 'round,pad=0.25',
                'facecolor': 'white',
                'edgecolor': 'none',
                'alpha': 0.88,
            },
        )

    for ax in axes[1, :]:
        ax.set_xlabel(r'$\beta_0$')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\beta_1$')

    savefig(fig, out_path)


def make_logistic_empirical_spurious_figure(out_path: Path) -> None:
    _import_stat221_utils()
    _configure_writable_caches()
    import matplotlib.pyplot as plt  # noqa: E402

    from stat221_viz import STAT221_COLORS, loss_landscape_contourf, savefig, set_stat221_style  # noqa: E402

    set_stat221_style()

    # A small synthetic dataset (x_i, y_i) in R^2 where the empirical squared-loss logistic
    # objective exhibits a spurious local minimum.
    X = np.array(
        [
            [-0.54, 1.57],
            [0.15, 2.62],
            [-0.89, 0.66],
            [-0.22, 0.64],
            [-0.01, -0.64],
            [-0.99, 3.48],
            [-0.09, -2.29],
            [-0.74, 0.77],
        ],
        dtype=float,
    )
    y = np.array([0, 1, 0, 0, 1, 1, 1, 1], dtype=float)

    beta1_grid = np.linspace(-6.0, 6.0, 241)
    beta2_grid = np.linspace(-6.0, 6.0, 241)

    nll, mse = _logistic_risks_on_grid(beta1_grid, beta2_grid, X, y)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), sharex=True, sharey=True)
    panels = [
        (axes[0], nll, 'Empirical logistic NLL (convex)', False),
        (axes[1], mse, 'Empirical MSE logistic (can have spurious minima)', True),
    ]

    for ax, z, title, show_cbar in panels:
        loss_landscape_contourf(
            ax,
            beta1_grid,
            beta2_grid,
            z,
            levels=44,
            log10_excess=True,
            eps=1e-6,
            add_colorbar=show_cbar,
        )

        j = int(np.argmin(np.min(z, axis=0)))
        i = int(np.argmin(z[:, j]))
        b1_hat = float(beta1_grid[j])
        b2_hat = float(beta2_grid[i])
        ax.scatter(
            [b1_hat],
            [b2_hat],
            s=85,
            marker='X',
            color=STAT221_COLORS['red'],
            edgecolor='black',
            linewidth=0.7,
            zorder=6,
        )
        ax.set_title(title)

    # Highlight a spurious local minimum on the empirical MSE landscape (if present).
    spurious_idx = None
    mins = _local_minima_2d(mse)
    if len(mins) >= 2:
        mins_sorted = sorted(mins, key=lambda ij: float(mse[ij]))
        global_min = mins_sorted[0]
        for iy, ix in mins_sorted[1:]:
            if float(mse[iy, ix]) <= float(mse[global_min]) + 5e-3:
                continue
            beta_cand = np.array([float(beta1_grid[ix]), float(beta2_grid[iy])])
            eig = np.linalg.eigvalsh(_logistic_mse_hessian(beta_cand, X, y))
            if float(np.min(eig)) <= 1e-4:
                continue
            if float(mse[iy, ix]) > float(mse[global_min]) + 1e-4:
                spurious_idx = (iy, ix)
                break
    if spurious_idx is not None:
        iy, ix = spurious_idx
        axes[1].scatter(
            [float(beta1_grid[ix])],
            [float(beta2_grid[iy])],
            s=130,
            facecolor='none',
            edgecolor=STAT221_COLORS['purple'],
            linewidth=2.2,
            marker='o',
            zorder=7,
        )
        axes[1].annotate(
            'spurious local min',
            xy=(float(beta1_grid[ix]), float(beta2_grid[iy])),
            xytext=(float(beta1_grid[ix]) - 2.3, float(beta2_grid[iy]) - 1.6),
            arrowprops={
                'arrowstyle': '->',
                'color': STAT221_COLORS['gray'],
                'alpha': 0.75,
                'lw': 1.2,
            },
            fontsize=10,
            color=STAT221_COLORS['gray'],
        )

    axes[0].set_xlabel(r'$\beta_1$')
    axes[1].set_xlabel(r'$\beta_1$')
    axes[0].set_ylabel(r'$\beta_2$')

    savefig(fig, out_path)


def make_logistic_empirical_basins_figure(out_path: Path) -> None:
    _import_stat221_utils()
    _configure_writable_caches()
    import matplotlib.pyplot as plt  # noqa: E402
    from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
    from matplotlib.patches import Patch  # noqa: E402

    from stat221_viz import STAT221_COLORS, loss_landscape_contourf, savefig, set_stat221_style  # noqa: E402

    set_stat221_style()

    X = np.array(
        [
            [-0.54, 1.57],
            [0.15, 2.62],
            [-0.89, 0.66],
            [-0.22, 0.64],
            [-0.01, -0.64],
            [-0.99, 3.48],
            [-0.09, -2.29],
            [-0.74, 0.77],
        ],
        dtype=float,
    )
    y = np.array([0, 1, 0, 0, 1, 1, 1, 1], dtype=float)

    # Find a global and a clearly suboptimal local minimum on a fine grid, then refine by GD.
    beta1_grid = np.linspace(-6.0, 6.0, 241)
    beta2_grid = np.linspace(-6.0, 6.0, 241)
    _, mse = _logistic_risks_on_grid(beta1_grid, beta2_grid, X, y)
    mins = _local_minima_2d(mse)
    if len(mins) < 2:
        raise RuntimeError('Expected at least two local minima on the grid.')

    mins_sorted = sorted(mins, key=lambda ij: float(mse[ij]))
    global_iy, global_ix = mins_sorted[0]
    beta_global0 = np.array([float(beta1_grid[global_ix]), float(beta2_grid[global_iy])])
    beta_spur0 = None
    for iy, ix in mins_sorted[1:]:
        if float(mse[iy, ix]) <= float(mse[global_iy, global_ix]) + 5e-3:
            continue
        beta_cand = np.array([float(beta1_grid[ix]), float(beta2_grid[iy])])
        eig = np.linalg.eigvalsh(_logistic_mse_hessian(beta_cand, X, y))
        if float(np.min(eig)) <= 1e-4:
            continue
        beta_spur0 = beta_cand
        break
    if beta_spur0 is None:
        raise RuntimeError('Failed to find a PD spurious local minimum on the grid.')

    step = 0.6
    refine_steps = 4000
    beta_global = beta_global0.copy()
    beta_spur = beta_spur0.copy()
    for _ in range(refine_steps):
        beta_global = beta_global - step * _logistic_mse_grad(beta_global, X, y)
        beta_spur = beta_spur - step * _logistic_mse_grad(beta_spur, X, y)

    # Basin-of-attraction map for fixed-step gradient descent on a coarse grid of initializations.
    b1 = np.linspace(-6.0, 6.0, 171)
    b2 = np.linspace(-6.0, 6.0, 171)
    B1, B2 = np.meshgrid(b1, b2)
    beta = np.column_stack([B1.ravel(), B2.ravel()])

    iters = 1400
    for _ in range(iters):
        z = beta @ X.T  # (N, n)
        p = _sigmoid(z)
        dp = p * (1.0 - p)
        w = (p - y[None, :]) * dp
        g = (2.0 / float(len(y))) * (w @ X)  # (N, 2)
        beta = beta - step * g

    d_global = np.sum((beta - beta_global[None, :]) ** 2, axis=1)
    d_spur = np.sum((beta - beta_spur[None, :]) ** 2, axis=1)
    labels = (d_spur < d_global).astype(int)
    # Mark points that did not land near either attractor as "other".
    tol2 = 0.6**2
    labels = np.where(np.minimum(d_global, d_spur) <= tol2, labels, 2)
    labels_grid = labels.reshape(B1.shape)

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 6.0))
    cmap = ListedColormap(
        [STAT221_COLORS['purple'], STAT221_COLORS['red'], STAT221_COLORS['gray_light']]
    )
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.imshow(
        labels_grid,
        origin='lower',
        extent=[float(b1[0]), float(b1[-1]), float(b2[0]), float(b2[-1])],
        cmap=cmap,
        norm=norm,
        interpolation='nearest',
        aspect='equal',
        alpha=0.92,
    )

    # Add faint contours of the objective for context.
    z_plot = np.log10(mse - float(np.min(mse)) + 1e-6)
    ax.contour(
        beta1_grid,
        beta2_grid,
        z_plot,
        levels=18,
        colors='k',
        alpha=0.12,
        linewidths=0.6,
    )

    # Mark the two minima.
    ax.scatter(
        [float(beta_global[0])],
        [float(beta_global[1])],
        s=90,
        marker='o',
        color=STAT221_COLORS['purple'],
        edgecolor='white',
        linewidth=0.9,
        zorder=6,
    )
    ax.scatter(
        [float(beta_spur[0])],
        [float(beta_spur[1])],
        s=90,
        marker='X',
        color=STAT221_COLORS['red'],
        edgecolor='black',
        linewidth=0.8,
        zorder=6,
    )

    # A few representative trajectories.
    traj_inits = [
        np.array([0.0, 0.0]),
        np.array([3.0, 3.5]),
        np.array([-3.5, 2.5]),
    ]
    for beta0 in traj_inits:
        path = [beta0.copy()]
        b = beta0.copy()
        for _ in range(260):
            b = b - step * _logistic_mse_grad(b, X, y)
            path.append(b.copy())
        path = np.asarray(path)
        ax.plot(path[:, 0], path[:, 1], color='white', alpha=0.85, lw=1.8, zorder=5)
        ax.scatter([path[0, 0]], [path[0, 1]], s=18, color='white', alpha=0.95, zorder=6)

    ax.set_title('Empirical MSE logistic: basins of attraction for gradient descent')
    ax.set_xlabel(r'$\beta_1$')
    ax.set_ylabel(r'$\beta_2$')

    legend_handles = [
        Patch(facecolor=STAT221_COLORS['purple'], edgecolor='none', label='converges to best basin'),
        Patch(facecolor=STAT221_COLORS['red'], edgecolor='none', label='converges to spurious basin'),
        Patch(facecolor=STAT221_COLORS['gray_light'], edgecolor='none', label='other / slow'),
    ]
    ax.legend(handles=legend_handles, loc='lower left', frameon=True, facecolor='white', framealpha=0.96)

    savefig(fig, out_path)


def make_phase_retrieval_landscape_figure(out_path: Path) -> None:
    _import_stat221_utils()
    _configure_writable_caches()
    import matplotlib.pyplot as plt  # noqa: E402

    from stat221_viz import STAT221_COLORS, loss_landscape_contourf, savefig, set_stat221_style  # noqa: E402

    set_stat221_style()

    x_star = np.array([1.0, 0.0])

    x1 = np.linspace(-1.6, 1.6, 300)
    x2 = np.linspace(-1.6, 1.6, 300)
    X1, X2 = np.meshgrid(x1, x2)

    r2 = X1**2 + X2**2
    inner = X1 * x_star[0] + X2 * x_star[1]
    f = 1.5 * (r2**2) - r2 * float(np.dot(x_star, x_star)) - 2.0 * (inner**2) + 1.5 * float(
        np.dot(x_star, x_star) ** 2
    )

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 6.2))
    loss_landscape_contourf(ax, x1, x2, f, levels=46, log10_excess=True, eps=1e-6)

    # Mark global minimizers, saddles, and the local maximum at 0.
    ax.scatter(
        [1.0, -1.0],
        [0.0, 0.0],
        s=95,
        marker='o',
        color=STAT221_COLORS['purple'],
        edgecolor='white',
        linewidth=0.9,
        zorder=6,
        label=r'global minima $\pm x^\star$',
    )
    saddle = 1.0 / np.sqrt(3.0)
    ax.scatter(
        [0.0, 0.0],
        [saddle, -saddle],
        s=85,
        marker='s',
        color=STAT221_COLORS['red'],
        edgecolor='white',
        linewidth=0.9,
        zorder=6,
        label='saddles',
    )
    ax.scatter(
        [0.0],
        [0.0],
        s=85,
        marker='X',
        color=STAT221_COLORS['gray'],
        edgecolor='white',
        linewidth=0.9,
        zorder=6,
        label='local max',
    )

    ax.set_title('Phase retrieval (population): symmetry, benign landscape')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.92)

    savefig(fig, out_path)


def make_frequency_estimation_landscape_figure(out_path: Path) -> None:
    _import_stat221_utils()
    _configure_writable_caches()
    import matplotlib.pyplot as plt  # noqa: E402

    from stat221_viz import (  # noqa: E402
        STAT221_COLORS,
        loss_landscape_contourf,
        savefig,
        set_stat221_style,
    )

    set_stat221_style()

    rng = np.random.default_rng(2)
    n = 80
    omega_star = 1.0
    c_star = 1.0
    sigma = 0.0

    t = np.arange(n, dtype=float)
    y = c_star * np.cos(omega_star * t) + sigma * rng.standard_normal(n)

    omega = np.linspace(0.0, 2.0 * np.pi, 820)
    c = np.linspace(-1.8, 1.8, 420)

    cos_mat = np.cos(t[:, None] * omega[None, :])
    a = np.sum(cos_mat**2, axis=0)
    b = np.sum(y[:, None] * cos_mat, axis=0)
    c0 = float(np.sum(y**2))

    # Quadratic form: F(omega, c) = 1/2 (||y||^2 - 2c <y,cos> + c^2 ||cos||^2).
    F = 0.5 * (c0 - 2.0 * c[:, None] * b[None, :] + (c[:, None] ** 2) * a[None, :])
    c_hat = b / a

    # Periodogram score I(omega) proportional to the reduction in the profiled objective.
    I = (b**2) / a
    I = I / float(np.max(I))
    I_plot = np.maximum(I, 1e-4)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    loss_landscape_contourf(
        axes[0],
        omega,
        c,
        F,
        levels=46,
        log10_excess=True,
        eps=1e-6,
        add_contours=False,
        add_colorbar=True,
        cbar_label=r'$\log_{10}(F-\min F+\varepsilon)$',
    )

    mask = (c_hat >= float(c[0])) & (c_hat <= float(c[-1]))
    axes[0].plot(omega[mask], c_hat[mask], color='white', alpha=0.86, lw=2.1, label=r'$\hat c(\omega)$')

    # Two equivalent global basins in omega: omega* and 2pi-omega*.
    omega_alt = float(2.0 * np.pi - omega_star)
    axes[0].scatter(
        [omega_star],
        [c_star],
        s=130,
        marker='X',
        color=STAT221_COLORS['red'],
        edgecolor='black',
        linewidth=0.8,
        zorder=8,
        label=r'$(\omega^\star,c^\star)$',
    )
    axes[0].scatter(
        [omega_alt],
        [c_star],
        s=130,
        marker='X',
        color=STAT221_COLORS['red'],
        edgecolor='black',
        linewidth=0.8,
        zorder=8,
        label='_nolegend_',
    )

    axes[0].set_title(r'Frequency estimation: many basins in $\omega$')
    axes[0].set_xlabel(r'$\omega$')
    axes[0].set_ylabel(r'$c$')
    axes[0].set_xlim(float(omega[0]), float(omega[-1]))
    axes[0].set_ylim(float(c[0]), float(c[-1]))
    axes[0].legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.92)

    axes[1].plot(omega, I_plot, color=STAT221_COLORS['blue'])
    axes[1].scatter(
        [omega_star, omega_alt],
        [float(I_plot[np.argmin(np.abs(omega - omega_star))]), float(I_plot[np.argmin(np.abs(omega - omega_alt))])],
        s=90,
        marker='o',
        color=STAT221_COLORS['purple'],
        edgecolor='white',
        linewidth=0.9,
        zorder=7,
    )
    axes[1].set_yscale('log')
    axes[1].set_title(r'Periodogram $I(\omega)$')
    axes[1].set_xlabel(r'$\omega$')
    axes[1].set_ylabel(r'$I(\omega)/\max I$ (log scale)')
    axes[1].set_xlim(float(omega[0]), float(omega[-1]))
    axes[1].set_ylim(1e-4, 1.6)

    savefig(fig, out_path)


def make_matrix_completion_rank1_balancing_figure(out_path: Path) -> None:
    _import_stat221_utils()
    _configure_writable_caches()
    import matplotlib.pyplot as plt  # noqa: E402

    from stat221_viz import (  # noqa: E402
        STAT221_COLORS,
        loss_landscape_contourf,
        savefig,
        set_stat221_style,
    )

    set_stat221_style()

    a = 1.0
    u = np.linspace(-2.6, 2.6, 320)
    v = np.linspace(-2.6, 2.6, 320)
    U, V = np.meshgrid(u, v)

    f = (U * V - a) ** 2
    f_bal = f + 0.125 * (U**2 - V**2) ** 2

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharex=True, sharey=True)

    loss_landscape_contourf(axes[0], u, v, f, levels=46, log10_excess=True, eps=1e-6, add_colorbar=False)
    loss_landscape_contourf(
        axes[1],
        u,
        v,
        f_bal,
        levels=46,
        log10_excess=True,
        eps=1e-6,
        add_colorbar=True,
        cbar_label=r'$\log_{10}(f-\min f+\varepsilon)$',
    )

    # Overlay global minimizer geometry and representative points.
    u_left = np.linspace(-2.6, -0.25, 500)
    u_right = np.linspace(0.25, 2.6, 500)
    v_left = a / u_left
    v_right = a / u_right

    axes[0].plot(u_left, v_left, color='white', alpha=0.72, lw=1.4, label=r'global minima $uv=a$')
    axes[0].plot(u_right, v_right, color='white', alpha=0.72, lw=1.4)

    u_marks = np.array([-2.0, -1.0, 1.0, 2.0])
    axes[0].scatter(
        u_marks,
        a / u_marks,
        s=80,
        marker='o',
        color=STAT221_COLORS['purple'],
        edgecolor='white',
        linewidth=0.9,
        zorder=7,
        label='global minima',
    )

    r = float(np.sqrt(a))
    axes[1].scatter(
        [r, -r],
        [r, -r],
        s=90,
        marker='o',
        color=STAT221_COLORS['purple'],
        edgecolor='white',
        linewidth=0.9,
        zorder=7,
        label='global minima',
    )

    for ax in axes:
        ax.scatter(
            [0.0],
            [0.0],
            s=110,
            marker='X',
            color=STAT221_COLORS['red'],
            edgecolor='black',
            linewidth=0.8,
            zorder=8,
            label='strict saddle',
        )

    axes[0].set_title(r'Unregularized: $f(u,v)=(uv-a)^2$')
    axes[1].set_title(r'Balanced: $f_{\mathrm{bal}}(u,v)$')
    for ax in axes:
        ax.set_xlabel(r'$u$')
        ax.set_aspect('equal', adjustable='box')
    axes[0].set_ylabel(r'$v$')
    axes[0].set_xlim(u[0], u[-1])
    axes[0].set_ylim(v[0], v[-1])

    axes[0].legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.92)
    axes[1].legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.92)

    savefig(fig, out_path)


def make_matrix_completion_observation_flatness_figure(out_path: Path) -> None:
    _import_stat221_utils()
    _configure_writable_caches()
    import matplotlib.pyplot as plt  # noqa: E402

    from stat221_viz import (  # noqa: E402
        STAT221_COLORS,
        loss_landscape_contourf,
        savefig,
        set_stat221_style,
    )

    set_stat221_style()

    x_star = np.array([1.0, 0.7], dtype=float)
    M_star = np.outer(x_star, x_star)

    x1 = np.linspace(-2.2, 2.2, 320)
    x2 = np.linspace(-2.2, 2.2, 320)
    X1, X2 = np.meshgrid(x1, x2)

    # Full observation: f(x) = 1/2 ||xx^T - M_star||_F^2.
    d11 = X1**2 - float(M_star[0, 0])
    d12 = X1 * X2 - float(M_star[0, 1])
    d22 = X2**2 - float(M_star[1, 1])
    f_full = 0.5 * (d11**2 + 2.0 * d12**2 + d22**2)

    # Single-entry observation Omega = {(1,1)}: f(x) = 1/2 (x1^2 - M11)^2.
    f_partial = 0.5 * d11**2

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharex=True, sharey=True)

    loss_landscape_contourf(
        axes[0],
        x1,
        x2,
        f_full,
        levels=46,
        log10_excess=True,
        eps=1e-6,
        add_colorbar=False,
    )
    loss_landscape_contourf(
        axes[1],
        x1,
        x2,
        f_partial,
        levels=46,
        log10_excess=True,
        eps=1e-6,
        add_contours=False,
        add_colorbar=True,
        cbar_label=r'$\log_{10}(f-\min f+\varepsilon)$',
    )

    # Mark the two global minimizers for the full-observation objective.
    axes[0].scatter(
        [float(x_star[0]), float(-x_star[0])],
        [float(x_star[1]), float(-x_star[1])],
        s=95,
        marker='o',
        color=STAT221_COLORS['purple'],
        edgecolor='white',
        linewidth=0.9,
        zorder=6,
        label=r'global minima $\pm x^\star$',
    )

    # In the partial-observation case, the set of global minimizers is two vertical lines.
    x1_star = float(abs(x_star[0]))
    for sgn in (-1.0, 1.0):
        axes[1].axvline(
            sgn * x1_star,
            color='white',
            alpha=0.8,
            lw=1.6,
            linestyle='--',
            zorder=6,
        )

    axes[0].set_title('Full observations: isolated minima')
    axes[1].set_title(r'Only $(1,1)$ observed: flat valleys')
    for ax in axes:
        ax.set_xlabel(r'$x_1$')
        ax.set_aspect('equal', adjustable='box')
    axes[0].set_ylabel(r'$x_2$')
    axes[0].set_xlim(x1[0], x1[-1])
    axes[0].set_ylim(x2[0], x2[-1])
    axes[0].legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.92)

    savefig(fig, out_path)


def make_matrix_completion_spurious_multimodal_figure(out_path: Path) -> None:
    _import_stat221_utils()
    _configure_writable_caches()
    import matplotlib.pyplot as plt  # noqa: E402

    from stat221_viz import (  # noqa: E402
        STAT221_COLORS,
        loss_landscape_contourf,
        savefig,
        set_stat221_style,
    )

    set_stat221_style()

    # A rank-one PSD completion toy in n=3 can have a spurious local minimum even when the
    # completion problem is identifiable. We use x* = (1, 1/2, -1) and observe the diagonal
    # plus the (1,2) and (2,3) off-diagonal pairs, but not (1,3).
    #
    # Objective: f(x) = 1/2 ||P_Omega(xx^T - x*x^T)||_F^2.
    x_star = np.array([1.0, 0.5, -1.0], dtype=float)
    M_star = np.outer(x_star, x_star)

    # Visualize a 2D slice by fixing x1 = 1 and varying (x2, x3).
    x1_fixed = 1.0
    x2 = np.linspace(-1.6, 1.6, 360)
    x3 = np.linspace(-1.6, 1.6, 360)
    X2, X3 = np.meshgrid(x2, x3)

    # Observation pattern Omega = diag + {(1,2),(2,1),(2,3),(3,2)}.
    d11 = x1_fixed**2 - float(M_star[0, 0])
    d22 = X2**2 - float(M_star[1, 1])
    d33 = X3**2 - float(M_star[2, 2])
    d12 = x1_fixed * X2 - float(M_star[0, 1])
    d23 = X2 * X3 - float(M_star[1, 2])

    f_slice = 0.5 * (d11**2 + d22**2 + d33**2) + d12**2 + d23**2

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 6.0))
    loss_landscape_contourf(
        ax,
        x2,
        x3,
        f_slice,
        levels=46,
        log10_excess=True,
        eps=1e-6,
        add_colorbar=True,
    )

    # In this slice, the global minimizer is x = (1, 1/2, -1) and a spurious local minimizer is x = (1, 0, 1).
    ax.scatter(
        [0.5],
        [-1.0],
        s=95,
        marker='o',
        color=STAT221_COLORS['purple'],
        edgecolor='white',
        linewidth=0.9,
        zorder=7,
        label=r'global min $(1,\frac{1}{2},-1)$',
    )
    ax.scatter(
        [0.0],
        [1.0],
        s=110,
        marker='X',
        color=STAT221_COLORS['red'],
        edgecolor='black',
        linewidth=0.8,
        zorder=8,
        label=r'spurious local min $(1,0,1)$',
    )
    ax.annotate(
        'spurious local min',
        xy=(0.0, 1.0),
        xytext=(-1.25, 1.2),
        arrowprops={
            'arrowstyle': '->',
            'color': STAT221_COLORS['gray'],
            'alpha': 0.85,
            'lw': 1.4,
        },
        fontsize=11,
        color='black',
        ha='left',
        va='bottom',
        bbox={
            'boxstyle': 'round,pad=0.25',
            'facecolor': 'white',
            'edgecolor': 'none',
            'alpha': 0.88,
        },
    )

    ax.set_title(r'Matrix completion: spurious basin from a missing entry ($x_1=1$ slice)')
    ax.set_xlabel(r'$x_2$')
    ax.set_ylabel(r'$x_3$')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.92)

    savefig(fig, out_path)


def main() -> None:
    repo = _repo_root()
    fig_dir = repo / 'Lecture Tex' / 'figures' / 'handouts' / 'convexity_optimization'

    # Figures used by `Lecture Tex/handouts/section 1/section_convexity_optimization.tex`.
    make_logistic_empirical_spurious_figure(fig_dir / 'fig_logistic_empirical_spurious.pdf')
    make_logistic_empirical_basins_figure(fig_dir / 'fig_logistic_empirical_basins.pdf')
    make_logistic_landscape_contourf_figure(fig_dir / 'fig_logistic_landscape_2d.pdf')
    make_phase_retrieval_landscape_figure(fig_dir / 'fig_phase_retrieval_landscape.pdf')
    make_frequency_estimation_landscape_figure(fig_dir / 'fig_frequency_estimation_landscape.pdf')
    print('Wrote handout figures to:', fig_dir)


if __name__ == '__main__':
    main()
