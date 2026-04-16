from __future__ import annotations

from pathlib import Path
import os

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/stat221_mplconfig")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/stat221_cache")

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.8,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "legend.fontsize": 8.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


GRAY = "#2F2F2F"
GRAY_LIGHT = "#F6F7FB"
LIGHT_BORDER = "#D5DAE5"
BLUE = "#1F4E79"
BLUE_LIGHT = "#EAF3FA"
GREEN = "#2E7D32"
GREEN_LIGHT = "#EDF8EF"
ORANGE = "#B45309"
ORANGE_LIGHT = "#FFF4E8"
RED = "#9F2D20"
RED_LIGHT = "#FDEEEE"
BBOX = {"facecolor": "white", "edgecolor": LIGHT_BORDER, "boxstyle": "round,pad=0.18", "alpha": 0.95}


def _style_axes(ax, *, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_facecolor(GRAY_LIGHT)
    ax.grid(True, color="white", linewidth=1.1)
    ax.tick_params(length=0, colors=GRAY)
    for spine in ax.spines.values():
        spine.set_color(LIGHT_BORDER)
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=GRAY)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=GRAY)


def _normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    std = float(std)
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2.0 * np.pi))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def _logistic_map_and_hessian(*, lam: float = 0.01):
    from scipy.optimize import minimize

    csv_path = _repo_root() / "Corrected and Solved with util functions" / "Problem Set 2" / "pset2.csv"
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    x_raw = data[:, :5]
    y = data[:, 5]
    x = np.column_stack([np.ones(x_raw.shape[0]), x_raw])

    def f(beta: np.ndarray) -> float:
        z = x @ beta
        return float(np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * lam * beta @ beta)

    def g(beta: np.ndarray) -> np.ndarray:
        p = _sigmoid(x @ beta)
        return x.T @ (p - y) + lam * beta

    def h(beta: np.ndarray) -> np.ndarray:
        p = _sigmoid(x @ beta)
        w = p * (1.0 - p)
        return x.T @ (w[:, None] * x) + lam * np.eye(x.shape[1])

    beta0 = np.zeros(x.shape[1], dtype=float)
    res = minimize(f, beta0, jac=g, method="BFGS")
    if not res.success:
        raise RuntimeError("BFGS failed for the Problem-Set-2 logistic posterior example.")
    return x, y, res.x, h(res.x)


def make_jko_tradeoff_figure(out_path: Path) -> None:
    plt = _configure_matplotlib()

    s_grid = np.linspace(0.0, 1.08, 500)
    move_term = 1.1 * s_grid**2
    free_term = 0.18 + 1.55 * (1.0 - s_grid) ** 2
    total_term = move_term + free_term
    s_star = float(s_grid[np.argmin(total_term)])
    total_star = float(np.min(total_term))

    fig, ax = plt.subplots(1, 1, figsize=(7.3, 3.4))
    _style_axes(ax, xlabel="fraction of the full move toward the target", ylabel="objective value")
    ax.plot(s_grid, move_term, color=BLUE, linewidth=2.2, label=r"movement penalty")
    ax.plot(s_grid, free_term, color=ORANGE, linewidth=2.2, label=r"free-energy term")
    ax.plot(s_grid, total_term, color=GREEN, linewidth=2.8, label="total objective")
    ax.axvline(0.0, color=BLUE, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(s_star, color=GREEN, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(1.0, color=GRAY, linestyle="--", linewidth=1.2, alpha=0.75)
    ax.scatter([s_star], [total_star], color=GREEN, s=34, zorder=5)
    ax.set_xlim(0.0, 1.08)
    ax.set_ylim(0.0, float(total_term.max()) * 1.08)
    ax.legend(loc="upper right", frameon=False)
    ax.text(
        s_star + 0.04,
        total_star + 0.07,
        "minimum total",
        ha="left",
        va="bottom",
        color=GREEN,
        bbox=BBOX,
    )
    y_text = 0.05
    ax.text(0.0, y_text, "current", color=BLUE, ha="left", va="bottom")
    ax.text(s_star, y_text, "JKO step", color=GREEN, ha="center", va="bottom")
    ax.text(1.0, y_text, "target", color=GRAY, ha="center", va="bottom")
    ax.text(0.17, 1.57, "moving farther raises the transport cost", color=BLUE, ha="left", va="center", bbox=BBOX)
    ax.text(0.72, 0.32, "moving toward $\\pi$ lowers free energy", color=ORANGE, ha="center", va="center", bbox=BBOX)
    ax.set_title("Why the JKO minimizer stops short", color=GRAY, pad=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_gaussian_jko_figure(out_path: Path) -> None:
    plt = _configure_matplotlib()

    h = 0.55
    lam = 1.0
    m = 3.2
    means = [m]
    for _ in range(5):
        m = m / (1.0 + h * lam)
        means.append(m)

    mode_lams = np.array([1.0, 9.0])
    mode_state = np.array([3.0, 3.0])
    mode_hist = [mode_state.copy()]
    for _ in range(6):
        mode_state = mode_state / (1.0 + h * mode_lams)
        mode_hist.append(mode_state.copy())
    mode_hist = np.asarray(mode_hist)

    colors = ["#173F5F", "#20639B", "#3CAEA3", "#7BC96F", "#B8DE6F", "#F6D55C"]

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.8))
    fig.subplots_adjust(wspace=0.28)

    ax = axes[0]
    _style_axes(ax, xlabel=r"state $x$")
    ax.set_title("Implicit-Euler mean iterates", color=GRAY, pad=10)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(-0.26, 0.48)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.axhline(0.0, color=LIGHT_BORDER, linewidth=1.8, zorder=1)
    ax.axvline(0.0, color=GRAY, linewidth=1.4, linestyle="--", alpha=0.7, zorder=1)
    ax.scatter([0.0], [0.0], s=58, color=GRAY, zorder=4)
    ax.text(0.0, 0.27, r"target mean", color=GRAY, ha="center", va="bottom", bbox=BBOX)
    ax.text(
        2.0,
        0.38,
        r"$m_{k+1}=(1+h\lambda)^{-1}m_k$",
        color=BLUE,
        ha="center",
        va="bottom",
        bbox=BBOX,
    )
    for k, mean in enumerate(means):
        ax.scatter([mean], [0.0], s=62, color=colors[k], zorder=5)
        if k < len(means) - 1:
            ax.annotate(
                "",
                xy=(means[k + 1], 0.0),
                xytext=(mean, 0.0),
                arrowprops={"arrowstyle": "->", "color": colors[k], "linewidth": 1.8, "shrinkA": 8, "shrinkB": 8},
            )
    ax.text(means[0], 0.18, r"$k=0$", color=colors[0], ha="center", va="bottom")
    ax.text(means[-1], 0.18, r"$k=5$", color=colors[-1], ha="center", va="bottom")
    ax.text(np.mean(means[1:5]), -0.16, r"$k=1,\ldots,4$", color=GRAY, ha="center", va="top")

    ax = axes[1]
    _style_axes(ax, xlabel=r"JKO step $k$", ylabel=r"$|m_{k,i}|$")
    ks = np.arange(mode_hist.shape[0], dtype=int)
    ax.plot(ks, np.abs(mode_hist[:, 0]), color=BLUE, linewidth=2.4, marker="o", markersize=4.5, label=r"slow mode $\lambda_1=1$")
    ax.plot(ks, np.abs(mode_hist[:, 1]), color=GREEN, linewidth=2.4, marker="o", markersize=4.5, label=r"stiff mode $\lambda_2=9$")
    ax.set_yscale("log")
    ax.set_xticks(ks)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Mode-by-mode contraction", color=GRAY, pad=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_gaussian_decay_figure(out_path: Path) -> None:
    plt = _configure_matplotlib()

    t = np.linspace(0.0, 3.0, 500)
    lam1 = 0.4
    lam2 = 10.0
    m0 = np.array([2.0 / np.sqrt(lam1), 2.0 / np.sqrt(lam2)], dtype=float)

    mean_iso = np.column_stack([m0[0] * np.exp(-lam1 * t), m0[1] * np.exp(-lam2 * t)])
    mean_white = np.column_stack([m0[0] * np.exp(-t), m0[1] * np.exp(-t)])

    def kl(mean: np.ndarray) -> np.ndarray:
        return 0.5 * (lam1 * mean[:, 0] ** 2 + lam2 * mean[:, 1] ** 2)

    def fisher(mean: np.ndarray) -> np.ndarray:
        return lam1**2 * mean[:, 0] ** 2 + lam2**2 * mean[:, 1] ** 2

    def w2_sq(mean: np.ndarray) -> np.ndarray:
        return mean[:, 0] ** 2 + mean[:, 1] ** 2

    curves = [
        ("KL", kl(mean_iso), kl(mean_white)),
        ("Fisher", fisher(mean_iso), fisher(mean_white)),
        (r"$W_2^2$", w2_sq(mean_iso), w2_sq(mean_white)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.45))
    fig.subplots_adjust(wspace=0.28, top=0.72)
    fig.suptitle(r"Exact decay for $H=\mathrm{diag}(0.4,10)$", color=GRAY, y=0.98)
    handles = None
    for ax, (title, iso, white) in zip(axes, curves):
        _style_axes(ax, xlabel=r"time $t$")
        line_iso = ax.plot(t, iso, color=BLUE, linewidth=2.4, label="isotropic")[0]
        line_white = ax.plot(t, white, color=GREEN, linewidth=2.4, linestyle="--", label="whitened")[0]
        handles = [line_iso, line_white]
        ax.set_yscale("log")
        ax.set_title(title, color=GRAY, pad=10)
        if ax is axes[0]:
            ax.set_ylabel("exact value")
    fig.legend(handles, ["isotropic", "whitened"], loc="upper center", bbox_to_anchor=(0.5, 0.88), ncol=2, frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_logistic_spectrum_figure(out_path: Path) -> None:
    plt = _configure_matplotlib()

    _, _, _, h_map = _logistic_map_and_hessian(lam=0.01)
    evals = np.linalg.eigvalsh(h_map)
    evals_desc = evals[::-1]
    cond = float(evals_desc[0] / evals_desc[-1])
    xs = np.arange(1, evals_desc.size + 1, dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(7.9, 3.35))
    fig.subplots_adjust(wspace=0.26)

    ax = axes[0]
    _style_axes(ax, xlabel="eigenvalue index", ylabel="eigenvalue")
    ax.plot(xs, evals_desc, color=BLUE, linewidth=2.4, marker="o", markersize=3.6)
    ax.set_xlim(0.7, xs[-1] + 0.3)
    ax.set_ylim(0.0, 1.16 * evals_desc[0])
    ax.set_title("MAP spectrum", color=GRAY, pad=10)
    ax.text(
        3.3,
        0.84 * evals_desc[0],
        rf"$\kappa\approx {cond:.1f}$",
        color=BLUE,
        bbox=BBOX,
    )

    ax = axes[1]
    _style_axes(ax, xlabel="eigenvalue index", ylabel="eigenvalue")
    ax.axhline(1.0, color=GREEN, linewidth=1.8, alpha=0.75)
    ax.plot(xs, np.ones_like(xs), color=GREEN, linewidth=2.0, marker="o", markersize=3.6)
    ax.set_xlim(0.7, xs[-1] + 0.3)
    ax.set_ylim(0.96, 1.04)
    ax.set_yticks([0.97, 1.0, 1.03])
    ax.set_title("Whitened spectrum", color=GRAY, pad=10)
    ax.text(2.9, 1.026, "all eigenvalues = 1", color=GREEN, bbox=BBOX)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_logistic_laplace_trajectory_figure(out_path: Path) -> None:
    plt = _configure_matplotlib()

    _, _, _, h_map = _logistic_map_and_hessian(lam=0.01)
    evals = np.linalg.eigvalsh(h_map)
    lam_soft = float(evals[0])
    lam_stiff = float(evals[-1])
    cond = lam_stiff / lam_soft

    radius = 2.7
    angles = np.deg2rad([20.0, 65.0, 110.0, 155.0, 250.0, 335.0])
    t = np.linspace(0.0, 3.2 / lam_soft, 400)

    def _bundle(whitened: bool) -> list[np.ndarray]:
        paths: list[np.ndarray] = []
        for angle in angles:
            z0 = np.array([radius * np.cos(angle), radius * np.sin(angle)], dtype=float)
            if whitened:
                path = np.column_stack([z0[0] * np.exp(-t), z0[1] * np.exp(-t)])
            else:
                path = np.column_stack([z0[0] * np.exp(-lam_stiff * t), z0[1] * np.exp(-lam_soft * t)])
            paths.append(path)
        return paths

    paths_iso = _bundle(False)
    paths_white = _bundle(True)

    g1 = np.linspace(-3.2, 3.2, 260)
    g2 = np.linspace(-3.2, 3.2, 260)
    xx, yy = np.meshgrid(g1, g2)
    dens = np.exp(-0.5 * (xx**2 + yy**2))
    levels = np.max(dens) * np.array([0.12, 0.24, 0.42, 0.62, 0.82])

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.95))
    fig.subplots_adjust(wspace=0.26)
    panel_data = [
        (
            "Isotropic local mean paths\n"
            + rf"$\lambda_{{\max}}/\lambda_{{\min}}\approx {cond:.1f}$",
            paths_iso,
            BLUE,
        ),
        (
            "Whitened local mean paths\n" + r"$M=H_{\mathrm{MAP}}^{-1}$ makes the rates equal",
            paths_white,
            GREEN,
        ),
    ]
    for ax, (title, paths, color) in zip(axes, panel_data):
        _style_axes(ax, xlabel="local eigen-coordinate 1", ylabel="local eigen-coordinate 2")
        ax.contourf(xx, yy, dens, levels=levels, cmap="Greys", alpha=0.18)
        ax.contour(xx, yy, dens, levels=levels, colors=["#9AA7B6"], linewidths=1.0)
        for path in paths:
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.8, alpha=0.9)
            ax.scatter(path[0, 0], path[0, 1], color=color, s=18, zorder=3)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, color=GRAY, pad=10)
        ax.set_xlim(g1.min(), g1.max())
        ax.set_ylim(g2.min(), g2.max())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_dir = _repo_root() / "Lecture Tex" / "figures" / "handouts" / "langevin_entropy_transport"
    fig_dir.mkdir(parents=True, exist_ok=True)

    make_jko_tradeoff_figure(fig_dir / "fig_jko_tradeoff.pdf")
    make_gaussian_jko_figure(fig_dir / "fig_gaussian_jko_steps.pdf")
    make_gaussian_decay_figure(fig_dir / "fig_gaussian_decay_preconditioning.pdf")
    make_logistic_spectrum_figure(fig_dir / "fig_logistic_hessian_spectrum.pdf")
    make_logistic_laplace_trajectory_figure(fig_dir / "fig_logistic_laplace_trajectories.pdf")
    print(f"Wrote figures to: {fig_dir}")


if __name__ == "__main__":
    main()
