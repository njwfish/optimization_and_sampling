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
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
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
PURPLE = "#6A3D9A"
PURPLE_LIGHT = "#F3EEFA"
RED = "#9F2D20"


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


def _find_best_gaussian_forward_kl(
    grid: np.ndarray, target_density: np.ndarray
) -> tuple[float, float, float]:
    mu_grid = np.linspace(-3.2, 3.2, 321)
    sigma_grid = np.linspace(0.18, 2.4, 241)
    best_mu = 0.0
    best_sigma = 1.0
    best_val = np.inf
    for mu in mu_grid:
        diffs = (grid[:, None] - mu) / sigma_grid[None, :]
        q_vals = np.exp(-0.5 * diffs**2) / (sigma_grid[None, :] * np.sqrt(2.0 * np.pi))
        integrand = q_vals * (np.log(np.maximum(q_vals, 1e-12)) - np.log(target_density[:, None]))
        vals = np.trapz(integrand, grid, axis=0)
        j = int(np.argmin(vals))
        if float(vals[j]) < best_val:
            best_val = float(vals[j])
            best_mu = float(mu)
            best_sigma = float(sigma_grid[j])
    return best_mu, best_sigma, best_val


def make_forward_kl_undercoverage_figure(out_path: Path) -> None:
    plt = _configure_matplotlib()

    x = np.linspace(-5.5, 5.5, 4001)
    mode_loc = 2.35
    mode_std = 0.55
    target = 0.5 * _normal_pdf(x, -mode_loc, mode_std) + 0.5 * _normal_pdf(x, mode_loc, mode_std)

    best_mu, best_sigma, best_kl = _find_best_gaussian_forward_kl(x, target)
    if best_mu < 0.0:
        best_mu = -best_mu
    q_mode = _normal_pdf(x, best_mu, best_sigma)

    centered_sigma_grid = np.linspace(0.35, 3.2, 321)
    centered_kl = []
    for sigma in centered_sigma_grid:
        q_center = _normal_pdf(x, 0.0, float(sigma))
        val = np.trapz(q_center * (np.log(np.maximum(q_center, 1e-12)) - np.log(target)), x)
        centered_kl.append(float(val))
    centered_kl = np.asarray(centered_kl)
    centered_sigma = float(centered_sigma_grid[int(np.argmin(centered_kl))])
    centered_best_kl = float(np.min(centered_kl))
    q_center = _normal_pdf(x, 0.0, centered_sigma)

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    _style_axes(ax, xlabel=r"latent coordinate $z$", ylabel="density")
    bridge_mask = np.abs(x) <= 0.8
    ax.axvspan(-0.8, 0.8, color=ORANGE_LIGHT, alpha=0.85, zorder=0)
    ax.plot(x, target, color=GRAY, linewidth=2.5, label=r"target posterior $p_\theta(z\mid x)$")
    ax.plot(
        x,
        q_mode,
        color=BLUE,
        linewidth=2.5,
        label=rf"best Gaussian under $\mathrm{{KL}}(q\|p)$",
    )
    ax.plot(
        x,
        q_center,
        color=ORANGE,
        linewidth=2.2,
        linestyle="--",
        label=r"best centered Gaussian",
    )
    y_top = 1.06 * max(float(np.max(target)), float(np.max(q_mode)), float(np.max(q_center)))
    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(0.0, y_top)
    ax.text(
        0.0,
        y_top - 0.02,
        "low-density bridge",
        ha="center",
        va="top",
        color=ORANGE,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": LIGHT_BORDER, "boxstyle": "round,pad=0.25"},
    )
    ax.text(
        2.05,
        0.60,
        rf"$\mathrm{{KL}}(q_{{\mathrm{{mode}}}}\|p)\approx {best_kl:.2f}$",
        color=BLUE,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": LIGHT_BORDER, "boxstyle": "round,pad=0.2"},
    )
    ax.text(
        -4.9,
        0.12,
        rf"$\mathrm{{KL}}(q_{{\mathrm{{center}}}}\|p)\approx {centered_best_kl:.2f}$",
        color=ORANGE,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": LIGHT_BORDER, "boxstyle": "round,pad=0.2"},
    )
    ax.legend(loc="upper left", frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _bivariate_gaussian_density(
    xx: np.ndarray, yy: np.ndarray, *, cov: np.ndarray
) -> np.ndarray:
    inv = np.linalg.inv(cov)
    det = float(np.linalg.det(cov))
    quad = (
        inv[0, 0] * xx**2
        + 2.0 * inv[0, 1] * xx * yy
        + inv[1, 1] * yy**2
    )
    return np.exp(-0.5 * quad) / (2.0 * np.pi * np.sqrt(det))


def make_mean_field_correlation_loss_figure(out_path: Path) -> None:
    plt = _configure_matplotlib()

    rho = 0.82
    cov_exact = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
    cov_prod = np.array([[1.0 - rho**2, 0.0], [0.0, 1.0 - rho**2]], dtype=float)

    grid = np.linspace(-3.1, 3.1, 350)
    xx, yy = np.meshgrid(grid, grid)
    exact = _bivariate_gaussian_density(xx, yy, cov=cov_exact)
    prod = _bivariate_gaussian_density(xx, yy, cov=cov_prod)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    fig.subplots_adjust(wspace=0.22)
    levels_exact = np.linspace(np.max(exact) * 0.15, np.max(exact) * 0.9, 5)
    levels_prod = np.linspace(np.max(prod) * 0.15, np.max(prod) * 0.9, 5)

    for ax in axes:
        _style_axes(ax, xlabel=r"$z_1$", ylabel=r"$z_2$")
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-3.0, 3.0)
        ax.set_aspect("equal", adjustable="box")

    axes[0].set_title("Exact correlated posterior", color=GRAY, pad=10)
    axes[0].contourf(xx, yy, exact, levels=levels_exact, cmap="Blues", alpha=0.25)
    axes[0].contour(xx, yy, exact, levels=levels_exact, colors=[BLUE], linewidths=1.4)
    axes[0].text(
        -2.7,
        -2.6,
        rf"$\mathrm{{Cov}}(Z_1,Z_2)={rho:.2f}$",
        color=BLUE,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": LIGHT_BORDER, "boxstyle": "round,pad=0.2"},
    )

    axes[1].set_title("Best product Gaussian", color=GRAY, pad=10)
    axes[1].contour(xx, yy, exact, levels=levels_exact, colors=["#9AA7B6"], linewidths=1.1, linestyles="--")
    axes[1].contourf(xx, yy, prod, levels=levels_prod, cmap="Greens", alpha=0.22)
    axes[1].contour(xx, yy, prod, levels=levels_prod, colors=[GREEN], linewidths=1.5)
    axes[1].text(
        -2.78,
        -2.6,
        rf"$\mathrm{{Var}}_q(Z_j)=1-\rho^2\approx {1.0 - rho**2:.2f}$",
        color=GREEN,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": LIGHT_BORDER, "boxstyle": "round,pad=0.2"},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_amortization_gap_figure(out_path: Path) -> None:
    plt = _configure_matplotlib()

    def m_star(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return 1.25 * np.sin(1.35 * x) + 0.28 * x

    xs = np.linspace(-3.0, 3.0, 13)
    y_star = m_star(xs)
    slope, intercept = np.polyfit(xs, y_star, deg=1)

    x_dense = np.linspace(-3.0, 3.0, 400)
    y_dense = m_star(x_dense)
    y_fit = slope * x_dense + intercept

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.85))
    fig.subplots_adjust(wspace=0.38, top=0.88)

    ax = axes[0]
    _style_axes(ax, xlabel=r"observation $x$", ylabel=r"posterior mean")
    ax.set_title("Per-datum optima vs\nshared affine encoder", color=GRAY, pad=8, fontsize=10.4)
    ax.plot(x_dense, y_dense, color=GRAY, linewidth=2.3, label=r"local optimum $\mu^\star(x)$")
    ax.plot(x_dense, y_fit, color=BLUE, linewidth=2.2, label=r"best affine encoder $\mu_\phi(x)$")
    ax.scatter(xs, y_star, color=PURPLE, s=28, zorder=3, edgecolor="white", linewidth=0.7)
    probe_idx = [1, 4, 8, 11]
    for idx in probe_idx:
        x0 = float(xs[idx])
        y0 = float(y_star[idx])
        y1 = float(slope * x0 + intercept)
        ax.plot([x0, x0], [y0, y1], color=BLUE, linewidth=1.0, linestyle=":")
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1]
    _style_axes(ax, xlabel=r"latent coordinate $z$")
    ax.set_title("Same local family,\nshared-map mismatch", color=GRAY, pad=8, fontsize=10.4)
    z_grid = np.linspace(-3.4, 3.4, 600)
    sigma = 0.32
    levels = np.array([3.25, 2.35, 1.45, 0.55])
    ax.set_yticks(levels)
    ax.set_yticklabels([r"$x=-2.5$", r"$x=-1.0$", r"$x=1.0$", r"$x=2.5$"])
    ax.set_ylim(0.0, 4.0)
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylabel("selected observations", color=GRAY, labelpad=10)
    ax.yaxis.set_label_position("right")
    for level, x0 in zip(levels, np.array([-2.5, -1.0, 1.0, 2.5])):
        true_mean = float(m_star(np.array([x0]))[0])
        fit_mean = float(slope * x0 + intercept)
        scale = 0.48
        ax.plot(z_grid, level + scale * _normal_pdf(z_grid, true_mean, sigma), color=GRAY, linewidth=2.0)
        ax.plot(
            z_grid,
            level + scale * _normal_pdf(z_grid, fit_mean, sigma),
            color=BLUE,
            linewidth=2.0,
            linestyle="--",
        )
        ax.scatter([true_mean], [level], color=GRAY, s=18, zorder=3)
        ax.scatter([fit_mean], [level], color=BLUE, s=18, zorder=3)
    ax.text(
        -3.18,
        3.78,
        "solid: local optimum\n dashed: shared encoder",
        ha="left",
        va="top",
        color=GRAY,
        fontsize=8.1,
        bbox={"facecolor": "white", "edgecolor": LIGHT_BORDER, "boxstyle": "round,pad=0.25"},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _make_bar_image(shift: float, *, n: int = 28) -> np.ndarray:
    coords = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(coords, coords)
    vertical = np.exp(-((xx - shift) ** 2) / (2.0 * 0.06**2))
    envelope = np.exp(-(yy**2) / (2.0 * 0.35**2))
    shoulder = 0.30 * np.exp(-((xx - shift) ** 2 + (yy + 0.45) ** 2) / (2.0 * 0.08**2))
    image = vertical * envelope + shoulder
    image = image / np.max(image)
    return image


def make_blur_figure(out_path: Path) -> None:
    plt = _configure_matplotlib()

    img_left = _make_bar_image(-0.22)
    img_right = _make_bar_image(0.18)
    img_mean = 0.5 * (img_left + img_right)

    rng = np.random.default_rng(221)
    z_left = rng.normal(loc=(-0.08, 0.06), scale=(0.18, 0.15), size=(60, 2))
    z_right = rng.normal(loc=(0.05, -0.02), scale=(0.18, 0.15), size=(60, 2))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(6.4, 6.0),
        gridspec_kw={"height_ratios": [1.0, 1.15]},
    )
    fig.subplots_adjust(wspace=0.18, hspace=0.24)

    for ax, image, title in [
        (axes[0, 0], img_left, r"sharp datum $x^{(1)}$"),
        (axes[0, 1], img_right, r"sharp datum $x^{(2)}$"),
        (axes[1, 1], img_mean, r"conditional mean $\mathbb{E}[X\mid Z=z]$"),
    ]:
        ax.imshow(image, cmap="gray_r", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(title, color=GRAY, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(GRAY_LIGHT)
        for spine in ax.spines.values():
            spine.set_color(LIGHT_BORDER)

    ax = axes[1, 0]
    _style_axes(ax, xlabel=r"$z_1$", ylabel=r"$z_2$")
    ax.set_title("overlapping encoder neighborhoods", color=GRAY, pad=8)
    ax.scatter(z_left[:, 0], z_left[:, 1], s=18, color=BLUE, alpha=0.78, edgecolors="none")
    ax.scatter(z_right[:, 0], z_right[:, 1], s=18, color=GREEN, alpha=0.78, edgecolors="none")
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect("equal", adjustable="box")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_posterior_collapse_figure(out_path: Path) -> None:
    plt = _configure_matplotlib()

    rng = np.random.default_rng(2210)
    cluster_means = np.array([[-1.8, 1.2], [1.7, 0.9], [-0.3, -2.0]], dtype=float)
    colors = [BLUE, GREEN, PURPLE]
    labels = ["group 1", "group 2", "group 3"]

    informative_points = []
    collapsed_points = []
    for mean in cluster_means:
        informative = rng.normal(loc=mean, scale=(0.26, 0.24), size=(85, 2))
        collapsed = rng.normal(loc=0.14 * mean, scale=(0.17, 0.16), size=(85, 2))
        informative_points.append(informative)
        collapsed_points.append(collapsed)

    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.6))
    fig.subplots_adjust(wspace=0.24)

    for ax, title, point_sets in [
        (axes[0], r"encoder uses $z$", informative_points),
        (axes[1], r"collapse toward the prior", collapsed_points),
    ]:
        _style_axes(ax, xlabel=r"$z_1$", ylabel=r"$z_2$")
        ax.set_title(title, color=GRAY, pad=10)
        circle_one = plt.Circle((0.0, 0.0), 1.0, fill=False, color=LIGHT_BORDER, linewidth=1.2)
        circle_two = plt.Circle((0.0, 0.0), 2.0, fill=False, color=LIGHT_BORDER, linewidth=1.0, linestyle="--")
        ax.add_patch(circle_one)
        ax.add_patch(circle_two)
        for points, color, label in zip(point_sets, colors, labels):
            ax.scatter(points[:, 0], points[:, 1], s=16, color=color, alpha=0.75, edgecolors="none", label=label)
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-3.0, 3.0)
        ax.set_aspect("equal", adjustable="box")

    axes[0].legend(loc="upper left", frameon=False)
    axes[1].text(
        -2.75,
        2.72,
        "circles: prior support",
        ha="left",
        va="top",
        color=GRAY,
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": LIGHT_BORDER, "boxstyle": "round,pad=0.22"},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo = _repo_root()
    handout_dir = repo / "Lecture Tex" / "figures" / "handouts" / "elbo_geometry"
    handout_dir.mkdir(parents=True, exist_ok=True)

    make_forward_kl_undercoverage_figure(handout_dir / "fig_forward_kl_undercoverage_applied.pdf")
    make_mean_field_correlation_loss_figure(handout_dir / "fig_mean_field_correlation_loss_applied.pdf")
    make_amortization_gap_figure(handout_dir / "fig_vi_amortization_applied.pdf")
    make_blur_figure(handout_dir / "fig_latent_overlap_blur_applied.pdf")
    make_posterior_collapse_figure(handout_dir / "fig_decoder_bypass_collapse_applied.pdf")

    print("Wrote handout figures to:", handout_dir)


if __name__ == "__main__":
    main()
