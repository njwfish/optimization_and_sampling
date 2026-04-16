from __future__ import annotations

from pathlib import Path
import os

import numpy as np


def _repo_root() -> Path:
    # This file lives at: Lecture Tex/figures/handouts/discrete_diffusion/<this_file>.py
    return Path(__file__).resolve().parents[4]


def _configure_writable_caches() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/stat221_mplconfig")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/stat221_cache")


def _configure_matplotlib():
    _configure_writable_caches()

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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


GRAY = "#2F2F2F"
GRAY_LIGHT = "#F6F7FB"
BLUE = "#1F4E79"
BLUE_LIGHT = "#EAF3FA"
TEAL = "#0F766E"
TEAL_LIGHT = "#EAF7F6"
GREEN = "#2E7D32"
GREEN_LIGHT = "#EDF8EF"
ORANGE = "#B45309"
ORANGE_LIGHT = "#FFF4E8"
PURPLE = "#6A3D9A"
PURPLE_LIGHT = "#F3EEFA"

STATE_FILL = {
    "BA": BLUE_LIGHT,
    "AA": GREEN_LIGHT,
    "BB": ORANGE_LIGHT,
    "AB": PURPLE_LIGHT,
}

STATE_EDGE = {
    "BA": BLUE,
    "AA": GREEN,
    "BB": ORANGE,
    "AB": PURPLE,
}


def _alpha(t: np.ndarray) -> np.ndarray:
    return 1.0 - np.asarray(t, dtype=float)


def _full_state_probabilities(t: np.ndarray) -> dict[str, np.ndarray]:
    alpha = _alpha(t)
    other = 0.25 * (1.0 - alpha)
    return {
        "clean": alpha + other,
        "one_token": other,
        "two_token": other,
    }


def _factorized_probabilities(t: np.ndarray) -> dict[str, np.ndarray]:
    alpha = _alpha(t)
    keep = 0.5 * (1.0 + alpha)
    flip = 0.5 * (1.0 - alpha)
    return {
        "clean": keep**2,
        "one_token": flip * keep,
        "two_token": flip**2,
    }


def _style_axes(ax, *, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_facecolor(GRAY_LIGHT)
    ax.grid(True, color="white", linewidth=1.1)
    ax.tick_params(length=0, colors=GRAY)
    for spine in ax.spines.values():
        spine.set_color("#D5DAE5")
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=GRAY)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=GRAY)


def _ring_centers(*, k: int, radius: float) -> np.ndarray:
    angles = 2.0 * np.pi * np.arange(k) / k
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def _sample_ring_mog(
    rng: np.random.Generator,
    n: int,
    *,
    k: int = 8,
    radius: float = 4.0,
    std: float = 0.35,
) -> np.ndarray:
    idx = rng.integers(0, k, size=n)
    angles = 2.0 * np.pi * idx / k
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    return centers + std * rng.standard_normal((n, 2))


def _discrete_ring_example(
    *,
    seed: int = 0,
    grid_extent: float = 6.0,
    grid_points: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    coords = np.linspace(-grid_extent, grid_extent, grid_points, dtype=float)
    step = float(coords[1] - coords[0])
    edges = np.concatenate(
        (
            [coords[0] - 0.5 * step],
            0.5 * (coords[:-1] + coords[1:]),
            [coords[-1] + 0.5 * step],
        )
    )

    ring_samples = _sample_ring_mog(rng, 240_000)
    hist_yx, _, _ = np.histogram2d(ring_samples[:, 1], ring_samples[:, 0], bins=[edges, edges])
    p0 = hist_yx / hist_yx.sum()

    base_1d = np.exp(-0.5 * coords**2)
    base_1d /= base_1d.sum()
    p1 = np.outer(base_1d, base_1d)
    return coords, p0, base_1d, p1


def _factorized_grid_marginal(p0: np.ndarray, base_1d: np.ndarray, *, alpha: float) -> np.ndarray:
    px = p0.sum(axis=0)
    py = p0.sum(axis=1)
    product_base = np.outer(base_1d, base_1d)
    keep_x_redraw_y = np.outer(base_1d, px)
    redraw_x_keep_y = np.outer(py, base_1d)
    return (
        (alpha**2) * p0
        + alpha * (1.0 - alpha) * (keep_x_redraw_y + redraw_x_keep_y)
        + ((1.0 - alpha) ** 2) * product_base
    )


def _style_grid_panel(ax, *, lim: float) -> None:
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-4, 0, 4])
    ax.set_yticks([-4, 0, 4])
    ax.tick_params(length=0, colors=GRAY)
    for spine in ax.spines.values():
        spine.set_color("#D5DAE5")
        spine.set_linewidth(0.9)


def _discrete_path_vertices(
    clean: tuple[float, float],
    target: tuple[float, float],
    *,
    tau_first: float,
    tau_second: float,
) -> list[tuple[float, float]]:
    points = [clean]
    if tau_first <= tau_second:
        mid = (target[0], clean[1])
    else:
        mid = (clean[0], target[1])
    for point in [mid, target]:
        if point != points[-1]:
            points.append(point)
    return points


def build_bridge_probability_figure(out_path: Path) -> None:
    plt = _configure_matplotlib()

    times = np.linspace(0.0, 1.0, 401)
    full = _full_state_probabilities(times)
    factorized = _factorized_probabilities(times)
    label_box = dict(facecolor=GRAY_LIGHT, edgecolor="none", pad=0.18, alpha=0.95)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharex=True, sharey=True)
    fig.subplots_adjust(top=0.84, bottom=0.18, left=0.11, right=0.98, wspace=0.18)

    for ax, title in zip(axes, ["Full-state bridge", "Factorized bridge"]):
        _style_axes(ax, xlabel=r"time $t$")
        ax.set_title(title, color=GRAY, pad=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    axes[0].set_ylabel("conditional mass", color=GRAY)

    axes[0].plot(times, full["clean"], color=BLUE, linewidth=2.7, solid_capstyle="round")
    axes[0].plot(times, full["one_token"], color="#747C88", linewidth=2.3, solid_capstyle="round")
    axes[0].text(
        0.14, 0.80, r"clean BA", color=BLUE, fontsize=8.8, ha="left", va="center", bbox=label_box
    )
    axes[0].text(
        0.46, 0.15, r"AA, AB, BB", color=GRAY, fontsize=8.6, ha="left", va="center", bbox=label_box
    )

    axes[1].plot(times, factorized["clean"], color=BLUE, linewidth=2.7, solid_capstyle="round")
    axes[1].plot(times, factorized["one_token"], color=TEAL, linewidth=2.7, solid_capstyle="round")
    axes[1].plot(times, factorized["two_token"], color=PURPLE, linewidth=2.7, solid_capstyle="round")
    axes[1].text(
        0.57, 0.50, r"clean BA", color=BLUE, fontsize=8.8, ha="left", va="center", bbox=label_box
    )
    axes[1].text(
        0.60, 0.33, r"AA, BB = $O(t)$", color=TEAL, fontsize=8.8, ha="left", va="center", bbox=label_box
    )
    axes[1].text(
        0.60, 0.12, r"AB = $O(t^2)$", color=PURPLE, fontsize=8.8, ha="left", va="center", bbox=label_box
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_discretized_spatial_bridge_figure(out_path: Path, *, seed: int = 0) -> None:
    plt = _configure_matplotlib()
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm

    coords, p0, base_1d, p1 = _discrete_ring_example(seed=seed)
    mid_t = 0.55
    alpha_mid = float(_alpha(np.array([mid_t]))[0])
    pmid = _factorized_grid_marginal(p0, base_1d, alpha=alpha_mid)

    edges = np.arange(coords[0] - 0.5, coords[-1] + 1.0, 1.0)
    extent = [edges[0], edges[-1], edges[0], edges[-1]]
    lim = float(coords[-1] + 0.5)

    cmap = LinearSegmentedColormap.from_list(
        "stat221_discrete_bridge_density",
        ["#FFFFFF", BLUE_LIGHT, BLUE],
        N=256,
    )
    norm = PowerNorm(
        gamma=0.56,
        vmin=0.0,
        vmax=0.84 * max(float(p0.max()), float(pmid.max()), float(p1.max())),
    )

    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.9))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.10, wspace=0.10)

    masses = [p0, pmid, p1]
    titles = [r"$t=0$", rf"$t={mid_t:.2f}$", r"$t=1$"]
    for ax, mass, title in zip(axes[:3], masses, titles):
        ax.imshow(
            mass,
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        _style_grid_panel(ax, lim=lim)
        ax.set_title(title, color=GRAY, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])

    clean = (4.0, 0.0)
    axes[0].scatter(
        [clean[0]],
        [clean[1]],
        s=58,
        marker="*",
        color=PURPLE,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )

    ax_paths = axes[3]
    ax_paths.imshow(
        p0,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        alpha=0.18,
        zorder=0,
    )
    _style_grid_panel(ax_paths, lim=lim)
    ax_paths.set_xticks(edges, minor=True)
    ax_paths.set_yticks(edges, minor=True)
    ax_paths.grid(which="minor", color="white", linewidth=0.35, alpha=0.55)
    ax_paths.tick_params(which="minor", length=0)
    ax_paths.set_title("Sample paths", color=GRAY, pad=8)
    ax_paths.set_xticks([-4, 0, 4])
    ax_paths.set_yticks([-4, 0, 4])

    rng = np.random.default_rng(seed + 7)
    flat_p0 = p0.reshape(-1)
    n_cols = len(coords)
    for idx in rng.choice(flat_p0.size, size=14, replace=True, p=flat_p0):
        row, col = divmod(int(idx), n_cols)
        clean_sample = (float(coords[col]), float(coords[row]))
        target = (
            float(rng.choice(coords, p=base_1d)),
            float(rng.choice(coords, p=base_1d)),
        )
        tau_first = float(rng.uniform())
        tau_second = float(rng.uniform())
        vertices = _discrete_path_vertices(clean_sample, target, tau_first=tau_first, tau_second=tau_second)
        xs = [point[0] for point in vertices]
        ys = [point[1] for point in vertices]
        ax_paths.plot(xs, ys, color=GRAY, linewidth=1.4, alpha=0.28, zorder=2)

    highlight_target = (0.0, -2.0)
    highlight_vertices = _discrete_path_vertices(
        clean,
        highlight_target,
        tau_first=0.68,
        tau_second=0.22,
    )
    highlight_x = [point[0] for point in highlight_vertices]
    highlight_y = [point[1] for point in highlight_vertices]
    ax_paths.plot(
        highlight_x,
        highlight_y,
        color=ORANGE,
        linewidth=2.4,
        solid_capstyle="round",
        zorder=4,
    )
    if len(highlight_vertices) == 3:
        ax_paths.scatter(
            [highlight_vertices[1][0]],
            [highlight_vertices[1][1]],
            s=36,
            facecolor="white",
            edgecolor=ORANGE,
            linewidth=1.1,
            zorder=5,
        )

    ax_paths.scatter(
        [clean[0]],
        [clean[1]],
        s=110,
        marker="*",
        color=PURPLE,
        edgecolor="white",
        linewidth=0.8,
        zorder=6,
    )
    ax_paths.scatter(
        [highlight_target[0]],
        [highlight_target[1]],
        s=46,
        color=TEAL,
        edgecolor="white",
        linewidth=0.7,
        zorder=6,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _full_state_segments(target: str, tau: float) -> list[tuple[float, float, str]]:
    return [(0.0, tau, "BA"), (tau, 1.0, target)]


def _factorized_segments(
    target: str, *, tau_first: float | None = None, tau_second: float | None = None
) -> list[tuple[float, float, str]]:
    state = ["B", "A"]
    events: list[tuple[float, int, str]] = []
    if target[0] != "B":
        if tau_first is None:
            raise ValueError("tau_first is required when the first token changes.")
        events.append((float(tau_first), 0, target[0]))
    if target[1] != "A":
        if tau_second is None:
            raise ValueError("tau_second is required when the second token changes.")
        events.append((float(tau_second), 1, target[1]))
    events.sort(key=lambda item: item[0])

    segments: list[tuple[float, float, str]] = []
    left = 0.0
    for tau, coord, token in events:
        segments.append((left, tau, "".join(state)))
        state[coord] = token
        left = tau
    segments.append((left, 1.0, "".join(state)))
    return segments


def _draw_path_panel(ax, rows: list[list[tuple[float, float, str]]], title: str) -> None:
    from matplotlib.patches import Rectangle

    _style_axes(ax, xlabel=r"time $t$")
    ax.set_title(title, color=GRAY, pad=10)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([])
    ax.set_axisbelow(True)

    for y, segments in enumerate(rows):
        for left, right, state in segments:
            rect = Rectangle(
                (left, y - 0.34),
                right - left,
                0.68,
                facecolor=STATE_FILL[state],
                edgecolor="white",
                linewidth=1.4,
            )
            ax.add_patch(rect)
            outline = Rectangle(
                (left, y - 0.34),
                right - left,
                0.68,
                facecolor="none",
                edgecolor=STATE_EDGE[state],
                linewidth=0.9,
            )
            ax.add_patch(outline)
            if right - left >= 0.14:
                ax.text(
                    0.5 * (left + right),
                    y,
                    state,
                    ha="center",
                    va="center",
                    color=GRAY,
                    fontsize=8.5,
                    weight="semibold",
                )


def build_bridge_coupling_figure(out_path: Path) -> None:
    plt = _configure_matplotlib()
    from matplotlib.patches import Patch

    full_rows = [
        _full_state_segments("AA", 0.14),
        _full_state_segments("AB", 0.23),
        _full_state_segments("BB", 0.33),
        _full_state_segments("AA", 0.48),
        _full_state_segments("AB", 0.60),
        _full_state_segments("BB", 0.72),
        _full_state_segments("AB", 0.86),
    ]
    factorized_rows = [
        _factorized_segments("AA", tau_first=0.18),
        _factorized_segments("AB", tau_first=0.21, tau_second=0.70),
        _factorized_segments("BB", tau_second=0.28),
        _factorized_segments("AB", tau_first=0.63, tau_second=0.19),
        _factorized_segments("AA", tau_first=0.54),
        _factorized_segments("AB", tau_first=0.16, tau_second=0.37),
        _factorized_segments("AB", tau_first=0.82, tau_second=0.41),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharex=True, sharey=True)
    fig.subplots_adjust(top=0.76, bottom=0.16, left=0.08, right=0.98, wspace=0.12)

    _draw_path_panel(axes[0], full_rows, "One bridge time\nfor the whole sequence")
    _draw_path_panel(axes[1], factorized_rows, "Independent bridge times\nby coordinate")
    axes[0].set_ylabel("sample path", color=GRAY)

    legend_handles = [
        Patch(facecolor=STATE_FILL[state], edgecolor=STATE_EDGE[state], label=state)
        for state in ["BA", "AA", "BB", "AB"]
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
        columnspacing=1.4,
        handlelength=1.5,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = _repo_root() / "Lecture Tex/figures/handouts/discrete_diffusion"
    build_bridge_probability_figure(out_dir / "fig_discrete_bridge_probabilities.pdf")
    build_bridge_coupling_figure(out_dir / "fig_discrete_bridge_couplings.pdf")
    build_discretized_spatial_bridge_figure(out_dir / "fig_discrete_spatial_bridge.pdf")


if __name__ == "__main__":
    main()
