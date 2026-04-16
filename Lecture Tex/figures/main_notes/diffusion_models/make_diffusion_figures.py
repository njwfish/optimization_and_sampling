#!/usr/bin/env python3
"""
Generate figures for the Diffusion Models unit.

Outputs are written to `Lecture Tex/figures/main_notes/diffusion_models/`.

Run:
  python3 "Lecture Tex/figures/main_notes/diffusion_models/make_diffusion_figures.py"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _hex_to_rgb01(h: str) -> tuple[float, float, float]:
    h = h.strip().lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Expected 6 hex digits, got: {h!r}")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return r, g, b


def _sample_ring_mog(rng: np.random.Generator, n: int, *, k: int = 8, radius: float = 4.0, std: float = 0.35) -> np.ndarray:
    """Mixture of k isotropic Gaussians on a ring (2D)."""
    idx = rng.integers(0, k, size=n)
    angles = 2.0 * np.pi * idx / k
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    return centers + std * rng.standard_normal((n, 2))


def _stat221_palette() -> dict[str, tuple[float, float, float]]:
    # Colors from `Lecture Tex/preamble.tex`.
    return {
        "blue": _hex_to_rgb01("1F4E79"),
        "blue_light": _hex_to_rgb01("EAF3FA"),
        "gray": _hex_to_rgb01("2F2F2F"),
        "gray_light": _hex_to_rgb01("F6F7FB"),
        "purple": _hex_to_rgb01("6A3D9A"),
        "purple_light": _hex_to_rgb01("F3EEFA"),
        "teal": _hex_to_rgb01("0F766E"),
        "teal_light": _hex_to_rgb01("EAF7F6"),
    }


def _configure_matplotlib() -> None:
    import matplotlib as mpl  # noqa: WPS433 - optional import for scripts

    mpl.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "mathtext.fontset": "stix",
            "font.family": "DejaVu Sans",
        }
    )


def fig_spatially_linear_bridge(*, out_path: Path, seed: int = 0) -> None:
    """
    A simple 2D illustration of a canonical Gaussian noising path.

    Bridge:
      X_t = (1-t) X_0 + t Z
    with $X_0$ from a toy "data" distribution and $Z\sim\mathcal{N}(0,I)$.
    """
    _configure_matplotlib()
    import matplotlib.pyplot as plt  # noqa: WPS433 - optional import for scripts
    from matplotlib.colors import LinearSegmentedColormap  # noqa: WPS433 - optional import for scripts

    colors = _stat221_palette()
    rng = np.random.default_rng(seed)

    n = 40_000
    x0 = _sample_ring_mog(rng, n)
    z = rng.standard_normal((n, 2))

    def I(t: float) -> np.ndarray:
        return (1.0 - t) * x0

    # "Movie strip" times, in the same spirit as Holderrieth--Erives.
    ts = [0.0, 0.25, 0.5, 0.75, 1.0]
    xs = [I(t) + t * z for t in ts]

    # Density colormap: white -> stat221Blue.
    cmap = LinearSegmentedColormap.from_list(
        "stat221_blue_density",
        [(1.0, 1.0, 1.0), colors["blue"]],
        N=256,
    )

    fig, axes = plt.subplots(1, len(ts), figsize=(10.2, 2.2), sharex=True, sharey=True)
    lim = 6.0
    bins = 220
    for ax, t, x in zip(axes, ts, xs):
        h, xedges, yedges = np.histogram2d(
            x[:, 0],
            x[:, 1],
            bins=bins,
            range=[[-lim, lim], [-lim, lim]],
        )
        h = (h / max(float(h.max()), 1.0)) ** 0.35
        ax.imshow(
            h.T,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
        )
        ax.set_title(f"$t={t:.2f}$")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color((*colors["gray"], 0.20))
            spine.set_linewidth(0.8)

    fig.tight_layout(pad=0.35, w_pad=0.55)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_conditional_vs_marginal(*, out_path: Path, seed: int = 0) -> None:
    """
    A 2D illustration of a Gaussian conditional path and its induced marginal path.

    We use the canonical noising map
      X_t = (1-t) X_0 + t Z,   Z ~ N(0, I),
    and show:
      - conditional: Law(X_t | X_0 = x_0) for a fixed x_0,
      - marginal: Law(X_t) when X_0 is random.
    """
    _configure_matplotlib()
    import matplotlib.pyplot as plt  # noqa: WPS433 - optional import for scripts
    from matplotlib.colors import LinearSegmentedColormap  # noqa: WPS433 - optional import for scripts

    colors = _stat221_palette()
    rng = np.random.default_rng(seed)

    # Density colormap: white -> stat221Blue.
    cmap = LinearSegmentedColormap.from_list(
        "stat221_blue_density",
        [(1.0, 1.0, 1.0), colors["blue"]],
        N=256,
    )

    ts = [0.0, 0.25, 0.5, 0.75, 1.0]
    lim = 6.0
    bins = 220

    # Conditional path for a fixed x0.
    x0_fixed = np.array([4.0, 0.0], dtype=float)
    n_cond = 90_000
    z_cond = rng.standard_normal((n_cond, 2))
    xs_cond = [(1.0 - t) * x0_fixed[None, :] + t * z_cond for t in ts]

    # Marginal path for a random x0.
    n_marg = 90_000
    x0 = _sample_ring_mog(rng, n_marg)
    z = rng.standard_normal((n_marg, 2))
    xs_marg = [(1.0 - t) * x0 + t * z for t in ts]

    fig, axes = plt.subplots(2, len(ts), figsize=(10.2, 4.2), sharex=True, sharey=True)

    def _draw(ax, x: np.ndarray, t: float, *, highlight: np.ndarray | None = None) -> None:
        h, xedges, yedges = np.histogram2d(
            x[:, 0],
            x[:, 1],
            bins=bins,
            range=[[-lim, lim], [-lim, lim]],
        )
        h = (h / max(float(h.max()), 1.0)) ** 0.35
        ax.imshow(
            h.T,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
        )
        ax.set_title(f"$t={t:.2f}$")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color((*colors["gray"], 0.20))
            spine.set_linewidth(0.8)
        if highlight is not None:
            ax.scatter(
                [float(highlight[0])],
                [float(highlight[1])],
                s=55,
                marker="*",
                color=colors["purple"],
                edgecolor=(1.0, 1.0, 1.0, 0.9),
                linewidth=0.6,
                zorder=3,
            )

    for j, t in enumerate(ts):
        _draw(axes[0, j], xs_cond[j], t, highlight=x0_fixed if j == 0 else None)
        _draw(axes[1, j], xs_marg[j], t)

    # Row labels.
    axes[0, 0].set_ylabel("Conditional", rotation=90, labelpad=18, fontsize=11)
    axes[1, 0].set_ylabel("Marginal", rotation=90, labelpad=18, fontsize=11)

    fig.tight_layout(pad=0.35, w_pad=0.55, h_pad=0.90)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _ring_centers(*, k: int, radius: float) -> np.ndarray:
    angles = 2.0 * np.pi * np.arange(k) / k
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def fig_reverse_dynamics_trajectories(*, out_path: Path, seed: int = 0) -> None:
    """
    A 2D illustration of reverse-time simulation: probability-flow ODE vs stochastic sampler.

    We pick a toy smooth data distribution p0: a mixture of Gaussians on a ring, and use the
    Gaussian noising bridge X_t=(1-t)X_0+tZ. For this p0, we can compute the exact velocity
    field b(t,x)=E[Z-X_0|X_t=x] and the exact score s(t,x)=∇log p_t(x), then simulate:

      ODE (ε≡0): dX_t = b(t,X_t) dt  (integrated backward t=1→0)
      SDE (ε>0): dX_t = (b-ε s)(t,X_t) dt + sqrt(2ε) dB_t  (backward)
    """
    _configure_matplotlib()
    import matplotlib.pyplot as plt  # noqa: WPS433 - optional import for scripts
    from matplotlib.colors import LinearSegmentedColormap  # noqa: WPS433 - optional import for scripts

    colors = _stat221_palette()
    rng = np.random.default_rng(seed)

    # Toy data distribution parameters.
    k = 8
    radius = 4.0
    sig0 = 0.35  # component std at t=0
    mus = _ring_centers(k=k, radius=radius)  # (k,2)

    # Background density for p0.
    n_bg = 120_000
    x0_bg = _sample_ring_mog(rng, n_bg, k=k, radius=radius, std=sig0)

    cmap = LinearSegmentedColormap.from_list(
        "stat221_blue_density",
        [(1.0, 1.0, 1.0), colors["blue"]],
        N=256,
    )

    lim = 6.0
    bins = 240

    def _draw_background(ax) -> None:
        h, xedges, yedges = np.histogram2d(
            x0_bg[:, 0],
            x0_bg[:, 1],
            bins=bins,
            range=[[-lim, lim], [-lim, lim]],
        )
        h = (h / max(float(h.max()), 1.0)) ** 0.33
        ax.imshow(
            h.T,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
            alpha=0.55,
        )
        ax.scatter(mus[:, 0], mus[:, 1], s=12, color=colors["blue"], alpha=0.55, linewidth=0.0, zorder=2)

    def _softmax(logw: np.ndarray) -> np.ndarray:
        logw = logw - logw.max(axis=1, keepdims=True)
        w = np.exp(logw)
        w_sum = w.sum(axis=1, keepdims=True)
        return w / w_sum

    def _posterior_m_and_score(t: float, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        For the bridge X_t=(1-t)X_0+tZ with mixture-of-Gaussians prior on X_0:
          return m(t,x)=E[X_0|X_t=x] and score s(t,x)=∇log p_t(x).
        """
        if not (0.0 < t <= 1.0):
            raise ValueError(f"t must be in (0,1], got t={t}")

        # Marginal p_t is a mixture with means (1-t) mu_j and isotropic variance:
        sig_xt2 = (1.0 - t) ** 2 * sig0**2 + t**2

        # Responsibilities w_j(x) proportional to exp(-||x-(1-t)mu_j||^2/(2 sig_xt2)).
        diff = x[:, None, :] - (1.0 - t) * mus[None, :, :]
        dist2 = np.sum(diff**2, axis=2)
        logw = -0.5 * dist2 / sig_xt2
        w = _softmax(logw)  # (n,k)

        # Componentwise posterior mean of X_0 given X_t=x and component j.
        # Prior: X_0 ~ N(mu_j, sig0^2 I), likelihood: X_t|X_0 ~ N((1-t)X_0, t^2 I).
        prec_post = (1.0 / sig0**2) + ((1.0 - t) ** 2) / (t**2)
        sig_post2 = 1.0 / prec_post
        mean_post = sig_post2 * (
            mus[None, :, :] / sig0**2 + ((1.0 - t) / (t**2)) * x[:, None, :]
        )  # (n,k,2)
        m = np.sum(w[:, :, None] * mean_post, axis=1)  # (n,2)

        # Score of the marginal mixture p_t (isotropic component covariance).
        mu_bar = np.sum(w[:, :, None] * ((1.0 - t) * mus[None, :, :]), axis=1)  # (n,2)
        s = (mu_bar - x) / sig_xt2  # (n,2)
        return m, s

    # Simulate reverse-time dynamics from t=1 down to t=0.
    n_particles = 2_000
    n_steps = 360
    ts = np.linspace(1.0, 0.0, n_steps + 1)

    x_init = rng.standard_normal((n_particles, 2))
    x_ode = x_init.copy()
    x_sde = x_init.copy()

    eps = 0.18

    # Record a subset of trajectories for plotting.
    n_traj = 70
    traj_idx = rng.choice(n_particles, size=n_traj, replace=False)
    traj_ode = np.empty((n_steps + 1, n_traj, 2))
    traj_sde = np.empty((n_steps + 1, n_traj, 2))
    traj_ode[0] = x_ode[traj_idx]
    traj_sde[0] = x_sde[traj_idx]

    for i in range(n_steps):
        t = float(ts[i])
        dt = float(ts[i + 1] - ts[i])  # negative

        m_ode, s_ode = _posterior_m_and_score(t, x_ode)
        b_ode = (x_ode - m_ode) / t
        x_ode = x_ode + dt * b_ode

        m_sde, s_sde = _posterior_m_and_score(t, x_sde)
        b_sde = (x_sde - m_sde) / t
        b_rev = b_sde - eps * s_sde
        x_sde = x_sde + dt * b_rev + np.sqrt(2.0 * eps * abs(dt)) * rng.standard_normal(x_sde.shape)

        traj_ode[i + 1] = x_ode[traj_idx]
        traj_sde[i + 1] = x_sde[traj_idx]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharex=True, sharey=True)
    titles = [r"probability-flow ODE ($\varepsilon\equiv 0$)", rf"stochastic sampler ($\varepsilon={eps:.2f}$)"]
    for ax, title in zip(axes, titles):
        _draw_background(ax)
        ax.set_title(title)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color((*colors["gray"], 0.25))
            spine.set_linewidth(0.8)

    # Trajectories.
    for j in range(n_traj):
        axes[0].plot(traj_ode[:, j, 0], traj_ode[:, j, 1], color=(*colors["gray"], 0.35), lw=0.8, zorder=3)
        axes[1].plot(traj_sde[:, j, 0], traj_sde[:, j, 1], color=(*colors["gray"], 0.35), lw=0.8, zorder=3)

    # Start/end markers.
    start = x_init[traj_idx]
    end_ode = traj_ode[-1]
    end_sde = traj_sde[-1]
    axes[0].scatter(start[:, 0], start[:, 1], s=16, color=colors["teal"], alpha=0.85, linewidth=0.0, zorder=4)
    axes[1].scatter(start[:, 0], start[:, 1], s=16, color=colors["teal"], alpha=0.85, linewidth=0.0, zorder=4)
    axes[0].scatter(end_ode[:, 0], end_ode[:, 1], s=18, color=colors["purple"], alpha=0.90, linewidth=0.0, zorder=5)
    axes[1].scatter(end_sde[:, 0], end_sde[:, 1], s=18, color=colors["purple"], alpha=0.90, linewidth=0.0, zorder=5)

    fig.tight_layout(pad=0.40, w_pad=1.00)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    out_dir = Path(__file__).resolve().parent
    fig_spatially_linear_bridge(out_path=out_dir / "fig_diffusion_spatially_linear_bridge.pdf")
    fig_conditional_vs_marginal(out_path=out_dir / "fig_diffusion_conditional_vs_marginal.pdf")
    fig_reverse_dynamics_trajectories(out_path=out_dir / "fig_diffusion_reverse_dynamics.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
