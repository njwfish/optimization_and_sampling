from __future__ import annotations

from pathlib import Path
import os

import numpy as np


def _repo_root() -> Path:
    # This file lives at: Lecture Tex/figures/main_notes/augmented_state_optimization/<this_file>.py
    return Path(__file__).resolve().parents[4]


def _configure_writable_caches() -> None:
    # Codex runs in a sandbox where the home directory may not be writable.
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/stat221_mplconfig')
    os.environ.setdefault('XDG_CACHE_HOME', '/tmp/stat221_cache')


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def _hermgauss_expectation(g, *, n: int = 140) -> float:
    """Compute E[g(Z)] for Z ~ N(0,1) via Gauss--Hermite quadrature."""

    x, w = np.polynomial.hermite.hermgauss(int(n))
    z = np.sqrt(2.0) * x
    return float(np.dot(w, g(z)) / np.sqrt(np.pi))


def _population_em_map(theta: float, *, mu: float, sigma: float) -> float:
    """Population EM map for the 1D symmetric two-component Gaussian mixture.

    Model: Y ~ 0.5 N(mu, sigma^2) + 0.5 N(-mu, sigma^2), with mu > 0.
    Parameter: theta (mean, with symmetry theta ~ -theta).
    """

    def g(z: np.ndarray) -> np.ndarray:
        y_pos = mu + sigma * z
        y_neg = -mu + sigma * z
        a_pos = 2.0 * theta * y_pos / (sigma**2)
        a_neg = 2.0 * theta * y_neg / (sigma**2)
        w_pos = _sigmoid(a_pos)
        w_neg = _sigmoid(a_neg)
        # M(theta) = 2 E[w_theta(Y) Y] - E[Y].  For the symmetric mixture, E[Y]=0,
        # but we keep the mixture average explicit.
        return 0.5 * (2.0 * w_pos * y_pos + 2.0 * w_neg * y_neg)

    return _hermgauss_expectation(g, n=160)


def _population_em_map_derivative(theta: float, *, mu: float, sigma: float) -> float:
    """Derivative d/dtheta M(theta) for the 1D symmetric two-component mixture."""

    def g(z: np.ndarray) -> np.ndarray:
        y_pos = mu + sigma * z
        y_neg = -mu + sigma * z

        def term(y: np.ndarray) -> np.ndarray:
            a = 2.0 * theta * y / (sigma**2)
            ww = _sigmoid(a)
            return ww * (1.0 - ww) * (2.0 * y / (sigma**2)) * y

        return 0.5 * (2.0 * term(y_pos) + 2.0 * term(y_neg))

    return _hermgauss_expectation(g, n=220)


def _sample_em_update(theta: float, y: np.ndarray, *, sigma: float) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    w = _sigmoid(2.0 * theta * y / (sigma**2))
    return float(2.0 * np.mean(w * y) - np.mean(y))


def _missing_covariates_population_map_1d(
    theta: float, *, theta_star: float, sigma: float, rho: float
) -> float:
    """Population EM map for 1D regression with missing covariates.

    Model:
      X ~ N(0,1),  Y = theta_star * X + eps,  eps ~ N(0, sigma^2).

    Observation:
      With prob (1-rho) we observe X; with prob rho we only observe Y and treat X as latent.

    This matches Eq. (missing-covariates-1d-pop-map) in the Section 3 EM handout.
    """

    theta = float(theta)
    theta_star = float(theta_star)
    sigma = float(sigma)
    rho = float(rho)

    num = (1.0 - rho) * theta_star + rho * (theta / (theta**2 + sigma**2)) * (
        theta_star**2 + sigma**2
    )
    den = (1.0 - rho) + rho * (
        sigma**2 / (theta**2 + sigma**2)
        + (theta**2 / (theta**2 + sigma**2) ** 2) * (theta_star**2 + sigma**2)
    )
    return float(num / den)


def _tex_table_xy(xs: np.ndarray, ys: np.ndarray) -> str:
    lines = ["x y \\\\"]
    for x, y in zip(xs, ys):
        lines.append(f"{float(x):.6f} {float(y):.12g} \\\\")
    return "\n".join(lines)


def _missing_covariates_regression_em_update(
    theta: np.ndarray,
    *,
    y: np.ndarray,
    x_obs: np.ndarray,
    obs_mask: np.ndarray,
    sigma: float,
    ridge: float = 1e-10,
) -> np.ndarray:
    """One sample-EM update for linear regression with missing covariates (d=2).

    Model for generating data:
      X ~ N(0, I_2),  Y = <X, theta_star> + eps,  eps ~ N(0, sigma^2).

    Observation pattern:
      Each coordinate of X may be missing; x_obs stores observed coordinates
      and uses 0.0 in missing positions. obs_mask is True where observed.

    The EM update treats missing covariates as latent and maximizes the
    expected complete-data log-likelihood under the conditional law of X
    given (x_obs, y) under the current parameter theta.
    """

    theta = np.asarray(theta, dtype=float).reshape(2)
    y = np.asarray(y, dtype=float).reshape(-1)
    x_obs = np.asarray(x_obs, dtype=float).reshape(-1, 2)
    obs_mask = np.asarray(obs_mask, dtype=bool).reshape(-1, 2)

    sum_a = np.zeros((2, 2), dtype=float)
    sum_b = np.zeros(2, dtype=float)

    both_obs = obs_mask[:, 0] & obs_mask[:, 1]
    only0 = obs_mask[:, 0] & (~obs_mask[:, 1])
    only1 = (~obs_mask[:, 0]) & obs_mask[:, 1]
    none_obs = (~obs_mask[:, 0]) & (~obs_mask[:, 1])

    # Case: both coordinates observed.
    if np.any(both_obs):
        x = x_obs[both_obs, :]
        y_b = y[both_obs]
        sum_a += x.T @ x
        sum_b += (y_b[:, None] * x).sum(axis=0)

    # Case: x0 observed, x1 missing.
    if np.any(only0):
        x0 = x_obs[only0, 0]
        y0 = y[only0]
        resid = y0 - theta[0] * x0
        den = theta[1] ** 2 + sigma**2
        m1 = (theta[1] / den) * resid
        var1 = sigma**2 / den

        sum_a[0, 0] += float(np.sum(x0**2))
        sum_a[0, 1] += float(np.sum(x0 * m1))
        sum_a[1, 0] += float(np.sum(x0 * m1))
        sum_a[1, 1] += float(np.sum(var1 + m1**2))

        sum_b[0] += float(np.sum(y0 * x0))
        sum_b[1] += float(np.sum(y0 * m1))

    # Case: x1 observed, x0 missing.
    if np.any(only1):
        x1 = x_obs[only1, 1]
        y1 = y[only1]
        resid = y1 - theta[1] * x1
        den = theta[0] ** 2 + sigma**2
        m0 = (theta[0] / den) * resid
        var0 = sigma**2 / den

        sum_a[1, 1] += float(np.sum(x1**2))
        sum_a[0, 1] += float(np.sum(m0 * x1))
        sum_a[1, 0] += float(np.sum(m0 * x1))
        sum_a[0, 0] += float(np.sum(var0 + m0**2))

        sum_b[1] += float(np.sum(y1 * x1))
        sum_b[0] += float(np.sum(y1 * m0))

    # Case: both coordinates missing.
    if np.any(none_obs):
        y2 = y[none_obs]
        den = float(np.dot(theta, theta) + sigma**2)
        # Cov[X|y,theta] = I - theta theta^T / (||theta||^2 + sigma^2).
        cov = np.eye(2, dtype=float) - np.outer(theta, theta) / den
        sum_a += float(y2.size) * cov

        y2_sq_sum = float(np.sum(y2**2))
        sum_a += (y2_sq_sum / (den**2)) * np.outer(theta, theta)
        sum_b += (y2_sq_sum / den) * theta

    sum_a += ridge * np.eye(2, dtype=float)
    return np.linalg.solve(sum_a, sum_b)


def make_em_operator_fixed_points_figure(out_path: Path) -> None:
    _configure_writable_caches()

    sigma = 1.0
    theta_grid = np.linspace(-3.0, 3.0, 241)

    theta0 = 0.2
    n_cobweb = 6

    settings = [
        (0.6, r"$\theta^\star/\sigma=0.6$ (weaker separation)"),
        (2.0, r"$\theta^\star/\sigma=2.0$ (stronger separation)"),
    ]

    panels: list[dict[str, object]] = []
    for mu, title in settings:
        m_vals = np.array([_population_em_map(float(t), mu=float(mu), sigma=sigma) for t in theta_grid])

        # Cobweb path starting from theta0.
        theta = float(theta0)
        cobweb_pts: list[tuple[float, float]] = [(theta, theta)]
        for _ in range(int(n_cobweb)):
            theta_next = _population_em_map(theta, mu=float(mu), sigma=sigma)
            cobweb_pts.append((theta, theta_next))
            cobweb_pts.append((theta_next, theta_next))
            theta = float(theta_next)
        cobweb_x = np.array([p[0] for p in cobweb_pts], dtype=float)
        cobweb_y = np.array([p[1] for p in cobweb_pts], dtype=float)

        slope = float(_population_em_map_derivative(float(mu), mu=float(mu), sigma=sigma))
        panels.append(
            {
                "mu": float(mu),
                "title": str(title),
                "m_table": _tex_table_xy(theta_grid, m_vals),
                "diag_table": _tex_table_xy(theta_grid, theta_grid),
                "cobweb_table": _tex_table_xy(cobweb_x, cobweb_y),
                "slope": slope,
            }
        )

    tex = rf"""
\begin{{tikzpicture}}
  \begin{{axis}}[
    name=left,
    width=0.48\linewidth,
    height=6.0cm,
    xmin=-3, xmax=3,
    ymin=-3, ymax=3,
    axis lines=left,
    xlabel={{$\theta$}},
    ylabel={{$M(\theta)$}},
    title={{{panels[0]["title"]}}},
    title style={{font=\small, yshift=-0.35em}},
    clip=false,
  ]
    \addplot[stat221Blue,thick] table[row sep=\\] {{
{panels[0]["m_table"]}
    }};
    \addplot[stat221Gray,densely dashed,thick,opacity=0.9] table[row sep=\\] {{
{panels[0]["diag_table"]}
    }};

    \addplot[stat221Orange,thick,mark=*,mark size=0.7pt,opacity=0.85] table[row sep=\\] {{
{panels[0]["cobweb_table"]}
    }};

    \addplot[only marks,mark=o,mark size=1.6pt,stat221Gray,fill=stat221Gray] coordinates {{(0,0)}};
    \addplot[only marks,mark=x,mark size=2.7pt,stat221Purple,very thick] coordinates {{(-{panels[0]["mu"]:.6f}, -{panels[0]["mu"]:.6f})}};
    \addplot[only marks,mark=x,mark size=2.7pt,stat221Red,very thick] coordinates {{({panels[0]["mu"]:.6f}, {panels[0]["mu"]:.6f})}};

    \node[anchor=south west] at (axis cs:{panels[0]["mu"]:.6f},{panels[0]["mu"]:.6f}) {{$\theta^\star$}};
    \node[anchor=south east] at (axis cs:-{panels[0]["mu"]:.6f},-{panels[0]["mu"]:.6f}) {{$-\theta^\star$}};
    \node[anchor=south west] at (axis cs:0,0) {{$0$}};

    \node[stat221Blue,anchor=south west,font=\small] at (axis cs:-2.85,2.55) {{$M(\theta)$}};
    \node[stat221Gray,anchor=south west,font=\small] at (axis cs:-2.85,2.25) {{$\theta$}};
    \node[stat221Orange,anchor=south west,font=\small] at (axis cs:-2.85,1.95) {{cobweb from $\theta^{(0)}={theta0:.1f}$}};
    \node[anchor=south west,font=\small] at (axis cs:-2.85,1.65) {{$|M'(\theta^\star)|\approx {abs(panels[0]["slope"]):.3f}$}};
  \end{{axis}}

  \begin{{axis}}[
    at={{(left.north east)}}, anchor=north west,
    xshift=0.45cm,
    width=0.48\linewidth,
    height=6.0cm,
    xmin=-3, xmax=3,
    ymin=-3, ymax=3,
    axis lines=left,
    xlabel={{$\theta$}},
    ylabel={{}},
    title={{{panels[1]["title"]}}},
    title style={{font=\small, yshift=-0.35em}},
    clip=false,
    yticklabels={{}},
  ]
    \addplot[stat221Blue,thick] table[row sep=\\] {{
{panels[1]["m_table"]}
    }};
    \addplot[stat221Gray,densely dashed,thick,opacity=0.9] table[row sep=\\] {{
{panels[1]["diag_table"]}
    }};

    \addplot[stat221Orange,thick,mark=*,mark size=0.7pt,opacity=0.85] table[row sep=\\] {{
{panels[1]["cobweb_table"]}
    }};

    \addplot[only marks,mark=o,mark size=1.6pt,stat221Gray,fill=stat221Gray] coordinates {{(0,0)}};
    \addplot[only marks,mark=x,mark size=2.7pt,stat221Purple,very thick] coordinates {{(-{panels[1]["mu"]:.6f}, -{panels[1]["mu"]:.6f})}};
    \addplot[only marks,mark=x,mark size=2.7pt,stat221Red,very thick] coordinates {{({panels[1]["mu"]:.6f}, {panels[1]["mu"]:.6f})}};

    \node[anchor=south west] at (axis cs:{panels[1]["mu"]:.6f},{panels[1]["mu"]:.6f}) {{$\theta^\star$}};
    \node[anchor=south east] at (axis cs:-{panels[1]["mu"]:.6f},-{panels[1]["mu"]:.6f}) {{$-\theta^\star$}};
    \node[anchor=south west] at (axis cs:0,0) {{$0$}};
    \node[anchor=south west,font=\small] at (axis cs:-2.85,1.65) {{$|M'(\theta^\star)|\approx {abs(panels[1]["slope"]):.3f}$}};
  \end{{axis}}
\end{{tikzpicture}}
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex.strip() + "\n", encoding="utf-8")


def make_em_population_vs_sample_convergence_figure(out_path: Path) -> None:
    _configure_writable_caches()

    rng = np.random.default_rng(3)
    mu = 2.0
    sigma = 1.0
    theta0 = 0.2
    T = 22

    # Population trajectory (deterministic).
    theta = float(theta0)
    pop_traj = [theta]
    for _ in range(T):
        theta = _population_em_map(theta, mu=mu, sigma=sigma)
        pop_traj.append(theta)
    pop_traj = np.asarray(pop_traj, dtype=float)
    pop_dist = np.maximum(np.abs(pop_traj - mu), 1e-10)

    # Sample trajectories (random), summarized by median + quantile band.
    n_list = [250, 1000, 4000]
    n_reps = 30

    it = np.arange(T + 1, dtype=int)

    curves: list[dict[str, object]] = [
        {"name": "pop", "table": _tex_table_xy(it, pop_dist)},
    ]

    for n, name in [(n_list[0], "n250"), (n_list[1], "n1000"), (n_list[2], "n4000")]:
        dists = np.zeros((n_reps, T + 1), dtype=float)
        for rep in range(n_reps):
            signs = rng.choice([-1.0, 1.0], size=n)
            y = signs * mu + sigma * rng.standard_normal(size=n)

            theta = float(theta0)
            traj = [theta]
            for _ in range(T):
                theta = _sample_em_update(theta, y, sigma=sigma)
                traj.append(theta)
            traj = np.asarray(traj, dtype=float)
            dists[rep, :] = np.maximum(np.abs(traj - mu), 1e-10)

        med = np.median(dists, axis=0)
        lo = np.quantile(dists, 0.2, axis=0)
        hi = np.quantile(dists, 0.8, axis=0)
        curves.append({"name": name, "table": _tex_table_xy(it, med)})
        curves.append({"name": f"{name}_lo", "table": _tex_table_xy(it, lo)})
        curves.append({"name": f"{name}_hi", "table": _tex_table_xy(it, hi)})

    def tbl(key: str) -> str:
        for item in curves:
            if item["name"] == key:
                return str(item["table"])
        raise KeyError(key)

    tex = rf"""
\begin{{tikzpicture}}
  \begin{{axis}}[
    width=0.92\linewidth,
    height=6.2cm,
    xmin=0, xmax={T},
    ymode=log,
    axis lines=left,
    xlabel={{EM iteration $t$}},
    ylabel={{distance $|\theta^{{(t)}}-\theta^\star|$}},
    title={{Gaussian mixture: geometric refinement until a sample-size floor}},
    title style={{font=\small, yshift=-0.35em}},
    legend style={{draw=stat221Gray!25, fill=white, fill opacity=0.95, text opacity=1}},
    legend pos=north east,
    clip=false,
  ]
    \addplot[stat221Gray,thick,densely dashed] table[row sep=\\] {{
{tbl("pop")}
    }};
    \addlegendentry{{Population EM}}

    % n=250
    \addplot[stat221Red,thick] table[row sep=\\] {{
{tbl("n250")}
    }};
    \addlegendentry{{$n=250$ (median)}}
    \addplot[stat221Red!50,thin] table[row sep=\\] {{
{tbl("n250_lo")}
    }};
    \addplot[stat221Red!50,thin] table[row sep=\\] {{
{tbl("n250_hi")}
    }};

    % n=1000
    \addplot[stat221Purple,thick] table[row sep=\\] {{
{tbl("n1000")}
    }};
    \addlegendentry{{$n=1000$ (median)}}
    \addplot[stat221Purple!55,thin] table[row sep=\\] {{
{tbl("n1000_lo")}
    }};
    \addplot[stat221Purple!55,thin] table[row sep=\\] {{
{tbl("n1000_hi")}
    }};

    % n=4000
    \addplot[stat221Blue,thick] table[row sep=\\] {{
{tbl("n4000")}
    }};
    \addlegendentry{{$n=4000$ (median)}}
    \addplot[stat221Blue!55,thin] table[row sep=\\] {{
{tbl("n4000_lo")}
    }};
    \addplot[stat221Blue!55,thin] table[row sep=\\] {{
{tbl("n4000_hi")}
    }};
  \end{{axis}}
\end{{tikzpicture}}
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex.strip() + "\n", encoding="utf-8")


def make_em_contraction_vs_snr_figure(out_path: Path) -> None:
    _configure_writable_caches()

    sigma = 1.0
    snr_grid = np.linspace(0.0, 3.0, 151)
    slopes = np.array(
        [
            abs(_population_em_map_derivative(float(mu), mu=float(mu), sigma=sigma))
            for mu in snr_grid
        ],
        dtype=float,
    )

    table = _tex_table_xy(snr_grid, slopes)

    mu_a = 0.6
    mu_b = 2.0
    slope_a = abs(_population_em_map_derivative(mu_a, mu=mu_a, sigma=sigma))
    slope_b = abs(_population_em_map_derivative(mu_b, mu=mu_b, sigma=sigma))

    tex = rf"""
\begin{{tikzpicture}}
  \begin{{axis}}[
    width=0.78\linewidth,
    height=5.8cm,
    xmin=0, xmax=3,
    ymin=0, ymax=1.02,
    axis lines=left,
    xlabel={{$\theta^\star/\sigma$}},
    ylabel={{local slope $|M'(\theta^\star)|$}},
    title={{Gaussian mixture: contractivity strengthens with separation}},
    title style={{font=\small, yshift=-0.35em}},
    clip=false,
  ]
    \addplot[stat221Blue,thick] table[row sep=\\] {{
{table}
    }};

    \addplot[only marks,mark=*,mark size=1.3pt,stat221Orange] coordinates {{
      ({mu_a:.6f},{slope_a:.12g})
      ({mu_b:.6f},{slope_b:.12g})
    }};
    \node[font=\small,anchor=south east] at (axis cs:{mu_a:.6f},{slope_a:.12g}) {{{mu_a:g}}};
    \node[font=\small,anchor=south west] at (axis cs:{mu_b:.6f},{slope_b:.12g}) {{{mu_b:g}}};
  \end{{axis}}
\end{{tikzpicture}}
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex.strip() + "\n", encoding="utf-8")


def make_em_missingness_contraction_figure(out_path: Path) -> None:
    _configure_writable_caches()

    rng = np.random.default_rng(7)

    d = 2
    sigma = 1.0
    theta_star = np.sqrt(2.0) * np.ones(d, dtype=float)  # ||theta*||=2.
    n_pop = 120_000
    h = 2e-3

    # Draw a single large population sample and reuse it across missingness
    # levels to keep the curve smooth (common random numbers).
    x = rng.standard_normal((n_pop, d))
    y = x @ theta_star + sigma * rng.standard_normal(n_pop)
    u = rng.random((n_pop, d))

    missing_grid = np.concatenate(
        [np.linspace(0.0, 0.9, 19), np.array([0.93, 0.95, 0.97, 0.98, 0.99], dtype=float)]
    )
    missing_grid = np.unique(np.sort(missing_grid))
    contractions: list[float] = []

    for miss_prob in missing_grid:
        obs_mask = u >= float(miss_prob)
        x_obs = x * obs_mask

        def M(theta: np.ndarray) -> np.ndarray:
            return _missing_covariates_regression_em_update(
                theta, y=y, x_obs=x_obs, obs_mask=obs_mask, sigma=sigma, ridge=1e-10
            )

        # Finite-difference Jacobian at theta_star.
        j = np.zeros((d, d), dtype=float)
        for k in range(d):
            e = np.zeros(d, dtype=float)
            e[k] = 1.0
            m_plus = M(theta_star + h * e)
            m_minus = M(theta_star - h * e)
            j[:, k] = (m_plus - m_minus) / (2.0 * h)

        svals = np.linalg.svd(j, compute_uv=False)
        contractions.append(float(svals[0]))

    table = _tex_table_xy(missing_grid, np.asarray(contractions, dtype=float))

    xmax = float(missing_grid.max())
    tex = rf"""
\begin{{tikzpicture}}
  \begin{{axis}}[
    width=0.78\linewidth,
    height=5.8cm,
    xmin=0, xmax={xmax:.2f},
    ymin=0, ymax=1.2,
    axis lines=left,
    xlabel={{missing probability $\rho$}},
    ylabel={{local linearization $\|DM(\theta^\star)\|_2$}},
    title={{Missing-covariates regression: contractivity degrades with missingness}},
    title style={{font=\small, yshift=-0.35em}},
    clip=false,
  ]
    \addplot[stat221Gray,densely dashed,thick,opacity=0.9] coordinates {{(0,1) ({xmax:.2f},1)}};
    \addplot[stat221Blue,thick] table[row sep=\\] {{
{table}
    }};
    \node[font=\small,stat221Gray,anchor=south west] at (axis cs:0.03,1.01) {{stability boundary}};
  \end{{axis}}
\end{{tikzpicture}}
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex.strip() + "\n", encoding="utf-8")


def make_em_missing_covariates_population_vs_sample_convergence_figure(out_path: Path) -> None:
    _configure_writable_caches()

    rng = np.random.default_rng(11)

    d = 2
    sigma = 1.0
    theta_star = np.sqrt(2.0) * np.ones(d, dtype=float)  # ||theta*||=2.
    rho = 0.95  # missing probability per coordinate

    theta0 = np.zeros(d, dtype=float)
    T = 30

    # Population trajectory via a large Monte Carlo approximation.
    n_pop = 220_000
    x_pop = rng.standard_normal((n_pop, d))
    y_pop = x_pop @ theta_star + sigma * rng.standard_normal(n_pop)
    mask_pop = rng.random((n_pop, d)) >= rho
    x_obs_pop = x_pop * mask_pop

    def M_pop(theta: np.ndarray) -> np.ndarray:
        return _missing_covariates_regression_em_update(
            theta, y=y_pop, x_obs=x_obs_pop, obs_mask=mask_pop, sigma=sigma, ridge=1e-10
        )

    theta = theta0.copy()
    pop_traj = [theta.copy()]
    for _ in range(T):
        theta = M_pop(theta)
        pop_traj.append(theta.copy())
    pop_traj = np.asarray(pop_traj, dtype=float)
    pop_dist = np.maximum(np.linalg.norm(pop_traj - theta_star[None, :], axis=1), 1e-10)

    it = np.arange(T + 1, dtype=int)

    curves: list[dict[str, object]] = [
        {"name": "pop", "table": _tex_table_xy(it, pop_dist)},
    ]

    # Sample trajectories (random), summarized by median + quantile band.
    n_list = [250, 1000, 4000]
    n_reps = 30

    for n, name in [(n_list[0], "n250"), (n_list[1], "n1000"), (n_list[2], "n4000")]:
        dists = np.zeros((n_reps, T + 1), dtype=float)
        for rep in range(n_reps):
            x = rng.standard_normal((n, d))
            y = x @ theta_star + sigma * rng.standard_normal(n)
            mask = rng.random((n, d)) >= rho
            x_obs = x * mask

            theta = theta0.copy()
            dists[rep, 0] = float(np.linalg.norm(theta - theta_star))
            for t in range(T):
                theta = _missing_covariates_regression_em_update(
                    theta, y=y, x_obs=x_obs, obs_mask=mask, sigma=sigma, ridge=1e-10
                )
                dists[rep, t + 1] = float(np.linalg.norm(theta - theta_star))

        dists = np.maximum(dists, 1e-10)
        med = np.median(dists, axis=0)
        lo = np.quantile(dists, 0.2, axis=0)
        hi = np.quantile(dists, 0.8, axis=0)
        curves.append({"name": name, "table": _tex_table_xy(it, med)})
        curves.append({"name": f"{name}_lo", "table": _tex_table_xy(it, lo)})
        curves.append({"name": f"{name}_hi", "table": _tex_table_xy(it, hi)})

    def tbl(key: str) -> str:
        for item in curves:
            if item["name"] == key:
                return str(item["table"])
        raise KeyError(key)

    tex = rf"""
\begin{{tikzpicture}}
  \begin{{axis}}[
    width=0.92\linewidth,
    height=6.2cm,
    xmin=0, xmax={T},
    ymode=log,
    axis lines=left,
    xlabel={{EM iteration $t$}},
    ylabel={{distance $\|\theta^{{(t)}}-\theta^\star\|_2$}},
    title={{Missing-covariates regression ($\rho={rho:g}$): geometric refinement until a sample-size floor}},
    title style={{font=\small, yshift=-0.35em}},
    legend style={{draw=stat221Gray!25, fill=white, fill opacity=0.95, text opacity=1}},
    legend pos=north east,
    clip=false,
  ]
    \addplot[stat221Gray,thick,densely dashed] table[row sep=\\] {{
{tbl("pop")}
    }};
    \addlegendentry{{Population EM}}

    % n=250
    \addplot[stat221Red,thick] table[row sep=\\] {{
{tbl("n250")}
    }};
    \addlegendentry{{$n=250$ (median)}}
    \addplot[stat221Red!50,thin] table[row sep=\\] {{
{tbl("n250_lo")}
    }};
    \addplot[stat221Red!50,thin] table[row sep=\\] {{
{tbl("n250_hi")}
    }};

    % n=1000
    \addplot[stat221Purple,thick] table[row sep=\\] {{
{tbl("n1000")}
    }};
    \addlegendentry{{$n=1000$ (median)}}
    \addplot[stat221Purple!55,thin] table[row sep=\\] {{
{tbl("n1000_lo")}
    }};
    \addplot[stat221Purple!55,thin] table[row sep=\\] {{
{tbl("n1000_hi")}
    }};

    % n=4000
    \addplot[stat221Blue,thick] table[row sep=\\] {{
{tbl("n4000")}
    }};
    \addlegendentry{{$n=4000$ (median)}}
    \addplot[stat221Blue!55,thin] table[row sep=\\] {{
{tbl("n4000_lo")}
    }};
    \addplot[stat221Blue!55,thin] table[row sep=\\] {{
{tbl("n4000_hi")}
    }};
  \end{{axis}}
\end{{tikzpicture}}
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex.strip() + "\n", encoding="utf-8")


def make_em_missing_covariates_1d_spurious_fixed_points_figure(out_path: Path) -> None:
    sigma = 1.0
    theta_star = 1.0
    rho = 0.95

    theta_grid = np.linspace(-2.5, 1.5, 401)
    m_vals = np.array(
        [
            _missing_covariates_population_map_1d(
                float(t), theta_star=theta_star, sigma=sigma, rho=rho
            )
            for t in theta_grid
        ],
        dtype=float,
    )

    # Find fixed points of f(theta)=M(theta)-theta on a fine grid.
    xs = np.linspace(float(theta_grid.min()), float(theta_grid.max()), 12001)
    f_vals = np.array(
        [
            _missing_covariates_population_map_1d(
                float(t), theta_star=theta_star, sigma=sigma, rho=rho
            )
            - float(t)
            for t in xs
        ],
        dtype=float,
    )
    roots: list[float] = []
    for i in range(xs.size - 1):
        if f_vals[i] == 0.0:
            roots.append(float(xs[i]))
            continue
        if f_vals[i] * f_vals[i + 1] < 0.0:
            a = float(xs[i])
            b = float(xs[i + 1])
            fa = float(f_vals[i])
            fb = float(f_vals[i + 1])
            for _ in range(70):
                m = 0.5 * (a + b)
                fm = _missing_covariates_population_map_1d(
                    m, theta_star=theta_star, sigma=sigma, rho=rho
                ) - m
                if fa * fm <= 0.0:
                    b, fb = m, float(fm)
                else:
                    a, fa = m, float(fm)
            roots.append(0.5 * (a + b))

    roots = sorted(roots)
    unique_roots: list[float] = []
    for r in roots:
        if not unique_roots or abs(r - unique_roots[-1]) > 5e-4:
            unique_roots.append(float(r))

    def slope(theta: float) -> float:
        h = 1e-6
        mp = _missing_covariates_population_map_1d(
            theta + h, theta_star=theta_star, sigma=sigma, rho=rho
        )
        mm = _missing_covariates_population_map_1d(
            theta - h, theta_star=theta_star, sigma=sigma, rho=rho
        )
        return float((mp - mm) / (2.0 * h))

    # Cobweb paths to visualize basins.
    def cobweb(theta0: float, *, n_steps: int = 7) -> str:
        theta = float(theta0)
        pts: list[tuple[float, float]] = [(theta, theta)]
        for _ in range(int(n_steps)):
            theta_next = _missing_covariates_population_map_1d(
                theta, theta_star=theta_star, sigma=sigma, rho=rho
            )
            pts.append((theta, theta_next))
            pts.append((theta_next, theta_next))
            theta = float(theta_next)
        xs_c = np.array([p[0] for p in pts], dtype=float)
        ys_c = np.array([p[1] for p in pts], dtype=float)
        return _tex_table_xy(xs_c, ys_c)

    cobweb_good = cobweb(0.5)
    cobweb_bad = cobweb(-2.0)

    # Mark the three roots (stable/unstable by slope).
    # We expect exactly three fixed points in this range for the chosen parameters.
    if len(unique_roots) != 3:
        raise RuntimeError(f"Expected 3 fixed points, found {len(unique_roots)}: {unique_roots}")
    r_bad, r_sep, r_star = unique_roots[0], unique_roots[1], unique_roots[2]
    slope_bad = abs(slope(r_bad))
    slope_sep = abs(slope(r_sep))
    slope_star = abs(slope(r_star))

    tex = rf"""
\begin{{tikzpicture}}
  \begin{{axis}}[
    width=0.82\linewidth,
    height=6.4cm,
    xmin=-2.5, xmax=1.5,
    ymin=-2.5, ymax=1.5,
    axis lines=left,
    xlabel={{$\theta$}},
    ylabel={{$M(\theta)$}},
    title={{Missing covariates (1D, $\rho={rho:g}$): a spurious stable fixed point}},
    title style={{font=\small, yshift=-0.35em}},
    clip=false,
  ]
    \addplot[stat221Blue,thick] table[row sep=\\] {{
{_tex_table_xy(theta_grid, m_vals)}
    }};
    \addplot[stat221Gray,densely dashed,thick,opacity=0.9] table[row sep=\\] {{
{_tex_table_xy(theta_grid, theta_grid)}
    }};

    \addplot[stat221Green,thick,mark=*,mark size=0.7pt,opacity=0.85] table[row sep=\\] {{
{cobweb_good}
    }};
    \addplot[stat221Orange,thick,mark=*,mark size=0.7pt,opacity=0.85] table[row sep=\\] {{
{cobweb_bad}
    }};

    % Fixed points.
    \addplot[only marks,mark=x,mark size=2.8pt,stat221Red,very thick] coordinates {{({r_star:.6f},{r_star:.6f})}};
    \addplot[only marks,mark=x,mark size=2.8pt,stat221Purple,very thick] coordinates {{({r_bad:.6f},{r_bad:.6f})}};
    \addplot[only marks,mark=o,mark size=2.0pt,stat221Gray,fill=white,very thick] coordinates {{({r_sep:.6f},{r_sep:.6f})}};

    \node[anchor=south west] at (axis cs:{r_star:.6f},{r_star:.6f}) {{$\theta^\star$}};
    \node[anchor=south east] at (axis cs:{r_bad:.6f},{r_bad:.6f}) {{$\bar\theta_{{\mathrm{{bad}}}}$}};
    \node[anchor=north east] at (axis cs:{r_sep:.6f},{r_sep:.6f}) {{$\bar\theta_{{\mathrm{{sep}}}}$}};

    \node[stat221Blue,anchor=south west,font=\small] at (axis cs:-2.35,1.15) {{$M(\theta)$}};
    \node[stat221Gray,anchor=south west,font=\small] at (axis cs:-2.35,0.95) {{$\theta$}};
    \node[stat221Green,anchor=south west,font=\small] at (axis cs:-2.35,0.75) {{cobweb from $\theta^{(0)}=0.5$}};
    \node[stat221Orange,anchor=south west,font=\small] at (axis cs:-2.35,0.55) {{cobweb from $\theta^{(0)}=-2$}};

    \node[anchor=south west,font=\small] at (axis cs:-2.35,0.30) {{$|M'(\theta^\star)|\approx {slope_star:.3f}$}};
    \node[anchor=south west,font=\small] at (axis cs:-2.35,0.10) {{$|M'(\bar\theta_{{\mathrm{{bad}}}})|\approx {slope_bad:.3f}$}};
    \node[anchor=south west,font=\small] at (axis cs:-2.35,-0.10) {{$|M'(\bar\theta_{{\mathrm{{sep}}}})|\approx {slope_sep:.3f}$}};
  \end{{axis}}
\end{{tikzpicture}}
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex.strip() + "\n", encoding="utf-8")


def make_gd_population_vs_sample_convergence_figure(out_path: Path) -> None:
    _configure_writable_caches()

    rng = np.random.default_rng(23)

    d = 2
    sigma = 1.0
    theta_star = np.sqrt(2.0) * np.ones(d, dtype=float)  # ||theta*||=2.

    theta0 = np.zeros(d, dtype=float)
    eta = 0.5
    T = 30

    it = np.arange(T + 1, dtype=int)

    # Population trajectory under the population risk:
    # f(θ) = 0.5 E[(Y - <X,θ>)^2] = 0.5||θ-θ*||^2 + const, so ∇f(θ)=θ-θ*.
    pop_dist = np.maximum((1.0 - eta) ** it * float(np.linalg.norm(theta_star - theta0)), 1e-10)
    curves: list[dict[str, object]] = [{"name": "pop", "table": _tex_table_xy(it, pop_dist)}]

    n_list = [250, 1000, 4000]
    n_reps = 30

    for n, name in [(n_list[0], "n250"), (n_list[1], "n1000"), (n_list[2], "n4000")]:
        dists = np.zeros((n_reps, T + 1), dtype=float)
        for rep in range(n_reps):
            x = rng.standard_normal((n, d))
            y = x @ theta_star + sigma * rng.standard_normal(n)

            theta = theta0.copy()
            dists[rep, 0] = float(np.linalg.norm(theta - theta_star))
            for t in range(T):
                grad = (x.T @ (x @ theta - y)) / float(n)
                theta = theta - eta * grad
                dists[rep, t + 1] = float(np.linalg.norm(theta - theta_star))

        dists = np.maximum(dists, 1e-10)
        med = np.median(dists, axis=0)
        lo = np.quantile(dists, 0.2, axis=0)
        hi = np.quantile(dists, 0.8, axis=0)
        curves.append({"name": name, "table": _tex_table_xy(it, med)})
        curves.append({"name": f"{name}_lo", "table": _tex_table_xy(it, lo)})
        curves.append({"name": f"{name}_hi", "table": _tex_table_xy(it, hi)})

    def tbl(key: str) -> str:
        for item in curves:
            if item["name"] == key:
                return str(item["table"])
        raise KeyError(key)

    tex = rf"""
\begin{{tikzpicture}}
  \begin{{axis}}[
    width=0.92\linewidth,
    height=6.2cm,
    xmin=0, xmax={T},
    ymode=log,
    axis lines=left,
    xlabel={{GD iteration $t$}},
    ylabel={{distance $\|\theta^{{(t)}}-\theta^\star\|_2$}},
    title={{Least squares (convex): geometric optimization until a statistical floor}},
    title style={{font=\small, yshift=-0.35em}},
    legend style={{draw=stat221Gray!25, fill=white, fill opacity=0.95, text opacity=1}},
    legend pos=north east,
    clip=false,
  ]
    \addplot[stat221Gray,thick,densely dashed] table[row sep=\\] {{
{tbl("pop")}
    }};
    \addlegendentry{{Population GD}}

    % n=250
    \addplot[stat221Red,thick] table[row sep=\\] {{
{tbl("n250")}
    }};
    \addlegendentry{{$n=250$ (median)}}
    \addplot[stat221Red!50,thin] table[row sep=\\] {{
{tbl("n250_lo")}
    }};
    \addplot[stat221Red!50,thin] table[row sep=\\] {{
{tbl("n250_hi")}
    }};

    % n=1000
    \addplot[stat221Purple,thick] table[row sep=\\] {{
{tbl("n1000")}
    }};
    \addlegendentry{{$n=1000$ (median)}}
    \addplot[stat221Purple!55,thin] table[row sep=\\] {{
{tbl("n1000_lo")}
    }};
    \addplot[stat221Purple!55,thin] table[row sep=\\] {{
{tbl("n1000_hi")}
    }};

    % n=4000
    \addplot[stat221Blue,thick] table[row sep=\\] {{
{tbl("n4000")}
    }};
    \addlegendentry{{$n=4000$ (median)}}
    \addplot[stat221Blue!55,thin] table[row sep=\\] {{
{tbl("n4000_lo")}
    }};
    \addplot[stat221Blue!55,thin] table[row sep=\\] {{
{tbl("n4000_hi")}
    }};
  \end{{axis}}
\end{{tikzpicture}}
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex.strip() + "\n", encoding="utf-8")


def main() -> None:
    repo = _repo_root()
    fig_root = repo / 'Lecture Tex' / 'figures'
    handout_dir = fig_root / 'handouts' / 'em_algorithm'
    main_note_dir = fig_root / 'main_notes' / 'augmented_state_optimization'

    # Shared figure used in the main notes and reused by the handout.
    make_em_operator_fixed_points_figure(main_note_dir / 'fig_em_operator_fixed_points.tex')

    # Figures used by `Lecture Tex/handouts/section 3/section_em_algorithm.tex`.
    make_em_contraction_vs_snr_figure(handout_dir / 'fig_em_contraction_vs_snr.tex')
    make_em_missingness_contraction_figure(handout_dir / 'fig_em_missingness_contraction.tex')
    make_em_missing_covariates_population_vs_sample_convergence_figure(
        handout_dir / 'fig_em_missing_covariates_population_vs_sample_convergence.tex'
    )
    make_em_missing_covariates_1d_spurious_fixed_points_figure(
        handout_dir / 'fig_em_missing_1d_spurious_fixed_points.tex'
    )
    make_gd_population_vs_sample_convergence_figure(
        handout_dir / 'fig_gd_population_vs_sample_convergence.tex'
    )
    make_em_population_vs_sample_convergence_figure(
        handout_dir / 'fig_em_population_vs_sample_convergence.tex'
    )
    print('Wrote main-note figure to:', main_note_dir)
    print('Wrote handout figures to:', handout_dir)


if __name__ == '__main__':
    main()
