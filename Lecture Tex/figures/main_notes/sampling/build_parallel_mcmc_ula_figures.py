#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np


def _write_table(path: Path, header: str, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(header.rstrip() + "\n")
        np.savetxt(f, data, fmt="%.10g")


# ---------------------------------------------------------------------
# 1D ULA example used for the figures
def _ula_1d_U(x: float) -> float:
    # U(x) = 0.5 x^2 + 0.1 x^4
    return 0.5 * (x**2) + 0.1 * (x**4)


def _ula_1d_gradU(x: float) -> float:
    # U(x) = 0.5 x^2 + 0.1 x^4
    return x + 0.4 * (x**3)


def _ula_1d_hessU(x: float) -> float:
    return 1.0 + 1.2 * (x**2)


def _ula_doublewell_gradU(x: float, a: float = 1.0) -> float:
    # Double-well: U(x) = 0.25 x^4 - 0.5 a x^2.
    return (x**3) - a * x


def _ula_doublewell_hessU(x: float, a: float = 1.0) -> float:
    return 3.0 * (x**2) - a


def _ula_rollout_1d(x0: float, eps: float, xi: np.ndarray, gradU=_ula_1d_gradU) -> np.ndarray:
    x = float(x0)
    out = np.empty_like(xi, dtype=float)
    for t in range(xi.size):
        x = x - eps * float(gradU(x)) + np.sqrt(2.0 * eps) * float(xi[t])
        out[t] = x
    return out


def build_ula_1d_energy_path_data(outdir: Path) -> None:
    rng = np.random.default_rng(0)

    x0 = 0.7
    eps = 0.12
    T = 80
    xi = rng.normal(size=T)

    x_seq = _ula_rollout_1d(x0, eps, xi)
    xs = np.concatenate([np.array([x0], dtype=float), x_seq])

    curve_x = np.linspace(-3.0, 3.0, 400, dtype=float)
    curve_u = 0.5 * (curve_x**2) + 0.1 * (curve_x**4)
    curve = np.column_stack([curve_x, curve_u])

    path_u = 0.5 * (xs**2) + 0.1 * (xs**4)
    path = np.column_stack([xs, path_u])

    _write_table(outdir / "data_parallel_mcmc_ula_1d_energy_curve.dat", "x U", curve)
    _write_table(outdir / "data_parallel_mcmc_ula_1d_energy_path.dat", "x U", path)
    _write_table(outdir / "data_parallel_mcmc_ula_1d_energy_start.dat", "x U", path[:1])
    _write_table(outdir / "data_parallel_mcmc_ula_1d_energy_end.dat", "x U", path[-1:])


def _newton_rollout_1d(
    *,
    x0: float,
    eps: float,
    xi: np.ndarray,
    x_init: np.ndarray,
    max_iters: int,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Returns:
      - x_seq: sequential rollout (length T)
      - xs: list of iterates x^{(k)} (each length T), starting with x_init
      - rnorms[k] = ||r(x^{(k)})||_2
      - errs[k] = ||x^{(k)} - x_seq||_2
    """
    T = xi.size

    def f(x: float, t: int) -> float:
        return x - eps * _ula_1d_gradU(x) + np.sqrt(2.0 * eps) * float(xi[t])

    x_seq = _ula_rollout_1d(x0, eps, xi)
    xk = x_init.astype(float).copy()

    xs: list[np.ndarray] = [xk.copy()]
    rnorms: list[float] = []
    errs: list[float] = []

    for _k in range(max_iters):
        r = np.empty(T, dtype=float)
        r[0] = xk[0] - f(x0, 0)
        for t in range(1, T):
            r[t] = xk[t] - f(float(xk[t - 1]), t)

        rnorms.append(float(np.linalg.norm(r)))
        errs.append(float(np.linalg.norm(xk - x_seq)))

        A = np.empty(T, dtype=float)
        A[0] = 0.0
        for t in range(1, T):
            A[t] = 1.0 - eps * _ula_1d_hessU(float(xk[t - 1]))

        dx = np.empty(T, dtype=float)
        dx[0] = -r[0]
        for t in range(1, T):
            dx[t] = A[t] * dx[t - 1] - r[t]

        xk = xk + dx
        xs.append(xk.copy())

        if rnorms[-1] < 1e-12:
            break

    return x_seq, xs, np.asarray(rnorms), np.asarray(errs)


def build_ula_newton_data(outdir: Path) -> None:
    rng = np.random.default_rng(0)

    x0 = 0.7
    eps = 0.12
    T = 80
    xi = rng.normal(size=T)

    x_init = np.full(T, x0, dtype=float)
    x_seq, xs, rnorms, errs = _newton_rollout_1d(x0=x0, eps=eps, xi=xi, x_init=x_init, max_iters=7)

    show_ks = [0, 1, 2, 3, min(4, len(xs) - 1)]

    timeseries = np.column_stack(
        [
            np.arange(1, T + 1, dtype=float),
            x_seq,
            xs[show_ks[0]],
            xs[show_ks[1]],
            xs[show_ks[2]],
            xs[show_ks[3]],
            xs[show_ks[4]],
        ]
    )
    _write_table(outdir / "data_parallel_mcmc_ula_timeseries.dat", "t seq k0 k1 k2 k3 k4", timeseries)

    metrics = np.column_stack([np.arange(rnorms.size, dtype=float), rnorms, errs])
    _write_table(outdir / "data_parallel_mcmc_ula_metrics.dat", "k rnorm err", metrics)


def _picard_rollout_1d(
    *,
    x0: float,
    eps: float,
    xi: np.ndarray,
    x_init: np.ndarray,
    omega: float,
    max_iters: int,
    tol: float,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Damped Picard iteration for the fixed-noise ULA rollout in 1D.

    Returns:
      - x_seq: sequential rollout (length T)
      - xs: list of iterates x^{(k)} (each length T), starting with x_init
      - rnorms[k] = ||r(x^{(k)})||_2
      - errs[k] = ||x^{(k)} - x_seq||_2
    """
    if not (0.0 < omega <= 1.0):
        raise ValueError("omega must be in (0, 1]")

    T = xi.size

    def f(x: float, t: int) -> float:
        return x - eps * _ula_1d_gradU(x) + np.sqrt(2.0 * eps) * float(xi[t])

    x_seq = _ula_rollout_1d(x0, eps, xi)
    xk = x_init.astype(float).copy()

    xs: list[np.ndarray] = [xk.copy()]
    rnorms: list[float] = []
    errs: list[float] = []

    for _k in range(max_iters):
        r = np.empty(T, dtype=float)
        r[0] = xk[0] - f(x0, 0)
        for t in range(1, T):
            r[t] = xk[t] - f(float(xk[t - 1]), t)

        rnorm = float(np.linalg.norm(r))
        rnorms.append(rnorm)
        errs.append(float(np.linalg.norm(xk - x_seq)))

        if rnorm < tol:
            break

        inc = np.empty(T, dtype=float)
        inc[0] = -eps * _ula_1d_gradU(float(x0)) + np.sqrt(2.0 * eps) * float(xi[0])
        for t in range(1, T):
            inc[t] = -eps * _ula_1d_gradU(float(xk[t - 1])) + np.sqrt(2.0 * eps) * float(xi[t])
        x_proposed = x0 + np.cumsum(inc)

        xk = (1.0 - omega) * xk + omega * x_proposed
        xs.append(xk.copy())

    return x_seq, xs, np.asarray(rnorms), np.asarray(errs)


def build_ula_picard_data(outdir: Path) -> None:
    rng = np.random.default_rng(0)

    x0 = 0.7
    eps = 0.12
    T = 80
    xi = rng.normal(size=T)

    x_init = np.full(T, x0, dtype=float)
    omega = 0.3
    x_seq, xs, rnorms, errs = _picard_rollout_1d(
        x0=x0, eps=eps, xi=xi, x_init=x_init, omega=omega, max_iters=120, tol=1e-12
    )

    show_ks = [0, 10, 20, min(40, len(xs) - 1)]
    timeseries = np.column_stack(
        [
            np.arange(1, T + 1, dtype=float),
            x_seq,
            xs[show_ks[0]],
            xs[show_ks[1]],
            xs[show_ks[2]],
            xs[show_ks[3]],
        ]
    )
    _write_table(outdir / "data_parallel_mcmc_ula_picard_timeseries.dat", "t seq k0 k10 k20 k40", timeseries)

    metrics = np.column_stack([np.arange(rnorms.size, dtype=float), rnorms, errs])
    _write_table(outdir / "data_parallel_mcmc_ula_picard_metrics.dat", "k rnorm err", metrics)


def _trajectory_merit_ula_1d(
    *,
    x0: float,
    eps: float,
    xi: np.ndarray,
    x: np.ndarray,
    gradU,
) -> float:
    """Return L(x)=0.5||r(x)||^2 for the fixed-noise 1D ULA rollout."""
    T = int(x.size)

    def f(z: float, t: int) -> float:
        return z - eps * float(gradU(z)) + np.sqrt(2.0 * eps) * float(xi[t])

    r0 = float(x[0]) - f(float(x0), 0)
    loss = 0.5 * (r0**2)
    for t in range(1, T):
        rt = float(x[t]) - f(float(x[t - 1]), t)
        loss += 0.5 * (rt**2)
    return loss


def _jtj_eigendirections_ula_1d(*, eps: float, x_star: np.ndarray, hessU) -> tuple[np.ndarray, np.ndarray]:
    """Return (v_min, v_max) eigen-directions of J(x_star)^T J(x_star) in 1D ULA."""
    T = int(x_star.size)
    a = np.empty(T, dtype=float)
    a[0] = 0.0
    for t in range(1, T):
        a[t] = 1.0 - eps * float(hessU(float(x_star[t - 1])))

    J = np.zeros((T, T), dtype=float)
    np.fill_diagonal(J, 1.0)
    for t in range(1, T):
        J[t, t - 1] = -a[t]

    jtj = J.T @ J
    _evals, evecs = np.linalg.eigh(jtj)
    v_min = evecs[:, 0].copy()
    v_max = evecs[:, -1].copy()

    # Normalize so that alpha/beta correspond to (roughly) max per-time deviation.
    v_min /= float(np.max(np.abs(v_min)))
    v_max /= float(np.max(np.abs(v_max)))
    return v_min, v_max


def build_trajectory_landscape_projection_data(outdir: Path) -> None:
    """Generate 2D slices of the trajectory-space merit function via J^T J eigen-directions."""
    rng = np.random.default_rng(0)

    alpha = np.linspace(-1.25, 1.25, 61, dtype=float)
    beta = np.linspace(-1.25, 1.25, 61, dtype=float)

    cases = [
        {
            "slug": "convex",
            "x0": 0.7,
            "eps": 0.12,
            "T": 60,
            "xi": rng.normal(size=60),
            "gradU": _ula_1d_gradU,
            "hessU": _ula_1d_hessU,
        },
        {
            "slug": "doublewell",
            "x0": 0.0,
            "eps": 0.30,
            "T": 60,
            "xi": np.zeros(60, dtype=float),
            "gradU": _ula_doublewell_gradU,
            "hessU": _ula_doublewell_hessU,
        },
    ]

    for case in cases:
        x0 = float(case["x0"])
        eps = float(case["eps"])
        T = int(case["T"])
        xi = np.asarray(case["xi"], dtype=float).reshape(T)
        gradU = case["gradU"]
        hessU = case["hessU"]

        x_star = _ula_rollout_1d(x0, eps, xi, gradU=gradU)
        v_min, v_max = _jtj_eigendirections_ula_1d(eps=eps, x_star=x_star, hessU=hessU)

        eps_vis = 1e-10

        rows: list[list[float]] = []
        for b in beta:
            for a in alpha:
                x_candidate = x_star + a * v_min + b * v_max
                loss = _trajectory_merit_ula_1d(x0=x0, eps=eps, xi=xi, x=x_candidate, gradU=gradU)
                rows.append([a, b, float(np.log10(loss + eps_vis))])

        data = np.asarray(rows, dtype=float)
        _write_table(
            outdir / f"data_parallel_mcmc_trajectory_landscape_projection_{case['slug']}.dat",
            "alpha beta logL",
            data,
        )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    fig_root = repo_root / "Lecture Tex" / "figures"
    main_note_dir = fig_root / "main_notes" / "sampling"
    handout_dir = fig_root / "handouts" / "parallel_mcmc"

    build_ula_1d_energy_path_data(main_note_dir)
    build_ula_newton_data(main_note_dir)
    build_ula_picard_data(handout_dir)
    build_trajectory_landscape_projection_data(handout_dir)

    print(f"Wrote shared sampling data to: {main_note_dir}")
    print(f"Wrote handout-only parallel MCMC data to: {handout_dir}")


if __name__ == "__main__":
    main()
