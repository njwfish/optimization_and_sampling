#!/usr/bin/env python3
"""
Regenerate the Python-generated assets used directly in the main notes.

Most main-note figures are authored as TikZ sources and are built as part of the
normal LaTeX compilation. This script handles the smaller generated subset:

- diffusion PDFs produced with Matplotlib;
- the EM fixed-point TikZ figure produced from Python;
- the ULA data tables consumed by the sampling TikZ figure.

Run:
  python3 "Lecture Tex/figures/main_notes/build_main_note_figures.py"
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/stat221_mplconfig")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/stat221_cache")

    diffusion = _load_module(
        "stat221_diffusion_figures",
        repo_root
        / "Lecture Tex"
        / "figures"
        / "main_notes"
        / "diffusion_models"
        / "make_diffusion_figures.py",
    )
    em = _load_module(
        "stat221_em_figures",
        repo_root
        / "Lecture Tex"
        / "figures"
        / "main_notes"
        / "augmented_state_optimization"
        / "make_em_figures.py",
    )
    parallel_mcmc = _load_module(
        "stat221_parallel_mcmc_figures",
        repo_root
        / "Lecture Tex"
        / "figures"
        / "main_notes"
        / "sampling"
        / "build_parallel_mcmc_ula_figures.py",
    )

    main_fig_dir = repo_root / "Lecture Tex" / "figures" / "main_notes"
    diffusion_dir = main_fig_dir / "diffusion_models"
    em_dir = main_fig_dir / "augmented_state_optimization"
    sampling_dir = main_fig_dir / "sampling"

    diffusion_dir.mkdir(parents=True, exist_ok=True)
    em_dir.mkdir(parents=True, exist_ok=True)
    sampling_dir.mkdir(parents=True, exist_ok=True)

    diffusion.fig_spatially_linear_bridge(
        out_path=diffusion_dir / "fig_diffusion_spatially_linear_bridge.pdf"
    )
    diffusion.fig_conditional_vs_marginal(
        out_path=diffusion_dir / "fig_diffusion_conditional_vs_marginal.pdf"
    )
    diffusion.fig_reverse_dynamics_trajectories(
        out_path=diffusion_dir / "fig_diffusion_reverse_dynamics.pdf"
    )
    em.make_em_operator_fixed_points_figure(
        em_dir / "fig_em_operator_fixed_points.tex"
    )
    parallel_mcmc.build_ula_1d_energy_path_data(sampling_dir)
    parallel_mcmc.build_ula_newton_data(sampling_dir)

    print("Regenerated main-note figure assets in:")
    print(f"  {diffusion_dir}")
    print(f"  {em_dir}")
    print(f"  {sampling_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
