#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HandoutBuild:
    section: int
    tex_relpath: str
    titled_stem: str

    @property
    def tex_path(self) -> Path:
        return Path(self.tex_relpath)

    @property
    def handout_dir(self) -> Path:
        return self.tex_path.parent

    @property
    def tex_name(self) -> str:
        return self.tex_path.name

    @property
    def stem(self) -> str:
        return self.tex_path.stem

    @property
    def student_jobname(self) -> str:
        return self.titled_stem

    @property
    def student_pdf_name(self) -> str:
        return f"{self.student_jobname}.pdf"

    @property
    def solutions_jobname(self) -> str:
        return f"Solutions - {self.titled_stem}"

    @property
    def solutions_pdf_name(self) -> str:
        return f"{self.solutions_jobname}.pdf"

    @property
    def legacy_build_basenames(self) -> tuple[str, ...]:
        return (
            self.stem,
            f"{self.stem}_solutions",
            f"{self.stem}_student",
        )


HANDOUTS: tuple[HandoutBuild, ...] = (
    HandoutBuild(
        section=1,
        tex_relpath="Lecture Tex/handouts/section 1/section_convexity_optimization.tex",
        titled_stem="When does nonconvexity make optimization hard",
    ),
    HandoutBuild(
        section=2,
        tex_relpath="Lecture Tex/handouts/section 2/section_parallel_mcmc.tex",
        titled_stem="Parallel-in-time evaluation of MCMC sample paths",
    ),
    HandoutBuild(
        section=3,
        tex_relpath="Lecture Tex/handouts/section 3/section_em_algorithm.tex",
        titled_stem="Why is EM hard to analyze",
    ),
    HandoutBuild(
        section=4,
        tex_relpath="Lecture Tex/handouts/section 4/section_hmc_geometry.tex",
        titled_stem="How Do Entropy and Transport Control Langevin Sampling",
    ),
    HandoutBuild(
        section=5,
        tex_relpath="Lecture Tex/handouts/section 5/section_elbo_geometry.tex",
        titled_stem="What makes variational inference fail",
    ),
    HandoutBuild(
        section=6,
        tex_relpath="Lecture Tex/handouts/section 6/section_discrete_diffusion.tex",
        titled_stem="Discrete diffusion as posterior learning",
    ),
)


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _cleanup_legacy_outputs(handout_dir: Path, handout: HandoutBuild) -> None:
    suffixes = (
        ".aux",
        ".bbl",
        ".blg",
        ".fdb_latexmk",
        ".fls",
        ".log",
        ".out",
        ".pdf",
        ".synctex.gz",
    )
    for basename in handout.legacy_build_basenames:
        for suffix in suffixes:
            _remove_if_exists(handout_dir / f"{basename}{suffix}")


def _build_handout(repo_root: Path, handout: HandoutBuild) -> None:
    handout_dir = repo_root / handout.handout_dir

    print(f"[section {handout.section}] building titled student PDF")
    _run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-jobname={handout.student_jobname}",
            handout.tex_name,
        ],
        cwd=handout_dir,
    )

    print(f"[section {handout.section}] building titled solutions PDF")
    _run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-jobname={handout.solutions_jobname}",
            r'-pdflatex=pdflatex %O "\def\STATshowsolutions{1}\input{%S}"',
            handout.tex_name,
        ],
        cwd=handout_dir,
    )

    student_pdf = handout_dir / handout.student_pdf_name
    solutions_pdf = handout_dir / handout.solutions_pdf_name

    print(f"[section {handout.section}] removing legacy section_* build products")
    _cleanup_legacy_outputs(handout_dir, handout)

    print(f"  built {student_pdf.name}")
    print(f"  built {solutions_pdf.name}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Stat 221 standalone handouts as titled student and solutions PDFs."
        )
    )
    parser.add_argument(
        "--section",
        type=int,
        action="append",
        choices=[handout.section for handout in HANDOUTS],
        help="Build only the specified section number. Repeat to build multiple sections.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    requested = set(args.section or [])

    for handout in HANDOUTS:
        if requested and handout.section not in requested:
            continue
        _build_handout(repo_root, handout)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
