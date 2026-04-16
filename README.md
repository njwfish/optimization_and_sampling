# Optimization and Sampling Course Notes

This repository contains the LaTeX source for the course notes, standalone
section handouts, and the figure assets needed to build them.

The tracked material is intentionally limited to the note-writing tree:

- `Lecture Tex/` for the main notes, handouts, figures, bibliography, and shared preamble,
- `scripts/build_handouts.py` for rebuilding the titled handout PDFs,
- `scripts/requirements-figures.txt` for the Python figure environment,
- `Makefile` for the main build entrypoints.

Problem sets, handwritten lecture packets, local reference libraries, extracted
reference text, and private workflow notes are left out of the Git repository.

## Course Structure

The notes are organized by course units rather than by lecture date. Roughly,
the course moves through three blocks:

1. Local methods:
   optimization and sampling.
2. Augmented-state methods:
   momentum, annealing, EM, HMC, NUTS, tempering, and sequential Monte Carlo.
3. Learned maps:
   variational inference and diffusion models.

The six top-level note units are:

- `Optimization`
- `Sampling`
- `Augmented-state optimization`
- `Augmented-state sampling`
- `Variational inference`
- `Diffusion models`

Within the source tree, those units live under `Lecture Tex/chapters/`, with a
small number of supporting chapter bundles for material that is logically nested
inside a larger unit.

## Repo Layout

- `Lecture Tex/main.tex`: entrypoint for the full notes.
- `Lecture Tex/preamble.tex`: shared macros and styles.
- `Lecture Tex/chapters/`: main course units and supporting chapter bundles.
- `Lecture Tex/figures/main_notes/`: assets used by the full notes.
- `Lecture Tex/figures/handouts/`: handout-only assets.
- `Lecture Tex/handouts/`: standalone section handouts.
- `Lecture Tex/references.bib`: bibliography for the notes.

## Build

From the repository root:

```sh
make figures-env
make figures
make notes
```

That creates the local figure environment, regenerates Python-backed figure
assets, and compiles the main notes to `Lecture Tex/Lecture Notes.pdf`.

To build the standalone handouts:

```sh
make handouts
```
