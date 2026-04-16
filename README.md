# Optimization and Sampling Course Notes

This repository contains the LaTeX source for the course notes, standalone
section handouts, and the figure assets needed to build them.

The tracked material is intentionally limited to the note-writing tree:

- `Lecture Tex/` for the main notes, handouts, figures, bibliography, and shared preamble,
- `scripts/build_handouts.py` for rebuilding the titled handout PDFs,
- `scripts/requirements-figures.txt` for the Python figure environment,
- `Makefile` and `ORGANIZATION.md` for the build entrypoints and source-tree map.

Problem sets, handwritten lecture packets, local reference libraries, extracted
reference text, and private workflow notes are left out of the Git repository.

## Build

From the repository root:

```sh
make figures-env
make figures
make notes
```

That creates the local figure environment, regenerates Python-backed figure
assets, and compiles the main notes.

To build the standalone handouts:

```sh
make handouts
```

For the detailed source-tree map, see [`ORGANIZATION.md`](ORGANIZATION.md). For
the LaTeX-specific build notes, see [`Lecture Tex/README.md`](Lecture%20Tex/README.md).
