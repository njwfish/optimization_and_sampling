# Stat 221 Lecture Notes (LaTeX)

This directory contains the main course notes plus six standalone section
handouts. The main notes and the handouts share the same preamble, notation, and
figure styles, but the handouts are compiled separately.

## Main Notes

From the repo root, the standard build path is:

```sh
make figures-env
make figures
make notes
```

This does three things:

1. creates the local `.venv-figures` environment for Python-generated figures,
2. regenerates the Python-backed figure assets used by the main notes,
3. compiles `Lecture Tex/main.tex`.

The direct `latexmk` command is:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd 'Lecture Tex/main.tex'
```

Output:

- `Lecture Tex/main.pdf`

To run the full main-note pipeline in one command:

```sh
make notes-all
```

## Handouts

The handouts live in:

- `Lecture Tex/handouts/section 1/section_convexity_optimization.tex`
- `Lecture Tex/handouts/section 2/section_parallel_mcmc.tex`
- `Lecture Tex/handouts/section 3/section_em_algorithm.tex`
- `Lecture Tex/handouts/section 4/section_hmc_geometry.tex`
- `Lecture Tex/handouts/section 5/section_elbo_geometry.tex`
- `Lecture Tex/handouts/section 6/section_discrete_diffusion.tex`

Each handout is a standalone document. The standard handout build compiles both
the titled student version and the titled solutions version directly, then
clears the legacy `section_*` build outputs in each section directory.

### Standard Handout Build

From the repo root:

```sh
make handouts
```

or equivalently:

```sh
python3 scripts/build_handouts.py
```

This produces, for every section:

- the titled student PDF `<Title>.pdf`,
- the titled solutions PDF `Solutions - <Title>.pdf`.

For example, Section 3 is refreshed as:

- `Lecture Tex/handouts/section 3/Why is EM hard to analyze.pdf`
- `Lecture Tex/handouts/section 3/Solutions - Why is EM hard to analyze.pdf`

To rebuild only one section, pass `--section`:

```sh
python3 scripts/build_handouts.py --section 3
```

The script is the preferred path because it makes the titled PDFs the real build
targets and clears the old `section_*` products. Direct `latexmk` commands still
work, but if you use them manually you should set `-jobname` to the titled
output you want.

## Figures

Figure assets are organized in two parallel trees:

- `Lecture Tex/figures/main_notes/<unit>/` for assets used by the main notes,
- `Lecture Tex/figures/handouts/<handout_slug>/` for handout-only assets.

Shared assets that are reused by both the main notes and a handout should stay
under `main_notes/<unit>/` and be referenced from the handout, rather than
duplicated.

More detail on figure organization and generator scripts lives in
[`figures/README.md`](figures/README.md).

## Clean Build Expectations

Before finishing note edits:

- the target document should compile with `latexmk -pdf -interaction=nonstopmode -halt-on-error`,
- the corresponding `.log` file should be free of errors and, as much as
  practical, free of warnings.
