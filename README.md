# Stat 221: Optimization and Sampling

This repository contains the user-facing course materials for Stat 221: the
typed lecture notes, the standalone handouts, the LaTeX source that builds
them, and the figure assets they depend on.

The main document is the full set of notes:

- [Lecture Tex/Lecture Notes.pdf](Lecture%20Tex/Lecture%20Notes.pdf)

Those notes are organized by topic rather than by lecture date. The handouts in
`Lecture Tex/handouts/` are shorter standalone documents built around one
question or one technique. Each handout is tracked in both a student version and
a solutions version.

This Git repository intentionally excludes problem sets, handwritten lecture
packets, local reference libraries, extracted reference text, and internal
workflow notes. It is meant to hold the material a reader of the course would
actually use.

## What the Course Covers

The course follows one running theme: local geometric information is often
enough to design algorithms, but the meaning of that local information changes
as we move from optimization to sampling, then to augmented-state methods,
variational inference, and diffusion models.

The notes are divided into three parts and six main units:

1. **Local methods**: Optimization covers gradients, Hessians, conditioning,
   line search, Newton methods, conjugate gradients, stochastic gradients, and
   constrained optimization. Sampling covers Monte Carlo, Markov kernels,
   Metropolis-Hastings, Gibbs, diagnostics, and gradient-based MCMC.
2. **Augmented-state methods**: Augmented-state optimization covers momentum
   methods, simulated annealing, and expectation-maximization. Augmented-state
   sampling covers Hamiltonian Monte Carlo, NUTS, tempering, and sequential
   Monte Carlo.
3. **Learned maps**: Variational inference covers the ELBO, mean-field
   approximations, stochastic optimization, and amortized inference. Diffusion
   models cover bridges, reverse-time dynamics, regression views of score
   learning, and corrected or guided samplers.

In the source tree, these live under `Lecture Tex/chapters/`. A few numbered
chapter folders are supporting bundles inside a larger unit rather than separate
top-level sections, so the repository layout is a little finer-grained than the
compiled table of contents.

## Notes and Handouts

The main notes are designed to be read as one coherent document. They collect
the course's core definitions, propositions, proofs, figures, and references in
one place.

The handouts are narrower. Each one takes a single question from the course and
pushes it further, usually by isolating one phenomenon, one proof strategy, or
one computational viewpoint. They are meant to be readable on their own while
still connecting back to the main notes.

The current handouts are:

- **Optimization**:
  [When does nonconvexity make optimization hard?](Lecture%20Tex/handouts/section%201/When%20does%20nonconvexity%20make%20optimization%20hard.pdf)
  and its
  [solutions](Lecture%20Tex/handouts/section%201/Solutions%20-%20When%20does%20nonconvexity%20make%20optimization%20hard.pdf).
  This handout separates basin structure from local geometry and uses concrete
  nonconvex examples to show when local methods remain reliable.
- **Sampling**:
  [Parallel-in-time evaluation of MCMC sample paths](Lecture%20Tex/handouts/section%202/Parallel-in-time%20evaluation%20of%20MCMC%20sample%20paths.pdf)
  and its
  [solutions](Lecture%20Tex/handouts/section%202/Solutions%20-%20Parallel-in-time%20evaluation%20of%20MCMC%20sample%20paths.pdf).
  This handout freezes the randomness in a sampler and studies trajectory-space
  fixed-point and Gauss-Newton viewpoints for parallel rollout.
- **Augmented-state optimization**:
  [Why is EM hard to analyze?](Lecture%20Tex/handouts/section%203/Why%20is%20EM%20hard%20to%20analyze.pdf)
  and its
  [solutions](Lecture%20Tex/handouts/section%203/Solutions%20-%20Why%20is%20EM%20hard%20to%20analyze.pdf).
  This handout treats EM through both free-energy monotonicity and the geometry
  of the induced update map.
- **Langevin geometry**:
  [How Do Entropy and Transport Control Langevin Sampling](Lecture%20Tex/handouts/section%204/How%20Do%20Entropy%20and%20Transport%20Control%20Langevin%20Sampling.pdf)
  and its
  [solutions](Lecture%20Tex/handouts/section%204/Solutions%20-%20How%20Do%20Entropy%20and%20Transport%20Control%20Langevin%20Sampling.pdf).
  This handout studies free-energy dissipation, transport inequalities, and
  preconditioning for Langevin convergence.
- **Variational inference**:
  [What makes variational inference fail?](Lecture%20Tex/handouts/section%205/What%20makes%20variational%20inference%20fail.pdf)
  and its
  [solutions](Lecture%20Tex/handouts/section%205/Solutions%20-%20What%20makes%20variational%20inference%20fail.pdf).
  This handout focuses on approximation error, amortization, under-coverage,
  blur, and posterior collapse.
- **Diffusion models**:
  [Discrete diffusion as posterior learning](Lecture%20Tex/handouts/section%206/Discrete%20diffusion%20as%20posterior%20learning.pdf)
  and its
  [solutions](Lecture%20Tex/handouts/section%206/Solutions%20-%20Discrete%20diffusion%20as%20posterior%20learning.pdf).
  This handout develops the bridge-first diffusion viewpoint on finite state
  spaces and connects reverse sampling to posterior learning.

## Repository Layout

- `Lecture Tex/main.tex`: entrypoint for the full notes.
- `Lecture Tex/Lecture Notes.pdf`: compiled main notes PDF.
- `Lecture Tex/preamble.tex`: shared macros, theorem styles, and layout.
- `Lecture Tex/chapters/`: source for the six main units plus supporting
  chapter bundles.
- `Lecture Tex/figures/main_notes/`: figure assets used in the full notes.
- `Lecture Tex/figures/handouts/`: handout-only figures and generated assets.
- `Lecture Tex/handouts/`: standalone handout source and tracked PDFs.
- `Lecture Tex/references.bib`: bibliography shared by the notes and handouts.
- `Makefile`: standard build entrypoints for figures, notes, and handouts.

## Build

From the repository root:

```sh
make figures-env
make figures
make notes
```

This creates the local Python environment for generated figures, rebuilds the
main-note figure assets, and compiles the full notes to
`Lecture Tex/Lecture Notes.pdf`.

To build the standalone handouts:

```sh
make handouts
```
