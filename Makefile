FIGURE_PYTHON ?= $(if $(wildcard .venv-figures/bin/python),.venv-figures/bin/python,python3)
LATEXMK ?= latexmk

.PHONY: figures-env figures notes notes-all handouts pset-solutions

figures-env:
	python3 -m venv .venv-figures
	.venv-figures/bin/pip install --upgrade pip
	.venv-figures/bin/pip install -r scripts/requirements-figures.txt

figures:
	$(FIGURE_PYTHON) 'Lecture Tex/figures/main_notes/build_main_note_figures.py'

notes:
	$(LATEXMK) -pdf -jobname='Lecture Notes' -interaction=nonstopmode -halt-on-error -cd 'Lecture Tex/main.tex'

notes-all: figures notes

handouts:
	python3 scripts/build_handouts.py

pset-solutions:
	python3 scripts/normalize_pset_solution_notebooks.py
