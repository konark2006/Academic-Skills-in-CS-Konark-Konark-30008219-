# An Empirical Study of Double Descent in Polynomial Regression

A short paper and the code that produced it. Written for the *Academic
Skills in Computer Science* course at Constructor University Bremen.

The paper reproduces the **double-descent** phenomenon
([Belkin et al., 2019](https://www.pnas.org/doi/10.1073/pnas.1903070116))
in roughly the simplest setting it is possible to demonstrate it: ordinary
least-squares polynomial regression on a one-dimensional toy problem.
Four experiments are included:

1. **Classical U-curve** — the textbook bias–variance trade-off in the
   under-parameterised regime.
2. **Double descent** — the test error spikes at the interpolation
   threshold `p = n` and then descends a second time.
3. **Bias–variance decomposition** — a Monte Carlo decomposition shows
   that *both* squared bias and variance blow up at the threshold.
4. **Ridge regularisation** — a small ridge penalty wipes out the spike.

## Repository layout

```
.
├── main.tex             # LaTeX source of the paper
├── jmlr2e.sty           # JMLR style file (re-implementation)
├── refs.bib             # Bibliography
├── main.pdf             # Compiled paper (6 pages, JMLR formatting)
├── experiments.py       # Python code that generates all four figures
├── figures/             # Compiled PDF figures used by main.tex
│   ├── fig_classical_ushape.pdf
│   ├── fig_double_descent.pdf
│   ├── fig_bias_variance_decomp.pdf
│   └── fig_ridge_regularization.pdf
├── requirements.txt     # Python dependencies (NumPy + matplotlib)
├── .gitignore           # Sensible Python + LaTeX ignore rules
├── LICENSE              # MIT for code, CC-BY 4.0 for the paper text
└── README.md            # This file
```

## Re-running the experiments

Set up a virtual environment and install the (very small) dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then regenerate every figure with a single command:

```bash
python3 experiments.py
```

All four PDF figures are written into `figures/`. The random seeds are
fixed inside the script, so the figures are bit-for-bit reproducible.
On a modern laptop the whole pipeline takes well under a minute.

## Re-compiling the paper

You need a TeX Live (or MacTeX / MiKTeX) installation with the standard
JMLR package set: `natbib`, `geometry`, `fancyhdr`, `times`, `graphicx`,
`booktabs`, `hyperref`, `xcolor`. Then:

```bash
pdflatex main.tex
bibtex   main
pdflatex main.tex
pdflatex main.tex
```

The result is `main.pdf` (6 pages, JMLR formatting).

## What's worth reading first

If you are skimming the repo, the two files to look at are
[`experiments.py`](experiments.py) (the actual code that produces
double descent) and [`main.pdf`](main.pdf) (the write-up). Everything
else is scaffolding.

## License

* The Python code (`experiments.py`) is released under the **MIT
  License** — see [`LICENSE`](LICENSE).
* The paper text and figures (`main.tex`, `refs.bib`, `main.pdf`, and
  the contents of `figures/`) are released under
  **CC-BY 4.0**.

## Author

**Konark**, Constructor University Bremen, 2026.
Contact: itsmekonark@gmail.com
