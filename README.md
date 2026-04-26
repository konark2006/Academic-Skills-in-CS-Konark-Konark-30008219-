# An Empirical Study of Double Descent in Polynomial Regression

This repository contains a short paper and the code that produced it. It was written for the *Academic Skills in Computer Science* course at Constructor University Bremen.

## Why I picked this topic

I chose this topic because I worked on something similar in my Machine Learning course this year. That class introduced me to ideas like overfitting, the bias–variance trade-off, and how a model's error changes as you make it more complex. Double descent is a surprising twist on that story, so it felt like a natural next step for me to explore and write about.

## What the paper is about

In machine learning, when you make a model more and more complex, you usually expect it to first get better and then get worse. The error curve looks like a U: it goes down, hits a sweet spot, and then climbs back up because the model starts memorising the training data instead of learning the pattern.

Around 2019, researchers noticed that this is not the full picture. If you keep adding even more parameters past the point where the model can perfectly fit the training data, something strange happens: the error suddenly drops again. The curve goes down, up, and then down a second time. This second dip is what people call **double descent**.

This paper shows that effect using one of the simplest models possible: fitting a polynomial to a small set of points. No deep learning, no fancy math — just a curve trying to pass through some dots.

## What is inside the paper

The paper walks through four small experiments:

1. **The classic U-shape** — the familiar story where a model with too few parameters underfits and one with too many starts to overfit.
2. **Double descent** — the moment the model has just enough parameters to perfectly fit the training data, the error spikes. Adding even more parameters surprisingly makes it better again.
3. **Why the spike happens** — the error is split into two parts (bias and variance) to show that both of them explode right at the spike.
4. **A simple fix** — adding a tiny bit of regularisation (a small penalty for using big numbers in the model) makes the spike disappear completely.

## Files in this repository

```
.
├── main.tex             # The paper, written in LaTeX
├── jmlr2e.sty           # The style file that controls how the paper looks
├── refs.bib             # The list of references used in the paper
├── main.pdf             # The finished paper (6 pages)
├── experiments.py       # Python code that creates all four figures
├── figures/             # The figures used in the paper
│   ├── fig_classical_ushape.pdf
│   ├── fig_double_descent.pdf
│   ├── fig_bias_variance_decomp.pdf
│   └── fig_ridge_regularization.pdf
├── requirements.txt     # The Python libraries you need (just NumPy and matplotlib)
├── LICENSE              # Licence info for the code and the paper
└── README.md            # The file you are reading right now
```

## How to run the code yourself

You only need Python and two small libraries. First, make a fresh environment and install them:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run the script that creates all the figures:

```bash
python3 experiments.py
```

The figures go into the `figures/` folder. The random numbers used in the script are fixed, so you should get exactly the same plots every time. On a normal laptop it takes less than a minute.

## How to rebuild the paper

You need a working LaTeX setup (TeX Live, MacTeX, or MiKTeX) with the usual packages: `natbib`, `geometry`, `fancyhdr`, `times`, `graphicx`, `booktabs`, `hyperref`, and `xcolor`. Then run:

```bash
pdflatex main.tex
bibtex   main
pdflatex main.tex
pdflatex main.tex
```

That gives you `main.pdf`, the same 6-page paper that is already included.

## Where to start reading

If you are just looking around the repo, the two files worth opening first are:

- [`main.pdf`](main.pdf) — the paper itself, which explains everything in detail.
- [`experiments.py`](experiments.py) — the Python code that actually produces the double-descent effect.

Everything else is supporting material.

## Licence

- The Python code (`experiments.py`) is released under the **MIT License** — see [`LICENSE`](LICENSE).
- The paper text and figures (`main.tex`, `refs.bib`, `main.pdf`, and everything inside `figures/`) are released under **CC-BY 4.0**.

## Author

**Konark**, Constructor University Bremen, 2026.
Contact: itsmekonark@gmail.com
