# FPL AI Thesis — LaTeX Project

## Overview

This directory contains the complete LaTeX source for the bachelor thesis:

**MK**: "Предиктивен систем за управување со тим во Фантази Премиер Лига базиран на хибридна вештачка интелигенција"

**EN**: "A Hybrid Artificial Intelligence System for Predictive Team Management in Fantasy Premier League"

**Student**: Андреј Андоновски (Andrej Andonovski), Index 196042  
**Faculty**: ФИНКИ — Факултет за информатички науки и компјутерско инженерство  
**University**: Универзитет „Св. Кирил и Методиј" во Скопје  
**Year**: 2026

---

## Project Structure

```
thesis/
├── main_mk.tex          # Root file — Macedonian version
├── main_en.tex          # Root file — English version
├── references.bib       # Shared BibTeX bibliography
├── figures/             # Place all figures/plots here
├── chapters_mk/         # Macedonian chapter files
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_methodology.tex
│   ├── 04_results.tex
│   ├── 05_future_work.tex
│   └── 06_conclusion.tex
└── chapters_en/         # English chapter files
    ├── 00_abstract.tex
    ├── 01_introduction.tex
    ├── 02_related_work.tex
    ├── 03_methodology.tex
    ├── 04_results.tex
    ├── 05_future_work.tex
    └── 06_conclusion.tex
```

---

## How to Compile on Overleaf (Recommended)

### Option A: Import from GitHub

1. Push this repository to GitHub (e.g., `github.com/yourname/fpl-ai-thesis`).
2. Go to [overleaf.com](https://www.overleaf.com) and log in.
3. Click **New Project** → **Import from GitHub**.
4. Select your repository.
5. Overleaf will clone the repo automatically.
6. In the project settings, set the **Main document** to either:
   - `main_mk.tex` for the Macedonian version, or
   - `main_en.tex` for the English version.
7. Set **Compiler** to `pdfLaTeX`.
8. Set **Bibliography tool** to `biber`.
9. Click **Recompile**.

### Option B: Upload ZIP

1. Zip the entire `thesis/` directory.
2. On Overleaf: **New Project** → **Upload Project** → upload the zip.
3. Set main document and compiler as above.

---

## Compiler Settings

| Setting | Value |
|---|---|
| Compiler | pdfLaTeX |
| Bibliography | biber |
| TeX Live version | 2023 or later |
| Main document (MK) | `main_mk.tex` |
| Main document (EN) | `main_en.tex` |

---

## Adding Figures

Place all figure files (PDF, PNG, JPG) in the `figures/` directory.  
Reference them in LaTeX with:

```latex
\includegraphics[width=0.8\textwidth]{figures/your_figure.pdf}
```

The `figures/` directory currently contains a `.gitkeep` placeholder.  
Replace placeholder `\fbox{...}` figure environments with real `\includegraphics` calls once plots are generated.

---

## Local Compilation

If compiling locally, run in order:

```bash
pdflatex main_mk.tex
biber main_mk
pdflatex main_mk.tex
pdflatex main_mk.tex
```

Or use `latexmk`:

```bash
latexmk -pdf -pdflatex="pdflatex" main_mk.tex
```

---

## Notes

- Both `main_mk.tex` and `main_en.tex` share `references.bib`.
- The bibliography uses `biblatex` with `backend=biber` and `style=numeric`.
- Macedonian text requires the `babel` package with the `macedonian` option.
- Unicode characters in `.tex` files require `inputenc` with `utf8`.
