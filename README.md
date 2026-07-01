# Semantic Integration Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.PLACEHOLDER.svg)](https://doi.org/10.5281/zenodo.PLACEHOLDER)

An interpretable, auditable pipeline for enterprise data integration. The
framework moves heterogeneous source records through four modules —
transparent normalization, hybrid entity resolution, lightweight knowledge
graph construction, and governed LLM annotation — while committing a complete
provenance trace at every step.

This repository implements the framework described in the paper *Bridging the
Data-Knowledge Gap in Enterprises: A Design Science Approach to Interpretable
Semantic Integration* (IEEE CBI 2026).

## Design principles, made concrete

The paper derives three design principles from field work. Each is visible in
the code, not just asserted:

- **Design transparency.** Every transformation rule lives in a versioned YAML
  file (`config/`), and every application of a rule is written to a provenance
  log (`src/provenance.py`). The output is not just a cleaner table; it is an
  auditable transformation that can be traced back to the rule, version, and
  evidence that produced it.
- **Modular adaptability.** The four modules have clean interfaces and share a
  single provenance log. A module can be replaced without touching the others.
- **Human-centered interpretability.** Module 4 turns the graph into
  natural-language descriptions, governed by three interception mechanisms so
  that output reaching a user is fact-checked, confidence-scored, and routed to
  human review when sensitive or uncertain.

## Architecture

```
raw records
  -> Module 1  src/module1_normalization.py    schema, types, imputation, noise
  -> Module 2  src/module2_entity_resolution.py hybrid similarity, consolidation
  -> Module 3  src/module3_knowledge_graph.py   typed RDF-style graph + exports
  -> Module 4  src/module4_annotation.py        LLM annotation + 3 controls
  =  knowledge graph + full provenance trace
```

Entity resolution combines two similarity signals:

```
S_hybrid(a, b) = alpha * S_lexical(a, b) + (1 - alpha) * S_semantic(a, b)
```

A pair is a match when `S_hybrid(a, b) >= t`. The defaults `alpha = 0.6` and
`t = 0.82` are the values reported in the paper. The parameter-selection
procedure (grid search with k-fold cross-validation) is reproduced in
`src/grid_search.py`; run on the demonstration data it selects the best
operating point for that data, which differs from the paper's original data.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+ is recommended. The pipeline runs end to end without any external
API. Live GPT-4 annotation is optional and activates only if `OPENAI_API_KEY`
is set; otherwise a deterministic, fact-consistent renderer is used and the
governance controls behave identically.

## Quick start

```bash
# Generate synthetic demonstration data (4 sources, injected failure modes)
python data/synthetic/generate_synthetic.py --entities 200 --seed 42

# To approximate the dataset scale reported in the paper (~8,440 records):
python data/synthetic/generate_synthetic.py --entities 4500 --seed 42

# Run the full pipeline on a small example
python -m src.pipeline

# Evaluate entity resolution against the synthetic ground truth
python -m src.evaluate

# Run the test suite
python -m pytest tests/
```

## Datasets and reproducibility

This package implements the method and is executable on:

- **A synthetic demonstration dataset**, produced by
  `data/synthetic/generate_synthetic.py`. It spans four source systems (ERP, CRM, API, and a legacy flat-file export)
  and injects the failure modes the paper analyzes (schema heterogeneity,
  abbreviations, missing values, light typos, country-code inconsistency). It is a structurally equivalent
  demonstration produced entirely by the provided script. The figures it yields
  characterize this demonstration data and are not the paper's reported
  results.
- **The public DBLP-ACM benchmark**, which can be downloaded from its original
  sources (see `data/dblp_acm/README.md`).

**What this package does not reproduce.** The enterprise pilot data and the
practitioner evaluations reported in the paper (interview transcripts, trust
and interpretability ratings) were collected under non-disclosure agreements
and involve human assessments. They cannot be redistributed or regenerated
computationally, and are therefore not included.

**On the numbers.** Figures produced by `src/evaluate.py` characterize the
synthetic demonstration data and the chosen generator settings. They are not
the paper's reported results and will differ from them. What transfers is the
*method* and the *grid-search procedure* for selecting `alpha` and `t`, not the
absolute scores, which depend on the original data.

## Blocking and recall

Entity resolution uses character n-gram blocking by default for scalability.
For full recall on small inputs or heavy abbreviation, set `use_blocking=False`
in `EntityResolver`.

## Repository layout

```
config/      versioned YAML rules and graph schema
src/         the four modules, provenance, pipeline, evaluation
data/        synthetic generator and DBLP-ACM instructions
prompts/     the annotation prompt template
tests/       unit tests for all modules
docs/        methodology mapping and ethical statement
examples/    example outputs (graph exports, provenance trace)
```

## Citation

If you use this software, please cite both the software and the paper.

For the software, see `CITATION.cff` (GitHub will render a "Cite this
repository" button using it). For the paper:

> A. Labhih, M. Hafsi, A. Toumi, and S. Fosso-Wamba, "Bridging the
> Data-Knowledge Gap in Enterprises: A Design Science Approach to
> Interpretable Semantic Integration," in *Proc. IEEE International
> Conference on Business Informatics (CBI)*, 2026.

The DOI for this software (issued by Zenodo upon release) is available in
`CITATION.cff` and at the top of this README.

## License

MIT. See `LICENSE`.
