"""
Grid search with k-fold cross-validation for the entity-resolution parameters.

Reproduces the parameter-selection procedure described in the paper: the grid
of (alpha, t) is evaluated with k-fold cross-validation, and the configuration
with the best mean validation F1 is selected. Cross-validation guards against
selecting parameters that only fit one particular split.

Run on the synthetic demonstration data, the absolute scores reflect that data,
not the paper's original results; the *procedure* is what the package
reproduces.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Set, Tuple

import pandas as pd
from sklearn.model_selection import KFold

from .module1_normalization import DataNormalizer
from .module2_entity_resolution import EntityResolver
from .evaluate import load_sources


def _score(records, truth_pairs: Set[Tuple[int, int]], alpha, t) -> Dict:
    resolver = EntityResolver(alpha=alpha, threshold=t, key_field="name",
                              use_blocking=True)
    resolver.resolve(records)
    predicted = {tuple(sorted(p)) for p in resolver.matched_pairs()}
    tp = len(predicted & truth_pairs)
    fp = len(predicted - truth_pairs)
    fn = len(truth_pairs - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def cross_validated_grid(outdir: str, rules: str, alphas: List[float],
                         thresholds: List[float], k: int = 5, seed: int = 42):
    records, truth = load_sources(outdir)
    norm = DataNormalizer(rules)
    normalized = norm.records(norm.normalize(pd.DataFrame(records)))
    normalized = [{kk: v for kk, v in r.items() if kk != "source"}
                  for r in normalized]

    n = len(normalized)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    gold_all = {tuple(sorted(p)) for p in truth}

    results = []
    for alpha in alphas:
        for t in thresholds:
            fold_f1 = []
            for _, test_idx in kf.split(range(n)):
                idx_set = set(int(i) for i in test_idx)
                fold_records = [normalized[i] for i in sorted(idx_set)]
                # Remap gold pairs to the fold's local indexing.
                remap = {orig: local for local, orig in enumerate(sorted(idx_set))}
                fold_gold = {
                    tuple(sorted((remap[a], remap[b])))
                    for (a, b) in gold_all if a in idx_set and b in idx_set
                }
                if not fold_gold:
                    continue
                res = _score(fold_records, fold_gold, alpha, t)
                fold_f1.append(res["f1"])
            mean_f1 = sum(fold_f1) / len(fold_f1) if fold_f1 else 0.0
            results.append({"alpha": alpha, "threshold": t,
                            "cv_mean_f1": round(mean_f1, 4),
                            "n_folds_scored": len(fold_f1)})
    results.sort(key=lambda r: r["cv_mean_f1"], reverse=True)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="data/synthetic")
    p.add_argument("--rules", default="config/normalization_rules.yaml")
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()

    # Grid ranges as reported in the paper (Section IV-C).
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresholds = [0.70, 0.75, 0.80, 0.82, 0.85, 0.90, 0.95]
    rows = cross_validated_grid(args.outdir, args.rules, alphas, thresholds,
                                k=args.folds)
    print(f"{args.folds}-fold cross-validated grid search "
          f"(synthetic demonstration data):")
    print(f"{'alpha':>6}{'t':>7}{'cv_mean_f1':>12}")
    for r in rows:
        print(f"{r['alpha']:>6}{r['threshold']:>7}{r['cv_mean_f1']:>12}")
    best = rows[0]
    print(f"\nBest by CV mean F1: alpha={best['alpha']}, t={best['threshold']}, "
          f"F1={best['cv_mean_f1']}")
    print("Note: scores characterize the synthetic data, not the paper.")
