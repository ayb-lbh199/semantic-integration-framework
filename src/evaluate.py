"""
Evaluation script.

Runs entity resolution on the synthetic demonstration data and computes
precision, recall, and F1 against the known ground-truth duplicate links.

This demonstrates that the method works. It does not claim to reproduce the
paper's exact figures: those were obtained on the enterprise study data, which
is governed by confidentiality agreements, and on human practitioner
evaluations. The numbers here depend on the synthetic generator's settings and
will differ from the paper. That is expected and disclosed.
"""

from __future__ import annotations

import argparse
import csv
from typing import Dict, List, Set, Tuple

import pandas as pd

from .module1_normalization import DataNormalizer
from .module2_entity_resolution import EntityResolver


def load_sources(outdir: str) -> Tuple[List[Dict], List[Tuple[int, int]]]:
    """Load the combined synthetic records in global emission order.

    The generator writes combined.csv in the same global order its ground-truth
    file is indexed against, so predicted and gold pairs line up exactly.
    Source-specific columns are merged into canonical raw fields here; schema
    alignment in Module 1 then maps them to canonical names.
    """
    df = pd.read_csv(f"{outdir}/combined.csv", dtype=str)
    records: List[Dict] = []
    for _, row in df.iterrows():
        rec = {}
        for k, v in row.items():
            if k == "source" or pd.isna(v) or v == "":
                continue
            rec[k] = v
        records.append(rec)
    truth: List[Tuple[int, int]] = []
    with open(f"{outdir}/ground_truth.csv", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            truth.append((int(r["record_a"]), int(r["record_b"])))
    return records, truth


def evaluate(outdir: str, rules_path: str, alpha: float, t: float) -> Dict:
    records, truth = load_sources(outdir)

    norm = DataNormalizer(rules_path)
    df = pd.DataFrame(records)
    normalized = norm.records(norm.normalize(df))

    resolver = EntityResolver(alpha=alpha, threshold=t, key_field="name",
                              use_blocking=True)
    resolver.resolve(normalized)
    predicted: Set[Tuple[int, int]] = {
        tuple(sorted(p)) for p in resolver.matched_pairs()}
    gold: Set[Tuple[int, int]] = {tuple(sorted(p)) for p in truth}

    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"alpha": alpha, "threshold": t, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "n_records": len(normalized),
            "n_gold_pairs": len(gold)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="data/synthetic")
    parser.add_argument("--rules", default="config/normalization_rules.yaml")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--threshold", type=float, default=0.82)
    args = parser.parse_args()
    result = evaluate(args.outdir, args.rules, args.alpha, args.threshold)
    print("Entity resolution on synthetic demonstration data:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("\nNote: these figures characterize the demonstration data, not the "
          "paper's original results.")
