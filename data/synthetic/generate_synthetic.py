"""
Synthetic enterprise data generator (demonstration).

Produces structurally equivalent demonstration data: four source systems
(ERP, CRM, API, LEGACY) describing overlapping organizations, with the failure modes
the paper analyzes injected at realistic rates: schema heterogeneity (different
field names per source), abbreviations, missing values, light typos, and
country code vs full-name inconsistency.

The figures obtained on this demonstration data do not reproduce the paper's
results; the paper's evaluation involved confidential pilot data that cannot be
redistributed. This generator provides a reproducible, structurally comparable
demonstration only. Realistic enterprise duplicates share most fields exactly
and differ in one or two; the generator reflects that, rather than stacking
every corruption on every record. A seed makes the output
reproducible, and ground truth is recorded against a single combined ordering
so the evaluation script can compute precision and recall unambiguously.
"""

from __future__ import annotations

import argparse
import os
import csv
import random
from typing import Dict, List, Tuple

BASE_NAMES = [
    "Acme Corporation", "Globex Industries", "Initech Solutions",
    "Umbrella Logistics", "Soylent Foods", "Hooli Technologies",
    "Vehement Capital", "Massive Dynamic", "Stark Manufacturing",
    "Wayne Enterprises", "Wonka Industries", "Cyberdyne Systems",
    "Pied Piper", "Aviato Holdings", "Gringotts Bank", "Tyrell Group",
]
CITIES = ["London", "Madrid", "Paris", "Berlin", "Milan", "Amsterdam"]
COUNTRY = {"London": ("UK", "United Kingdom"), "Madrid": ("ES", "Spain"),
           "Paris": ("FR", "France"), "Berlin": ("DE", "Germany"),
           "Milan": ("IT", "Italy"), "Amsterdam": ("NL", "Netherlands")}
SUFFIXES = ["", " Ltd", " Inc", " SA", " GmbH"]


def _light_typo(s: str, rng: random.Random) -> str:
    """Swap two adjacent characters once, occasionally."""
    if len(s) < 5 or rng.random() > 0.15:
        return s
    i = rng.randint(1, len(s) - 3)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def generate(n_entities: int, dup_rate: float, seed: int):
    rng = random.Random(seed)
    combined: List[Dict] = []
    truth: List[Tuple[int, int]] = []

    for eid in range(n_entities):
        base = f"{rng.choice(BASE_NAMES)} {eid}"
        city = rng.choice(CITIES)
        code, full = COUNTRY[city]
        employees = rng.randint(50, 5000)

        # ERP: the clean anchor record, full country name.
        erp_idx = len(combined)
        revenue = rng.randint(1, 500) * 100000
        combined.append({"source": "ERP", "company": base + rng.choice(SUFFIXES),
                         "ville": city, "pays": full,
                         "headcount": str(employees), "revenue": str(revenue),
                         "currency": rng.choice(["EUR", "USD", "GBP"])})

        if rng.random() < dup_rate:
            # CRM duplicate: same core name, realistic minor variation.
            crm_name = _light_typo(base, rng) + rng.choice(SUFFIXES)
            crm_idx = len(combined)
            combined.append({
                "source": "CRM", "client_name": crm_name, "location": city,
                "nation": code,  # country code instead of full name
                "headcount": str(employees + rng.randint(-3, 3))})
            truth.append((erp_idx, crm_idx))

            # Sometimes an API record too, occasionally missing the country.
            if rng.random() < 0.4:
                api_idx = len(combined)
                combined.append({
                    "source": "API", "org": base, "location": city,
                    "nation": "" if rng.random() < 0.3 else code})
                truth.append((erp_idx, api_idx))
                truth.append((crm_idx, api_idx))

            # A legacy flat-file export with a fourth, distinct schema.
            if rng.random() < 0.25:
                legacy_idx = len(combined)
                combined.append({
                    "source": "LEGACY", "entity_name": base.upper(),
                    "town": city, "country_code": code,
                    "staff": str(employees)})
                truth.append((erp_idx, legacy_idx))

    return combined, truth


def write_combined(rows: List[Dict], path: str) -> None:
    fields = ["source", "company", "client_name", "org", "entity_name",
              "ville", "location", "town", "pays", "nation", "country_code",
              "headcount", "staff", "revenue", "currency"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_per_source(rows: List[Dict], outdir: str) -> None:
    for src, fname, cols in [
        ("ERP", "erp.csv", ["company", "ville", "pays", "headcount",
                             "revenue", "currency"]),
        ("CRM", "crm.csv", ["client_name", "location", "nation", "headcount"]),
        ("API", "api.csv", ["org", "location", "nation"]),
        ("LEGACY", "legacy.csv", ["entity_name", "town", "country_code",
                                  "staff"]),
    ]:
        subset = [r for r in rows if r["source"] == src]
        with open(f"{outdir}/{fname}", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in subset:
                w.writerow({k: r.get(k, "") for k in cols})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate synthetic demo data.")
    p.add_argument("--entities", type=int, default=200)
    p.add_argument("--dup-rate", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="data/synthetic")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    combined, truth = generate(args.entities, args.dup_rate, args.seed)
    write_combined(combined, f"{args.outdir}/combined.csv")
    write_per_source(combined, args.outdir)
    with open(f"{args.outdir}/ground_truth.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["record_a", "record_b"])
        w.writerows(truth)
    print(f"Generated {len(combined)} records across 4 sources, "
          f"{len(truth)} ground-truth duplicate links (seed={args.seed}).")
