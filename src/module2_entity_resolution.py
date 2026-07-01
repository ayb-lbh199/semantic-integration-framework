"""
Module 2: Hybrid Entity Resolution with Empirical Validation.

Implements the hybrid similarity model from the paper:

    S_hybrid(a, b) = alpha * S_lexical(a, b) + (1 - alpha) * S_semantic(a, b)

A pair is a match when S_hybrid(a, b) >= t. The defaults alpha = 0.6 and
t = 0.82 are the values reported in the paper. The parameter-selection
procedure (grid search with k-fold cross-validation) is reproduced in
grid_search.py; on the demonstration data it selects the best operating point
for that data, which differs from the paper's original calibration.

- S_lexical : normalized Levenshtein similarity averaged over shared fields,
              so abbreviations and name variants resolve.
- S_semantic: TF-IDF cosine similarity over tokenized field values.

The paper speaks of *resolved entities*, not isolated pairs. Match decisions
are therefore consolidated transitively: if a~b and b~c, then {a, b, c} form
one entity. Every match decision is committed to the provenance log, and each
consolidated entity carries the evidence that produced it.

A character n-gram blocking step keeps the comparison tractable; it can be
disabled for full recall on small inputs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Tuple

import pandas as pd
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .provenance import ProvenanceEntry, ProvenanceLog


def _norm(s) -> str:
    return str(s or "").strip().lower()


def field_levenshtein(s1: str, s2: str) -> float:
    s1, s2 = _norm(s1), _norm(s2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return 1.0 - Levenshtein.normalized_distance(s1, s2)


# Fields that must not contribute to lexical similarity: provenance/source
# tags carry no entity identity, and numeric fields are better compared by
# value than by edit distance ("2303" vs "2454" is not a name difference).
NON_LEXICAL_FIELDS = {"source", "employees", "revenue", "headcount"}


def lexical_similarity(a: Dict[str, str], b: Dict[str, str]) -> float:
    shared = [
        k for k in a.keys()
        if k in b and k not in NON_LEXICAL_FIELDS
    ]
    if not shared:
        return 0.0
    return sum(field_levenshtein(a[k], b[k]) for k in shared) / len(shared)


def _record_to_text(record: Dict[str, str]) -> str:
    return " ".join(
        str(v) for v in record.values()
        if v is not None and not (isinstance(v, float) and pd.isna(v))
        and str(v) not in ("<NA>", "nan", "NaN", "None")
    )


def build_tfidf_matrix(records: List[Dict[str, str]]):
    corpus = [_record_to_text(r) for r in records]
    vec = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
    return vec.fit_transform(corpus), vec


def semantic_similarity(matrix, i: int, j: int) -> float:
    sim = cosine_similarity(matrix[i], matrix[j])[0, 0]
    return float(max(0.0, min(1.0, sim)))


def s_hybrid(lexical: float, semantic: float, alpha: float = 0.6) -> float:
    return alpha * lexical + (1.0 - alpha) * semantic


def is_match(score: float, t: float = 0.82) -> bool:
    return score >= t


def char_ngram_keys(record: Dict[str, str], key_field: str, n: int = 3) -> set:
    text = _norm(record.get(key_field, ""))
    if not text:
        return {""}
    keys = set()
    for tok in text.split():
        keys.add(tok[:n] if len(tok) >= n else tok)
    return keys or {""}


def candidate_pairs(records, key_field, use_blocking, n=3):
    if not use_blocking or key_field is None:
        return list(combinations(range(len(records)), 2))
    key_to_records = defaultdict(list)
    for idx, rec in enumerate(records):
        for k in char_ngram_keys(rec, key_field, n):
            key_to_records[k].append(idx)
    pairs = set()
    for indices in key_to_records.values():
        for i, j in combinations(sorted(indices), 2):
            pairs.add((i, j))
    return sorted(pairs)


@dataclass
class MatchDecision:
    index_a: int
    index_b: int
    lexical: float
    semantic: float
    hybrid: float
    matched: bool
    alpha: float
    threshold: float

    def as_dict(self) -> dict:
        return {"index_a": self.index_a, "index_b": self.index_b,
                "lexical": round(self.lexical, 4), "semantic": round(self.semantic, 4),
                "hybrid": round(self.hybrid, 4), "matched": self.matched,
                "alpha": self.alpha, "threshold": self.threshold}


class _UnionFind:
    """Transitive consolidation of matched pairs into entities."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def clusters(self) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            out[self.find(i)].append(i)
        return out


@dataclass
class ResolvedEntity:
    """A consolidated entity with the record indices it was built from."""

    entity_id: str
    member_indices: List[int]
    canonical: Dict[str, str]


@dataclass
class EntityResolver:
    alpha: float = 0.6
    threshold: float = 0.82
    key_field: str | None = None
    use_blocking: bool = True
    ngram: int = 3
    provenance: ProvenanceLog = field(default_factory=ProvenanceLog)
    decisions: List[MatchDecision] = field(default_factory=list)

    def resolve(self, records: List[Dict[str, str]]) -> List[MatchDecision]:
        self.decisions = []
        if len(records) < 2:
            return self.decisions
        matrix, _ = build_tfidf_matrix(records)
        for i, j in candidate_pairs(records, self.key_field, self.use_blocking, self.ngram):
            lex = lexical_similarity(records[i], records[j])
            sem = semantic_similarity(matrix, i, j)
            score = s_hybrid(lex, sem, self.alpha)
            matched = is_match(score, self.threshold)
            self.decisions.append(MatchDecision(
                i, j, lex, sem, score, matched, self.alpha, self.threshold))
            if matched:
                self.provenance.record(ProvenanceEntry(
                    module="M2", operation="entity_match",
                    rule_id="ER_HYBRID_01", rule_version="1.0",
                    subject=f"pair_{i}_{j}",
                    evidence={"lexical": round(lex, 4), "semantic": round(sem, 4),
                              "hybrid": round(score, 4), "alpha": self.alpha,
                              "threshold": self.threshold}))
        return self.decisions

    def matched_pairs(self) -> List[Tuple[int, int]]:
        return [(d.index_a, d.index_b) for d in self.decisions if d.matched]

    def consolidate(self, records: List[Dict[str, str]]) -> List[ResolvedEntity]:
        """Group matched records into entities via transitive closure.

        The paper resolves *entities*, not pairs: if a matches b and b matches
        c, the three records describe one real-world entity. The canonical
        record keeps, per field, the most frequent non-null value across members.
        """
        uf = _UnionFind(len(records))
        for a, b in self.matched_pairs():
            uf.union(a, b)
        entities: List[ResolvedEntity] = []
        for root, members in sorted(uf.clusters().items()):
            canonical: Dict[str, str] = {}
            all_fields = {k for idx in members for k in records[idx].keys()}
            for fld in all_fields:
                values = [records[idx].get(fld) for idx in members
                          if records[idx].get(fld) not in (None, "")]
                if values:
                    canonical[fld] = max(set(values), key=values.count)
            ent_id = f"ENT_{root:06d}"
            entities.append(ResolvedEntity(ent_id, sorted(members), canonical))
            if len(members) > 1:
                self.provenance.record(ProvenanceEntry(
                    module="M2", operation="entity_consolidation",
                    rule_id="ER_CONSOLIDATE_01", rule_version="1.0",
                    subject=ent_id,
                    evidence={"members": sorted(members), "size": len(members)}))
        return entities


if __name__ == "__main__":
    records = [
        {"name": "Acme Corporation", "city": "London", "country": "United Kingdom"},
        {"name": "Acme Corporation Ltd", "city": "London", "country": "United Kingdom"},
        {"name": "Acme Corp", "city": "London", "country": "United Kingdom"},
        {"name": "Globex SA", "city": "Madrid", "country": "Spain"},
    ]
    r = EntityResolver(key_field="name", use_blocking=False)
    r.resolve(records)
    print("matched pairs:", r.matched_pairs())
    for ent in r.consolidate(records):
        print(f"  {ent.entity_id}: members={ent.member_indices} canonical={ent.canonical}")
    print("provenance entries:", len(r.provenance))
