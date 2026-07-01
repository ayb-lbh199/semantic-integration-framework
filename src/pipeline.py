"""
Pipeline orchestrator.

Runs the four modules in sequence on a single shared provenance log, so the
output is a knowledge graph plus a complete, auditable trace from raw record to
final annotation. This is the artifact-level instantiation of the paper's claim
that transparency is architectural.

Relations are added to the graph (entities located in the same city are linked
with a co-location predicate), so the graph carries relational semantics and
the annotation step receives adjacent-node context, as the paper describes.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

import pandas as pd

from .module1_normalization import DataNormalizer
from .module2_entity_resolution import EntityResolver
from .module3_knowledge_graph import KnowledgeGraphBuilder
from .module4_annotation import AnnotationEngine
from .provenance import ProvenanceLog

# Fields that identify provenance/source rather than entity content; they must
# not be passed to the resolver as comparable attributes.
DROP_BEFORE_RESOLUTION = {"source"}


class Pipeline:
    def __init__(self, rules_path: str, schema_path: str, template_path: str,
                 alpha: float = 0.6, threshold: float = 0.82,
                 key_field: str = "name", use_blocking: bool = True):
        self.provenance = ProvenanceLog()
        self.normalizer = DataNormalizer(rules_path, self.provenance)
        self.resolver = EntityResolver(
            alpha=alpha, threshold=threshold, key_field=key_field,
            use_blocking=use_blocking, provenance=self.provenance)
        self.kg = KnowledgeGraphBuilder(schema_path, self.provenance)
        self.annotator = AnnotationEngine(template_path, self.provenance)

    # Demonstration heuristic only. The paper states that the annotation step
    # consumes adjacent-node context but does not prescribe a specific relation
    # predicate; co-location is a deterministic relation derivable from the data
    # itself, used here purely to give the graph relational structure for the
    # demo. To avoid a combinatorial explosion in dense cities, each entity is
    # linked to a bounded number of co-located neighbours.
    MAX_COLOCATION_NEIGHBORS = 3

    def _add_colocation_relations(self, entities) -> int:
        """Link each entity to a bounded set of co-located entities (demo only)."""
        by_city: Dict[str, List[str]] = defaultdict(list)
        for ent in entities:
            city = ent.canonical.get("city")
            if city:
                by_city[str(city)].append(ent.entity_id)
        edges = 0
        for _city, ids in by_city.items():
            for i, src_id in enumerate(ids):
                # Link to the next few entities in the city only.
                for obj_id in ids[i + 1:i + 1 + self.MAX_COLOCATION_NEIGHBORS]:
                    self.kg.add_relation(src_id, "coLocatedWith", obj_id)
                    edges += 1
        return edges

    def _neighbors(self, entity_id: str, limit: int = 3) -> List[str]:
        names = []
        for _, nbr, data in self.kg.graph.edges(entity_id, data=True):
            nbr_name = self.kg.graph.nodes[nbr].get("name", nbr)
            names.append(f"{data.get('predicate', 'related')} {nbr_name}")
            if len(names) >= limit:
                break
        return names

    def run(self, raw: pd.DataFrame) -> Dict[str, Any]:
        # Module 1
        normalized = self.normalizer.normalize(raw)
        records = self.normalizer.records(normalized)
        # Drop provenance/source tags before resolution.
        records = [{k: v for k, v in r.items() if k not in DROP_BEFORE_RESOLUTION}
                   for r in records]
        # Module 2
        self.resolver.resolve(records)
        entities = self.resolver.consolidate(records)
        # Module 3
        for ent in entities:
            self.kg.add_entity(ent.entity_id, ent.canonical)
        n_edges = self._add_colocation_relations(entities)
        # Module 4 (with adjacent-node context)
        annotations: List[Dict[str, Any]] = []
        for ent in entities:
            etype = self.kg.graph.nodes[ent.entity_id].get("node_type", "Entity")
            ann = self.annotator.annotate(
                ent.entity_id, etype, ent.canonical, self._neighbors(ent.entity_id))
            annotations.append(ann.as_dict())
        return {
            "n_input": len(raw),
            "n_entities": len(entities),
            "n_relations": n_edges,
            "graph_metrics": self.kg.quality_metrics(),
            "annotations": annotations,
            "review_queue": self.annotator.review_queue,
            "provenance_entries": len(self.provenance),
        }


def run_from_csv(csv_path: str, rules: str, schema: str, template: str,
                 use_blocking: bool = True) -> Dict[str, Any]:
    """Run the pipeline on a generated combined.csv (used by the demo)."""
    df = pd.read_csv(csv_path, dtype=str)
    records = [{k: v for k, v in row.items()
                if pd.notna(v) and v != ""}
               for _, row in df.iterrows()]
    pipe = Pipeline(rules, schema, template, use_blocking=use_blocking)
    return pipe.run(pd.DataFrame(records))


if __name__ == "__main__":
    # If a generated CSV exists, demonstrate on it; otherwise use a small inline sample.
    csv_path = "data/synthetic/combined.csv"
    if os.path.exists(csv_path):
        result = run_from_csv(csv_path, "config/normalization_rules.yaml",
                              "config/graph_schema.yaml",
                              "prompts/annotation_template.txt")
        print(json.dumps({k: v for k, v in result.items()
                          if k != "annotations"}, indent=2, default=str))
        print("\nSample annotations:")
        for a in result["annotations"][:5]:
            print("  ", a["text"], "| conf:", a["confidence"])
        sys.exit(0)

    raw = pd.DataFrame([
        {"source": "ERP", "company": "Acme Corporation", "ville": "London",
         "pays": "UK", "headcount": "450"},
        {"source": "CRM", "client_name": "Acme Corporation Ltd",
         "location": "London", "nation": "GB", "headcount": "455"},
        {"source": "ERP", "company": "Globex SA", "ville": "Madrid",
         "pays": "Spain", "headcount": "200"},
    ])
    pipe = Pipeline("config/normalization_rules.yaml",
                    "config/graph_schema.yaml",
                    "prompts/annotation_template.txt",
                    use_blocking=False)
    result = pipe.run(raw)
    print(json.dumps({k: v for k, v in result.items() if k != "annotations"},
                     indent=2, default=str))
    print("\nannotations:")
    for a in result["annotations"]:
        print("  ", a["text"], "| conf:", a["confidence"])
    os.makedirs("examples/output", exist_ok=True)
    pipe.provenance.to_json("examples/output/provenance_full.json")
    print("\nfull provenance written:", len(pipe.provenance), "entries")
