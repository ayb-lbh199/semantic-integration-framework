"""
Module 3: Lightweight Semantic Graph Construction.

Resolved entities are encoded as subject-predicate-object triples following an
RDF-inspired schema and assembled into a property graph with NetworkX. The
paper deliberately rejects OWL ontologies: meaning is carried by typed nodes
and labeled edges documented in plain text, not by formal axioms that require
ontology engineers to maintain.

The schema is explicit and versioned (config/graph_schema.yaml). The graph is
exportable to GraphML and JSON-LD, so it is an interoperable artifact rather
than an in-memory structure. Every triple is committed to the provenance log.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import networkx as nx
import yaml

from .provenance import ProvenanceEntry, ProvenanceLog


@dataclass
class Triple:
    """An RDF-inspired subject-predicate-object statement."""

    subject: str
    predicate: str
    obj: str

    def as_tuple(self):
        return (self.subject, self.predicate, self.obj)


class KnowledgeGraphBuilder:
    """Build a typed property graph from resolved entities and a schema."""

    def __init__(self, schema_path: str, provenance: ProvenanceLog | None = None):
        if not os.path.exists(schema_path):
            raise FileNotFoundError(
                f"Graph schema not found: {schema_path}")
        with open(schema_path, "r", encoding="utf-8") as fh:
            self.schema = yaml.safe_load(fh)
        self.version = str(self.schema.get("version", "1.0"))
        self.graph = nx.MultiDiGraph()
        self.triples: List[Triple] = []
        self.provenance = provenance if provenance is not None else ProvenanceLog()

    def _entity_type(self, canonical: Dict[str, Any]) -> str:
        """Infer node type from the schema's typing rules (no OWL reasoning)."""
        rules = self.schema.get("entity_typing", {})
        for type_name, required in rules.items():
            if all(canonical.get(fld) not in (None, "") for fld in required):
                return type_name
        return self.schema.get("default_entity_type", "Entity")

    def add_entity(self, entity_id: str, canonical: Dict[str, Any]) -> None:
        etype = self._entity_type(canonical)
        self.graph.add_node(entity_id, node_type=etype, **canonical)
        # Type assertion as a triple.
        self._add_triple(entity_id, "rdf:type", etype)
        # Attribute triples following the schema's predicate map.
        for fld, predicate in self.schema.get("attribute_predicates", {}).items():
            if canonical.get(fld) not in (None, ""):
                self._add_triple(entity_id, predicate, str(canonical[fld]))

    def add_relation(self, subj_id: str, predicate: str, obj_id: str) -> None:
        # Guard against phantom nodes: both endpoints must already exist.
        if subj_id not in self.graph or obj_id not in self.graph:
            self.provenance.record(ProvenanceEntry(
                module="M3", operation="relation_rejected",
                rule_id="KG_GUARD_01", rule_version=self.version,
                subject=f"{subj_id}->{obj_id}",
                evidence={"reason": "endpoint missing", "predicate": predicate}))
            return
        if predicate not in self.schema.get("relation_predicates", []):
            # Schema is extensible: new predicates are allowed but flagged.
            self.provenance.record(ProvenanceEntry(
                module="M3", operation="schema_extension",
                rule_id="KG_SCHEMA_EXT", rule_version=self.version,
                subject=predicate,
                evidence={"note": "predicate not in base schema; added"}))
        self.graph.add_edge(subj_id, obj_id, key=predicate, predicate=predicate)
        self._add_triple(subj_id, predicate, obj_id)

    def _add_triple(self, s: str, p: str, o: str) -> None:
        triple = Triple(s, p, o)
        self.triples.append(triple)
        self.provenance.record(ProvenanceEntry(
            module="M3", operation="triple_assertion",
            rule_id="KG_TRIPLE_01", rule_version=self.version,
            subject=s, after=f"{p} {o}",
            evidence={"predicate": p, "object": o}))

    # --- quality metrics (Table VI of the paper) -------------------------- #
    def quality_metrics(self) -> Dict[str, Any]:
        """Compute the graph quality metrics reported in Table VI of the paper.

        - Node coverage: fraction of entities successfully typed (i.e., placed
          under a schema entity type rather than the generic fallback).
        - Triplet accuracy: fraction of triples whose predicate is declared in
          the schema (attribute or relation), i.e., schema-valid statements.
        - Schema coherence: fraction of relation edges whose predicate is in the
          declared relation vocabulary (domain/range respected at the
          predicate level).
        - Connected components: number of weakly connected components.
        """
        n_nodes = self.graph.number_of_nodes()
        n_typed = sum(1 for _, d in self.graph.nodes(data=True)
                      if d.get("node_type", "Entity") != "Entity")
        node_coverage = round(n_typed / n_nodes, 4) if n_nodes else 0.0

        declared_attr = set(self.schema.get("attribute_predicates", {}).values())
        declared_rel = set(self.schema.get("relation_predicates", []))
        declared = declared_attr | declared_rel | {"rdf:type"}
        valid_triples = sum(1 for tr in self.triples if tr.predicate in declared)
        triplet_accuracy = (round(valid_triples / len(self.triples), 4)
                            if self.triples else 0.0)

        rel_edges = [d.get("predicate") for _, _, d in
                     self.graph.edges(data=True)]
        coherent = sum(1 for p in rel_edges if p in declared_rel)
        schema_coherence = (round(coherent / len(rel_edges), 4)
                            if rel_edges else 1.0)

        components = (nx.number_weakly_connected_components(self.graph)
                      if n_nodes else 0)

        return {
            "node_coverage": node_coverage,
            "triplet_accuracy": triplet_accuracy,
            "schema_coherence": schema_coherence,
            "connected_components": components,
            "nodes": n_nodes,
            "edges": self.graph.number_of_edges(),
            "triples": len(self.triples),
        }

    # --- interoperable exports -------------------------------------------- #
    def export_graphml(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        # Flatten attributes to strings for GraphML compatibility.
        g = nx.MultiDiGraph()
        for n, d in self.graph.nodes(data=True):
            g.add_node(n, **{k: str(v) for k, v in d.items()})
        for u, v, k, d in self.graph.edges(keys=True, data=True):
            g.add_edge(u, v, key=str(k), **{kk: str(vv) for kk, vv in d.items()})
        nx.write_graphml(g, path)

    def export_jsonld(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        doc = {"@context": self.schema.get("jsonld_context", {}), "@graph": []}
        for n, d in self.graph.nodes(data=True):
            node = {"@id": n, "@type": d.get("node_type", "Entity")}
            for k, v in d.items():
                if k != "node_type":
                    node[k] = v
            doc["@graph"].append(node)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    schema_path = "config/graph_schema.yaml"
    builder = KnowledgeGraphBuilder(schema_path)
    builder.add_entity("ENT_000000",
                       {"name": "Acme Corporation", "city": "London",
                        "country": "United Kingdom", "employees": 450})
    builder.add_entity("ENT_000003",
                       {"name": "Globex SA", "city": "Madrid", "country": "Spain"})
    builder.add_relation("ENT_000000", "operatesIn", "ENT_000003")
    print("metrics:", builder.quality_metrics())
    os.makedirs("examples/output", exist_ok=True)
    builder.export_jsonld("examples/output/graph.jsonld")
    builder.export_graphml("examples/output/graph.graphml")
    print("exported JSON-LD and GraphML")
    print("provenance entries:", len(builder.provenance))
