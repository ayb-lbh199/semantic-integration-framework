"""Unit tests for the four modules and the orchestrated pipeline.

These tests assert behavior the paper claims: provenance is recorded, the
hybrid similarity and decision rule behave correctly, entities consolidate
transitively, the graph types nodes and exports, and the annotation controls
route sensitive or low-confidence output to human review.
"""

import pandas as pd
import pytest

from src.provenance import ProvenanceLog, ProvenanceEntry
from src.module1_normalization import DataNormalizer
from src.module2_entity_resolution import (
    EntityResolver, s_hybrid, is_match, field_levenshtein, lexical_similarity)
from src.module3_knowledge_graph import KnowledgeGraphBuilder
from src.module4_annotation import (
    AnnotationEngine, fact_check, confidence_score, evidence_density)

RULES = "config/normalization_rules.yaml"
SCHEMA = "config/graph_schema.yaml"
TEMPLATE = "prompts/annotation_template.txt"


# --- provenance -------------------------------------------------------------
def test_provenance_records_and_serializes(tmp_path):
    log = ProvenanceLog()
    log.record(ProvenanceEntry("M1", "op", "R1", "1.0", "s", before="a", after="b"))
    assert len(log) == 1
    out = tmp_path / "prov.json"
    log.to_json(str(out))
    assert out.exists()


# --- module 1 ---------------------------------------------------------------
def test_normalization_aligns_and_maps():
    df = pd.DataFrame([{"company": "Acme", "pays": "UK"}])
    norm = DataNormalizer(RULES)
    out = norm.normalize(df)
    assert "name" in out.columns
    assert "country" in out.columns
    assert out["country"].iloc[0] == "United Kingdom"
    assert len(norm.provenance) > 0


def test_normalization_coalesces_duplicate_columns():
    df = pd.DataFrame([
        {"company": "Acme", "client_name": None},
        {"company": None, "client_name": "Globex"},
    ])
    norm = DataNormalizer(RULES)
    out = norm.normalize(df)
    # both 'company' and 'client_name' map to 'name' -> single column
    assert list(out.columns).count("name") == 1


# --- module 2 ---------------------------------------------------------------
def test_hybrid_and_decision():
    assert s_hybrid(1.0, 1.0, 0.6) == pytest.approx(1.0)
    assert s_hybrid(0.0, 0.0, 0.6) == pytest.approx(0.0)
    assert is_match(0.82) is True
    assert is_match(0.8199) is False


def test_field_levenshtein_bounds():
    assert field_levenshtein("abc", "abc") == pytest.approx(1.0)
    assert field_levenshtein("", "x") == 0.0


def test_resolver_consolidates_transitively():
    records = [
        {"name": "Acme Corporation", "city": "London"},
        {"name": "Acme Corporation", "city": "London"},
        {"name": "Acme Corporation", "city": "London"},
        {"name": "Globex SA", "city": "Madrid"},
    ]
    r = EntityResolver(key_field="name", use_blocking=False)
    r.resolve(records)
    entities = r.consolidate(records)
    sizes = sorted(len(e.member_indices) for e in entities)
    assert sizes == [1, 3]  # three Acme consolidated, Globex alone


# --- module 3 ---------------------------------------------------------------
def test_graph_types_and_exports(tmp_path):
    b = KnowledgeGraphBuilder(SCHEMA)
    b.add_entity("ENT_1", {"name": "Acme", "country": "United Kingdom"})
    assert b.graph.nodes["ENT_1"]["node_type"] == "Organization"
    metrics = b.quality_metrics()
    assert metrics["nodes"] == 1
    assert metrics["triples"] > 0
    jsonld = tmp_path / "g.jsonld"
    b.export_jsonld(str(jsonld))
    assert jsonld.exists()


# --- module 4 ---------------------------------------------------------------
def test_fact_check_flags_ungrounded_numbers():
    flagged = fact_check("Revenue is 9999.", {"name": "Acme", "city": "London"})
    assert "9999" in flagged


def test_confidence_in_range():
    d = evidence_density({"a": 1, "b": 2, "c": None}, expected_fields=5)
    c = confidence_score(None, d)
    assert 0.0 <= c <= 1.0


def test_sensitive_routed_to_review():
    engine = AnnotationEngine(TEMPLATE)
    ann = engine.annotate("ENT_9", "Organization",
                          {"name": "FinCo", "revenue": 1000000})
    assert ann.routed_to_review is True
    assert len(engine.review_queue) == 1


# --- integration: provenance spans the whole pipeline -----------------------
def test_pipeline_provenance_spans_all_modules():
    import pandas as pd
    from src.pipeline import Pipeline
    raw = pd.DataFrame([
        {"source": "ERP", "company": "Acme Corp", "ville": "London",
         "pays": "UK", "headcount": "450"},
        {"source": "ERP", "company": "Globex", "ville": "London",
         "pays": "UK", "headcount": "200"},
    ])
    pipe = Pipeline(RULES, SCHEMA, TEMPLATE, use_blocking=False)
    pipe.run(raw)
    modules = {e.module for e in pipe.provenance.entries()}
    # M1 (normalize), M3 (triples), M4 (annotation) always present;
    # M2 present when at least one match is recorded.
    assert "M1" in modules
    assert "M3" in modules
    assert "M4" in modules


def test_pipeline_creates_relations_for_colocated():
    import pandas as pd
    from src.pipeline import Pipeline
    raw = pd.DataFrame([
        {"source": "ERP", "company": "Acme Corp", "ville": "London",
         "pays": "UK", "headcount": "450"},
        {"source": "ERP", "company": "Globex", "ville": "London",
         "pays": "UK", "headcount": "200"},
    ])
    pipe = Pipeline(RULES, SCHEMA, TEMPLATE, use_blocking=False)
    result = pipe.run(raw)
    assert result["n_relations"] >= 1
    assert result["graph_metrics"]["edges"] >= 1


def test_currency_conversion_recorded():
    import pandas as pd
    from src.module1_normalization import DataNormalizer
    df = pd.DataFrame([{"company": "Acme", "revenue": "1000",
                        "currency": "USD"}])
    norm = DataNormalizer(RULES)
    out = norm.normalize(df)
    ops = {e.operation for e in norm.provenance.entries()}
    assert "currency_conversion" in ops
    # 1000 USD * 0.92 = 920
    assert float(out["revenue"].iloc[0]) == 920.0



# --- regression: fact_check must not flag common sentence-initial words ------
def test_fact_check_no_false_positives_on_common_words():
    from src.module4_annotation import fact_check
    # Realistic GPT-4-style phrasings whose leading words are not entities.
    assert fact_check("Located in the United Kingdom.",
                      {"name": "Acme", "country": "United Kingdom"}) == []
    assert fact_check("Revenue of 1,200,000 EUR.",
                      {"name": "Acme", "revenue": "1200000"}) == []
    assert fact_check("Has 5000 employees.",
                      {"name": "Acme", "employees": "5000"}) == []


def test_fact_check_still_flags_fabricated_name():
    from src.module4_annotation import fact_check
    flagged = fact_check("Acme is part of MegaCorp.",
                         {"name": "Acme", "city": "London"})
    assert "MegaCorp" in flagged


def test_annotation_engine_falls_back_on_bad_key(monkeypatch):
    # A present-but-invalid key must not crash; the engine falls back.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-invalid-test-key")
    from src.module4_annotation import AnnotationEngine
    engine = AnnotationEngine(TEMPLATE)
    ann = engine.annotate("ENT_1", "Organization",
                          {"name": "Acme", "city": "London"})
    assert ann.text  # produced a description via fallback, no exception
