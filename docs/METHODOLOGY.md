# Methodology: how the code maps to the paper

| Paper element | Code |
|---|---|
| Design transparency (DR2) | `src/provenance.py`, versioned YAML in `config/` |
| Module 1: normalization (schema, types, imputation, noise) | `src/module1_normalization.py` |
| Module 2: hybrid entity resolution, Eq. (1) | `src/module2_entity_resolution.py` |
| Resolved entities (transitive consolidation) | `EntityResolver.consolidate` |
| Module 3: RDF-style triples, NetworkX, no OWL | `src/module3_knowledge_graph.py` |
| Graph quality metrics: node coverage, triplet accuracy, schema coherence, connected components (Table VI) | `KnowledgeGraphBuilder.quality_metrics` |
| Module 4: GPT-4 annotation | `src/module4_annotation.py` |
| Three hallucination controls | `fact_check`, `confidence_score`, review queue |
| Parameter selection by grid search with k-fold cross-validation, grid alpha in [0.1, 0.9] and t in [0.70, 0.95] | `src/grid_search.py` |
| Single-configuration evaluation | `src/evaluate.py` |

## Parameters

- `alpha = 0.6` (lexical weight), `t = 0.82` (decision threshold), per the paper.
- GPT-4 annotation: `temperature = 0.1`, `max_tokens = 512`.
- Confidence: `confidence = lambda * p_norm + (1 - lambda) * d`, `lambda = 0.6`,
  review threshold `0.75`.

These parameters are defaults in code and can be overridden at call sites.

## Confidence score

`p_norm` is the normalized mean token log-probability returned by the model
(when available); `d` is evidence density, the fraction of expected source
fields that are present. When log-probabilities are unavailable, `p_norm` falls
back to `d` so the score remains meaningful.
