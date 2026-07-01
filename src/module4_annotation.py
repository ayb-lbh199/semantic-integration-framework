"""
Module 4: LLM-based Annotation with Quality Controls.

GPT-4 produces natural-language descriptions of graph entities using structured
prompts that inject graph context (attributes and adjacent relations). Because
uncontrolled LLM output is hazardous in enterprise settings, the paper specifies
three interception mechanisms, all implemented here:

  1. Fact-checking: a deterministic validator that checks every numeric and
     named value in the generated text against the source attributes. Anything
     not grounded in the source is flagged.
  2. Confidence scoring: confidence = lambda * p_norm + (1 - lambda) * d, where
     p_norm is the normalized mean token log-probability and d is evidence
     density (non-null source fields / expected fields). Annotations below the
     review threshold are quarantined.
  3. Human validation queue: entities in sensitive categories, or with low
     confidence, are written to a review queue instead of entering the graph.

The GPT-4 call is optional. Without an API key the module uses a deterministic
template renderer, so the pipeline is runnable end-to-end by anyone. The control
logic (the part the paper claims as a contribution) is identical either way.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .provenance import ProvenanceEntry, ProvenanceLog

SENSITIVE_FIELDS = {"revenue", "iban", "ssn", "tax_id", "compliance_status"}
REVIEW_THRESHOLD = 0.75
LAMBDA = 0.6


@dataclass
class Annotation:
    entity_id: str
    text: str
    confidence: float
    grounded: bool
    flagged_terms: List[str] = field(default_factory=list)
    routed_to_review: bool = False
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "entity_id": self.entity_id, "text": self.text,
            "confidence": round(self.confidence, 4), "grounded": self.grounded,
            "flagged_terms": self.flagged_terms,
            "routed_to_review": self.routed_to_review, "reason": self.reason,
        }


def _render_template(template: str, entity_id: str, etype: str,
                     attributes: Dict[str, Any], relations: List[str]) -> str:
    attr_lines = "\n".join(f"- {k}: {v}" for k, v in attributes.items())
    rel_lines = "\n".join(f"- {r}" for r in relations) or "- none"
    return template.format(entity_id=entity_id, entity_type=etype,
                           attributes=attr_lines, relations=rel_lines)


def _deterministic_description(etype: str, attributes: Dict[str, Any],
                              relations: List[str]) -> str:
    """Fallback generator used when no LLM API key is available.

    It only restates grounded facts, so it is always fact-consistent. This keeps
    the package runnable without external calls while exercising the same
    control logic.
    """
    name = attributes.get("name", "This entity")
    article = "an" if etype[:1].lower() in "aeiou" else "a"
    parts = [f"{name} is {article} {etype.lower()}"]
    if attributes.get("city") or attributes.get("country"):
        loc = ", ".join(str(attributes[k]) for k in ("city", "country")
                        if attributes.get(k))
        parts.append(f"based in {loc}")
    if attributes.get("employees"):
        parts.append(f"with {attributes['employees']} employees")
    text = " ".join(parts) + "."
    if relations:
        text += " Related: " + "; ".join(relations) + "."
    return text


# Common words that may be capitalized without being entity names.
_STOPWORDS = {
    "The", "This", "It", "A", "An", "And", "Or", "But", "Its",
    "Related", "Based", "With", "Has", "Have", "Had", "Is", "Are",
    "Located", "Founded", "Established", "Headquartered", "Operating",
    "Phone", "Email", "Address", "Revenue", "Total", "Number", "Currency",
    "EUR", "USD", "GBP", "JPY", "Employees", "Staff", "Of", "In", "At",
}


def fact_check(text: str, attributes: Dict[str, Any],
               extra_context: List[str] | None = None) -> List[str]:
    """Return terms in the text not grounded in the source attributes.

    Two checks run: every number in the description must appear among the
    source values (catching fabricated quantities), and every multi-character
    capitalized token (a candidate proper noun) must appear in the source
    (catching fabricated names such as an invented parent company).
    """
    source_values = {str(v).lower() for v in attributes.values() if v is not None}
    if extra_context:
        source_values |= {str(c).lower() for c in extra_context}
    source_blob = " ".join(source_values)
    # Normalize thousands separators so "1,200,000" matches "1200000".
    source_blob_digits = source_blob.replace(",", "")
    flagged: List[str] = []
    for number in re.findall(r"\b\d[\d,\.]*\b", text):
        bare = number.replace(",", "")
        if bare.lower() not in source_blob_digits and number.lower() not in source_blob:
            flagged.append(number)
    for token in re.findall(r"\b[A-Z][A-Za-z]{2,}\b", text):
        if token in _STOPWORDS:
            continue
        if token.lower() not in source_blob:
            flagged.append(token)
    return flagged


def evidence_density(attributes: Dict[str, Any], expected_fields: int) -> float:
    present = sum(1 for v in attributes.values() if v not in (None, ""))
    if expected_fields <= 0:
        return 0.0
    return min(1.0, present / expected_fields)


def confidence_score(mean_logprob: Optional[float], density: float) -> float:
    """confidence = lambda * p_norm + (1 - lambda) * d.

    p_norm maps a mean log-probability into [0, 1] via exp(logprob). When the
    LLM does not return log-probabilities (or the fallback is used), p_norm
    falls back to the evidence density so the score stays meaningful.
    """
    if mean_logprob is None:
        p_norm = density
    else:
        p_norm = max(0.0, min(1.0, math.exp(mean_logprob)))
    return LAMBDA * p_norm + (1.0 - LAMBDA) * density


class AnnotationEngine:
    """Generate and govern entity annotations."""

    def __init__(self, template_path: str, provenance: ProvenanceLog | None = None,
                 expected_fields: int = 5):
        """expected_fields is the denominator of evidence density: the number of
        source fields a fully-populated entity is expected to carry. It is a
        calibration parameter of the confidence score, set to the typical
        attribute count of the demonstration schema; adjust it to match the
        attribute richness of a given deployment.
        """
        with open(template_path, "r", encoding="utf-8") as fh:
            self.template = fh.read()
        self.provenance = provenance if provenance is not None else ProvenanceLog()
        self.expected_fields = expected_fields
        self.review_queue: List[Dict[str, Any]] = []
        self.use_llm = bool(os.environ.get("OPENAI_API_KEY"))

    def _generate(self, entity_id: str, etype: str, attributes: Dict[str, Any],
                  relations: List[str]):
        """Return (text, mean_logprob). Uses GPT-4 if a key is present."""
        if not self.use_llm:
            return _deterministic_description(etype, attributes, relations), None
        # Real GPT-4 path. temperature=0.1, max_tokens=512 (fixed in the spec).
        try:
            from openai import OpenAI
            client = OpenAI()
            prompt = _render_template(self.template, entity_id, etype,
                                      attributes, relations)
            resp = client.chat.completions.create(
                model="gpt-4-turbo-2024-04-09",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=512, logprobs=True)
            choice = resp.choices[0]
            text = choice.message.content.strip()
            mean_lp = None
            if choice.logprobs and choice.logprobs.content:
                lps = [t.logprob for t in choice.logprobs.content]
                mean_lp = sum(lps) / len(lps) if lps else None
            return text, mean_lp
        except Exception as exc:
            # Any LLM-side failure (auth, rate limit, network) must fall back,
            # never break the pipeline.
            print(f"[module4] LLM call failed ({exc!r}); using fallback.",
                  file=sys.stderr)
            return _deterministic_description(etype, attributes, relations), None

    def annotate(self, entity_id: str, etype: str, attributes: Dict[str, Any],
                 relations: List[str] | None = None) -> Annotation:
        relations = relations or []
        text, mean_lp = self._generate(entity_id, etype, attributes, relations)

        # Control 1: fact-checking. Relations are legitimate graph context,
        # so they count as grounded evidence alongside the entity attributes.
        flagged = fact_check(text, attributes, extra_context=relations)
        grounded = not flagged

        # Control 2: confidence scoring.
        density = evidence_density(attributes, self.expected_fields)
        confidence = confidence_score(mean_lp, density)

        # Control 3: human validation routing.
        sensitive = bool(set(attributes.keys()) & SENSITIVE_FIELDS)
        routed = (confidence < REVIEW_THRESHOLD) or (not grounded) or sensitive
        reason = ""
        if not grounded:
            reason = "ungrounded terms: " + ", ".join(flagged)
        elif sensitive:
            reason = "sensitive category"
        elif confidence < REVIEW_THRESHOLD:
            reason = f"confidence {confidence:.2f} < {REVIEW_THRESHOLD}"

        annotation = Annotation(entity_id, text, confidence, grounded,
                                flagged, routed, reason)

        if routed:
            self.review_queue.append(annotation.as_dict())

        self.provenance.record(ProvenanceEntry(
            module="M4", operation="annotation", rule_id="LLM_ANNOTATE_01",
            rule_version="1.0", subject=entity_id, after=text,
            evidence={"confidence": round(confidence, 4), "grounded": grounded,
                      "routed_to_review": routed, "reason": reason}))
        return annotation

    def flush_review_queue(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.review_queue, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    engine = AnnotationEngine("prompts/annotation_template.txt")
    a1 = engine.annotate("ENT_000000", "Organization",
                         {"name": "Acme Corporation", "city": "London",
                          "country": "United Kingdom", "employees": 450})
    print("A1:", a1.as_dict())
    # Sensitive: routed to review regardless of confidence.
    a2 = engine.annotate("ENT_000009", "Organization",
                         {"name": "FinCo", "revenue": 1200000})
    print("A2:", a2.as_dict())
    print("review queue size:", len(engine.review_queue))
