"""
Provenance: the auditable backbone of the framework.

The paper's central claim is that transparency is *architectural*, not a logging
layer bolted on afterward. Every module therefore commits a provenance entry
before passing output downstream, so that any artifact (a normalized value, a
merged entity, a graph edge, an annotation) can be traced back to the rule and
the evidence that produced it.

This module defines the single provenance record type used across all four
modules and a writer that persists the full chain.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ProvenanceEntry:
    """One auditable step, attributable to a specific rule and module.

    The fields are deliberately uniform across modules so that an auditor reads
    a single, consistent trace from raw input to final annotation.
    """

    module: str
    operation: str
    rule_id: str
    rule_version: str
    subject: str
    before: Any = None
    after: Any = None
    evidence: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=_now)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProvenanceLog:
    """An ordered, serializable chain of provenance entries."""

    def __init__(self) -> None:
        self._entries: List[ProvenanceEntry] = []

    def record(self, entry: ProvenanceEntry) -> None:
        self._entries.append(entry)

    def extend(self, entries: List[ProvenanceEntry]) -> None:
        self._entries.extend(entries)

    def entries(self) -> List[ProvenanceEntry]:
        return list(self._entries)

    def for_subject(self, subject: str) -> List[ProvenanceEntry]:
        """Return the full trace for one subject (e.g., a single entity id)."""
        return [e for e in self._entries if e.subject == subject]

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([e.as_dict() for e in self._entries], fh, indent=2,
                      ensure_ascii=False, default=str)

    def __len__(self) -> int:
        return len(self._entries)
