from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .interfaces import FeatureProducer
from .models import Document, Head, TrustedContext

_INSTRUCTION_PATTERNS = (
    r"\bignore (all|any|the|previous|prior)\b",
    r"\bsystem prompt\b",
    r"\bdeveloper message\b",
    r"\boverride\b",
    r"\bdo not follow\b",
    r"\bact as\b",
    r"\breveal\b.*\b(prompt|secret|policy)\b",
)

_DANGEROUS_BOUNDARY = re.compile(r"</?(POLICY|REQUEST|EVIDENCE|DOC)\b", re.I)
_WORD = re.compile(r"\w+", re.UNICODE)


def _clip(value: float) -> float:
    if not math.isfinite(value):
        return 1.0
    return min(1.0, max(0.0, value))


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def _script_anomaly(text: str) -> float:
    scripts = Counter()
    suspicious = 0
    for char in text:
        if char.isspace() or char.isdigit() or unicodedata.category(char).startswith("P"):
            continue
        name = unicodedata.name(char, "")
        if "BENGALI" in name:
            scripts["bengali"] += 1
        elif "LATIN" in name:
            scripts["latin"] += 1
        elif name:
            scripts["other"] += 1
        if char in "\u202e\u202d\u2066\u2067\u2068\u2069\u200b\u200c\u200d":
            suspicious += 1
    total = sum(scripts.values())
    if total == 0:
        return 0.0
    mixed = 1.0 - max(scripts.values()) / total
    bidi = min(1.0, suspicious / max(1, len(text)) * 20)
    return _clip(0.7 * mixed + 0.3 * bidi)


def _source_concentration(documents: Sequence[Document]) -> float:
    if not documents:
        return 1.0
    counts = Counter(doc.source for doc in documents)
    return max(counts.values()) / len(documents)


def _language_mismatch(context: TrustedContext, documents: Sequence[Document]) -> float:
    known = [doc.language for doc in documents if doc.language]
    if not known:
        return 0.0
    majority = Counter(known).most_common(1)[0][0]
    user_script = "bn" if any("\u0980" <= c <= "\u09ff" for c in context.user) else "en"
    return 0.0 if majority.startswith(user_script) else 1.0


class HeuristicFeatureProducer(FeatureProducer):
    """Safe fallback feature producer. Replace with frozen learned components in production."""

    def produce(
        self,
        head: Head,
        context: TrustedContext,
        documents: Sequence[Document],
        response: str | None,
        *,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, float | None]:
        if head is Head.INSTRUCTION:
            user = context.user
            matches = sum(bool(re.search(p, user, re.I)) for p in _INSTRUCTION_PATTERNS)
            role_conflict = _clip(matches / 3)
            boundary = 1.0 if _DANGEROUS_BOUNDARY.search(user) else 0.0
            task_inconsistency = _clip(1.0 - _jaccard(user, f"{context.system} {context.developer}"))
            uncertainty = float(metadata.get("instruction_uncertainty", 0.0))
            return {
                "uncertainty": _clip(uncertainty),
                "task_inconsistency": task_inconsistency,
                "role_conflict": role_conflict,
                "boundary_violation": boundary,
                "script_anomaly": _script_anomaly(user),
            }

        if head is Head.RETRIEVAL:
            joined = " ".join(d.text for d in documents)
            query_overlap = _jaccard(context.user, joined) if documents else 0.0
            poison_hits = sum(
                bool(re.search(p, d.text, re.I))
                for d in documents
                for p in _INSTRUCTION_PATTERNS
            )
            poison = _clip(poison_hits / max(1, len(documents)))
            return {
                "facet_gap": 1.0 - query_overlap,
                "rank_instability": _clip(float(metadata.get("rank_instability", 0.0))),
                "source_concentration": _source_concentration(documents),
                "query_evidence_mismatch": 1.0 - query_overlap,
                "poison_likelihood": poison,
                "language_mismatch": _language_mismatch(context, documents),
            }

        if head is Head.EVIDENCE:
            if not documents:
                return {
                    "support_gap": 1.0,
                    "contradiction_density": 0.0,
                    "provenance_gap": 1.0,
                    "staleness": 0.0,
                    "fragmentation": 1.0,
                }
            analyzer_features = metadata.get("evidence_features", {})
            return {
                "support_gap": _clip(float(analyzer_features.get("support_gap", 0.5))),
                "contradiction_density": _clip(
                    float(analyzer_features.get("contradiction_density", 0.0))
                ),
                "provenance_gap": 1.0 - (1.0 / _source_concentration(documents)),
                "staleness": _clip(float(analyzer_features.get("staleness", 0.0))),
                "fragmentation": _clip(float(analyzer_features.get("fragmentation", 0.0))),
            }

        verifier = metadata.get("verification_features", {})
        if response is None:
            return {
                "unsupported_insufficient": 0.0,
                "unsupported_contradicted": 0.0,
                "unsupported_conflicting": 0.0,
                "dependency_unsupported": 0.0,
                "no_claim": 1.0,
                "segmentation_failure": 0.0,
                "verifier_failure": 0.0,
            }
        return {
            "unsupported_insufficient": _clip(
                float(verifier.get("unsupported_insufficient", 0.0))
            ),
            "unsupported_contradicted": _clip(
                float(verifier.get("unsupported_contradicted", 0.0))
            ),
            "unsupported_conflicting": _clip(
                float(verifier.get("unsupported_conflicting", 0.0))
            ),
            "dependency_unsupported": _clip(
                float(verifier.get("dependency_unsupported", 0.0))
            ),
            "no_claim": 1.0 if not _WORD.search(response) else 0.0,
            "segmentation_failure": _clip(float(verifier.get("segmentation_failure", 0.0))),
            "verifier_failure": _clip(float(verifier.get("verifier_failure", 0.0))),
        }
