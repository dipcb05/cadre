from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from .errors import CalibrationError


class Lineage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    template_id: str = ""
    instantiation_id: str = ""
    context_id: str = ""
    example_id: str = ""
    lineage_id: str = ""

    def compute_lineage_id(self) -> str:
        key = json.dumps(
            [self.source_id, self.template_id, self.instantiation_id, self.context_id],
            sort_keys=True,
        )
        return hashlib.sha256(key.encode()).hexdigest()[:16]


class LineageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lineage: Lineage
    features: dict[str, float | None] = Field(default_factory=dict)
    label: int = 0


def grouped_split(
    records: Sequence[LineageRecord],
    *,
    ratios: tuple[float, ...] = (0.4, 0.15, 0.15, 0.15, 0.15),
    names: tuple[str, ...] = ("D_fit", "D_prob", "D_threshold", "D_policy", "D_test"),
) -> dict[str, list[LineageRecord]]:
    if len(ratios) != len(names):
        raise CalibrationError("ratios and names must have equal length")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise CalibrationError("ratios must sum to 1.0")

    groups: dict[str, list[LineageRecord]] = defaultdict(list)
    for record in records:
        lid = record.lineage.lineage_id or record.lineage.compute_lineage_id()
        groups[lid].append(record)

    sorted_ids = sorted(groups.keys())
    total = len(sorted_ids)
    splits: dict[str, list[LineageRecord]] = {name: [] for name in names}
    cursor = 0
    for i, name in enumerate(names):
        if i == len(names) - 1:
            count = total - cursor
        else:
            count = max(1, round(ratios[i] * total)) if cursor < total else 0
        for lid in sorted_ids[cursor : cursor + count]:
            splits[name].extend(groups[lid])
        cursor += count

    return splits


def assert_no_overlap(*splits: Sequence[LineageRecord]) -> None:
    seen: dict[str, int] = {}
    for i, split in enumerate(splits):
        ids = {r.lineage.lineage_id or r.lineage.compute_lineage_id() for r in split}
        for lid in ids:
            if lid in seen:
                raise CalibrationError(
                    f"lineage '{lid}' appears in splits {seen[lid]} and {i}"
                )
            seen[lid] = i
