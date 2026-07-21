from __future__ import annotations

import hashlib
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .errors import CadreError


class ArtifactError(CadreError):
    pass


class ModelArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    code_version: str = ""
    data_version: str = ""
    feature_schema_version: str = ""
    risk_head_versions: dict[str, str] = Field(default_factory=dict)
    calibrator_versions: dict[str, str] = Field(default_factory=dict)
    threshold_versions: dict[str, str] = Field(default_factory=dict)
    policy_version: str = ""
    hashes: dict[str, str] = Field(default_factory=dict)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_artifact(
    obj: Any, path: Path, *, manifest: ModelArtifactManifest | None = None
) -> str:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    digest = _sha256(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if manifest is not None:
        manifest_path = path.with_suffix(".manifest.json")
        updated = manifest.model_copy(
            update={"hashes": {**manifest.hashes, path.name: digest}}
        )
        manifest_path.write_text(updated.model_dump_json(indent=2))
    return digest


def load_artifact(path: Path, *, expected_hash: str | None = None) -> Any:
    if not path.exists():
        raise ArtifactError(f"artifact not found: {path}")
    data = path.read_bytes()
    if expected_hash is not None:
        actual = _sha256(data)
        if actual != expected_hash:
            raise ArtifactError(
                f"hash mismatch for {path}: expected {expected_hash}, got {actual}"
            )
    return pickle.loads(data)  # noqa: S301


def load_manifest(path: Path) -> ModelArtifactManifest:
    if not path.exists():
        raise ArtifactError(f"manifest not found: {path}")
    return ModelArtifactManifest.model_validate_json(path.read_text())
