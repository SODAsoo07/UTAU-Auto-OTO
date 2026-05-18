from __future__ import annotations

from dataclasses import dataclass, field


ANCHOR_FIELDS: tuple[str, ...] = (
    "offset_abs",
    "overlap_abs",
    "pre_abs",
    "consonant_abs",
    "cutoff_abs",
)

ROLE_VOCAB: tuple[str, ...] = (
    "other",
    "-cv",
    "cv",
    "v",
    "special",
    "vc",
    "vv",
    "v-cv",
    "v-",
    "cv-",
    "br",
    "endbr",
)

ALIAS_TYPE_VOCAB: tuple[str, ...] = (
    "other",
    "cv",
    "-cv",
    "v",
    "vc",
    "vv",
    "v-cv",
    "v-",
    "cv-",
    "br",
    "endbr",
    "special",
)

LANGUAGE_VOCAB: tuple[str, ...] = ("other", "korean", "japanese", "english")
FORMAT_VOCAB: tuple[str, ...] = ("other", "cv", "cvc", "cvvc", "vcv", "cmpx")


@dataclass(frozen=True)
class Stage2FeatureRow:
    numeric: tuple[float, ...]
    role_id: int
    prev_role_id: int
    next_role_id: int
    alias_type_id: int
    language_id: int
    format_id: int
    base_anchors_ms: tuple[float, float, float, float, float]


@dataclass(frozen=True)
class Stage2FeatureBatch:
    rows: tuple[Stage2FeatureRow, ...]
    numeric_dim: int


@dataclass(frozen=True)
class Stage2Decision:
    anchors_ms: tuple[float, float, float, float, float]
    confidence: float
    accepted: bool
    reason: str
    max_candidate_distance_ms: float = 0.0


@dataclass(frozen=True)
class Stage2ModelConfig:
    numeric_dim: int
    hidden_dim: int = 128
    role_count: int = len(ROLE_VOCAB)
    alias_type_count: int = len(ALIAS_TYPE_VOCAB)
    language_count: int = len(LANGUAGE_VOCAB)
    format_count: int = len(FORMAT_VOCAB)
    role_emb_dim: int = 16
    alias_type_emb_dim: int = 12
    language_emb_dim: int = 8
    format_emb_dim: int = 8
    gru_layers: int = 2

    @classmethod
    def from_mapping(cls, data: dict[str, object]) -> "Stage2ModelConfig":
        return cls(
            numeric_dim=int(data.get("numeric_dim", 0) or 0),
            hidden_dim=int(data.get("hidden_dim", 128) or 128),
            role_count=int(data.get("role_count", len(ROLE_VOCAB)) or len(ROLE_VOCAB)),
            alias_type_count=int(data.get("alias_type_count", len(ALIAS_TYPE_VOCAB)) or len(ALIAS_TYPE_VOCAB)),
            language_count=int(data.get("language_count", len(LANGUAGE_VOCAB)) or len(LANGUAGE_VOCAB)),
            format_count=int(data.get("format_count", len(FORMAT_VOCAB)) or len(FORMAT_VOCAB)),
            role_emb_dim=int(data.get("role_emb_dim", 16) or 16),
            alias_type_emb_dim=int(data.get("alias_type_emb_dim", 12) or 12),
            language_emb_dim=int(data.get("language_emb_dim", 8) or 8),
            format_emb_dim=int(data.get("format_emb_dim", 8) or 8),
            gru_layers=int(data.get("gru_layers", 2) or 2),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "numeric_dim": int(self.numeric_dim),
            "hidden_dim": int(self.hidden_dim),
            "role_count": int(self.role_count),
            "alias_type_count": int(self.alias_type_count),
            "language_count": int(self.language_count),
            "format_count": int(self.format_count),
            "role_emb_dim": int(self.role_emb_dim),
            "alias_type_emb_dim": int(self.alias_type_emb_dim),
            "language_emb_dim": int(self.language_emb_dim),
            "format_emb_dim": int(self.format_emb_dim),
            "gru_layers": int(self.gru_layers),
        }


@dataclass(frozen=True)
class Stage2CheckpointMeta:
    feature_schema_version: int = 1
    target_mode: str = "absolute_anchor_from_boundary_candidates"
    role_vocab: tuple[str, ...] = field(default_factory=lambda: ROLE_VOCAB)
    alias_type_vocab: tuple[str, ...] = field(default_factory=lambda: ALIAS_TYPE_VOCAB)
    language_vocab: tuple[str, ...] = field(default_factory=lambda: LANGUAGE_VOCAB)
    format_vocab: tuple[str, ...] = field(default_factory=lambda: FORMAT_VOCAB)

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_schema_version": int(self.feature_schema_version),
            "target_mode": str(self.target_mode),
            "role_vocab": list(self.role_vocab),
            "alias_type_vocab": list(self.alias_type_vocab),
            "language_vocab": list(self.language_vocab),
            "format_vocab": list(self.format_vocab),
        }


def vocab_index(value: object, vocab: tuple[str, ...]) -> int:
    text = str(value or "").strip().lower()
    if not text:
        return 0
    try:
        return int(vocab.index(text))
    except ValueError:
        return 0


__all__ = [
    "ALIAS_TYPE_VOCAB",
    "ANCHOR_FIELDS",
    "FORMAT_VOCAB",
    "LANGUAGE_VOCAB",
    "ROLE_VOCAB",
    "Stage2CheckpointMeta",
    "Stage2Decision",
    "Stage2FeatureBatch",
    "Stage2FeatureRow",
    "Stage2ModelConfig",
    "vocab_index",
]
