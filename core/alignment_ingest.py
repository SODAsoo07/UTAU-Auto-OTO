from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AlignmentIngestSnapshot:
    language: str
    real_wav_name: str = ""
    tg_path: str = ""
    phones: list = field(default_factory=list)
    phones_all: list = field(default_factory=list)
    words: list = field(default_factory=list)
    textgrid_trust_score: float = 0.0
    textgrid_trust_tier: str = "low"
    prefer_filename_sequence: bool = False
    phone_quality: dict[str, object] = field(default_factory=dict)
    low_quality_reasons: list[str] = field(default_factory=list)
    low_phone_quality: bool = False
    wav_duration_ms: float = 0.0
    mel_ctx_for_file: object = None
    timeline_meta: dict[str, object] = field(default_factory=dict)
    extra: dict[str, object] = field(default_factory=dict)


def _base_snapshot(language, file_ctx, loop_prep):
    return AlignmentIngestSnapshot(
        language=str(language or "").strip().lower(),
        real_wav_name=str(getattr(file_ctx, "real_wav_name", "") or ""),
        tg_path=str(getattr(file_ctx, "tg_path", "") or ""),
        textgrid_trust_score=float(getattr(loop_prep, "textgrid_trust_score", 0.0) or 0.0),
        textgrid_trust_tier=str(getattr(loop_prep, "textgrid_trust_tier", "low") or "low"),
        prefer_filename_sequence=bool(getattr(loop_prep, "prefer_filename_sequence", False)),
        phone_quality=dict(getattr(loop_prep, "phone_quality", {}) or {}),
        low_quality_reasons=list(getattr(loop_prep, "low_quality_reasons", []) or []),
        low_phone_quality=bool(getattr(loop_prep, "low_phone_quality", False)),
        wav_duration_ms=float(getattr(loop_prep, "wav_duration_ms", getattr(file_ctx, "wav_duration_ms", 0.0)) or 0.0),
        mel_ctx_for_file=getattr(file_ctx, "mel_ctx_for_file", None),
    )


def build_ja_alignment_ingest(file_ctx, loop_prep):
    snapshot = _base_snapshot("japanese", file_ctx, loop_prep)
    snapshot.phones = list(getattr(loop_prep, "ph_intervals", []) or [])
    snapshot.words = list(getattr(loop_prep, "wd_intervals", []) or [])
    snapshot.timeline_meta = {
        "timeline_start_ms": float(getattr(loop_prep, "timeline_start_ms", 0.0) or 0.0),
        "timeline_end_ms": float(getattr(loop_prep, "timeline_end_ms", 0.0) or 0.0),
        "effective_end_ms": float(getattr(loop_prep, "effective_end_ms", 0.0) or 0.0),
        "boundary_points_ms": list(getattr(loop_prep, "boundary_points_ms", []) or []),
        "phone_spans_ms": list(getattr(loop_prep, "phone_spans_ms", []) or []),
    }
    snapshot.extra = {
        "filename_syllables": list(getattr(loop_prep, "filename_syllables", []) or []),
        "cv_targets": list(getattr(loop_prep, "cv_targets", []) or []),
        "detected_format": str(getattr(loop_prep, "detected_format", "") or ""),
        "format_type": str(getattr(loop_prep, "format_type", "") or ""),
        "ja_style_profile": getattr(loop_prep, "ja_style_profile", None),
        "forced_words_mapping": bool(getattr(loop_prep, "forced_words_mapping", False)),
        "conf_th": float(getattr(loop_prep, "conf_th", 0.0) or 0.0),
        "sinsy_label_entries": list(getattr(file_ctx, "sinsy_label_entries", []) or []),
        "sinsy_label_path": str(getattr(file_ctx, "sinsy_label_path", "") or ""),
    }
    return snapshot


def build_kr_alignment_ingest(file_ctx, loop_prep):
    snapshot = _base_snapshot("korean", file_ctx, loop_prep)
    snapshot.phones = list(getattr(loop_prep, "ph_intervals", []) or [])
    snapshot.phones_all = list(getattr(loop_prep, "ph_intervals_all", []) or [])
    snapshot.words = list(getattr(loop_prep, "wd_intervals", []) or [])
    snapshot.timeline_meta = {
        "timeline_start_ms": float(getattr(loop_prep, "timeline_start_ms", 0.0) or 0.0),
        "timeline_end_ms": float(getattr(loop_prep, "timeline_end_ms", 0.0) or 0.0),
    }
    snapshot.extra = {
        "file_format": str(getattr(loop_prep, "file_format", "") or ""),
        "file_mapping_conf_th": float(getattr(loop_prep, "file_mapping_conf_th", 0.0) or 0.0),
        "filename_cv_targets": list(getattr(loop_prep, "filename_cv_targets", []) or []),
        "targets_for_build": list(getattr(loop_prep, "targets_for_build", []) or []),
        "force_words_phone_fill": bool(getattr(loop_prep, "force_words_phone_fill", False)),
        "sinsy_label_entries": list(getattr(file_ctx, "sinsy_label_entries", []) or []),
        "sinsy_label_path": str(getattr(file_ctx, "sinsy_label_path", "") or ""),
    }
    return snapshot


__all__ = [
    "AlignmentIngestSnapshot",
    "build_ja_alignment_ingest",
    "build_kr_alignment_ingest",
]
