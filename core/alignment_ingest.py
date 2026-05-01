from __future__ import annotations

from dataclasses import dataclass, field


def _seq_seg_log_enabled() -> bool:
    import os
    return str(os.environ.get("UTOA_SEQ_SEGMENT_LOG", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}


def _apply_viterbi_correction(phones: list, file_ctx, loop_prep) -> list:
    """Apply sequence-segment Viterbi boundary correction when alignment trust is low/mid."""
    wav_path = str(getattr(file_ctx, "wav_path_for_signal", "") or "")
    if not wav_path or not phones:
        return phones
    trust_tier = str(getattr(loop_prep, "textgrid_trust_tier", "low") or "low")
    prefer_seq = bool(getattr(loop_prep, "prefer_filename_sequence", False))
    try:
        from core.sequence_segment import correct_phone_intervals_by_viterbi
        corrected = correct_phone_intervals_by_viterbi(
            phones,
            wav_path,
            trust_tier=trust_tier,
            prefer_sequence=prefer_seq,
        )
    except Exception:
        return phones

    if corrected is phones or corrected == phones:
        return phones

    if _seq_seg_log_enabled() and len(phones) == len(corrected):
        wav_name = str(getattr(file_ctx, "real_wav_name", "") or "")
        shifts = []
        for orig, corr in zip(phones, corrected):
            orig_start = float(getattr(orig, "minTime", 0.0)) * 1000.0
            corr_start = float(getattr(corr, "minTime", 0.0)) * 1000.0
            orig_end = float(getattr(orig, "maxTime", 0.0)) * 1000.0
            corr_end = float(getattr(corr, "maxTime", 0.0)) * 1000.0
            shifts.append(abs(corr_start - orig_start) + abs(corr_end - orig_end))
        avg_shift = sum(shifts) / len(shifts) if shifts else 0.0
        max_shift = max(shifts) if shifts else 0.0
        print(
            f"[SeqSeg] {wav_name} trust={trust_tier} prefer_seq={prefer_seq}"
            f" phones={len(phones)} avg_shift={avg_shift:.1f}ms max_shift={max_shift:.1f}ms"
        )
        if max_shift > 5.0:
            details = []
            for orig, corr in zip(phones, corrected):
                os_ms = float(getattr(orig, "minTime", 0.0)) * 1000.0
                cs_ms = float(getattr(corr, "minTime", 0.0)) * 1000.0
                oe_ms = float(getattr(orig, "maxTime", 0.0)) * 1000.0
                ce_ms = float(getattr(corr, "maxTime", 0.0)) * 1000.0
                shift = abs(cs_ms - os_ms) + abs(ce_ms - oe_ms)
                if shift > 5.0:
                    details.append(
                        f"  [{orig.mark}] {os_ms:.0f}-{oe_ms:.0f}ms → {cs_ms:.0f}-{ce_ms:.0f}ms"
                    )
            for line in details:
                print(line)

    return corrected


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
    phones = list(getattr(loop_prep, "ph_intervals", []) or [])
    snapshot.phones = _apply_viterbi_correction(phones, file_ctx, loop_prep)
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
        "alignment_source": str(getattr(loop_prep, "alignment_source", getattr(file_ctx, "alignment_source", "")) or ""),
        "alignment_source_reason": str(
            getattr(loop_prep, "alignment_source_reason", getattr(file_ctx, "alignment_source_reason", "")) or ""
        ),
        "alignment_source_meta": dict(
            getattr(loop_prep, "alignment_source_meta", getattr(file_ctx, "alignment_source_meta", {})) or {}
        ),
        "ja_style_profile": getattr(loop_prep, "ja_style_profile", None),
        "forced_words_mapping": bool(getattr(loop_prep, "forced_words_mapping", False)),
        "conf_th": float(getattr(loop_prep, "conf_th", 0.0) or 0.0),
        "sinsy_label_entries": list(getattr(file_ctx, "sinsy_label_entries", []) or []),
        "sinsy_label_path": str(getattr(file_ctx, "sinsy_label_path", "") or ""),
    }
    return snapshot


def build_kr_alignment_ingest(file_ctx, loop_prep):
    snapshot = _base_snapshot("korean", file_ctx, loop_prep)
    phones = list(getattr(loop_prep, "ph_intervals", []) or [])
    snapshot.phones = _apply_viterbi_correction(phones, file_ctx, loop_prep)
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
        "alignment_source": str(getattr(file_ctx, "alignment_source", "") or ""),
        "alignment_source_reason": str(getattr(file_ctx, "alignment_source_reason", "") or ""),
        "alignment_source_meta": dict(getattr(file_ctx, "alignment_source_meta", {}) or {}),
        "sinsy_label_entries": list(getattr(file_ctx, "sinsy_label_entries", []) or []),
        "sinsy_label_path": str(getattr(file_ctx, "sinsy_label_path", "") or ""),
    }
    return snapshot


__all__ = [
    "AlignmentIngestSnapshot",
    "build_ja_alignment_ingest",
    "build_kr_alignment_ingest",
]
