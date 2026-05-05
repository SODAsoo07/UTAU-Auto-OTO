from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional


def append_preserved_lines(
    final_lines: List[str],
    lines: Iterable[str],
    *,
    alias_suffix: str,
    apply_suffix_to_oto_line_fn: Callable[[str, str], str],
) -> None:
    final_lines.extend([apply_suffix_to_oto_line_fn(line, alias_suffix) for line in (lines or [])])


def prepare_file_context_with_sinsy(
    *,
    fname: str,
    lines: List[str],
    prepare_file_context_fn: Callable[..., Any],
    prepare_kwargs: Dict[str, Any],
    use_sinsy_labels: bool,
    load_sinsy_label_entries_fn: Optional[Callable[..., List[Dict[str, Any]]]],
    sinsy_label_path: str,
) -> Any:
    file_ctx = prepare_file_context_fn(
        fname=fname,
        lines=lines,
        **dict(prepare_kwargs or {}),
    )
    file_ctx.sinsy_label_entries = []
    file_ctx.sinsy_label_path = ""
    if use_sinsy_labels and callable(load_sinsy_label_entries_fn):
        try:
            file_ctx.sinsy_label_entries = load_sinsy_label_entries_fn(
                tg_path=str(getattr(file_ctx, "tg_path", "") or ""),
                real_wav_name=str(getattr(file_ctx, "real_wav_name", "") or ""),
                explicit_path=str(sinsy_label_path or ""),
            )
            if file_ctx.sinsy_label_entries:
                file_ctx.sinsy_label_path = str(sinsy_label_path or "")
        except Exception:
            file_ctx.sinsy_label_entries = []
            file_ctx.sinsy_label_path = ""
    return file_ctx


def load_named_tiers_for_generation(
    *,
    file_ctx: Any,
    load_named_tiers_fn: Callable[..., Any],
    load_textgrid_fn: Callable[[str], Any],
    preloaded_tg_by_path: Optional[Dict[str, Any]] = None,
    tier_predicate: Optional[Callable[[Any], bool]] = None,
) -> Any:
    kwargs: Dict[str, Any] = {
        "load_textgrid_fn": load_textgrid_fn,
    }
    if preloaded_tg_by_path is not None:
        kwargs["preloaded_tg_by_path"] = preloaded_tg_by_path
    if tier_predicate is not None:
        kwargs["tier_predicate"] = tier_predicate
    return load_named_tiers_fn(file_ctx, **kwargs)


def resolve_mapping_failure_reason(
    *,
    low_quality_reasons,
    low_phone_quality: bool,
    has_word_intervals: bool,
    has_phone_intervals: bool = True,
) -> str:
    reasons = {str(x or "").strip() for x in (low_quality_reasons or []) if str(x or "").strip()}
    reason = "mapping_failed"
    if "spn_heavy" in reasons:
        reason = "mapping_failed_spn_heavy"
    elif ("insufficient_phones" in reasons) or ("insufficient_vowel_phones" in reasons):
        reason = "mapping_failed_insufficient_phones"
    elif low_phone_quality and (not has_word_intervals):
        reason = "mapping_failed_no_words_support"
    elif not has_phone_intervals:
        reason = "mapping_failed_empty_intervals"
    return reason


def handle_mapping_failure_fallback(
    *,
    fail_reason: str,
    fname: str,
    lines: List[str],
    final_lines: List[str],
    alias_suffix: str,
    log_fn: Callable[[str], None],
    failure_log_message: str,
    record_unset_lines_fn: Callable[..., None],
    meta: Optional[Dict[str, Any]],
    apply_suffix_to_oto_line_fn: Callable[[str, str], str],
) -> None:
    if str(failure_log_message or "").strip():
        log_fn(str(failure_log_message))
    record_unset_lines_fn(
        str(fail_reason or "mapping_failed"),
        fname,
        lines,
        meta=dict(meta or {}),
    )
    append_preserved_lines(
        final_lines,
        lines,
        alias_suffix=alias_suffix,
        apply_suffix_to_oto_line_fn=apply_suffix_to_oto_line_fn,
    )


def handle_mapping_abstain_fallback(
    *,
    fname: str,
    lines: List[str],
    final_lines: List[str],
    alias_suffix: str,
    log_fn: Callable[[str], None],
    abstain_log_message: str,
    record_unset_lines_fn: Callable[..., None],
    meta: Optional[Dict[str, Any]],
    apply_suffix_to_oto_line_fn: Callable[[str, str], str],
) -> None:
    if str(abstain_log_message or "").strip():
        log_fn(str(abstain_log_message))
    record_unset_lines_fn(
        "mapping_v2_abstain",
        fname,
        lines,
        meta=dict(meta or {}),
    )
    append_preserved_lines(
        final_lines,
        lines,
        alias_suffix=alias_suffix,
        apply_suffix_to_oto_line_fn=apply_suffix_to_oto_line_fn,
    )


def resolve_mapping_line_alias(
    *,
    line: str,
    fname: str,
    real_wav_name: str,
    final_lines: List[str],
    alias_suffix: str,
    record_unset_fn: Callable[[str, str, str], None],
    apply_suffix_to_oto_line_fn: Callable[[str, str], str],
) -> Optional[str]:
    parts = line.split("=", 1)
    if len(parts) < 2:
        record_unset_fn("malformed_line", fname, line)
        final_lines.append(apply_suffix_to_oto_line_fn(line, alias_suffix))
        return None
    alias = parts[1].split(",", 1)[0].strip()
    if alias:
        return alias
    # Preserve malformed alias rows (`wav=,offset,...`) to avoid dropping user lines.
    record_unset_fn("empty_alias", fname, line)
    preserved = f"{real_wav_name}={parts[1]}"
    final_lines.append(apply_suffix_to_oto_line_fn(preserved, alias_suffix))
    return None


def handle_kr_file_context_status(
    *,
    file_ctx: Any,
    fname: str,
    lines: List[str],
    final_lines: List[str],
    alias_suffix: str,
    log_fn: Callable[[str], None],
    record_unset_lines_fn: Callable[..., None],
    apply_suffix_to_oto_line_fn: Callable[[str, str], str],
) -> bool:
    status = str(getattr(file_ctx, "status", "") or "").strip().lower()
    if status == "textgrid_missing":
        log_fn(f"[ERROR] {fname}: TextGrid를 찾지 못해 자동 설정을 중단합니다.")
        record_unset_lines_fn("textgrid_missing", fname, lines)
        append_preserved_lines(
            final_lines,
            lines,
            alias_suffix=alias_suffix,
            apply_suffix_to_oto_line_fn=apply_suffix_to_oto_line_fn,
        )
        return True
    if status == "textgrid_load_failed":
        err_msg = str(getattr(file_ctx, "error_message", "") or "")
        log_fn(f"[ERROR] {fname}: TextGrid 로드 실패로 자동 설정을 중단합니다. ({err_msg})")
        record_unset_lines_fn("textgrid_load_failed", fname, lines)
        append_preserved_lines(
            final_lines,
            lines,
            alias_suffix=alias_suffix,
            apply_suffix_to_oto_line_fn=apply_suffix_to_oto_line_fn,
        )
        return True
    if status == "tier_missing":
        log_fn(f"[ERROR] {fname}: phones tier가 없어 자동 설정을 중단합니다.")
        record_unset_lines_fn("tier_missing", fname, lines)
        append_preserved_lines(
            final_lines,
            lines,
            alias_suffix=alias_suffix,
            apply_suffix_to_oto_line_fn=apply_suffix_to_oto_line_fn,
        )
        return True
    return False


def handle_ja_file_context_status(
    *,
    file_ctx: Any,
    fname: str,
    lines: List[str],
    final_lines: List[str],
    alias_suffix: str,
    log_fn: Callable[[str], None],
    record_unset_lines_fn: Callable[..., None],
    apply_suffix_to_oto_line_fn: Callable[[str, str], str],
    debug_reason_logging: bool,
) -> bool:
    status = str(getattr(file_ctx, "status", "") or "").strip().lower()
    if status == "textgrid_missing":
        miss_diag = dict(getattr(file_ctx, "diagnostics", {}) or {})
        log_fn(
            f"❌ {fname}: TextGrid 없음 → 자동 설정 중단 "
            f"(norm={miss_diag.get('lookup_norm', '')}, 후보={len(miss_diag.get('norm_candidates', []) or [])})"
        )
        if debug_reason_logging:
            for path in list(miss_diag.get("candidate_paths", []) or [])[:3]:
                log_fn(f"   ㄴ 후보 경로: {path}")
        record_unset_lines_fn("textgrid_missing", fname, lines, meta=miss_diag)
        append_preserved_lines(
            final_lines,
            lines,
            alias_suffix=alias_suffix,
            apply_suffix_to_oto_line_fn=apply_suffix_to_oto_line_fn,
        )
        return True
    if status == "textgrid_load_failed":
        err_msg = str(getattr(file_ctx, "error_message", "") or "")
        log_fn(f"❌ {fname}: TextGrid 로드 실패 → 자동 설정 중단 ({err_msg})")
        record_unset_lines_fn("textgrid_load_failed", fname, lines)
        append_preserved_lines(
            final_lines,
            lines,
            alias_suffix=alias_suffix,
            apply_suffix_to_oto_line_fn=apply_suffix_to_oto_line_fn,
        )
        return True
    if status == "tier_missing":
        log_fn(f"❌ {fname}: phones 티어 없음 → 자동 설정 중단")
        record_unset_lines_fn("tier_missing", fname, lines)
        append_preserved_lines(
            final_lines,
            lines,
            alias_suffix=alias_suffix,
            apply_suffix_to_oto_line_fn=apply_suffix_to_oto_line_fn,
        )
        return True
    return False


def handle_kr_loop_prep_status(
    *,
    loop_prep: Any,
    fname: str,
    lines: List[str],
    final_lines: List[str],
    alias_suffix: str,
    log_fn: Callable[[str], None],
    record_unset_lines_fn: Callable[..., None],
    apply_suffix_to_oto_line_fn: Callable[[str, str], str],
) -> bool:
    status = str(getattr(loop_prep, "status", "") or "").strip().lower()
    if status == "empty_intervals":
        log_fn(f"[ERROR] {fname}: 유효한 음소 구간이 없어 자동 설정을 중단합니다.")
        record_unset_lines_fn("mapping_failed_empty_intervals", fname, lines)
        append_preserved_lines(
            final_lines,
            lines,
            alias_suffix=alias_suffix,
            apply_suffix_to_oto_line_fn=apply_suffix_to_oto_line_fn,
        )
        return True
    if status == "no_valid_alias":
        log_fn(f"[ERROR] {fname}: 유효한 alias를 해석하지 못해 자동 설정을 중단합니다.")
        record_unset_lines_fn("no_valid_alias", fname, lines)
        append_preserved_lines(
            final_lines,
            lines,
            alias_suffix=alias_suffix,
            apply_suffix_to_oto_line_fn=apply_suffix_to_oto_line_fn,
        )
        return True
    return False


def _ja_empty_interval_reason(loop_prep: Any) -> str:
    low_quality_reasons = list(getattr(loop_prep, "low_quality_reasons", []) or [])
    reason = "mapping_failed_empty_intervals"
    low_phone_quality = bool(getattr(loop_prep, "low_phone_quality", False))
    wd_intervals = list(getattr(loop_prep, "wd_intervals", []) or [])
    if low_phone_quality and "spn_heavy" in low_quality_reasons:
        reason = "mapping_failed_spn_heavy"
    elif low_phone_quality and any(
        entry in {"insufficient_phones", "insufficient_vowel_phones"} for entry in low_quality_reasons
    ):
        reason = "mapping_failed_insufficient_phones"
    if low_phone_quality and not wd_intervals:
        reason = "mapping_failed_no_words_support"
    return reason


def handle_ja_loop_prep_status(
    *,
    loop_prep: Any,
    fname: str,
    lines: List[str],
    final_lines: List[str],
    alias_suffix: str,
    log_fn: Callable[[str], None],
    record_unset_lines_fn: Callable[..., None],
    apply_suffix_to_oto_line_fn: Callable[[str, str], str],
) -> bool:
    status = str(getattr(loop_prep, "status", "") or "").strip().lower()
    if status == "no_valid_alias":
        log_fn(f"❌ {fname}: 유효한 에일리어스가 없어 자동 설정 중단")
        record_unset_lines_fn("no_valid_alias", fname, lines)
        append_preserved_lines(
            final_lines,
            lines,
            alias_suffix=alias_suffix,
            apply_suffix_to_oto_line_fn=apply_suffix_to_oto_line_fn,
        )
        return True
    if status == "empty_intervals":
        reason = _ja_empty_interval_reason(loop_prep)
        meta = dict(getattr(loop_prep, "meta", {}) or {})
        log_fn(f"❌ {fname}: 음소 정보 없음 → 자동 설정 중단")
        record_unset_lines_fn(reason, fname, lines, meta=meta)
        append_preserved_lines(
            final_lines,
            lines,
            alias_suffix=alias_suffix,
            apply_suffix_to_oto_line_fn=apply_suffix_to_oto_line_fn,
        )
        return True
    return False
