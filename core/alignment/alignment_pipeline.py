from __future__ import annotations

import os
from typing import Dict, List

from core.sequence_aligner import check_sequence_aligner_ready, get_last_sequence_align_meta, run_sequence_align
from core.wfl_runner import check_wfl_ready, get_last_wfl_align_meta, run_wfl_align
from core.pipeline_status import (
    ALIGN_OUTPUT_EMPTY,
    ALIGN_RUN_FAILED,
    ALIGN_USING_EXISTING,
    OK,
    classify_alignment_error,
    compute_alignment_quality_score,
    has_textgrid_files,
    make_runtime_report,
    normalize_aligner_name,
    resolve_aligner_chain,
)


def _emit(callback, message: str) -> None:
    if not callback:
        return
    try:
        callback(message)
    except Exception:
        return


def _clear_existing_textgrids(path: str, callback=None) -> tuple[int, int]:
    removed = 0
    failed = 0
    if not path or not os.path.isdir(path):
        return removed, failed
    for root, _dirs, files in os.walk(path):
        for name in files:
            if not str(name).lower().endswith(".textgrid"):
                continue
            full_path = os.path.join(root, name)
            try:
                os.remove(full_path)
                removed += 1
            except Exception as exc:
                failed += 1
                _emit(callback, f"[Align] failed to remove existing TextGrid: {full_path} ({exc})")
    return removed, failed


def run_alignment_with_fallback(
    *,
    language: str,
    wav_folder: str,
    dictionary_path: str,
    output_folder: str,
    format_hint: str = "",
    primary_aligner: str = "wfl",
    fallback_aligner: str = "",
    overwrite_existing_textgrids: bool = False,
    callback=None,
    # Legacy MFA parameters — accepted but ignored for backward compatibility.
    mfa_path: str = "",  # noqa: ARG001
    mfa_align_profile: str = "",  # noqa: ARG001
    mfa_runtime_options: Dict[str, object] | None = None,  # noqa: ARG001
) -> Dict[str, object]:
    lang = str(language or "korean").strip().lower()
    attempts = []

    normalized_primary = normalize_aligner_name(primary_aligner, default="wfl")
    resolved_fallback = fallback_aligner
    chain = resolve_aligner_chain(primary_aligner, resolved_fallback)
    # Filter out any "mfa" entries that may appear in legacy configurations.
    chain = [e for e in chain if e != "mfa"]
    if not chain:
        chain = ["wfl"]
    primary = chain[0]

    if has_textgrid_files(output_folder):
        if not bool(overwrite_existing_textgrids):
            return make_runtime_report(
                "align",
                ALIGN_USING_EXISTING,
                "TextGrid already exists.",
                ok=True,
                primary_engine=primary,
                used_engine="existing",
                attempts=attempts,
                fallback_used=False,
                attempt_count=0,
            )
        removed, failed = _clear_existing_textgrids(output_folder, callback=callback)
        _emit(callback, f"[Align] existing TextGrid overwrite requested: removed={removed}, failed={failed}")
        if has_textgrid_files(output_folder):
            return make_runtime_report(
                "align",
                ALIGN_RUN_FAILED,
                "Failed to overwrite existing TextGrid files.",
                ok=False,
                primary_engine=primary,
                used_engine="",
                attempts=attempts,
                fallback_used=False,
                attempt_count=0,
            )

    run_attempt_count = 0
    last_err = ""
    last_code = ALIGN_RUN_FAILED
    used_engine = ""
    fallback_notes: List[str] = []
    used_confidence = 0.0
    used_warnings: List[str] = []
    used_fallback_hint = ""

    for engine in chain:
        if engine == "none":
            attempts.append(
                make_runtime_report(
                    "align",
                    ALIGN_USING_EXISTING,
                    "Alignment skipped by aligner setting.",
                    engine="none",
                    attempt_index=len(attempts) + 1,
                    ready=True,
                    ok=True,
                )
            )
            used_engine = "none"
            break

        if engine == "hsmm_oto":
            attempts.append(
                make_runtime_report(
                    "align",
                    ALIGN_USING_EXISTING,
                    "Alignment skipped; HSMM OTO generation runs in the OTO stage.",
                    engine="hsmm_oto",
                    attempt_index=len(attempts) + 1,
                    ready=True,
                    ok=True,
                )
            )
            used_engine = "hsmm_oto"
            break

        if engine == "sequence":
            ready = check_sequence_aligner_ready(
                language=lang,
                wav_folder=wav_folder,
                callback=callback,
            )
            ready["engine"] = "sequence"
            ready["attempt_index"] = len(attempts) + 1
            attempts.append(dict(ready))
            if str(ready.get("code", OK)).upper() != OK:
                _emit(
                    callback,
                    f"[Align] not ready engine=sequence code={ready.get('code')} message={ready.get('message', '')}",
                )
                last_err = str(ready.get("message", "") or "sequence aligner not ready")
                last_code = str(ready.get("code", ALIGN_RUN_FAILED) or ALIGN_RUN_FAILED)
                continue

            run_attempt_count += 1
            _emit(callback, "[Align] start engine=sequence attempt=1/1")
            ok, err = run_sequence_align(
                "",
                wav_folder,
                dictionary_path,
                output_folder,
                language=lang,
                format_hint=format_hint,
                callback=callback,
            )
            sequence_meta = get_last_sequence_align_meta()
            sequence_confidence = float(sequence_meta.get("confidence", 1.0) or 1.0)
            sequence_warnings = list(sequence_meta.get("warnings") or [])
            sequence_fallback_hint = str(sequence_meta.get("fallback_hint", "") or "")
            if ok and not has_textgrid_files(output_folder):
                ok = False
                err = "TextGrid output missing after alignment."
                code = ALIGN_OUTPUT_EMPTY
            else:
                code = OK if ok else classify_alignment_error("sequence", err)

            attempts.append(
                make_runtime_report(
                    "align",
                    code,
                    err or ("alignment complete" if ok else ""),
                    engine="sequence",
                    attempt_index=len(attempts) + 1,
                    ready=True,
                    ok=bool(ok),
                    confidence=sequence_confidence,
                    warnings=sequence_warnings,
                    fallback_hint=sequence_fallback_hint,
                )
            )
            if ok:
                used_engine = "sequence"
                used_confidence = sequence_confidence
                used_warnings = sequence_warnings
                used_fallback_hint = sequence_fallback_hint
                break

            last_err = str(err or "")
            last_code = str(code or ALIGN_RUN_FAILED)
            _emit(callback, f"[Align] failed engine=sequence code={code} message={err}")
            continue

        if engine == "wfl":
            ready = check_wfl_ready(
                language=lang,
                wav_folder=wav_folder,
                callback=callback,
            )
            ready["engine"] = "wfl"
            ready["attempt_index"] = len(attempts) + 1
            attempts.append(dict(ready))
            if str(ready.get("code", OK)).upper() != OK:
                _emit(
                    callback,
                    f"[Align] not ready engine=wfl code={ready.get('code')} message={ready.get('message', '')}",
                )
                last_err = str(ready.get("message", "") or "wfl aligner not ready")
                last_code = str(ready.get("code", ALIGN_RUN_FAILED) or ALIGN_RUN_FAILED)
                continue

            run_attempt_count += 1
            _emit(callback, "[Align] start engine=wfl attempt=1/1")
            ok, err = run_wfl_align(
                "",
                wav_folder,
                dictionary_path,
                output_folder,
                language=lang,
                format_hint=format_hint,
                callback=callback,
            )
            wfl_meta = get_last_wfl_align_meta()
            wfl_confidence = float(wfl_meta.get("confidence", 1.0) or 1.0)
            wfl_warnings = list(wfl_meta.get("warnings") or [])
            wfl_fallback_hint = str(wfl_meta.get("fallback_hint", "") or "")
            if ok and not has_textgrid_files(output_folder):
                ok = False
                err = "TextGrid output missing after alignment."
                code = ALIGN_OUTPUT_EMPTY
            else:
                code = OK if ok else classify_alignment_error("wfl", err)

            attempts.append(
                make_runtime_report(
                    "align",
                    code,
                    err or ("alignment complete" if ok else ""),
                    engine="wfl",
                    attempt_index=len(attempts) + 1,
                    ready=True,
                    ok=bool(ok),
                    confidence=wfl_confidence,
                    warnings=wfl_warnings,
                    fallback_hint=wfl_fallback_hint,
                )
            )
            if ok:
                used_engine = "wfl"
                used_confidence = wfl_confidence
                used_warnings = wfl_warnings
                used_fallback_hint = wfl_fallback_hint
                break

            last_err = str(err or "")
            last_code = str(code or ALIGN_RUN_FAILED)
            _emit(callback, f"[Align] failed engine=wfl code={code} message={err}")
            continue

        # Unknown engine — skip with a warning.
        _emit(callback, f"[Align] unknown engine={engine}, skipping.")
        last_err = f"unsupported aligner engine: {engine}"
        last_code = ALIGN_RUN_FAILED
        continue

    if used_engine:
        if used_engine != primary:
            fallback_notes.insert(0, f"engine:{primary}->{used_engine}")
        fallback_path = ",".join(fallback_notes)
        fallback_used = bool(fallback_notes)
        # TICKET-001: Compute and attach a quality score so downstream stages
        # (OTO generation, UI) can calibrate based on alignment confidence.
        alignment_quality_score = compute_alignment_quality_score(used_engine, fallback_notes)
        if used_engine not in {"sequence", "wfl"}:
            used_confidence = float(alignment_quality_score)
        message = "alignment complete"
        if fallback_used:
            message = f"alignment complete (fallback: {fallback_path})"
        return make_runtime_report(
            "align",
            OK,
            message,
            ok=True,
            primary_engine=primary,
            used_engine=used_engine,
            attempts=attempts,
            fallback_used=fallback_used,
            attempt_count=max(run_attempt_count, 1),
            fallback_path=fallback_path,
            alignment_quality_score=alignment_quality_score,
            confidence=float(used_confidence),
            warnings=list(dict.fromkeys([*used_warnings, *fallback_notes])),
            fallback_hint=(used_fallback_hint or ("none" if not fallback_used else "wfl")),
        )

    return make_runtime_report(
        "align",
        str(last_code or ALIGN_RUN_FAILED),
        str(last_err or "alignment failed"),
        ok=False,
        primary_engine=primary,
        used_engine="",
        attempts=attempts,
        fallback_used=False,
        attempt_count=max(run_attempt_count, 1),
        fallback_path="",
        confidence=0.0,
        warnings=["alignment_failed"],
        fallback_hint="none",
    )


__all__ = ["run_alignment_with_fallback"]
