from __future__ import annotations

import os
from typing import Dict, List

from core.mfa_runner import check_mfa_ready, run_mfa_align
from core.sequence_aligner import check_sequence_aligner_ready, run_sequence_align
from core.pipeline_status import (
    ALIGN_EXEC_MISSING,
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

_MFA_SOFT_GATE_FAIL_CACHE: Dict[str, int] = {}


def _emit(callback, message: str) -> None:
    if not callback:
        return
    try:
        callback(message)
    except Exception:
        return


def _should_retry_mfa_with_fallback(message: str) -> bool:
    text = str(message or "").lower()
    if not text:
        return False
    if "missing korean tokenizer dependencies" in text:
        return False
    if "missing japanese tokenizer dependencies" in text:
        return False
    if "mfa executable not found" in text:
        return False
    return (
        "code: 3221225477" in text
        or "code: -1073741819" in text
        or "access violation" in text
    )


def _mfa_profile_chain(language: str, requested_profile: str) -> List[str]:
    _ = str(language or "").strip().lower()
    req = str(requested_profile or "").strip().lower()
    valid = ["default", "accurate", "accurate_adapted", "fast"]
    if req not in valid:
        req = "default"
    fallback = "fast"
    if req in {"fast", "accurate", "accurate_adapted"}:
        fallback = "default"
    if fallback == req:
        return [req]
    return [req, fallback]


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
    primary_aligner: str = "mfa",
    fallback_aligner: str = "",
    mfa_path: str = "",
    mfa_align_profile: str = "default",
    mfa_runtime_options: Dict[str, object] | None = None,
    overwrite_existing_textgrids: bool = False,
    callback=None,
) -> Dict[str, object]:
    lang = str(language or "korean").strip().lower()
    attempts = []

    normalized_primary = normalize_aligner_name(primary_aligner, default="mfa")
    resolved_fallback = fallback_aligner
    if normalized_primary == "sequence" and not str(fallback_aligner or "").strip():
        # Dedicated sequence aligner is heuristic-first; keep MFA fallback for runtime safety.
        resolved_fallback = "mfa"

    chain = resolve_aligner_chain(primary_aligner, resolved_fallback)
    primary = chain[0] if chain else "mfa"

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
                )
            )
            if ok:
                used_engine = "sequence"
                break

            last_err = str(err or "")
            last_code = str(code or ALIGN_RUN_FAILED)
            _emit(callback, f"[Align] failed engine=sequence code={code} message={err}")
            continue

        # default: MFA
        ready = check_mfa_ready(language=lang, mfa_path=mfa_path)
        ready["engine"] = "mfa"
        ready["attempt_index"] = len(attempts) + 1
        attempts.append(dict(ready))
        ready_code = str(ready.get("code", OK) or OK).upper()
        if ready_code != OK:
            _emit(
                callback,
                f"[Align] precheck engine=mfa code={ready.get('code')} message={ready.get('message', '')}",
            )
            # Hybrid relaxed policy:
            # - Hard-block only when MFA executable is missing.
            # - For model/dependency precheck failures, attempt actual MFA run first.
            if ready_code == ALIGN_EXEC_MISSING:
                last_err = str(ready.get("message", "") or "mfa not ready")
                last_code = str(ready.get("code", ALIGN_RUN_FAILED) or ALIGN_RUN_FAILED)
                continue
            try:
                soft_limit = max(1, int(str(os.environ.get("UTOA_MFA_SOFT_GATE_RETRY_LIMIT", "2") or "2")))
            except Exception:
                soft_limit = 2
            soft_key = "|".join(
                [
                    str(lang or ""),
                    str(mfa_path or ready.get("mfa_path", "") or ""),
                    str(ready_code or ""),
                ]
            )
            soft_count = int(_MFA_SOFT_GATE_FAIL_CACHE.get(soft_key, 0)) + 1
            _MFA_SOFT_GATE_FAIL_CACHE[soft_key] = soft_count
            if soft_count > soft_limit:
                _emit(
                    callback,
                    f"[Align] precheck soft-gate limit exceeded ({soft_count-1}>{soft_limit}); switching to hard block.",
                )
                last_err = str(ready.get("message", "") or "mfa not ready")
                last_code = str(ready.get("code", ALIGN_RUN_FAILED) or ALIGN_RUN_FAILED)
                continue
            _emit(callback, "[Align] precheck failed but continuing with runtime attempt (soft gate).")

        mfa_exec = mfa_path or str(ready.get("mfa_path", "") or "")
        profile_chain = _mfa_profile_chain(lang, mfa_align_profile)
        used_profile = ""

        for idx, profile in enumerate(profile_chain, start=1):
            if idx > 1 and not _should_retry_mfa_with_fallback(last_err):
                break
            run_attempt_count += 1
            _emit(callback, f"[Align] start engine=mfa attempt={idx}/{len(profile_chain)} profile={profile}")
            ok, err = run_mfa_align(
                mfa_exec,
                wav_folder,
                dictionary_path,
                output_folder,
                language=lang,
                callback=callback,
                align_profile=profile,
                runtime_options=mfa_runtime_options,
            )
            if ok and not has_textgrid_files(output_folder):
                ok = False
                err = "TextGrid output missing after alignment."
                code = ALIGN_OUTPUT_EMPTY
            else:
                code = OK if ok else classify_alignment_error("mfa", err)

            attempts.append(
                make_runtime_report(
                    "align",
                    code,
                    err or ("alignment complete" if ok else ""),
                    engine="mfa",
                    attempt_index=len(attempts) + 1,
                    ready=True,
                    ok=bool(ok),
                    profile=profile,
                )
            )
            if ok:
                try:
                    soft_key = "|".join(
                        [
                            str(lang or ""),
                            str(mfa_exec or ""),
                            str(ready_code or ""),
                        ]
                    )
                    _MFA_SOFT_GATE_FAIL_CACHE.pop(soft_key, None)
                except Exception:
                    pass
                used_profile = profile
                break

            last_err = str(err or "")
            last_code = str(code or ALIGN_RUN_FAILED)
            _emit(callback, f"[Align] failed engine=mfa code={code} profile={profile} message={err}")

        if used_profile:
            used_engine = "mfa"
            if used_profile != profile_chain[0]:
                fallback_notes.append(f"mfa_profile:{profile_chain[0]}->{used_profile}")
            break

    if used_engine:
        if used_engine != primary:
            fallback_notes.insert(0, f"engine:{primary}->{used_engine}")
        fallback_path = ",".join(fallback_notes)
        fallback_used = bool(fallback_notes)
        # TICKET-001: Compute and attach a quality score so downstream stages
        # (OTO generation, UI) can calibrate based on alignment confidence.
        alignment_quality_score = compute_alignment_quality_score(used_engine, fallback_notes)
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
    )


__all__ = ["run_alignment_with_fallback"]
