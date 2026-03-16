from __future__ import annotations

from typing import Dict, List

from core.mfa_runner import check_mfa_ready, run_mfa_align
from core.pipeline_status import (
    ALIGN_OUTPUT_EMPTY,
    ALIGN_RUN_FAILED,
    ALIGN_USING_EXISTING,
    OK,
    classify_alignment_error,
    has_textgrid_files,
    make_runtime_report,
    resolve_aligner_chain,
)


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
    req = str(requested_profile or "").strip().lower()
    valid = ["default", "accurate", "fast"]
    if req not in valid:
        req = "default"
    # Retry policy: one retry at most (total max 2 attempts).
    # Prefer a lower-load fallback to reduce repeated crashes.
    fallback = "fast"
    if req == "fast":
        fallback = "default"
    elif req == "accurate":
        fallback = "default"
    if fallback == req:
        return [req]
    return [req, fallback]


def run_alignment_with_fallback(
    *,
    language: str,
    wav_folder: str,
    dictionary_path: str,
    output_folder: str,
    primary_aligner: str = "mfa",
    fallback_aligner: str = "",
    mfa_path: str = "",
    mfa_align_profile: str = "default",
    callback=None,
) -> Dict[str, object]:
    lang = str(language or "korean").strip().lower()
    attempts = []
    chain = resolve_aligner_chain(primary_aligner, fallback_aligner)
    primary = chain[0] if chain else "mfa"

    if has_textgrid_files(output_folder):
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

    if primary != "mfa":
        primary = "mfa"

    ready = check_mfa_ready(language=lang, mfa_path=mfa_path)
    ready["engine"] = "mfa"
    ready["attempt_index"] = 1
    attempts.append(dict(ready))
    if str(ready.get("code", OK)).upper() != OK:
        _emit(
            callback,
            f"[Align] not ready engine=mfa code={ready.get('code')} message={ready.get('message', '')}",
        )
        return make_runtime_report(
            "align",
            str(ready.get("code", ALIGN_RUN_FAILED) or ALIGN_RUN_FAILED),
            str(ready.get("message", "") or "alignment not ready"),
            ok=False,
            primary_engine=primary,
            used_engine="",
            attempts=attempts,
            fallback_used=False,
            attempt_count=1,
        )

    mfa_exec = mfa_path or str(ready.get("mfa_path", "") or "")
    profile_chain = _mfa_profile_chain(lang, mfa_align_profile)
    last_err = ""
    last_code = ALIGN_RUN_FAILED
    used_profile = ""
    attempt_count = 0
    for idx, profile in enumerate(profile_chain, start=1):
        if idx > 1 and not _should_retry_mfa_with_fallback(last_err):
            break
        attempt_count += 1
        _emit(callback, f"[Align] start engine=mfa attempt={idx}/{len(profile_chain)} profile={profile}")
        ok, err = run_mfa_align(
            mfa_exec,
            wav_folder,
            dictionary_path,
            output_folder,
            language=lang,
            callback=callback,
            align_profile=profile,
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
                attempt_index=idx,
                ready=True,
                ok=bool(ok),
                profile=profile,
            )
        )
        if ok:
            used_profile = profile
            break
        last_err = str(err or "")
        last_code = str(code or ALIGN_RUN_FAILED)
        _emit(callback, f"[Align] failed engine=mfa code={code} profile={profile} message={err}")

    if used_profile:
        fallback_used = bool(used_profile != profile_chain[0] and attempt_count > 1)
        message = "alignment complete"
        if fallback_used:
            message = f"alignment complete (fallback profile: {used_profile})"
        return make_runtime_report(
            "align",
            OK,
            message,
            ok=True,
            primary_engine=primary,
            used_engine="mfa",
            attempts=attempts,
            fallback_used=fallback_used,
            attempt_count=attempt_count,
            fallback_path=f"mfa_profile:{profile_chain[0]}->{used_profile}" if fallback_used else "",
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
        attempt_count=max(attempt_count, 1),
        fallback_path="",
    )


__all__ = ["run_alignment_with_fallback"]
