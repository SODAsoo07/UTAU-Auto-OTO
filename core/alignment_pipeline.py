from __future__ import annotations

from typing import Dict

from core.mfa_runner import check_mfa_ready, run_mfa_align
from core.pipeline_status import (
    ALIGN_OUTPUT_EMPTY,
    ALIGN_SKIPPED,
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
            str(ready.get("code", ALIGN_SKIPPED) or ALIGN_SKIPPED),
            str(ready.get("message", "") or "alignment not ready"),
            ok=False,
            primary_engine=primary,
            used_engine="",
            attempts=attempts,
            fallback_used=False,
            attempt_count=1,
        )

    _emit(callback, "[Align] start engine=mfa attempt=1/1")
    ok, err = run_mfa_align(
        mfa_path or str(ready.get("mfa_path", "") or ""),
        wav_folder,
        dictionary_path,
        output_folder,
        language=lang,
        callback=callback,
        align_profile=mfa_align_profile,
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
            attempt_index=1,
            ready=True,
            ok=bool(ok),
        )
    )

    if ok:
        return make_runtime_report(
            "align",
            OK,
            "alignment complete",
            ok=True,
            primary_engine=primary,
            used_engine="mfa",
            attempts=attempts,
            fallback_used=False,
            attempt_count=1,
            fallback_path="",
        )

    _emit(callback, f"[Align] failed engine=mfa code={code} message={err}")
    return make_runtime_report(
        "align",
        code,
        str(err or "alignment failed"),
        ok=False,
        primary_engine=primary,
        used_engine="",
        attempts=attempts,
        fallback_used=False,
        attempt_count=1,
        fallback_path="",
    )


__all__ = ["run_alignment_with_fallback"]
