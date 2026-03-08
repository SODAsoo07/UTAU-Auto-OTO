from __future__ import annotations

from typing import Dict, Optional

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
from core.sofa_runner import check_sofa_ready_for_alignment, run_sofa_align


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
    mfa_align_profile: str = "accurate",
    sofa_kwargs: Optional[Dict[str, object]] = None,
    callback=None,
) -> Dict[str, object]:
    lang = str(language or "korean").strip().lower()
    attempts = []
    chain = resolve_aligner_chain(primary_aligner, fallback_aligner)
    primary = chain[0] if chain else "mfa"
    sofa_kwargs = dict(sofa_kwargs or {})

    if has_textgrid_files(output_folder):
        return make_runtime_report(
            "align",
            ALIGN_USING_EXISTING,
            "기존 TextGrid를 그대로 사용합니다.",
            ok=True,
            primary_engine=primary,
            used_engine="existing",
            attempts=attempts,
            fallback_used=False,
            attempt_count=0,
        )

    for index, engine in enumerate(chain, start=1):
        if engine == "mfa":
            ready = check_mfa_ready(language=lang, mfa_path=mfa_path)
        else:
            ready = check_sofa_ready_for_alignment(
                ckpt_path=str(sofa_kwargs.get("ckpt_path", "") or ""),
                dictionary_path=str(sofa_kwargs.get("dictionary_path", dictionary_path) or ""),
                mfa_path=mfa_path,
                sofa_python=str(sofa_kwargs.get("sofa_python", "") or ""),
                sofa_repo_dir=str(sofa_kwargs.get("sofa_repo_dir", "") or ""),
            )
        ready["engine"] = engine
        ready["attempt_index"] = index
        attempts.append(dict(ready))
        if str(ready.get("code", OK)).upper() != OK:
            _emit(
                callback,
                f"[Align] 준비 실패 engine={engine} code={ready.get('code')} message={ready.get('message', '')}",
            )
            continue

        _emit(callback, f"[Align] 실행 engine={engine} attempt={index}/{len(chain)}")
        if engine == "mfa":
            ok, err = run_mfa_align(
                mfa_path or str(ready.get("mfa_path", "") or ""),
                wav_folder,
                dictionary_path,
                output_folder,
                language=lang,
                callback=callback,
                align_profile=mfa_align_profile,
            )
        else:
            ok, err = run_sofa_align(
                wav_folder=wav_folder,
                output_folder=output_folder,
                ckpt_path=str(sofa_kwargs.get("ckpt_path", "") or ""),
                dictionary_path=str(sofa_kwargs.get("dictionary_path", dictionary_path) or ""),
                mfa_path=mfa_path,
                sofa_python=str(sofa_kwargs.get("sofa_python", "") or ""),
                callback=callback,
                sofa_repo_dir=str(sofa_kwargs.get("sofa_repo_dir", "") or ""),
                mode=str(sofa_kwargs.get("mode", "force") or "force"),
                g2p=str(sofa_kwargs.get("g2p", "Dictionary") or "Dictionary"),
                ap_detector=str(
                    sofa_kwargs.get("ap_detector", "LoudnessSpectralcentroidAPDetector")
                    or "LoudnessSpectralcentroidAPDetector"
                ),
                ap_detector_config=str(sofa_kwargs.get("ap_detector_config", "") or ""),
                save_confidence=sofa_kwargs.get("save_confidence", False),
                out_formats=sofa_kwargs.get("out_formats", "TextGrid"),
                extra_infer_args=sofa_kwargs.get("extra_infer_args", ""),
                two_pass_retry=sofa_kwargs.get("two_pass_retry", False),
                two_pass_retry_mode=str(sofa_kwargs.get("two_pass_retry_mode", "match") or "match"),
                confidence_threshold=sofa_kwargs.get("confidence_threshold", 0.55),
                low_confidence_max_files=sofa_kwargs.get("low_confidence_max_files", 0),
            )

        if ok and not has_textgrid_files(output_folder):
            ok = False
            err = "정렬은 완료됐지만 TextGrid 결과를 찾지 못했습니다."
            code = ALIGN_OUTPUT_EMPTY
        else:
            code = OK if ok else classify_alignment_error(engine, err)
        attempt = make_runtime_report(
            "align",
            code,
            err or ("정렬 완료" if ok else ""),
            engine=engine,
            attempt_index=index,
            ready=True,
            ok=bool(ok),
        )
        attempts.append(attempt)
        if ok:
            fallback_used = engine != primary
            message = "정렬 완료"
            if fallback_used:
                message = f"primary={primary} 실패 후 {engine}로 정렬 완료"
            return make_runtime_report(
                "align",
                OK,
                message,
                ok=True,
                primary_engine=primary,
                used_engine=engine,
                attempts=attempts,
                fallback_used=fallback_used,
                attempt_count=index,
                fallback_path=f"align:{primary}->{engine}" if fallback_used else "",
            )
        _emit(callback, f"[Align] 실패 engine={engine} code={code} message={err}")

    last_attempt = attempts[-1] if attempts else {}
    code = str(last_attempt.get("code", ALIGN_SKIPPED) or ALIGN_SKIPPED)
    message = str(last_attempt.get("message", "") or "정렬을 실행하지 못했습니다.")
    tried_fallback = any(int((attempt or {}).get("attempt_index", 0) or 0) > 1 for attempt in attempts)
    return make_runtime_report(
        "align",
        code,
        message,
        ok=False,
        primary_engine=primary,
        used_engine="",
        attempts=attempts,
        fallback_used=tried_fallback,
        attempt_count=len(chain),
        fallback_path=(f"align:{primary}->{chain[-1]}" if tried_fallback and len(chain) > 1 else ""),
    )


__all__ = ["run_alignment_with_fallback"]
