"""Product UI for running the real voicebank OTO generator."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping

from .bank_detection import FormatDetection, check_language, detect_format
from .batch import BankFolder, discover_bank_folders, run_batch
from .generator import DIAGNOSTIC_FILE_OPTIONS, GenerationOptions, _discover_wavs, generate_voicebank, inspect_voicebank
from .oto_io import parse_oto
from .oto_style import OtoStyle, load_oto_style, save_oto_style


LANGUAGE_CHOICES: Mapping[str, str | None] = {
    "한국어": "korean",
    "일본어": "japanese",
}
AUTO_FORMAT_CHOICE = "자동 감지"
FORMAT_CHOICES: Mapping[str, str | None] = {
    AUTO_FORMAT_CHOICE: None,
    "CV": "cv",
    "CVC": "cvc",
    "CVVC": "cvvc",
    "VCV": "vcv",
}
ENCODING_CHOICES = ("preserve", "utf-8", "cp932", "shift_jis", "cp949", "euc-kr")
VC_END_CHOICES: Mapping[str, str] = {
    "다음 자음 안에서 끝냄 (권장)": "consonant",
    "템플릿 기본값": "template",
}
RELEASE_PRE_CHOICES: Mapping[str, str] = {
    "모음 끝 (권장)": "vowel_end",
    "템플릿 기본값": "template",
}
CONFIDENCE_LABELS = {"high": "높음", "medium": "보통", "low": "낮음", "none": "없음"}
DETECTION_SOURCE_LABELS = {
    "base_oto": "기존 oto.ini 별칭",
    "base_oto+folder_name": "기존 oto.ini + 폴더 이름",
    "folder_name": "폴더 이름",
    "filenames": "WAV 파일 이름",
    "sibling_folders": "다른 폴더",
    "none": "근거 없음",
}
# Advanced style fields: (OtoStyle field, label, hint). Blank keeps the template value.
STYLE_FIELDS = (
    ("cv_overlap_ratio", "CV 오버랩 비율", "0~1, 오버랩 = 비율 × 선행발성"),
    ("vc_overlap_ratio", "VC 오버랩 비율", "0~1"),
    ("vcv_overlap_ratio", "VCV 오버랩 비율", "0~1"),
    ("cv_pre_shift_ms", "CV 선행발성 이동", "ms, + = 늦게"),
    ("vc_pre_shift_ms", "VC 선행발성 이동", "ms"),
    ("vcv_pre_shift_ms", "VCV 선행발성 이동", "ms"),
    ("consonant_extra_ms", "고정 범위 연장", "ms, 0 이상"),
    ("cutoff_margin_ms", "오른쪽 블랭크 여유", "ms, 0 이상"),
)
# Product UI generation settings, measured against hand-made OTOs on nine banks
# (docs/BANK_ACCURACY_2026-10-09.md, docs/SYLLABLE_POSITION_2026-10-09.md,
# docs/SYLLABLE_DIAGNOSIS_AND_FIXES_2026-10-11.md). Korean-only rules in the profile stay
# off for other languages; pause-aware gaps, the exact (unpruned) HSMM search, the
# dense-speech activity floor, the three-pass phone identity templates, held syllabic
# nasals, filename pause states and the VC/release policies apply to all.
RECOMMENDED_DEVELOPMENT_PROFILE = "cvvc_slot_rhythm_v16"
RECOMMENDED_VC_END_POLICY = "consonant"
RECOMMENDED_RELEASE_PRE_POLICY = "vowel_end"
# Rows the parser cannot bind to a recording are left out and listed instead
# of stopping the whole bank; the log says so after every generation.
PRODUCT_UNMAPPED_ALIAS_POLICY = "exclude_row"
EXCLUDED_ROW_LOG_SAMPLES = 10
# Diagnostic files the product UI can write next to oto.ini. All are off unless the user
# turns them on in the advanced options; they never change the OTO rows.
DIAGNOSTIC_FILE_LABELS: Mapping[str, str] = {
    "write_unmapped_aliases_csv": "unmapped_aliases.csv (생성되지 않은 행 목록)",
    "write_generation_report": "generation_report.json (생성 보고서)",
    "write_row_report": "row_report.jsonl (행별 근거)",
    "write_timelines": "timelines/ (WAV별 음소 경계)",
    "write_project_manifest": "project_manifest.json (입력 해석 결과)",
}

# Kept verbatim from the existing AutoOTO Credits tab.
CREDIT_LINES = (
    "카메비 Kamebit",
    "22 twotwosibi",
    "촙타러/익명의 음성제공자 infinityecho00&Anonymous Voice Provider",
    "빈빈 2xbin",
    "류시아 Lyusia",
    "HUEY",
    "집냥 Ironic_SP",
    "라리우 RARIU",
    "서딕호 neohyajach",
    "나미니 Namini",
    "노와노 Nowano",
    "실안개 AngaeSil1115",
    "사토 야나기 Sato Yanagi",
    "혜성 Comet",
    "도살 BUTCHER_TUNING",
    "펜촉 ianharuesaone",
    "다유 DAU_Multiverse",
    "집냥 Blue",
)

# Colors reused from the existing AutoOTO light/dark palette.
PANEL_BG = ("#EDE9E4", "#1F232B")
PANEL_BORDER = ("#B3B0AA", "#303744")
PRIMARY = ("#9CB3C2", "#2F7EB8")
PRIMARY_HOVER = ("#8DA6B8", "#286C9D")
TEXT = ("#2B2D2E", "#FFFFFF")
HINT = ("#4E5866", "#DCE9F8")
HEADER = ("#394045", "#EAF4FF")


@dataclass(frozen=True)
class UIOperationResult:
    title: str
    payload: Mapping[str, object]


def build_generation_options(
    *,
    bank: str | Path,
    output: str | Path,
    base_oto: str | Path | None = None,
    recursive: bool = False,
    output_encoding: str = "preserve",
    review_threshold: str | float = 0.45,
    allow_heuristic_clean: bool = False,
    max_files: str | int | None = None,
    force: bool = False,
    allow_in_place_output: bool = False,
    format_hint: str | None = None,
    language_hint: str | None = None,
    backend: str = "legacy_heuristic",
    development_alignment_profile: str | None = None,
    vc_end_policy: str = "template",
    release_pre_policy: str = "template",
    unmapped_alias_policy: str = "fail",
    diagnostic_files: Mapping[str, bool] | None = None,
    generate_missing_aliases: bool = False,
    oto_style: OtoStyle | None = None,
    bpm: float | None = None,
    estimate_bpm: bool = False,
) -> GenerationOptions:
    """Convert UI values into the same options object used by the CLI.

    Diagnostic files are written unless named as False in ``diagnostic_files``.
    """

    bank_text = str(bank).strip()
    output_text = str(output).strip()
    if not bank_text:
        raise ValueError("WAV 보이스뱅크 폴더를 지정하세요.")
    if not output_text:
        raise ValueError("출력 폴더를 지정하세요.")
    threshold = float(review_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("검수 임계값은 0.0~1.0 범위여야 합니다.")
    parsed_max_files: int | None
    if max_files is None or not str(max_files).strip():
        parsed_max_files = None
    else:
        parsed_max_files = int(str(max_files).strip())
        if parsed_max_files < 1:
            raise ValueError("최대 WAV 수는 1 이상이어야 합니다.")
    base_path = None
    if base_oto is not None and str(base_oto).strip():
        base_path = Path(str(base_oto).strip()).expanduser()
    return GenerationOptions(
        bank=Path(bank_text).expanduser(),
        output=Path(output_text).expanduser(),
        base_oto=base_path,
        recursive=bool(recursive),
        output_encoding=output_encoding,
        review_threshold=threshold,
        allow_heuristic_clean=bool(allow_heuristic_clean),
        max_files=parsed_max_files,
        force=bool(force),
        allow_in_place_output=bool(allow_in_place_output),
        format_hint=format_hint or None,
        language_hint=language_hint or None,
        backend=backend,
        development_alignment_profile=development_alignment_profile,
        vc_end_policy=vc_end_policy,
        release_pre_policy=release_pre_policy,
        unmapped_alias_policy=unmapped_alias_policy,
        **_diagnostic_file_flags(diagnostic_files, default=True),
        generate_missing_aliases=generate_missing_aliases,
        oto_style=oto_style,
        bpm=bpm,
        estimate_bpm=estimate_bpm,
    )


def _diagnostic_file_flags(choices: Mapping[str, bool] | None, *, default: bool) -> dict[str, bool]:
    choices = dict(choices or {})
    unknown = set(choices) - set(DIAGNOSTIC_FILE_OPTIONS)
    if unknown:
        raise ValueError(f"알 수 없는 진단 파일 옵션입니다: {', '.join(sorted(unknown))}")
    return {name: bool(choices.get(name, default)) for name in DIAGNOSTIC_FILE_OPTIONS}


def build_product_generation_options(
    *,
    bank: str | Path,
    output: str | Path,
    base_oto: str | Path | None = None,
    recursive: bool = False,
    output_encoding: str = "preserve",
    language_choice: str = "한국어",
    format_choice: str = "CVVC",
    force: bool = False,
    overwrite_existing_oto: bool = False,
    diagnostic_files: Mapping[str, bool] | None = None,
    detected_format: str | None = None,
    bpm: str | float | None = None,
    estimate_bpm: bool = False,
    oto_style: OtoStyle | None = None,
    vc_end_choice: str | None = None,
    release_pre_choice: str | None = None,
) -> GenerationOptions:
    """Build the safe production options exposed by the product UI.

    Diagnostic files are off unless named as True in ``diagnostic_files``.
    """

    if language_choice not in LANGUAGE_CHOICES:
        raise ValueError(f"지원하지 않는 언어 선택입니다: {language_choice}")
    if format_choice not in FORMAT_CHOICES:
        raise ValueError(f"지원하지 않는 형식 선택입니다: {format_choice}")
    format_hint = FORMAT_CHOICES[format_choice] or detected_format
    if not format_hint:
        raise ValueError("형식을 자동으로 알아내지 못했습니다. 포맷을 직접 선택하세요.")
    vc_end_policy = VC_END_CHOICES[vc_end_choice] if vc_end_choice else RECOMMENDED_VC_END_POLICY
    release_pre_policy = (RELEASE_PRE_CHOICES[release_pre_choice] if release_pre_choice
                          else RECOMMENDED_RELEASE_PRE_POLICY)
    has_base_oto = base_oto is not None and bool(str(base_oto).strip())
    if output_encoding not in ENCODING_CHOICES:
        raise ValueError(f"지원하지 않는 출력 인코딩입니다: {output_encoding}")
    return build_generation_options(
        bank=bank,
        output=output,
        base_oto=base_oto,
        recursive=recursive,
        output_encoding=output_encoding,
        review_threshold=0.45,
        allow_heuristic_clean=False,
        max_files=None,
        force=force or overwrite_existing_oto,
        allow_in_place_output=overwrite_existing_oto,
        format_hint=format_hint,
        language_hint=LANGUAGE_CHOICES[language_choice],
        backend="routed_hsmm",
        development_alignment_profile=RECOMMENDED_DEVELOPMENT_PROFILE,
        vc_end_policy=vc_end_policy,
        release_pre_policy=release_pre_policy,
        unmapped_alias_policy=PRODUCT_UNMAPPED_ALIAS_POLICY,
        diagnostic_files=_diagnostic_file_flags(diagnostic_files, default=False),
        generate_missing_aliases=not has_base_oto,
        oto_style=None if oto_style is None or oto_style.is_default else oto_style,
        bpm=parse_bpm(bpm),
        estimate_bpm=bool(estimate_bpm),
    )


def parse_bpm(value: str | float | None) -> float | None:
    """Blank means "not given"; otherwise a tempo between 60 and 240 BPM."""

    if value is None or not str(value).strip():
        return None
    try:
        bpm = float(str(value).strip())
    except ValueError as error:
        raise ValueError("BPM은 숫자로 입력하세요 (예: 120).") from error
    if not 60.0 <= bpm <= 240.0:
        raise ValueError("BPM은 60~240 범위로 입력하세요 (보통 100~160).")
    return bpm


def build_oto_style(values: Mapping[str, str]) -> OtoStyle:
    """Advanced settings entries -> OtoStyle (blank fields keep the template value)."""

    payload: dict[str, float | None] = {}
    for name, label, _ in STYLE_FIELDS:
        text = str(values.get(name, "") or "").strip()
        if not text:
            continue
        try:
            payload[name] = float(text)
        except ValueError as error:
            raise ValueError(f"{label}: 숫자를 입력하세요.") from error
    try:
        return OtoStyle.from_dict(payload)
    except ValueError as error:
        raise ValueError(f"고급 설정 값이 범위를 벗어났습니다: {error}") from error


def style_entry_values(style: OtoStyle) -> dict[str, str]:
    values = style.to_dict()
    result = {}
    for name, _, _ in STYLE_FIELDS:
        value = values[name]
        blank = value is None or (not name.endswith("_ratio") and value == 0.0)
        result[name] = "" if blank else f"{value:g}"
    return result


def _bank_names_and_aliases(bank: Path, base_oto: Path | None, recursive: bool) -> tuple[list[str], list[str]]:
    wavs = [path.name for path in _discover_wavs(bank, recursive)] if bank.is_dir() else []
    aliases: list[str] = []
    if base_oto is None and (bank / "oto.ini").is_file():
        # Only read for detection; generation still uses the base OTO the user chose.
        base_oto = bank / "oto.ini"
    if base_oto is not None and base_oto.is_file():
        try:
            aliases = [row.alias for row in parse_oto(base_oto)[0]]
        except (OSError, ValueError):
            aliases = []
    return wavs, aliases


def _optional_path(value: str | Path | None) -> Path | None:
    return Path(str(value).strip()).expanduser() if value is not None and str(value).strip() else None


def detect_bank_format(
    bank: str | Path,
    base_oto: str | Path | None,
    language_choice: str,
    *,
    recursive: bool = False,
) -> FormatDetection:
    bank_path = Path(str(bank).strip()).expanduser()
    wavs, aliases = _bank_names_and_aliases(bank_path, _optional_path(base_oto), recursive)
    return detect_format(bank_path, wavs, base_aliases=aliases or None, language=LANGUAGE_CHOICES.get(language_choice))


def bank_language_warning(
    bank: str | Path,
    base_oto: str | Path | None,
    language_choice: str,
    *,
    recursive: bool = False,
) -> str | None:
    bank_path = Path(str(bank).strip()).expanduser()
    wavs, aliases = _bank_names_and_aliases(bank_path, _optional_path(base_oto), recursive)
    return check_language(LANGUAGE_CHOICES.get(language_choice) or "", wavs, aliases).warning


def format_detection_text(detection: FormatDetection) -> str:
    if detection.format is None:
        return "형식 감지: 알 수 없음 - 포맷을 직접 선택하세요."
    return (f"형식 감지: {detection.format.upper()} "
            f"(확신 {CONFIDENCE_LABELS.get(detection.confidence, detection.confidence)}, "
            f"근거: {DETECTION_SOURCE_LABELS.get(detection.source, detection.source)})")


def tempo_log_lines(payload: Mapping[str, object]) -> list[str]:
    tempo = payload.get("recording_tempo")
    if not isinstance(tempo, Mapping):
        return []
    estimate = tempo.get("estimate") if isinstance(tempo.get("estimate"), Mapping) else None
    if tempo.get("source") == "user_bpm":
        return [f"[BPM] 입력한 {float(tempo['user_bpm']):g} BPM 사용 (음절 간격 {float(tempo['syllable_period_ms']):.0f} ms)"]
    if tempo.get("source") == "estimated" and estimate:
        return [f"[BPM] 자동 추정 {estimate.get('bpm')} BPM 사용 ({estimate.get('reason')})"]
    if tempo.get("estimate_enabled") and estimate:
        return [f"[BPM] 추정 안 함: {estimate.get('reason')} BPM 없이 생성했습니다."]
    return []


def alias_generation_log_lines(payload: Mapping[str, object]) -> list[str]:
    count = int(payload.get("generated_alias_wavs", 0) or 0)
    if count <= 0:
        return []
    lines = [f"[별칭 생성] 기존 oto.ini 없이 WAV {count}개의 파일 이름에서 형식에 맞는 별칭을 만들었습니다."]
    skipped = int(payload.get("alias_generation_skipped_count", 0) or 0)
    if skipped:
        lines.append(f"[주의] 이름을 읽지 못한 WAV {skipped}개는 빠졌습니다.")
        for item in payload.get("alias_generation_skipped") or []:
            lines.append(f"  - {item.get('wav', '')} ({item.get('detail', '')})")
    return lines


def summarize_generation_result(report: Mapping[str, object], *, output: Path) -> dict[str, object]:
    """Build a compact summary while retaining the full report on disk."""

    oto_path = output / "oto.ini"
    written = dict(report.get("diagnostic_files", {}) or {})
    summary: dict[str, object] = {
        "oto": str(oto_path.resolve()),
        "oto_exists": oto_path.is_file(),
        "wav_count": report.get("wav_count", 0),
        "generated_rows": report.get("generated_rows", 0),
        "clean_rows": report.get("clean_rows", 0),
        "review_rows": report.get("review_rows", 0),
        "output": str(output.resolve()),
        "generation_backend": report.get("generation_backend"),
        "in_place_output": report.get("in_place_output", False),
        "alignment_backend": report.get("alignment_backend"),
        "feature_backend": report.get("feature_backend"),
        "reason_counts": dict(report.get("reason_counts", {}) or {}),
        "report_path": (str((output / "generation_report.json").resolve())
                        if written.get("generation_report", True) else None),
        "row_report_path": (str((output / "row_report.jsonl").resolve())
                            if written.get("row_report", True) else None),
        "clean_path": str((output / "oto.clean.ini").resolve()),
        "review_path": str((output / "oto.review.ini").resolve()),
        "review_notice": "review 행은 자동 검수 보류이며 오류 확정이 아닙니다.",
        "generation_settings": {
            key: (report.get("provenance") or {}).get(key)
            for key in ("development_alignment_profile", "vc_end_policy", "release_pre_policy")
        },
        "usage_review": dict(report.get("usage_review", {}) or {}),
        "excluded_alias_rows": int(report.get("excluded_alias_rows", 0) or 0),
        "excluded_alias_rows_csv": report.get("excluded_alias_rows_csv"),
        "recording_tempo": report.get("recording_tempo"),
        "generated_alias_wavs": int(report.get("generated_alias_wavs", 0) or 0),
        "alias_generation_skipped": list(report.get("alias_generation_skipped", []) or [])[:EXCLUDED_ROW_LOG_SAMPLES],
        "alias_generation_skipped_count": len(report.get("alias_generation_skipped", []) or []),
        "excluded_alias_samples": list(report.get("excluded_alias_samples", []) or [])[:EXCLUDED_ROW_LOG_SAMPLES],
    }
    if oto_path.is_file():
        summary["oto_sha256"] = hashlib.sha256(oto_path.read_bytes()).hexdigest()
    return summary


def excluded_rows_log_lines(payload: Mapping[str, object]) -> list[str]:
    """Log lines telling the user which base-OTO rows were not generated."""

    count = int(payload.get("excluded_alias_rows", 0) or 0)
    if count <= 0:
        return []
    lines = [
        f"[주의] 생성되지 않은 행 {count}개: 원본 OTO의 별칭을 녹음 파일 이름과 연결하지 못해 oto.ini에서 뺐습니다.",
        (f"  전체 목록: {payload['excluded_alias_rows_csv']}" if payload.get("excluded_alias_rows_csv")
         else "  전체 목록을 파일로 받으려면 고급 출력 옵션에서 unmapped_aliases.csv 저장을 켜세요."),
    ]
    samples = payload.get("excluded_alias_samples") or []
    for sample in samples:
        lines.append(f"  - {sample.get('wav', '')} = {sample.get('alias', '')}")
    if count > len(samples):
        lines.append(f"  … 외 {count - len(samples)}개")
    return lines


def format_result_text(title: str, payload: Mapping[str, object]) -> str:
    """Format an operation result for the human-readable Result tab."""

    lines = [title]
    if "generated_rows" in payload:
        lines.extend(
            [
                f"WAV: {payload.get('wav_count', 0)}",
                f"생성 행: {payload.get('generated_rows', 0)}",
                f"자동 clean: {payload.get('clean_rows', 0)}",
                f"검수 보류: {payload.get('review_rows', 0)}",
                str(payload.get("review_notice", "")),
            ]
        )
        excluded = excluded_rows_log_lines(payload)
        if excluded:
            lines.extend(("", *excluded))
        reasons = payload.get("reason_counts")
        usage = payload.get("usage_review")
        if isinstance(usage, Mapping) and usage.get("csv"):
            counts = usage.get("category_counts", {})
            lines.extend(("", f"먼저 확인: {counts.get('inspect_first', 0)}행",
                f"연결 확인: {counts.get('connection_check', 0)}행",
                f"표본 확인: {counts.get('spot_check', 0)}행", f"검수 순서: {usage['csv']}"))
        if isinstance(reasons, Mapping) and reasons:
            lines.extend(("", "주요 검수 사유"))
            for reason, count in sorted(reasons.items(), key=lambda item: (-int(item[1]), str(item[0])))[:8]:
                lines.append(f"- {reason}: {count}")
    elif "valid" in payload:
        lines.append(f"구조 검증: {'통과' if payload.get('valid') else '실패'}")
        lines.append(f"OTO 행: {payload.get('rows', 0)}")
        lines.append(f"오류: {len(payload.get('errors', []) or [])}")
    elif "wav_count" in payload:
        lines.append(f"발견한 WAV: {payload.get('wav_count', 0)}")
        if "mapped_wavs" in payload:
            lines.append(f"기존 OTO 매핑: {payload.get('mapped_wavs', 0)}")
            lines.append(f"미매핑 WAV: {payload.get('unmapped_wavs', 0)}")
    lines.extend(("", "상세", json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str)))
    return "\n".join(line for line in lines if line is not None)


def launch_oto_generator_ui(
    *,
    initial_bank: Path | None = None,
    initial_output: Path | None = None,
) -> dict[str, object]:
    """Launch the AutoOTO product UI and return its last operation result."""

    try:
        import customtkinter as ctk
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except ImportError as error:  # pragma: no cover - platform packaging issue
        raise RuntimeError(
            "이 UI에는 Tkinter와 customtkinter가 필요합니다. 프로젝트를 다시 설치하세요."
        ) from error

    ctk.set_default_color_theme("blue")
    ctk.set_appearance_mode("dark")

    class OtoGeneratorApp(ctk.CTk):
        def __init__(self) -> None:
            super().__init__()
            self.title("AutoOTO v3")
            self.geometry("1100x800")
            self.minsize(900, 680)
            self.configure(fg_color=PANEL_BG)
            self._events: queue.Queue[tuple[str, object]] = queue.Queue()
            self._busy = False
            self._advanced_visible = False
            self._log_lines: list[str] = []
            self.last_summary: dict[str, object] = {}
            self.bank_var = tk.StringVar(value=str(initial_bank or ""))
            self.base_oto_var = tk.StringVar()
            self.output_var = tk.StringVar(value=str(initial_output or ""))
            self.recursive_var = tk.BooleanVar(value=False)
            self.batch_var = tk.BooleanVar(value=False)
            self.folder_oto_var = tk.BooleanVar(value=True)
            self.bpm_var = tk.StringVar()
            self.estimate_bpm_var = tk.BooleanVar(value=False)
            self.vc_end_var = tk.StringVar(value=next(iter(VC_END_CHOICES)))
            self.release_pre_var = tk.StringVar(value=next(iter(RELEASE_PRE_CHOICES)))
            self.style_vars = {name: tk.StringVar() for name, _, _ in STYLE_FIELDS}
            self.detection_var = tk.StringVar(value="형식 감지: 보이스뱅크 폴더를 고르면 표시됩니다.")
            self._detection: FormatDetection | None = None
            self._language_acknowledged: set[tuple[str, str]] = set()
            self.overwrite_oto_var = tk.BooleanVar(value=False)
            self.encoding_var = tk.StringVar(value="preserve")
            self.language_var = tk.StringVar(value="한국어")
            self.format_var = tk.StringVar(value=AUTO_FORMAT_CHOICE)
            self.diagnostic_vars = {name: tk.BooleanVar(value=False) for name in DIAGNOSTIC_FILE_LABELS}
            self.theme_var = tk.StringVar(value="다크 모드")
            self.status_var = tk.StringVar(value="대기 중")
            self._buttons: list[object] = []
            self._build_ui(ctk)
            self.after(100, self._poll_events)

        def _build_ui(self, ctk_module) -> None:
            header = ctk_module.CTkFrame(self, fg_color="transparent")
            header.pack(fill="x", padx=18, pady=(14, 8))
            ctk_module.CTkLabel(
                header, text="AutoOTO v3", font=("", 22, "bold"), text_color=HEADER
            ).pack(side="left")
            ctk_module.CTkLabel(
                header, text="WAV를 분석해 새 oto.ini를 생성합니다.", text_color=HINT
            ).pack(side="left", padx=14)
            ctk_module.CTkOptionMenu(
                header,
                values=["다크 모드", "라이트 모드"],
                variable=self.theme_var,
                command=self._set_theme,
                width=120,
            ).pack(side="right")

            self.tabview = ctk_module.CTkTabview(
                self,
                fg_color=PANEL_BG,
                segmented_button_selected_color=PRIMARY,
                segmented_button_selected_hover_color=PRIMARY_HOVER,
            )
            self.tabview.pack(fill="both", expand=True, padx=16, pady=(0, 10))
            for name in ("생성", "결과", "로그", "크레딧"):
                self.tabview.add(name)
            self._build_generate_tab(ctk_module, self.tabview.tab("생성"))
            self._build_result_tab(ctk_module, self.tabview.tab("결과"))
            self._build_log_tab(ctk_module, self.tabview.tab("로그"))
            self._build_credits_tab(ctk_module, self.tabview.tab("크레딧"))

            bottom = ctk_module.CTkFrame(self, fg_color="transparent")
            bottom.pack(fill="x", padx=18, pady=(0, 14))
            self.progress = ctk_module.CTkProgressBar(bottom, mode="indeterminate", height=10)
            self.progress.set(0)
            self.progress.pack(side="left", fill="x", expand=True)
            ctk_module.CTkLabel(bottom, textvariable=self.status_var, text_color=TEXT, width=180).pack(
                side="right", padx=(12, 0)
            )

        def _build_generate_tab(self, ctk_module, tab) -> None:
            scroll = ctk_module.CTkScrollableFrame(tab, fg_color="transparent")
            scroll.pack(fill="both", expand=True, padx=8, pady=8)
            paths = self._section(
                ctk_module,
                scroll,
                "입력과 출력",
                "기본은 별도 폴더 저장입니다. 직접 저장은 아래 덮어쓰기 토글을 켠 경우에만 허용됩니다.",
            )
            self._path_row(ctk_module, paths, "보이스뱅크 폴더", self.bank_var, self._choose_bank)
            self._path_row(
                ctk_module, paths, "기존 oto.ini (선택)", self.base_oto_var, self._choose_base_oto
            )
            ctk_module.CTkLabel(
                paths,
                text=("기존 OTO는 에일리어스·인코딩 참고용이며 timing을 정답으로 재사용하지 않습니다.\n"
                      "없으면 형식에 맞는 별칭(CV·VC·VCV)을 WAV 파일 이름에서 만듭니다."),
                text_color=HINT,
                anchor="w",
                justify="left",
            ).pack(fill="x", padx=132, pady=(0, 4))
            self._path_row(ctk_module, paths, "출력 폴더", self.output_var, self._choose_output)
            ctk_module.CTkCheckBox(
                paths,
                text="기존 oto.ini 덮어쓰기 허용 (보이스뱅크 폴더 직접 저장)",
                variable=self.overwrite_oto_var,
            ).pack(anchor="w", padx=132, pady=(4, 2))
            ctk_module.CTkLabel(
                paths,
                text="켜면 기존 oto.ini는 타임스탬프 .bak으로 백업된 뒤 교체됩니다.",
                text_color=HINT,
                anchor="w",
            ).pack(fill="x", padx=132, pady=(0, 4))
            ctk_module.CTkCheckBox(
                paths, text="하위 폴더 포함 (한 번에 처리)", variable=self.recursive_var,
                command=lambda: self._exclusive_folder_mode("recursive"),
            ).pack(anchor="w", padx=132, pady=(4, 2))
            ctk_module.CTkCheckBox(
                paths, text="하위 폴더별 일괄 생성 (WAV가 있는 폴더마다 oto.ini를 따로 만듦)",
                variable=self.batch_var, command=lambda: self._exclusive_folder_mode("batch"),
            ).pack(anchor="w", padx=132, pady=2)
            ctk_module.CTkCheckBox(
                paths, text="일괄 생성 때 각 폴더의 oto.ini를 별칭 참고용으로 사용", variable=self.folder_oto_var
            ).pack(anchor="w", padx=158, pady=(2, 10))

            profile = self._section(
                ctk_module,
                scroll,
                "생성 프로필",
                "언어는 직접 고르세요. 포맷은 기존 oto.ini 별칭·파일 이름·폴더 이름으로 자동 감지하거나 직접 고를 수 있습니다.",
            )
            profile_row = ctk_module.CTkFrame(profile, fg_color="transparent")
            profile_row.pack(fill="x", padx=12, pady=(0, 4))
            self.language_menu = self._option_menu(
                ctk_module, profile_row, "언어", self.language_var, tuple(LANGUAGE_CHOICES)
            )
            self.format_menu = self._option_menu(
                ctk_module, profile_row, "포맷", self.format_var, tuple(FORMAT_CHOICES)
            )
            ctk_module.CTkLabel(profile, textvariable=self.detection_var, text_color=HINT, anchor="w").pack(
                fill="x", padx=12, pady=(2, 8)
            )
            tempo_row = ctk_module.CTkFrame(profile, fg_color="transparent")
            tempo_row.pack(fill="x", padx=12, pady=(0, 2))
            ctk_module.CTkLabel(tempo_row, text="녹음 BPM (선택, 실험적)", anchor="w").pack(side="left")
            ctk_module.CTkEntry(tempo_row, textvariable=self.bpm_var, width=80, placeholder_text="예: 120").pack(
                side="left", padx=(10, 16)
            )
            ctk_module.CTkCheckBox(
                tempo_row, text="BPM 자동 추정", variable=self.estimate_bpm_var
            ).pack(side="left")
            ctk_module.CTkLabel(
                profile,
                text=("BPM을 넣으면 음절 위치를 박자 격자 쪽으로 당깁니다. 비우면 사용하지 않습니다(기본).\n"
                      "뱅크에 따라 좋아지기도 나빠지기도 하니 결과를 들어 보고 쓰세요. 쉼 표기 파일과 박자가 안 맞는 파일은 자동 제외합니다.\n"
                      "자동 추정은 박자가 일정한 파일이 충분할 때만 값을 내며, 가이드 없이 녹음했다면 끄세요."),
                text_color=HINT,
                anchor="w",
                justify="left",
            ).pack(fill="x", padx=12, pady=(2, 12))

            self.advanced_toggle_btn = ctk_module.CTkButton(
                scroll,
                text="▶ 고급 설정",
                width=180,
                fg_color=("#D7DEE5", "#2B313D"),
                hover_color=("#CAD4DE", "#3A4251"),
                text_color=("#334151", "#DCE9F8"),
                command=self._toggle_advanced,
            )
            self.advanced_toggle_btn.pack(anchor="e", padx=10, pady=(4, 4))
            self.advanced_frame = self._section(
                ctk_module,
                scroll,
                "고급 설정",
                "비워 둔 칸은 자동(템플릿) 값을 씁니다. 음소 위치는 그대로 두고 OTO 행 모양만 바꿉니다.",
                pack=False,
            )
            option_row = ctk_module.CTkFrame(self.advanced_frame, fg_color="transparent")
            option_row.pack(fill="x", padx=12, pady=(0, 8))
            self.encoding_menu = self._option_menu(
                ctk_module, option_row, "인코딩", self.encoding_var, ENCODING_CHOICES
            )
            self._option_menu(ctk_module, option_row, "VC 끝 위치", self.vc_end_var, tuple(VC_END_CHOICES))
            self._option_menu(
                ctk_module, option_row, "'-' 선행발성", self.release_pre_var, tuple(RELEASE_PRE_CHOICES)
            )
            style_grid = ctk_module.CTkFrame(self.advanced_frame, fg_color="transparent")
            style_grid.pack(fill="x", padx=12, pady=(0, 4))
            for index, (name, label, hint) in enumerate(STYLE_FIELDS):
                row, column = divmod(index, 2)
                ctk_module.CTkLabel(style_grid, text=label, anchor="w", width=130).grid(
                    row=row, column=column * 3, sticky="w", padx=(0, 6), pady=2
                )
                ctk_module.CTkEntry(style_grid, textvariable=self.style_vars[name], width=80,
                                    placeholder_text="자동").grid(row=row, column=column * 3 + 1, pady=2)
                ctk_module.CTkLabel(style_grid, text=hint, text_color=HINT, anchor="w").grid(
                    row=row, column=column * 3 + 2, sticky="w", padx=(6, 18), pady=2
                )
            preset_row = ctk_module.CTkFrame(self.advanced_frame, fg_color="transparent")
            preset_row.pack(fill="x", padx=12, pady=(4, 12))
            for label, command in (
                ("프리셋 불러오기", self._load_style_preset),
                ("프리셋 저장", self._save_style_preset),
                ("기본값으로", self._reset_advanced),
            ):
                ctk_module.CTkButton(
                    preset_row, text=label, width=120, command=command,
                    fg_color=PRIMARY, hover_color=PRIMARY_HOVER,
                ).pack(side="left", padx=(0, 8))
            ctk_module.CTkLabel(
                self.advanced_frame,
                text="진단 파일 저장 (기본 꺼짐, 켜도 oto.ini 결과는 같습니다)",
                text_color=HINT,
                anchor="w",
            ).pack(fill="x", padx=12, pady=(0, 2))
            for name, label in DIAGNOSTIC_FILE_LABELS.items():
                ctk_module.CTkCheckBox(
                    self.advanced_frame, text=label, variable=self.diagnostic_vars[name]
                ).pack(anchor="w", padx=24, pady=2)
            ctk_module.CTkFrame(self.advanced_frame, fg_color="transparent", height=10).pack()

            actions = self._section(
                ctk_module, scroll, "실행", "생성 백엔드는 routed_hsmm으로 고정됩니다."
            )
            self.actions_frame = actions
            action_row = ctk_module.CTkFrame(actions, fg_color="transparent")
            action_row.pack(fill="x", padx=12, pady=(0, 12))
            for label, command in (
                ("입력 검사", self._inspect),
                ("OTO 생성", self._generate),
                ("출력 검증", self._validate_output),
                ("출력 폴더 열기", self._open_output),
            ):
                button = ctk_module.CTkButton(
                    action_row,
                    text=label,
                    command=command,
                    fg_color=PRIMARY,
                    hover_color=PRIMARY_HOVER,
                )
                button.pack(side="left", padx=(0, 8))
                self._buttons.append(button)

        def _build_result_tab(self, ctk_module, tab) -> None:
            toolbar = ctk_module.CTkFrame(tab, fg_color="transparent")
            toolbar.pack(fill="x", padx=8, pady=(8, 4))
            ctk_module.CTkLabel(
                toolbar,
                text="review는 자동 검수 보류이며 오류 확정이 아닙니다.",
                text_color=HINT,
            ).pack(side="left")
            ctk_module.CTkButton(
                toolbar,
                text="생성 보고서 열기",
                width=140,
                command=self._open_report,
                fg_color=PRIMARY,
                hover_color=PRIMARY_HOVER,
            ).pack(side="right")
            self.result_text = ctk_module.CTkTextbox(tab, wrap="word")
            self.result_text.pack(fill="both", expand=True, padx=8, pady=(4, 8))
            self.result_text.insert("1.0", "아직 실행 결과가 없습니다.")
            self.result_text.configure(state="disabled")

        def _build_log_tab(self, ctk_module, tab) -> None:
            toolbar = ctk_module.CTkFrame(tab, fg_color="transparent")
            toolbar.pack(fill="x", padx=8, pady=(8, 4))
            ctk_module.CTkButton(
                toolbar,
                text="로그 복사",
                width=110,
                command=self._copy_log,
                fg_color=PRIMARY,
                hover_color=PRIMARY_HOVER,
            ).pack(side="right")
            self.log = ctk_module.CTkTextbox(tab, wrap="word")
            self.log.pack(fill="both", expand=True, padx=8, pady=(4, 8))
            self.log.configure(state="disabled")

        def _build_credits_tab(self, ctk_module, tab) -> None:
            scroll = ctk_module.CTkScrollableFrame(tab, fg_color="transparent")
            scroll.pack(fill="both", expand=True, padx=12, pady=12)
            ctk_module.CTkLabel(
                scroll,
                text="보정 모델 데이터 제공에 도움 주신 분들",
                font=("", 16, "bold"),
                text_color=HEADER,
                anchor="w",
            ).pack(fill="x", padx=10, pady=(8, 12))
            for line in CREDIT_LINES:
                ctk_module.CTkLabel(scroll, text=line, text_color=TEXT, anchor="w").pack(
                    fill="x", padx=12, pady=2
                )
            ctk_module.CTkLabel(
                scroll,
                text="감사합니다!",
                font=("", 14, "bold"),
                text_color=HEADER,
                anchor="w",
            ).pack(fill="x", padx=10, pady=(16, 8))

        def _section(self, ctk_module, parent, title: str, subtitle: str, *, pack: bool = True):
            frame = ctk_module.CTkFrame(
                parent,
                fg_color=PANEL_BG,
                border_width=1,
                border_color=PANEL_BORDER,
            )
            if pack:
                frame.pack(fill="x", padx=10, pady=6)
            ctk_module.CTkLabel(
                frame,
                text=title,
                font=("", 14, "bold"),
                text_color=HEADER,
                anchor="w",
            ).pack(fill="x", padx=12, pady=(10, 2))
            ctk_module.CTkLabel(
                frame,
                text=subtitle,
                text_color=HINT,
                anchor="w",
                justify="left",
            ).pack(fill="x", padx=12, pady=(0, 8))
            return frame

        def _path_row(self, ctk_module, parent, label: str, variable, command) -> None:
            row = ctk_module.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=4)
            ctk_module.CTkLabel(row, text=label, width=110, anchor="w").pack(side="left")
            ctk_module.CTkEntry(row, textvariable=variable).pack(
                side="left", fill="x", expand=True, padx=(10, 8)
            )
            ctk_module.CTkButton(
                row,
                text="찾아보기",
                width=90,
                command=command,
                fg_color=PRIMARY,
                hover_color=PRIMARY_HOVER,
            ).pack(side="right")

        def _option_menu(self, ctk_module, parent, label: str, variable, values):
            group = ctk_module.CTkFrame(parent, fg_color="transparent")
            group.pack(side="left", fill="x", expand=True, padx=(0, 10))
            ctk_module.CTkLabel(group, text=label, anchor="w").pack(anchor="w")
            menu = ctk_module.CTkOptionMenu(group, variable=variable, values=list(values))
            menu.pack(fill="x", pady=(2, 0))
            return menu

        def _toggle_advanced(self) -> None:
            self._advanced_visible = not self._advanced_visible
            if self._advanced_visible:
                self.advanced_frame.pack(fill="x", padx=10, pady=6, before=self.actions_frame)
                self.advanced_toggle_btn.configure(text="▼ 고급 설정")
            else:
                self.advanced_frame.pack_forget()
                self.advanced_toggle_btn.configure(text="▶ 고급 설정")

        def _set_theme(self, choice: str) -> None:
            ctk.set_appearance_mode("light" if choice == "라이트 모드" else "dark")

        def _choose_bank(self) -> None:
            selected = filedialog.askdirectory(title="WAV 보이스뱅크 폴더 선택")
            if selected:
                self.bank_var.set(selected)
                if not self.output_var.get().strip():
                    bank = Path(selected)
                    self.output_var.set(str(bank.parent / f"{bank.name}_auto_oto"))
                self._refresh_detection()

        def _choose_base_oto(self) -> None:
            selected = filedialog.askopenfilename(
                title="기존 oto.ini 선택",
                filetypes=(("OTO files", "*.ini"), ("All files", "*.*")),
            )
            if selected:
                self.base_oto_var.set(selected)
                self._refresh_detection()

        def _choose_output(self) -> None:
            selected = filedialog.askdirectory(title="출력 폴더 선택", mustexist=False)
            if selected:
                self.output_var.set(selected)

        def _append_log(self, text: str) -> None:
            line = text.rstrip()
            self._log_lines.append(line)
            self.log.configure(state="normal")
            self.log.insert("end", line + "\n")
            self.log.see("end")
            self.log.configure(state="disabled")

        def _show_result(self, title: str, payload: Mapping[str, object]) -> None:
            self.result_text.configure(state="normal")
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", format_result_text(title, payload))
            self.result_text.configure(state="disabled")
            self.tabview.set("결과")

        def _set_busy(self, busy: bool) -> None:
            self._busy = busy
            for button in self._buttons:
                button.configure(state="disabled" if busy else "normal")
            if busy:
                self.progress.start()
            else:
                self.progress.stop()
                self.progress.set(0)

        def _run(self, title: str, operation: Callable[[], Mapping[str, object]]) -> None:
            if self._busy:
                return
            self._set_busy(True)
            self.status_var.set(f"{title} 실행 중...")
            self._append_log(f"[시작] {title}")

            def worker() -> None:
                try:
                    self._events.put(("success", UIOperationResult(title, dict(operation()))))
                except Exception as error:
                    self._events.put(("error", f"{type(error).__name__}: {error}"))

            threading.Thread(target=worker, name="auto-oto-ui-worker", daemon=True).start()

        def _exclusive_folder_mode(self, changed: str) -> None:
            if changed == "batch" and self.batch_var.get():
                self.recursive_var.set(False)
            elif changed == "recursive" and self.recursive_var.get():
                self.batch_var.set(False)

        def _style(self) -> OtoStyle:
            return build_oto_style({name: var.get() for name, var in self.style_vars.items()})

        def _load_style_preset(self) -> None:
            selected = filedialog.askopenfilename(
                title="고급 설정 프리셋 불러오기", filetypes=(("JSON", "*.json"), ("All files", "*.*"))
            )
            if not selected:
                return
            try:
                style = load_oto_style(Path(selected))
            except Exception as error:
                messagebox.showerror("프리셋 오류", str(error), parent=self)
                return
            for name, value in style_entry_values(style).items():
                self.style_vars[name].set(value)
            self._append_log(f"[고급 설정] 프리셋을 불러왔습니다: {selected}")

        def _save_style_preset(self) -> None:
            try:
                style = self._style()
            except ValueError as error:
                messagebox.showerror("입력 오류", str(error), parent=self)
                return
            selected = filedialog.asksaveasfilename(
                title="고급 설정 프리셋 저장", defaultextension=".json", filetypes=(("JSON", "*.json"),)
            )
            if selected:
                save_oto_style(Path(selected), style)
                self._append_log(f"[고급 설정] 프리셋을 저장했습니다: {selected}")

        def _reset_advanced(self) -> None:
            for var in self.style_vars.values():
                var.set("")
            self.encoding_var.set("preserve")
            self.vc_end_var.set(next(iter(VC_END_CHOICES)))
            self.release_pre_var.set(next(iter(RELEASE_PRE_CHOICES)))

        def _refresh_detection(self) -> FormatDetection | None:
            bank = self.bank_var.get().strip()
            if not bank or not Path(bank).expanduser().is_dir():
                return None
            try:
                if self.batch_var.get():
                    self.detection_var.set("형식 감지: 일괄 생성은 폴더마다 따로 감지합니다.")
                    return None
                detection = detect_bank_format(
                    bank, self.base_oto_var.get(), self.language_var.get(), recursive=self.recursive_var.get()
                )
            except Exception as error:
                self.detection_var.set(f"형식 감지 실패: {error}")
                return None
            self._detection = detection
            self.detection_var.set(format_detection_text(detection))
            return detection

        def _confirm_language(self, bank: str, base_oto: str | None) -> bool:
            """Warn once per bank and language when the names look like another language."""

            key = (str(Path(bank).expanduser().resolve()), self.language_var.get())
            if key in self._language_acknowledged:
                return True
            try:
                warning = bank_language_warning(
                    bank, base_oto, self.language_var.get(),
                    recursive=self.recursive_var.get() or self.batch_var.get(),
                )
            except Exception:
                return True
            if warning is None:
                return True
            self._append_log(f"[언어 확인] {warning}")
            if not messagebox.askyesno("언어 확인", f"{warning}\n\n이 언어로 계속할까요?", parent=self):
                return False
            self._language_acknowledged.add(key)
            return True

        def _detected_format_for_generation(self) -> str | None:
            if self.format_var.get() != AUTO_FORMAT_CHOICE:
                return None
            detection = self._refresh_detection()
            if detection is None or detection.format is None:
                return None
            self._append_log(f"[{format_detection_text(detection)}]")
            for line in detection.evidence:
                self._append_log(f"  - {line}")
            return detection.format

        def _generation_options(self, *, force: bool = False, detected_format: str | None = None) -> GenerationOptions:
            return build_product_generation_options(
                bank=self.bank_var.get(),
                output=self.output_var.get(),
                base_oto=self.base_oto_var.get(),
                recursive=self.recursive_var.get(),
                output_encoding=self.encoding_var.get(),
                language_choice=self.language_var.get(),
                format_choice=self.format_var.get(),
                force=force,
                overwrite_existing_oto=self.overwrite_oto_var.get(),
                diagnostic_files={name: var.get() for name, var in self.diagnostic_vars.items()},
                detected_format=detected_format,
                bpm=self.bpm_var.get(),
                estimate_bpm=self.estimate_bpm_var.get(),
                oto_style=self._style(),
                vc_end_choice=self.vc_end_var.get(),
                release_pre_choice=self.release_pre_var.get(),
            )

        def _read_options(self, *, force: bool = False, detected_format: str | None = None) -> GenerationOptions | None:
            try:
                return self._generation_options(force=force, detected_format=detected_format)
            except Exception as error:
                messagebox.showerror("입력 오류", str(error), parent=self)
                self._append_log(f"[입력 오류] {type(error).__name__}: {error}")
                return None

        def _inspect(self) -> None:
            bank = self.bank_var.get().strip()
            if not bank:
                messagebox.showerror("입력 오류", "WAV 보이스뱅크 폴더를 지정하세요.", parent=self)
                return
            if self.batch_var.get():
                language = LANGUAGE_CHOICES[self.language_var.get()]
                use_folder_oto = self.folder_oto_var.get()

                def batch_inspect() -> Mapping[str, object]:
                    folders = discover_bank_folders(Path(bank).expanduser(), language=language, use_folder_oto=use_folder_oto)
                    return {"batch_folders": [folder.to_dict() for folder in folders], "folder_count": len(folders)}

                self._run("입력 검사 (일괄)", batch_inspect)
                return
            detection = self._refresh_detection()
            self._confirm_language(bank, self.base_oto_var.get())
            options = self._read_options(detected_format=detection.format if detection else None)
            if options is None:
                return

            def operation() -> Mapping[str, object]:
                payload = dict(inspect_voicebank(options.bank, options.base_oto, options.recursive))
                if detection is not None:
                    payload["format_detection"] = detection.to_dict()
                return payload

            self._run("입력 검사", operation)

        def _generate(self) -> None:
            bank = self.bank_var.get().strip()
            if bank and not self._confirm_language(bank, self.base_oto_var.get()):
                return
            if self.batch_var.get():
                self._generate_batch()
                return
            detected = self._detected_format_for_generation()
            options = self._read_options(detected_format=detected)
            if options is None:
                return
            if options.output.resolve() == options.bank.resolve() and not options.allow_in_place_output:
                messagebox.showerror(
                    "직접 저장 차단",
                    "출력 폴더가 보이스뱅크와 같습니다. 기존 oto.ini 덮어쓰기 허용 토글을 켜세요.",
                    parent=self,
                )
                return
            output_has_files = options.output.exists() and any(options.output.iterdir())
            if output_has_files and not options.allow_in_place_output:
                confirmed = messagebox.askyesno(
                    "출력 폴더 확인",
                    "출력 폴더가 비어 있지 않습니다. 기존 생성 산출물은 백업 후 교체될 수 있습니다. 계속할까요?",
                    parent=self,
                )
                if not confirmed:
                    return
                options = self._read_options(force=True, detected_format=detected)
                if options is None:
                    return

            def operation() -> Mapping[str, object]:
                report = generate_voicebank(options)
                return summarize_generation_result(report, output=options.output)

            self._run("OTO 생성", operation)

        def _generate_batch(self) -> None:
            root = Path(self.bank_var.get().strip()).expanduser()
            output_root = Path(self.output_var.get().strip()).expanduser() if self.output_var.get().strip() else None
            in_place = self.overwrite_oto_var.get()
            if output_root is None and not in_place:
                messagebox.showerror("입력 오류", "출력 폴더를 지정하세요.", parent=self)
                return
            if output_root is None:
                self.output_var.set(str(root))
            try:
                template = self._generation_options(force=True, detected_format="cvvc")
                folders = discover_bank_folders(
                    root, language=LANGUAGE_CHOICES[self.language_var.get()], use_folder_oto=self.folder_oto_var.get()
                )
            except Exception as error:
                messagebox.showerror("입력 오류", str(error), parent=self)
                return
            if not folders:
                messagebox.showerror("입력 오류", "WAV가 있는 폴더를 찾지 못했습니다.", parent=self)
                return
            selected = self._choose_batch_folders(folders)
            if not selected:
                return
            fixed_format = FORMAT_CHOICES[self.format_var.get()]
            use_folder_oto = self.folder_oto_var.get()
            events = self._events

            def generate_one(folder: BankFolder, output: Path) -> Mapping[str, object]:
                format_hint = fixed_format or folder.format
                if not format_hint:
                    raise ValueError("형식을 알 수 없습니다. 포맷을 직접 고르고 다시 실행하세요.")
                folder_oto = folder.path / "oto.ini"
                base_oto = folder_oto if use_folder_oto and folder_oto.is_file() else None
                options = replace(
                    template,
                    bank=folder.path,
                    output=output,
                    base_oto=base_oto,
                    recursive=False,
                    format_hint=format_hint,
                    generate_missing_aliases=base_oto is None,
                    allow_in_place_output=in_place,
                    force=True,
                )
                report = generate_voicebank(options)
                return summarize_generation_result(report, output=output)

            def progress(index: int, total: int, folder: BankFolder) -> None:
                events.put(("log", f"[일괄 {index}/{total}] {folder.relative} ({(fixed_format or folder.format or '?').upper()})"))

            def operation() -> Mapping[str, object]:
                return run_batch(
                    selected, root=root, output_root=output_root or root, in_place=in_place,
                    generate_one=generate_one, progress=progress,
                )

            self._run("일괄 OTO 생성", operation)

        def _choose_batch_folders(self, folders: list[BankFolder]) -> list[BankFolder]:
            dialog = ctk.CTkToplevel(self)
            dialog.title("일괄 생성할 폴더 선택")
            dialog.geometry("820x520")
            dialog.transient(self)
            dialog.grab_set()
            ctk.CTkLabel(
                dialog,
                text="폴더마다 따로 oto.ini를 만듭니다. 형식을 알 수 없는 폴더는 기본으로 꺼져 있습니다.",
                text_color=HINT, anchor="w",
            ).pack(fill="x", padx=14, pady=(12, 6))
            frame = ctk.CTkScrollableFrame(dialog)
            frame.pack(fill="both", expand=True, padx=12, pady=6)
            fixed = FORMAT_CHOICES[self.format_var.get()]
            variables = []
            for folder in folders:
                var = tk.BooleanVar(value=bool(fixed or folder.format))
                variables.append(var)
                fmt = (fixed or folder.format or "?").upper()
                note = "직접 선택" if fixed else format_detection_text(folder.detection).replace("형식 감지: ", "")
                text = (f"{folder.relative}   WAV {folder.wav_count}   "
                        f"{'oto.ini 있음' if folder.has_oto else 'oto.ini 없음'}   → {fmt}   ({note})")
                ctk.CTkCheckBox(frame, text=text, variable=var).pack(anchor="w", pady=2)
            chosen: list[BankFolder] = []

            def accept() -> None:
                chosen.extend(folder for folder, var in zip(folders, variables) if var.get())
                dialog.destroy()

            buttons = ctk.CTkFrame(dialog, fg_color="transparent")
            buttons.pack(fill="x", padx=12, pady=(6, 12))
            ctk.CTkButton(buttons, text="취소", width=100, command=dialog.destroy).pack(side="right")
            ctk.CTkButton(buttons, text="생성", width=100, command=accept,
                          fg_color=PRIMARY, hover_color=PRIMARY_HOVER).pack(side="right", padx=8)
            self.wait_window(dialog)
            return chosen

        def _validate_output(self) -> None:
            options = self._read_options()
            if options is None:
                return

            def operation() -> Mapping[str, object]:
                from .cli import validate_oto

                oto_path = options.output / "oto.ini"
                if not oto_path.is_file():
                    raise FileNotFoundError(f"생성된 oto.ini가 없습니다: {oto_path}")
                return validate_oto(oto_path, options.bank)

            self._run("출력 검증", operation)

        def _open_output(self) -> None:
            output = Path(self.output_var.get().strip()).expanduser()
            if not output.exists():
                messagebox.showwarning("출력 폴더 없음", "먼저 OTO를 생성하세요.", parent=self)
                return
            os.startfile(str(output))

        def _open_report(self) -> None:
            report = Path(self.output_var.get().strip()).expanduser() / "generation_report.json"
            if self.last_summary.get("oto") and self.last_summary.get("report_path") is None:
                messagebox.showwarning(
                    "보고서 없음",
                    "생성 보고서 저장이 꺼져 있었습니다. 고급 출력 옵션에서 켜고 다시 생성하세요.",
                    parent=self,
                )
                return
            if not report.is_file():
                messagebox.showwarning("보고서 없음", "먼저 OTO를 생성하세요.", parent=self)
                return
            os.startfile(str(report))

        def _copy_log(self) -> None:
            text = "\n".join(self._log_lines)
            self.clipboard_clear()
            self.clipboard_append(text)
            self.status_var.set("로그 복사 완료")

        def _poll_events(self) -> None:
            try:
                while True:
                    kind, value = self._events.get_nowait()
                    if kind == "log":
                        self._append_log(str(value))
                        self.status_var.set(str(value)[:60])
                        continue
                    self._set_busy(False)
                    if kind == "success":
                        result = value
                        assert isinstance(result, UIOperationResult)
                        self.last_summary = dict(result.payload)
                        self.status_var.set(f"{result.title} 완료")
                        self._append_log(
                            json.dumps(dict(result.payload), ensure_ascii=False, indent=2, default=str)
                        )
                        for line in (*tempo_log_lines(result.payload), *alias_generation_log_lines(result.payload)):
                            self._append_log(line)
                        for item in result.payload.get("results", []) or []:
                            state = "완료" if item.get("ok") else f"실패: {item.get('error')}"
                            self._append_log(f"[일괄] {item.get('folder')}: {state}")
                        excluded = excluded_rows_log_lines(result.payload)
                        for line in excluded:
                            self._append_log(line)
                        if excluded:
                            self.status_var.set(
                                f"{result.title} 완료 (생성되지 않은 행 {result.payload.get('excluded_alias_rows')}개, 로그 참고)")
                        self._show_result(result.title, result.payload)
                    else:
                        self.status_var.set("오류")
                        self._append_log(f"[오류] {value}")
                        self.tabview.set("로그")
                        messagebox.showerror("작업 실패", str(value), parent=self)
            except queue.Empty:
                pass
            self.after(100, self._poll_events)

    app = OtoGeneratorApp()
    app.mainloop()
    return app.last_summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="AutoOTO v3 desktop UI")
    parser.add_argument("--bank", type=Path, help="Pre-fill the voicebank folder")
    parser.add_argument("--output", type=Path, help="Pre-fill the output folder")
    args = parser.parse_args(argv)
    try:
        launch_oto_generator_ui(initial_bank=args.bank, initial_output=args.output)
    except Exception as error:
        print(f"ERROR: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
