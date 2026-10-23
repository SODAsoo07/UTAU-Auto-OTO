"""Product UI for running the real voicebank OTO generator."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping

from . import __version__, display_version
from .app_icon import CTK_TOPLEVEL_ICON_DELAY_MS, apply_app_icon
from .bank_detection import FormatDetection, check_language, detect_format
from .batch import BankFolder, discover_bank_folders, run_batch
from .language_pack import find_language_pack
from .i18n import LANGUAGE_NAMES, get_language, load_ui_language, machine_translation_notice, save_ui_language, t, translate
from .generator import (DIAGNOSTIC_FILE_OPTIONS, GenerationCancelled, GenerationOptions, _discover_wavs,
                        generate_voicebank, inspect_voicebank)
from .oto_io import apply_alias_suffix_to_oto, normalize_alias_suffix, parse_oto, with_alias_suffix, write_lines_atomic
from .oto_preview import launch_oto_preview
from .oto_style import OtoStyle, load_oto_style, save_oto_style
from .pack_style import NO_STYLE


APP_NAME = f"AutoOTO v{display_version()}"

LANGUAGE_CHOICES: Mapping[str, str | None] = {
    "한국어": "korean",
    "일본어": "japanese",
    "영어": "english",
}
AUTO_FORMAT_CHOICE = "자동 감지"
FORMAT_CHOICES: Mapping[str, str | None] = {
    AUTO_FORMAT_CHOICE: None,
    "CV": "cv",
    "CVC": "cvc",
    "CVVC": "cvvc",
    "VCV": "vcv",
    "VCCV": "vccv",
}
# Formats the format menu hides for a language (no route or pack for that pair).
HIDDEN_FORMATS: Mapping[str, frozenset[str]] = {
    "korean": frozenset({"vccv"}),
    "japanese": frozenset({"cvc", "vccv"}),
    "english": frozenset({"cv", "cvc", "cvvc", "vcv"}),
}


def format_choices_for(language_choice: str | None) -> tuple[str, ...]:
    """Format menu entries for a language choice; automatic detection is always offered."""

    hidden = HIDDEN_FORMATS.get(LANGUAGE_CHOICES.get(language_choice or "") or "", frozenset())
    return tuple(key for key, value in FORMAT_CHOICES.items() if value is None or value not in hidden)


ENCODING_CHOICES = ("preserve", "utf-8", "cp932", "shift_jis", "cp949", "euc-kr")
VC_END_CHOICES: Mapping[str, str] = {
    "다음 자음 안에서 끝냄 (권장)": "consonant",
    "템플릿 기본값": "template",
}
# Base-OTO rows that could not be generated: keep the base values (default) or leave them out.
UNMAPPED_ROW_CHOICES: Mapping[str, str] = {
    "기존 oto 값 유지 (권장)": "keep_base",
    "oto.ini에서 빼기": "exclude",
}
RELEASE_PRE_CHOICES: Mapping[str, str] = {
    "모음 끝 (권장)": "vowel_end",
    "템플릿 기본값": "template",
}
STABLE_WINDOW_CHOICES: Mapping[str, str] = {
    "안정된 부분만 (권장)": "stable",
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
# v17 adds level-robust activity: a softly recorded take's room/reverb tail is not taken for
# speech (run_outputs/row_risk_20261019/eval_level.txt: quiet folders 55% -> 33% rows off by
# > 80 ms, no bank worse in eval_all_v17.txt).
# v18 adds the speech span limit (user's idea, run_outputs/row_risk_20261019/variant_span.txt:
# 13.49% -> 13.10% rows off by > 80 ms on 18 folders, syllable shifts 7.82% -> 7.46%).
RECOMMENDED_DEVELOPMENT_PROFILE = "cvvc_slot_rhythm_v19"
RECOMMENDED_VC_END_POLICY = "consonant"
RECOMMENDED_RELEASE_PRE_POLICY = "vowel_end"
RECOMMENDED_STABLE_WINDOW_POLICY = "stable"
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
    "다다쿤 DADAKUN",
)

# Colors reused from the existing AutoOTO light/dark palette.
PANEL_BG = ("#EDE9E4", "#1F232B")
PANEL_BORDER = ("#B3B0AA", "#303744")
PRIMARY = ("#9CB3C2", "#2F7EB8")
PRIMARY_HOVER = ("#8DA6B8", "#286C9D")
TEXT = ("#2B2D2E", "#FFFFFF")
HINT = ("#4E5866", "#DCE9F8")
HEADER = ("#394045", "#EAF4FF")
# The main action (OTO 생성) gets its own color; the other run buttons are muted.
GENERATE = ("#3C8D5A", "#2E9E5B")
GENERATE_HOVER = ("#337A4D", "#26864D")
SECONDARY = ("#D7DEE5", "#2B313D")
SECONDARY_HOVER = ("#CAD4DE", "#3A4251")
SECONDARY_TEXT = ("#334151", "#DCE9F8")


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
    stable_window_policy: str = "template",
    unmapped_alias_policy: str = "fail",
    unmapped_row_output: str = "exclude",
    diagnostic_files: Mapping[str, bool] | None = None,
    generate_missing_aliases: bool = False,
    oto_style: OtoStyle | None = None,
    bpm: float | None = None,
    estimate_bpm: bool = False,
    style_profile: str | None = None,
    alias_suffix: str = "",
) -> GenerationOptions:
    """Convert UI values into the same options object used by the CLI.

    Diagnostic files are written unless named as False in ``diagnostic_files``.
    """

    bank_text = str(bank).strip()
    output_text = str(output).strip()
    if not bank_text:
        raise ValueError(t("WAV 보이스뱅크 폴더를 지정하세요."))
    if not output_text:
        raise ValueError(t("출력 폴더를 지정하세요."))
    threshold = float(review_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(t("검수 임계값은 0.0~1.0 범위여야 합니다."))
    parsed_max_files: int | None
    if max_files is None or not str(max_files).strip():
        parsed_max_files = None
    else:
        parsed_max_files = int(str(max_files).strip())
        if parsed_max_files < 1:
            raise ValueError(t("최대 WAV 수는 1 이상이어야 합니다."))
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
        stable_window_policy=stable_window_policy,
        unmapped_alias_policy=unmapped_alias_policy,
        unmapped_row_output=unmapped_row_output,
        **_diagnostic_file_flags(diagnostic_files, default=True),
        generate_missing_aliases=generate_missing_aliases,
        oto_style=oto_style,
        bpm=bpm,
        estimate_bpm=estimate_bpm,
        style_profile=_style_profile_value(style_profile),
        alias_suffix=parse_alias_suffix(alias_suffix),
        alignment_workers=0,  # automatic: parallel alignment, identical rows
    )


def _style_profile_value(value: str | None) -> Path | str | None:
    """Blank = the language pack's default style, 'none' = templates only, else a fit-style file."""
    text = str(value or "").strip()
    if not text:
        return None
    if text.casefold() == NO_STYLE:
        return NO_STYLE
    path = Path(text).expanduser()
    if not path.is_file():
        raise ValueError(t("스타일 프로필 파일이 없습니다: {path}", path=path))
    return path


def _diagnostic_file_flags(choices: Mapping[str, bool] | None, *, default: bool) -> dict[str, bool]:
    choices = dict(choices or {})
    unknown = set(choices) - set(DIAGNOSTIC_FILE_OPTIONS)
    if unknown:
        raise ValueError(t("알 수 없는 진단 파일 옵션입니다: {names}", names=", ".join(sorted(unknown))))
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
    style_profile: str | None = None,
    alias_suffix: str = "",
    unmapped_row_choice: str | None = None,
    stable_window_choice: str | None = None,
) -> GenerationOptions:
    """Build the safe production options exposed by the product UI.

    Diagnostic files are off unless named as True in ``diagnostic_files``.
    """

    if language_choice not in LANGUAGE_CHOICES:
        raise ValueError(t("지원하지 않는 언어 선택입니다: {choice}", choice=language_choice))
    if format_choice not in FORMAT_CHOICES:
        raise ValueError(t("지원하지 않는 형식 선택입니다: {choice}", choice=format_choice))
    format_hint = FORMAT_CHOICES[format_choice] or detected_format
    if not format_hint:
        raise ValueError(t("형식을 자동으로 알아내지 못했습니다. 포맷을 직접 선택하세요."))
    vc_end_policy = VC_END_CHOICES[vc_end_choice] if vc_end_choice else RECOMMENDED_VC_END_POLICY
    release_pre_policy = (RELEASE_PRE_CHOICES[release_pre_choice] if release_pre_choice
                          else RECOMMENDED_RELEASE_PRE_POLICY)
    stable_window_policy = (STABLE_WINDOW_CHOICES[stable_window_choice] if stable_window_choice
                            else RECOMMENDED_STABLE_WINDOW_POLICY)
    has_base_oto = base_oto is not None and bool(str(base_oto).strip())
    if output_encoding not in ENCODING_CHOICES:
        raise ValueError(t("지원하지 않는 출력 인코딩입니다: {encoding}", encoding=output_encoding))
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
        development_alignment_profile=recommended_alignment_profile(LANGUAGE_CHOICES[language_choice], format_hint),
        vc_end_policy=vc_end_policy,
        release_pre_policy=release_pre_policy,
        stable_window_policy=stable_window_policy,
        unmapped_alias_policy=PRODUCT_UNMAPPED_ALIAS_POLICY,
        unmapped_row_output=UNMAPPED_ROW_CHOICES[unmapped_row_choice or next(iter(UNMAPPED_ROW_CHOICES))],
        diagnostic_files=_diagnostic_file_flags(diagnostic_files, default=False),
        generate_missing_aliases=not has_base_oto,
        oto_style=None if oto_style is None or oto_style.is_default else oto_style,
        bpm=parse_bpm(bpm),
        estimate_bpm=bool(estimate_bpm),
        style_profile=style_profile,
        alias_suffix=alias_suffix,
    )


def recommended_alignment_profile(language_hint: str | None, format_hint: str | None) -> str:
    """The language pack's profile for pack formats (English VCCV stays on v16), else the GUI's."""

    pack = find_language_pack(language_hint, format_hint) if language_hint and format_hint else None
    return pack.alignment_profile if pack is not None and pack.alignment_profile else RECOMMENDED_DEVELOPMENT_PROFILE


def parse_bpm(value: str | float | None) -> float | None:
    """Blank means "not given"; otherwise a tempo between 60 and 240 BPM."""

    if value is None or not str(value).strip():
        return None
    try:
        bpm = float(str(value).strip())
    except ValueError as error:
        raise ValueError(t("BPM은 숫자로 입력하세요 (예: 120).")) from error
    if not 60.0 <= bpm <= 240.0:
        raise ValueError(t("BPM은 60~240 범위로 입력하세요 (보통 100~160)."))
    return bpm


def parse_alias_suffix(value: str | None) -> str:
    """Suffix entry -> the text after '_' ("C4" and "_C4" -> "C4"); blank means none."""

    try:
        return normalize_alias_suffix(value)
    except ValueError as error:
        raise ValueError(t("접미사에는 쉼표(,), = 기호, 줄바꿈을 쓸 수 없습니다.")) from error


def apply_suffix_to_output(oto: Path, suffix: str | None) -> dict[str, object]:
    """Add the suffix to an already generated oto.ini, its clean/review files and row_report.jsonl.

    Aliases that already end with the suffix are kept, so applying it twice changes nothing.
    """

    normalized = parse_alias_suffix(suffix)
    if not normalized:
        raise ValueError(t("적용할 접미사를 입력하세요."))
    if not oto.is_file():
        raise FileNotFoundError(t("생성된 oto.ini가 없습니다: {path}", path=oto))
    files: dict[str, list[int]] = {}
    changed, total = apply_alias_suffix_to_oto(oto, normalized)
    files[oto.name] = [changed, total]
    for companion in (oto.with_name(f"{oto.stem}.clean{oto.suffix}"), oto.with_name(f"{oto.stem}.review{oto.suffix}")):
        if companion.is_file():
            files[companion.name] = list(apply_alias_suffix_to_oto(companion, normalized, backup_existing=False))
    report = oto.with_name("row_report.jsonl")
    if report.is_file():  # the preview matches rows to their risk by alias
        entries = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines() if line.strip()]
        for entry in entries:
            entry["alias"] = with_alias_suffix(str(entry.get("alias", "")), normalized)
        write_lines_atomic(report, (json.dumps(entry, ensure_ascii=False, sort_keys=True) for entry in entries),
                           encoding="utf-8", newline="\n", backup_existing=False)
    return {"alias_suffix": f"_{normalized}", "suffixed_oto": str(oto), "changed_aliases": changed, "aliases": total,
            "files": files}


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
            raise ValueError(t("{label}: 숫자를 입력하세요.", label=t(label))) from error
    try:
        return OtoStyle.from_dict(payload)
    except ValueError as error:
        raise ValueError(t("고급 설정 값이 범위를 벗어났습니다: {error}", error=error)) from error


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
    return t("형식 감지: {detail}", detail=format_detection_detail(detection))


def format_detection_detail(detection: FormatDetection) -> str:
    if detection.format is None:
        return t("알 수 없음 - 포맷을 직접 선택하세요.")
    return t("{format} (확신 {confidence}, 근거: {source})", format=detection.format.upper(),
             confidence=t(CONFIDENCE_LABELS.get(detection.confidence, detection.confidence)),
             source=t(DETECTION_SOURCE_LABELS.get(detection.source, detection.source)))


def tempo_log_lines(payload: Mapping[str, object]) -> list[str]:
    tempo = payload.get("recording_tempo")
    if not isinstance(tempo, Mapping):
        return []
    estimate = tempo.get("estimate") if isinstance(tempo.get("estimate"), Mapping) else None
    if tempo.get("source") == "user_bpm":
        return [t("[BPM] 입력한 {bpm:g} BPM 사용 (음절 간격 {period:.0f} ms)", bpm=float(tempo["user_bpm"]),
                  period=float(tempo["syllable_period_ms"]))]
    if tempo.get("source") == "estimated" and estimate:
        return [t("[BPM] 자동 추정 {bpm} BPM 사용 ({reason})", bpm=estimate.get("bpm"), reason=estimate.get("reason"))]
    if tempo.get("estimate_enabled") and estimate:
        return [t("[BPM] 추정 안 함: {reason} BPM 없이 생성했습니다.", reason=estimate.get("reason"))]
    return []


def encoding_log_lines(payload: Mapping[str, object]) -> list[str]:
    fallback = payload.get("output_encoding_fallback")
    if not isinstance(fallback, Mapping):
        return []
    return [t("[주의] 파일 이름이나 별칭을 {requested}로 저장할 수 없어 oto.ini를 UTF-8로 저장했습니다.",
              requested=fallback.get("requested"))]


def alias_generation_log_lines(payload: Mapping[str, object]) -> list[str]:
    count = int(payload.get("generated_alias_wavs", 0) or 0)
    if count <= 0:
        return []
    lines = [t("[별칭 생성] 기존 oto.ini 없이 WAV {count}개의 파일 이름에서 형식에 맞는 별칭을 만들었습니다.", count=count)]
    skipped = int(payload.get("alias_generation_skipped_count", 0) or 0)
    if skipped:
        lines.append(t("[주의] 이름을 읽지 못한 WAV {count}개는 빠졌습니다.", count=skipped))
        for item in payload.get("alias_generation_skipped") or []:
            lines.append(f"  - {item.get('wav', '')} ({item.get('detail', '')})")
    return lines


def preview_oto_path(*, last_oto: object, output: Path | None, bank: Path | None) -> Path | None:
    """The oto.ini the preview opens: the last generated one, else the output folder's, else the bank's."""

    candidates = [Path(str(last_oto))] if last_oto else []
    candidates += [folder / "oto.ini" for folder in (output, bank) if folder is not None]
    return next((path for path in candidates if path.is_file()), None)


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
        "review_notice": t("review 행은 자동 검수 보류이며 오류 확정이 아닙니다."),
        "generation_settings": {
            key: (report.get("provenance") or {}).get(key)
            for key in ("development_alignment_profile", "vc_end_policy", "release_pre_policy",
                        "stable_window_policy")
        },
        "usage_review": dict(report.get("usage_review", {}) or {}),
        "style_sample": report.get("style_sample"),
        "excluded_alias_rows": int(report.get("excluded_alias_rows", 0) or 0),
        "kept_base_rows": int(report.get("kept_base_rows", 0) or 0),
        "excluded_alias_rows_csv": report.get("excluded_alias_rows_csv"),
        "recording_tempo": report.get("recording_tempo"),
        "output_encoding_fallback": report.get("output_encoding_fallback"),
        "generated_alias_wavs": int(report.get("generated_alias_wavs", 0) or 0),
        "alias_generation_skipped": list(report.get("alias_generation_skipped", []) or [])[:EXCLUDED_ROW_LOG_SAMPLES],
        "alias_generation_skipped_count": len(report.get("alias_generation_skipped", []) or []),
        "excluded_alias_samples": list(report.get("excluded_alias_samples", []) or [])[:EXCLUDED_ROW_LOG_SAMPLES],
        "failed_recordings": [item.get("wav", "") for item in report.get("failed_recordings", []) or []],
    }
    if oto_path.is_file():
        summary["oto_sha256"] = hashlib.sha256(oto_path.read_bytes()).hexdigest()
    return summary


OperationCancelled = GenerationCancelled  # raised from the progress callback when 취소 is pressed


RENDER_SHARE = 0.08  # the OTO writing stage after alignment


def progress_fraction(stage: str, done: int, total: int) -> float:
    """Overall 0..1 progress from a runtime stage ("align:k/n" or "render")."""
    part = (done / total) if total else 1.0
    if stage.startswith("align:"):
        k, n = (int(value) for value in stage.split(":", 1)[1].split("/"))
        return (1.0 - RENDER_SHARE) * ((k - 1) + part) / max(n, 1)
    if stage == "render":
        return 1.0 - RENDER_SHARE + RENDER_SHARE * part
    return 0.0


def progress_text(stage: str, done: int, total: int, elapsed_s: float, fraction: float, prefix: str = "") -> str:
    if stage.startswith("align:"):
        label = t("정렬 {step}단계 {done}/{total}", step=stage.split(":", 1)[1], done=done, total=total)
    elif stage == "render":
        label = t("OTO 작성 {done}/{total}", done=done, total=total)
    else:
        label = stage
    remaining = ""
    if fraction >= 0.03 and elapsed_s >= 3.0:
        seconds = elapsed_s * (1.0 - fraction) / fraction
        remaining = (t(" · 남은 시간 약 {minutes}분", minutes=max(1, round(seconds / 60))) if seconds >= 90
                     else t(" · 남은 시간 약 {seconds}초", seconds=max(1, round(seconds))))
    return f"{prefix}{label}{remaining}"


def excluded_rows_log_lines(payload: Mapping[str, object]) -> list[str]:
    """Log lines telling the user which base-OTO rows were not generated."""

    count = int(payload.get("excluded_alias_rows", 0) or 0)
    if count <= 0:
        return []
    failed = list(payload.get("failed_recordings") or [])
    kept = int(payload.get("kept_base_rows", 0) or 0)
    lines = [
        t("[주의] 생성하지 못한 행 {count}개: 원본 OTO의 별칭을 녹음 파일 이름과 연결하지 못했거나 녹음을 분석하지 못해 "
          "기존 oto.ini 값을 그대로 넣었습니다 (oto.review.ini에도 포함).", count=count) if kept
        else t("[주의] 생성되지 않은 행 {count}개: 원본 OTO의 별칭을 녹음 파일 이름과 연결하지 못했거나 "
               "녹음을 분석하지 못해 oto.ini에서 뺐습니다.", count=count),
        *([t("  분석하지 못한 녹음 {count}개(무음·잘린 녹음일 수 있음): {names}", count=len(failed),
             names=", ".join(failed))] if failed else []),
        (t("  전체 목록: {path}", path=payload["excluded_alias_rows_csv"]) if payload.get("excluded_alias_rows_csv")
         else t("  전체 목록을 파일로 받으려면 고급 출력 옵션에서 unmapped_aliases.csv 저장을 켜세요.")),
    ]
    samples = payload.get("excluded_alias_samples") or []
    for sample in samples:
        lines.append(f"  - {sample.get('wav', '')} = {sample.get('alias', '')}")
    if count > len(samples):
        lines.append(t("  … 외 {count}개", count=count - len(samples)))
    return lines


def format_result_text(title: str, payload: Mapping[str, object]) -> str:
    """Format an operation result for the human-readable Result tab."""

    lines = [title]
    if "generated_rows" in payload:
        lines.extend(
            [
                f"WAV: {payload.get('wav_count', 0)}",
                t("생성 행: {count}", count=payload.get("generated_rows", 0)),
                t("자동 clean: {count}", count=payload.get("clean_rows", 0)),
                t("검수 보류: {count}", count=payload.get("review_rows", 0)),
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
            lines.extend(("", t("먼저 확인: {count}행", count=counts.get("inspect_first", 0)),
                t("연결 확인: {count}행", count=counts.get("connection_check", 0)),
                t("표본 확인: {count}행", count=counts.get("spot_check", 0)), t("검수 순서: {path}", path=usage["csv"])))
        sample = payload.get("style_sample")
        if isinstance(sample, Mapping) and sample.get("rows"):
            lines.extend(("", t("스타일 샘플: {count}행 ({path})", count=sample["rows"], path=sample.get("path")),
                          t("이 행들만 oto.ini에서 확인·수정한 뒤 고급 설정의 '샘플로 학습'을 누르고 다시 생성하면"
                            " 내 타이밍 습관이 반영됩니다.")))
        if isinstance(reasons, Mapping) and reasons:
            lines.extend(("", t("주요 검수 사유")))
            for reason, count in sorted(reasons.items(), key=lambda item: (-int(item[1]), str(item[0])))[:8]:
                lines.append(f"- {reason}: {count}")
    elif "changed_aliases" in payload:
        lines.append(t("접미사 {suffix}: 별칭 {changed}/{total}개에 붙였습니다 (이미 붙은 별칭은 그대로).",
                       suffix=payload.get("alias_suffix"), changed=payload.get("changed_aliases", 0),
                       total=payload.get("aliases", 0)))
    elif "valid" in payload:
        lines.append(t("구조 검증: {state}", state=t("통과") if payload.get("valid") else t("실패")))
        lines.append(t("OTO 행: {count}", count=payload.get("rows", 0)))
        lines.append(t("오류: {count}", count=len(payload.get("errors", []) or [])))
    elif "wav_count" in payload:
        lines.append(t("발견한 WAV: {count}", count=payload.get("wav_count", 0)))
        if "mapped_wavs" in payload:
            lines.append(t("기존 OTO 매핑: {count}", count=payload.get("mapped_wavs", 0)))
            lines.append(t("미매핑 WAV: {count}", count=payload.get("unmapped_wavs", 0)))
    lines.extend(("", t("상세"), json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str)))
    return "\n".join(line for line in lines if line is not None)


def launch_oto_generator_ui(
    *,
    initial_bank: Path | None = None,
    initial_output: Path | None = None,
) -> dict[str, object]:
    """Launch the AutoOTO product UI and return its last operation result."""

    load_ui_language()
    try:
        import customtkinter as ctk
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except ImportError as error:  # pragma: no cover - platform packaging issue
        raise RuntimeError(
            t("이 UI에는 Tkinter와 customtkinter가 필요합니다. 프로젝트를 다시 설치하세요.")
        ) from error

    ctk.set_default_color_theme("blue")
    ctk.set_appearance_mode("dark")

    class OtoGeneratorApp(ctk.CTk):
        def __init__(self) -> None:
            super().__init__()
            self.title(APP_NAME)
            self.geometry("1100x800")
            self.minsize(900, 680)
            self.configure(fg_color=PANEL_BG)
            apply_app_icon(self)
            self._events: queue.Queue[tuple[str, object]] = queue.Queue()
            self._busy = False
            self._cancel = threading.Event()
            self._started_at = 0.0
            self._advanced_visible = False
            self._log_lines: list[str] = []
            self.last_summary: dict[str, object] = {}
            self._preview_window = None
            self.bank_var = tk.StringVar(value=str(initial_bank or ""))
            self.base_oto_var = tk.StringVar()
            self.output_var = tk.StringVar(value=str(initial_output or ""))
            self.recursive_var = tk.BooleanVar(value=False)
            self.batch_var = tk.BooleanVar(value=False)
            self.folder_oto_var = tk.BooleanVar(value=True)
            self.bpm_var = tk.StringVar()
            self.style_profile_var = tk.StringVar()
            self.estimate_bpm_var = tk.BooleanVar(value=False)
            self.alias_suffix_var = tk.StringVar()
            self.vc_end_var = tk.StringVar(value=next(iter(VC_END_CHOICES)))
            self.release_pre_var = tk.StringVar(value=next(iter(RELEASE_PRE_CHOICES)))
            self.stable_window_var = tk.StringVar(value=next(iter(STABLE_WINDOW_CHOICES)))
            self.unmapped_row_var = tk.StringVar(value=next(iter(UNMAPPED_ROW_CHOICES)))
            self.style_vars = {name: tk.StringVar() for name, _, _ in STYLE_FIELDS}
            self.detection_var = tk.StringVar(value=t("형식 감지: 보이스뱅크 폴더를 고르면 표시됩니다."))
            self._detection: FormatDetection | None = None
            self._language_acknowledged: set[tuple[str, str]] = set()
            self.overwrite_oto_var = tk.BooleanVar(value=False)
            self.encoding_var = tk.StringVar(value="preserve")
            self.language_var = tk.StringVar(value="한국어")
            self.format_var = tk.StringVar(value=AUTO_FORMAT_CHOICE)
            self.diagnostic_vars = {name: tk.BooleanVar(value=False) for name in DIAGNOSTIC_FILE_LABELS}
            self.theme_var = tk.StringVar(value="다크 모드")
            self.status_var = tk.StringVar(value=t("대기 중"))
            self.ui_language_var = tk.StringVar(value=LANGUAGE_NAMES[get_language()])
            self.ui_language_hint_var = tk.StringVar()
            self._buttons: list[object] = []
            self._build_ui(ctk)
            self.after(100, self._poll_events)

        def _build_ui(self, ctk_module) -> None:
            notice = machine_translation_notice()
            if notice:  # English / Japanese: say the UI text is machine-translated
                ctk_module.CTkLabel(
                    self, text=f"⚠ {notice}", font=("", 11), text_color=HINT, anchor="w"
                ).pack(fill="x", padx=18, pady=(6, 0))
            header = ctk_module.CTkFrame(self, fg_color="transparent")
            header.pack(fill="x", padx=18, pady=(4 if notice else 14, 8))
            ctk_module.CTkLabel(
                header, text=APP_NAME, font=("", 22, "bold"), text_color=HEADER
            ).pack(side="left")
            ctk_module.CTkLabel(
                header, text=t("WAV를 분석해 새 oto.ini를 생성합니다."), text_color=HINT
            ).pack(side="left", padx=14)
            self._choice_menu(
                ctk_module, header, self.theme_var, ("다크 모드", "라이트 모드"), command=self._set_theme, width=120,
            ).pack(side="right")
            ctk_module.CTkOptionMenu(
                header,
                values=list(LANGUAGE_NAMES.values()),
                variable=self.ui_language_var,
                command=self._set_ui_language,
                width=100,
            ).pack(side="right", padx=(0, 8))
            ctk_module.CTkLabel(header, text=t("UI 언어"), text_color=HINT).pack(side="right", padx=(0, 6))
            ctk_module.CTkLabel(header, textvariable=self.ui_language_hint_var, text_color=HINT).pack(
                side="right", padx=(0, 10)
            )

            self.tabview = ctk_module.CTkTabview(
                self,
                fg_color=PANEL_BG,
                segmented_button_selected_color=PRIMARY,
                segmented_button_selected_hover_color=PRIMARY_HOVER,
            )
            self.tabview.pack(fill="both", expand=True, padx=16, pady=(0, 10))
            for name in ("생성", "결과", "로그", "크레딧"):
                self.tabview.add(t(name))
            self._build_generate_tab(ctk_module, self.tabview.tab(t("생성")))
            self._build_result_tab(ctk_module, self.tabview.tab(t("결과")))
            self._build_log_tab(ctk_module, self.tabview.tab(t("로그")))
            self._build_credits_tab(ctk_module, self.tabview.tab(t("크레딧")))

            bottom = ctk_module.CTkFrame(self, fg_color="transparent")
            bottom.pack(fill="x", padx=18, pady=(0, 14))
            self.progress = ctk_module.CTkProgressBar(bottom, mode="indeterminate", height=10)
            self.progress.set(0)
            self.progress.pack(side="left", fill="x", expand=True)
            self.cancel_button = ctk_module.CTkButton(bottom, text=t("취소"), width=70, state="disabled",
                                                      command=self._request_cancel)
            self.cancel_button.pack(side="right", padx=(8, 0))
            ctk_module.CTkLabel(bottom, textvariable=self.status_var, text_color=TEXT, width=360, anchor="e").pack(
                side="right", padx=(12, 0)
            )

        def _build_generate_tab(self, ctk_module, tab) -> None:
            self._build_action_bar(ctk_module, tab)  # packed first so the scroll area cannot squeeze it
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
                text=t("기존 OTO는 에일리어스·인코딩 참고용이며 timing을 정답으로 재사용하지 않습니다.\n"
                       "없으면 형식에 맞는 별칭(CV·VC·VCV)을 WAV 파일 이름에서 만듭니다."),
                text_color=HINT,
                anchor="w",
                justify="left",
            ).pack(fill="x", padx=132, pady=(0, 4))
            self._path_row(ctk_module, paths, "출력 폴더", self.output_var, self._choose_output)
            ctk_module.CTkCheckBox(
                paths,
                text=t("기존 oto.ini 덮어쓰기 허용 (보이스뱅크 폴더 직접 저장)"),
                variable=self.overwrite_oto_var,
            ).pack(anchor="w", padx=132, pady=(4, 2))
            ctk_module.CTkLabel(
                paths,
                text=t("켜면 기존 oto.ini는 타임스탬프 .bak으로 백업된 뒤 교체됩니다."),
                text_color=HINT,
                anchor="w",
            ).pack(fill="x", padx=132, pady=(0, 4))
            ctk_module.CTkCheckBox(
                paths, text=t("하위 폴더 포함 (한 번에 처리)"), variable=self.recursive_var,
                command=lambda: self._exclusive_folder_mode("recursive"),
            ).pack(anchor="w", padx=132, pady=(4, 2))
            ctk_module.CTkCheckBox(
                paths, text=t("하위 폴더별 일괄 생성 (WAV가 있는 폴더마다 oto.ini를 따로 만듦)"),
                variable=self.batch_var, command=lambda: self._exclusive_folder_mode("batch"),
            ).pack(anchor="w", padx=132, pady=2)
            ctk_module.CTkCheckBox(
                paths, text=t("일괄 생성 때 각 폴더의 oto.ini를 별칭 참고용으로 사용"), variable=self.folder_oto_var
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
                ctk_module, profile_row, "언어", self.language_var, tuple(LANGUAGE_CHOICES), translated=True
            )
            self.format_menu = self._option_menu(
                ctk_module, profile_row, "포맷", self.format_var, format_choices_for(self.language_var.get()),
                translated=True
            )
            self.language_var.trace_add("write", lambda *_: self._sync_format_choices())
            ctk_module.CTkLabel(profile, textvariable=self.detection_var, text_color=HINT, anchor="w").pack(
                fill="x", padx=12, pady=(2, 8)
            )
            tempo_row = ctk_module.CTkFrame(profile, fg_color="transparent")
            tempo_row.pack(fill="x", padx=12, pady=(0, 2))
            ctk_module.CTkLabel(tempo_row, text=t("녹음 BPM (선택, 실험적)"), anchor="w").pack(side="left")
            ctk_module.CTkEntry(tempo_row, textvariable=self.bpm_var, width=80, placeholder_text=t("예: 120")).pack(
                side="left", padx=(10, 16)
            )
            ctk_module.CTkCheckBox(
                tempo_row, text=t("BPM 자동 추정"), variable=self.estimate_bpm_var
            ).pack(side="left")
            ctk_module.CTkLabel(
                profile,
                text=t("BPM을 넣으면 음절 위치를 박자 격자 쪽으로 당깁니다. 비우면 사용하지 않습니다(기본).\n"
                       "뱅크에 따라 좋아지기도 나빠지기도 하니 결과를 들어 보고 쓰세요. 쉼 표기 파일과 박자가 안 맞는 파일은 자동 제외합니다.\n"
                       "자동 추정은 박자가 일정한 파일이 충분할 때만 값을 내며, 가이드 없이 녹음했다면 끄세요."),
                text_color=HINT,
                anchor="w",
                justify="left",
            ).pack(fill="x", padx=12, pady=(2, 8))
            suffix_row = ctk_module.CTkFrame(profile, fg_color="transparent")
            suffix_row.pack(fill="x", padx=12, pady=(0, 2))
            ctk_module.CTkLabel(suffix_row, text=t("별칭 접미사 (선택)"), anchor="w").pack(side="left")
            ctk_module.CTkEntry(suffix_row, textvariable=self.alias_suffix_var, width=110,
                                placeholder_text=t("예: C4")).pack(side="left", padx=(10, 10))
            suffix_button = ctk_module.CTkButton(
                suffix_row, text=t("생성된 oto에 적용"), width=140, command=self._apply_suffix,
                fg_color=PRIMARY, hover_color=PRIMARY_HOVER,
            )
            suffix_button.pack(side="left")
            self._buttons.append(suffix_button)
            ctk_module.CTkLabel(
                profile,
                text=t("모든 별칭 끝에 _C4 형태로 붙입니다. 생성할 때 적용되고, 이미 만든 oto.ini에는 버튼으로 붙일 수 있습니다.\n"
                       "이미 같은 접미사로 끝나는 별칭은 그대로 둡니다."),
                text_color=HINT,
                anchor="w",
                justify="left",
            ).pack(fill="x", padx=12, pady=(2, 12))

            self.advanced_toggle_btn = ctk_module.CTkButton(
                scroll,
                text=t("▶ 고급 설정"),
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
            self._option_menu(ctk_module, option_row, "VC 끝 위치", self.vc_end_var, tuple(VC_END_CHOICES),
                              translated=True)
            self._option_menu(
                ctk_module, option_row, "'-' 선행발성", self.release_pre_var, tuple(RELEASE_PRE_CHOICES),
                translated=True,
            )
            self._option_menu(
                ctk_module, option_row, "고정·늘림 구간", self.stable_window_var, tuple(STABLE_WINDOW_CHOICES),
                translated=True,
            )
            self._option_menu(
                ctk_module, option_row, "생성하지 못한 기존 oto 행", self.unmapped_row_var, tuple(UNMAPPED_ROW_CHOICES),
                translated=True,
            )
            style_grid = ctk_module.CTkFrame(self.advanced_frame, fg_color="transparent")
            style_grid.pack(fill="x", padx=12, pady=(0, 4))
            for index, (name, label, hint) in enumerate(STYLE_FIELDS):
                row, column = divmod(index, 2)
                ctk_module.CTkLabel(style_grid, text=t(label), anchor="w", width=130).grid(
                    row=row, column=column * 3, sticky="w", padx=(0, 6), pady=2
                )
                ctk_module.CTkEntry(style_grid, textvariable=self.style_vars[name], width=80,
                                    placeholder_text=t("자동")).grid(row=row, column=column * 3 + 1, pady=2)
                ctk_module.CTkLabel(style_grid, text=t(hint), text_color=HINT, anchor="w").grid(
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
                    preset_row, text=t(label), width=120, command=command,
                    fg_color=PRIMARY, hover_color=PRIMARY_HOVER,
                ).pack(side="left", padx=(0, 8))
            labeler_row = ctk_module.CTkFrame(self.advanced_frame, fg_color="transparent")
            labeler_row.pack(fill="x", padx=12, pady=(4, 2))
            ctk_module.CTkLabel(labeler_row, text=t("라벨러 스타일 (영어 등 언어팩)"), anchor="w").pack(side="left")
            ctk_module.CTkEntry(labeler_row, textvariable=self.style_profile_var, width=260,
                                placeholder_text=t("비우면 언어팩 기본")).pack(side="left", padx=(10, 8))
            for label, command in ((t("파일 선택"), self._choose_style_profile),
                                   (t("내 OTO로 학습"), self._fit_style_profile),
                                   (t("샘플로 학습"), self._fit_style_sample)):
                ctk_module.CTkButton(labeler_row, text=label, width=110, command=command,
                                     fg_color=PRIMARY, hover_color=PRIMARY_HOVER).pack(side="left", padx=(0, 8))
            ctk_module.CTkLabel(
                self.advanced_frame,
                text=t("내가 손으로 맞춘 oto.ini가 있는 폴더(하위 폴더 포함)에서 내 타이밍 습관을 배워 새 음원에 적용합니다.\n"
                       "샘플로 학습: 생성 후 출력 폴더의 style_sample.csv에 적힌 행(모양마다 5행)만 oto.ini에서 고치고 누르세요.\n"
                       "'none'을 넣으면 스타일 없이 템플릿만 씁니다."),
                text_color=HINT, anchor="w", justify="left",
            ).pack(fill="x", padx=12, pady=(0, 10))
            ctk_module.CTkLabel(
                self.advanced_frame,
                text=t("진단 파일 저장 (기본 꺼짐, 켜도 oto.ini 결과는 같습니다)"),
                text_color=HINT,
                anchor="w",
            ).pack(fill="x", padx=12, pady=(0, 2))
            for name, label in DIAGNOSTIC_FILE_LABELS.items():
                ctk_module.CTkCheckBox(
                    self.advanced_frame, text=t(label), variable=self.diagnostic_vars[name]
                ).pack(anchor="w", padx=24, pady=2)
            ctk_module.CTkFrame(self.advanced_frame, fg_color="transparent", height=10).pack()

        def _build_action_bar(self, ctk_module, tab) -> None:
            """Run buttons pinned under the scrolling settings, so they stay visible however far
            the settings (or the opened advanced section) extend. OTO 생성 is the main action."""

            bar = ctk_module.CTkFrame(tab, fg_color=PANEL_BG, border_width=1, border_color=PANEL_BORDER)
            bar.pack(side="bottom", fill="x", padx=18, pady=(0, 8))
            for label, command in (
                ("입력 검사", self._inspect),
                ("출력 검증", self._validate_output),
                ("출력 폴더 열기", self._open_output),
                ("OTO 미리보기", self._open_preview),
            ):
                button = ctk_module.CTkButton(
                    bar, text=t(label), width=110, height=32, command=command,
                    fg_color=SECONDARY, hover_color=SECONDARY_HOVER, text_color=SECONDARY_TEXT,
                )
                button.pack(side="left", padx=(10, 0), pady=10)
                self._buttons.append(button)
            generate = ctk_module.CTkButton(
                bar, text=t("OTO 생성"), width=200, height=44, font=("", 16, "bold"), command=self._generate,
                fg_color=GENERATE, hover_color=GENERATE_HOVER, text_color="#FFFFFF",
            )
            generate.pack(side="right", padx=10, pady=10)
            self._buttons.append(generate)

        def _build_result_tab(self, ctk_module, tab) -> None:
            toolbar = ctk_module.CTkFrame(tab, fg_color="transparent")
            toolbar.pack(fill="x", padx=8, pady=(8, 4))
            ctk_module.CTkLabel(
                toolbar,
                text=t("review는 자동 검수 보류이며 오류 확정이 아닙니다."),
                text_color=HINT,
            ).pack(side="left")
            ctk_module.CTkButton(
                toolbar,
                text=t("OTO 미리보기"),
                width=120,
                command=self._open_preview,
                fg_color=PRIMARY,
                hover_color=PRIMARY_HOVER,
            ).pack(side="right", padx=(8, 0))
            ctk_module.CTkButton(
                toolbar,
                text=t("생성 보고서 열기"),
                width=140,
                command=self._open_report,
                fg_color=PRIMARY,
                hover_color=PRIMARY_HOVER,
            ).pack(side="right")
            self.result_text = ctk_module.CTkTextbox(tab, wrap="word")
            self.result_text.pack(fill="both", expand=True, padx=8, pady=(4, 8))
            self.result_text.insert("1.0", t("아직 실행 결과가 없습니다."))
            self.result_text.configure(state="disabled")

        def _build_log_tab(self, ctk_module, tab) -> None:
            toolbar = ctk_module.CTkFrame(tab, fg_color="transparent")
            toolbar.pack(fill="x", padx=8, pady=(8, 4))
            ctk_module.CTkButton(
                toolbar,
                text=t("로그 복사"),
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
                text=t("보정 모델 데이터 제공에 도움 주신 분들"),
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
                text=t("감사합니다!"),
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
                text=t(title),
                font=("", 14, "bold"),
                text_color=HEADER,
                anchor="w",
            ).pack(fill="x", padx=12, pady=(10, 2))
            ctk_module.CTkLabel(
                frame,
                text=t(subtitle),
                text_color=HINT,
                anchor="w",
                justify="left",
            ).pack(fill="x", padx=12, pady=(0, 8))
            return frame

        def _path_row(self, ctk_module, parent, label: str, variable, command) -> None:
            row = ctk_module.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=4)
            ctk_module.CTkLabel(row, text=t(label), width=110, anchor="w").pack(side="left")
            ctk_module.CTkEntry(row, textvariable=variable).pack(
                side="left", fill="x", expand=True, padx=(10, 8)
            )
            ctk_module.CTkButton(
                row,
                text=t("찾아보기"),
                width=90,
                command=command,
                fg_color=PRIMARY,
                hover_color=PRIMARY_HOVER,
            ).pack(side="right")

        def _option_menu(self, ctk_module, parent, label: str, variable, values, *, translated: bool = False):
            group = ctk_module.CTkFrame(parent, fg_color="transparent")
            group.pack(side="left", fill="x", expand=True, padx=(0, 10))
            ctk_module.CTkLabel(group, text=t(label), anchor="w").pack(anchor="w")
            menu = (self._choice_menu(ctk_module, group, variable, values) if translated
                    else ctk_module.CTkOptionMenu(group, variable=variable, values=list(values)))
            menu.pack(fill="x", pady=(2, 0))
            return menu

        def _choice_menu(self, ctk_module, parent, variable, keys, *, command=None, **kwargs):
            """Option menu showing translated labels while `variable` keeps the Korean key."""

            by_label = {t(key): key for key in keys}
            shown = tk.StringVar(value=t(variable.get()))
            variable.trace_add("write", lambda *_: shown.set(t(variable.get())))

            def chosen(label: str) -> None:
                variable.set(by_label.get(label, label))
                if command is not None:
                    command(variable.get())

            menu = ctk_module.CTkOptionMenu(parent, variable=shown, values=list(by_label), command=chosen, **kwargs)

            def set_keys(new_keys) -> None:
                by_label.clear()
                by_label.update({t(key): key for key in new_keys})
                menu.configure(values=list(by_label))

            menu.set_choice_keys = set_keys
            return menu

        def _sync_format_choices(self) -> None:
            """Show only the formats of the chosen language; a hidden choice falls back to detection."""

            keys = format_choices_for(self.language_var.get())
            self.format_menu.set_choice_keys(keys)
            if self.format_var.get() not in keys:
                self.format_var.set(AUTO_FORMAT_CHOICE)

        def _set_ui_language(self, choice: str) -> None:
            language = next(code for code, name in LANGUAGE_NAMES.items() if name == choice)
            try:
                save_ui_language(language)
            except OSError as error:
                messagebox.showerror(t("오류"), str(error), parent=self)
                return
            self.ui_language_hint_var.set(
                "" if language == get_language() else translate("재시작 후 적용됩니다.", language)
            )

        def _toggle_advanced(self) -> None:
            self._advanced_visible = not self._advanced_visible
            if self._advanced_visible:
                self.advanced_frame.pack(fill="x", padx=10, pady=6)
                self.advanced_toggle_btn.configure(text=t("▼ 고급 설정"))
            else:
                self.advanced_frame.pack_forget()
                self.advanced_toggle_btn.configure(text=t("▶ 고급 설정"))

        def _set_theme(self, choice: str) -> None:
            ctk.set_appearance_mode("light" if choice == "라이트 모드" else "dark")

        def _choose_bank(self) -> None:
            selected = filedialog.askdirectory(title=t("WAV 보이스뱅크 폴더 선택"))
            if selected:
                self.bank_var.set(selected)
                if not self.output_var.get().strip():
                    bank = Path(selected)
                    self.output_var.set(str(bank.parent / f"{bank.name}_auto_oto"))
                self._refresh_detection()

        def _choose_base_oto(self) -> None:
            selected = filedialog.askopenfilename(
                title=t("기존 oto.ini 선택"),
                filetypes=(("OTO files", "*.ini"), ("All files", "*.*")),
            )
            if selected:
                self.base_oto_var.set(selected)
                self._refresh_detection()

        def _choose_output(self) -> None:
            selected = filedialog.askdirectory(title=t("출력 폴더 선택"), mustexist=False)
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
            self.tabview.set(t("결과"))

        def _set_busy(self, busy: bool) -> None:
            self._busy = busy
            for button in self._buttons:
                button.configure(state="disabled" if busy else "normal")
            self.cancel_button.configure(state="normal" if busy else "disabled")
            if busy:
                self._cancel.clear()
                self._started_at = time.monotonic()
                self.progress.configure(mode="indeterminate")
                self.progress.start()
            else:
                self.progress.stop()
                self.progress.configure(mode="determinate")
                self.progress.set(0)

        def _request_cancel(self) -> None:
            if self._busy and not self._cancel.is_set():
                self._cancel.set()
                self.cancel_button.configure(state="disabled")
                self.status_var.set(t("취소하는 중... (진행 중인 파일을 마치고 멈춥니다)"))
                self._append_log(t("[취소 요청]"))

        def _progress_callback(self, prefix: str = "") -> Callable[[str, int, int], None]:
            """Runs in the worker thread: forwards progress and stops the run once 취소 was pressed."""
            events, cancel = self._events, self._cancel

            def progress(stage: str, done: int, total: int) -> None:
                if cancel.is_set():
                    raise OperationCancelled()
                events.put(("progress", (prefix, stage, done, total)))
            return progress

        def _run(self, title: str, operation: Callable[[], Mapping[str, object]]) -> None:
            if self._busy:
                return
            self._set_busy(True)
            self.status_var.set(t("{title} 실행 중...", title=title))
            self._append_log(t("[시작] {title}", title=title))

            def worker() -> None:
                try:
                    self._events.put(("success", UIOperationResult(title, dict(operation()))))
                except OperationCancelled:
                    self._events.put(("cancelled", title))
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
                title=t("고급 설정 프리셋 불러오기"), filetypes=(("JSON", "*.json"), ("All files", "*.*"))
            )
            if not selected:
                return
            try:
                style = load_oto_style(Path(selected))
            except Exception as error:
                messagebox.showerror(t("프리셋 오류"), str(error), parent=self)
                return
            for name, value in style_entry_values(style).items():
                self.style_vars[name].set(value)
            self._append_log(t("[고급 설정] 프리셋을 불러왔습니다: {path}", path=selected))

        def _save_style_preset(self) -> None:
            try:
                style = self._style()
            except ValueError as error:
                messagebox.showerror(t("입력 오류"), str(error), parent=self)
                return
            selected = filedialog.asksaveasfilename(
                title=t("고급 설정 프리셋 저장"), defaultextension=".json", filetypes=(("JSON", "*.json"),)
            )
            if selected:
                save_oto_style(Path(selected), style)
                self._append_log(t("[고급 설정] 프리셋을 저장했습니다: {path}", path=selected))

        def _choose_style_profile(self) -> None:
            selected = filedialog.askopenfilename(
                title=t("스타일 프로필 선택"), filetypes=(("YAML", "*.yaml *.yml"), ("All files", "*.*"))
            )
            if selected:
                self.style_profile_var.set(selected)

        def _fit_style_profile(self) -> None:
            root = Path(self.bank_var.get().strip()).expanduser()
            language = LANGUAGE_CHOICES.get(self.language_var.get())
            format_hint = FORMAT_CHOICES.get(self.format_var.get()) or self._detected_format_for_generation()
            if not root.is_dir() or not format_hint:
                messagebox.showerror(t("입력 오류"), t("보이스뱅크 폴더와 포맷을 먼저 정하세요."), parent=self)
                return
            banks = [folder for folder in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]
                     if (folder / "oto.ini").is_file() and (folder / "oto.ini").stat().st_size and any(folder.glob("*.wav"))]
            if not banks:
                messagebox.showerror(t("입력 오류"), t("손으로 맞춘 oto.ini와 WAV가 같이 있는 폴더가 없습니다."), parent=self)
                return
            from .style_fitting import template_like_reference

            template = template_like_reference(banks)
            if template is not None and not messagebox.askokcancel(
                t("템플릿 OTO 확인"),
                t("oto.ini {rows}행의 offset이 {count}가지뿐입니다 (예: {examples} ms).\n\n"
                  "음원에 맞추지 않은 템플릿 OTO라면 잘못된 스타일을 배웁니다."
                  " 직접 맞춘 OTO가 맞는지 확인하세요.\n\n그래도 학습할까요?",
                  rows=template["rows"], count=template["distinct_offsets"],
                  examples=", ".join(f"{value:g}" for value in template["most_common_offsets"])),
                icon="warning", parent=self):
                return
            target = filedialog.asksaveasfilename(
                title=t("스타일 프로필 저장"), defaultextension=".yaml", filetypes=(("YAML", "*.yaml"),)
            )
            if not target:
                return

            def operation() -> Mapping[str, object]:
                from .style_fitting import fit_style_from_banks

                return fit_style_from_banks(banks, output=Path(target), format_hint=format_hint,
                                            language_hint=language or "")

            # The next generation uses the profile; until the fit finishes the file does not exist yet.
            self.style_profile_var.set(target)
            self._append_log(t("[스타일 학습] oto.ini가 있는 폴더 {count}개 -> {path}", count=len(banks), path=target))
            self._run(t("스타일 학습"), operation)

        def _fit_style_sample(self) -> None:
            from .style_sample import find_style_samples, fit_style_from_sample

            output = Path(self.output_var.get().strip()).expanduser()
            samples = find_style_samples(output) if output.is_dir() else []
            if not samples:
                messagebox.showerror(
                    t("입력 오류"), t("출력 폴더에 style_sample.csv가 없습니다. 언어팩 형식(영어 VCCV)으로 먼저 생성하세요."),
                    parent=self)
                return
            in_place = messagebox.askyesnocancel(
                t("샘플로 학습"),
                t("style_sample.csv {count}개에 적힌 행을 확인·수정했나요?\n\n"
                  "예: 출력 폴더의 oto.ini를 직접 고쳤습니다.\n"
                  "아니오: 결과를 보이스뱅크 폴더로 옮겨 거기서 고쳤습니다(폴더를 고릅니다).", count=len(samples)),
                parent=self)
            if in_place is None:
                return
            edited: Path | None = None
            if not in_place:
                chosen = filedialog.askdirectory(title=t("고친 oto.ini가 있는 보이스뱅크 폴더"))
                if not chosen:
                    return
                edited = Path(chosen)
            target = filedialog.asksaveasfilename(
                title=t("스타일 프로필 저장"), defaultextension=".yaml", filetypes=(("YAML", "*.yaml"),))
            if not target:
                return

            def operation() -> Mapping[str, object]:
                return fit_style_from_sample(samples, Path(target), edited=edited, root=output)

            self.style_profile_var.set(target)
            self._append_log(t("[샘플 스타일 학습] 샘플 {count}개, 수정본 {edited} -> {path} (학습 후 다시 생성하세요)",
                               count=len(samples), edited=edited or output, path=target))
            self._run(t("샘플 스타일 학습"), operation)

        def _reset_advanced(self) -> None:
            for var in self.style_vars.values():
                var.set("")
            self.encoding_var.set("preserve")
            self.vc_end_var.set(next(iter(VC_END_CHOICES)))
            self.release_pre_var.set(next(iter(RELEASE_PRE_CHOICES)))
            self.stable_window_var.set(next(iter(STABLE_WINDOW_CHOICES)))
            self.style_profile_var.set("")
            self.unmapped_row_var.set(next(iter(UNMAPPED_ROW_CHOICES)))

        def _refresh_detection(self) -> FormatDetection | None:
            bank = self.bank_var.get().strip()
            if not bank or not Path(bank).expanduser().is_dir():
                return None
            try:
                if self.batch_var.get():
                    self.detection_var.set(t("형식 감지: 일괄 생성은 폴더마다 따로 감지합니다."))
                    return None
                detection = detect_bank_format(
                    bank, self.base_oto_var.get(), self.language_var.get(), recursive=self.recursive_var.get()
                )
            except Exception as error:
                self.detection_var.set(t("형식 감지 실패: {error}", error=error))
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
            self._append_log(t("[언어 확인] {warning}", warning=warning))
            if not messagebox.askyesno(t("언어 확인"), t("{warning}\n\n이 언어로 계속할까요?", warning=warning),
                                       parent=self):
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
                style_profile=self.style_profile_var.get(),
                alias_suffix=self.alias_suffix_var.get(),
                unmapped_row_choice=self.unmapped_row_var.get(),
                stable_window_choice=self.stable_window_var.get(),
            )

        def _read_options(self, *, force: bool = False, detected_format: str | None = None) -> GenerationOptions | None:
            try:
                return self._generation_options(force=force, detected_format=detected_format)
            except Exception as error:
                messagebox.showerror(t("입력 오류"), str(error), parent=self)
                self._append_log(t("[입력 오류] {error}", error=f"{type(error).__name__}: {error}"))
                return None

        def _inspect(self) -> None:
            bank = self.bank_var.get().strip()
            if not bank:
                messagebox.showerror(t("입력 오류"), t("WAV 보이스뱅크 폴더를 지정하세요."), parent=self)
                return
            if self.batch_var.get():
                language = LANGUAGE_CHOICES[self.language_var.get()]
                use_folder_oto = self.folder_oto_var.get()

                def batch_inspect() -> Mapping[str, object]:
                    folders = discover_bank_folders(Path(bank).expanduser(), language=language, use_folder_oto=use_folder_oto)
                    return {"batch_folders": [folder.to_dict() for folder in folders], "folder_count": len(folders)}

                self._run(t("입력 검사 (일괄)"), batch_inspect)
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

            self._run(t("입력 검사"), operation)

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
                    t("직접 저장 차단"),
                    t("출력 폴더가 보이스뱅크와 같습니다. 기존 oto.ini 덮어쓰기 허용 토글을 켜세요."),
                    parent=self,
                )
                return
            output_has_files = options.output.exists() and any(options.output.iterdir())
            if output_has_files and not options.allow_in_place_output:
                confirmed = messagebox.askyesno(
                    t("출력 폴더 확인"),
                    t("출력 폴더가 비어 있지 않습니다. 기존 생성 산출물은 백업 후 교체될 수 있습니다. 계속할까요?"),
                    parent=self,
                )
                if not confirmed:
                    return
                options = self._read_options(force=True, detected_format=detected)
                if options is None:
                    return

            options = replace(options, progress=self._progress_callback())

            def operation() -> Mapping[str, object]:
                report = generate_voicebank(options)
                return summarize_generation_result(report, output=options.output)

            self._run(t("OTO 생성"), operation)

        def _generate_batch(self) -> None:
            root = Path(self.bank_var.get().strip()).expanduser()
            output_root = Path(self.output_var.get().strip()).expanduser() if self.output_var.get().strip() else None
            in_place = self.overwrite_oto_var.get()
            if output_root is None and not in_place:
                messagebox.showerror(t("입력 오류"), t("출력 폴더를 지정하세요."), parent=self)
                return
            if output_root is None:
                self.output_var.set(str(root))
            try:
                template = self._generation_options(force=True, detected_format="cvvc")
                folders = discover_bank_folders(
                    root, language=LANGUAGE_CHOICES[self.language_var.get()], use_folder_oto=self.folder_oto_var.get()
                )
            except Exception as error:
                messagebox.showerror(t("입력 오류"), str(error), parent=self)
                return
            if not folders:
                messagebox.showerror(t("입력 오류"), t("WAV가 있는 폴더를 찾지 못했습니다."), parent=self)
                return
            selected = self._choose_batch_folders(folders)
            if not selected:
                return
            fixed_format = FORMAT_CHOICES[self.format_var.get()]
            use_folder_oto = self.folder_oto_var.get()
            events = self._events
            position = {"index": 0, "total": len(selected)}

            def generate_one(folder: BankFolder, output: Path) -> Mapping[str, object]:
                format_hint = fixed_format or folder.format
                if not format_hint:
                    raise ValueError(t("형식을 알 수 없습니다. 포맷을 직접 고르고 다시 실행하세요."))
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
                    progress=self._progress_callback(f"[{position['index']}/{position['total']}] "),
                )
                report = generate_voicebank(options)
                return summarize_generation_result(report, output=output)

            def progress(index: int, total: int, folder: BankFolder) -> None:
                position["index"] = index
                events.put(("log", t("[일괄 {index}/{total}] {folder} ({format})", index=index, total=total,
                                     folder=folder.relative, format=(fixed_format or folder.format or "?").upper())))

            def operation() -> Mapping[str, object]:
                return run_batch(
                    selected, root=root, output_root=output_root or root, in_place=in_place,
                    generate_one=generate_one, progress=progress,
                )

            self._run(t("일괄 OTO 생성"), operation)

        def _choose_batch_folders(self, folders: list[BankFolder]) -> list[BankFolder]:
            dialog = ctk.CTkToplevel(self)
            apply_app_icon(dialog, delay_ms=CTK_TOPLEVEL_ICON_DELAY_MS)
            dialog.title(t("일괄 생성할 폴더 선택"))
            dialog.geometry("820x520")
            dialog.transient(self)
            dialog.grab_set()
            ctk.CTkLabel(
                dialog,
                text=t("폴더마다 따로 oto.ini를 만듭니다. 형식을 알 수 없는 폴더는 기본으로 꺼져 있습니다."),
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
                note = t("직접 선택") if fixed else format_detection_detail(folder.detection)
                text = (f"{folder.relative}   WAV {folder.wav_count}   "
                        f"{t('oto.ini 있음') if folder.has_oto else t('oto.ini 없음')}   → {fmt}   ({note})")
                ctk.CTkCheckBox(frame, text=text, variable=var).pack(anchor="w", pady=2)
            chosen: list[BankFolder] = []

            def accept() -> None:
                chosen.extend(folder for folder, var in zip(folders, variables) if var.get())
                dialog.destroy()

            buttons = ctk.CTkFrame(dialog, fg_color="transparent")
            buttons.pack(fill="x", padx=12, pady=(6, 12))
            ctk.CTkButton(buttons, text=t("취소"), width=100, command=dialog.destroy).pack(side="right")
            ctk.CTkButton(buttons, text=t("생성"), width=100, command=accept,
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
                    raise FileNotFoundError(t("생성된 oto.ini가 없습니다: {path}", path=oto_path))
                return validate_oto(oto_path, options.bank)

            self._run(t("출력 검증"), operation)

        def _apply_suffix(self) -> None:
            output = self.output_var.get().strip()
            oto = Path(str(self.last_summary.get("oto"))) if self.last_summary.get("oto") else None
            if oto is None or not oto.is_file():
                oto = Path(output).expanduser() / "oto.ini" if output else None
            if oto is None or not oto.is_file():
                messagebox.showwarning(t("출력 폴더 없음"), t("먼저 OTO를 생성하세요."), parent=self)
                return
            suffix = self.alias_suffix_var.get()
            try:
                parse_alias_suffix(suffix)
            except ValueError as error:
                messagebox.showerror(t("입력 오류"), str(error), parent=self)
                return
            self._run(t("접미사 적용"), lambda: apply_suffix_to_output(oto, suffix))

        def _open_output(self) -> None:
            output = Path(self.output_var.get().strip()).expanduser()
            if not output.exists():
                messagebox.showwarning(t("출력 폴더 없음"), t("먼저 OTO를 생성하세요."), parent=self)
                return
            os.startfile(str(output))

        def _open_preview(self) -> None:
            bank = self.bank_var.get().strip()
            output = self.output_var.get().strip()
            oto = preview_oto_path(
                last_oto=self.last_summary.get("oto"),
                output=Path(output).expanduser() if output else None,
                bank=Path(bank).expanduser() if bank else None,
            )
            if oto is None:
                chosen = filedialog.askopenfilename(
                    parent=self, title=t("미리볼 oto.ini 선택"), filetypes=[("oto.ini", "*.ini"), (t("모든 파일"), "*.*")]
                )
                if not chosen:
                    return
                oto = Path(chosen)
            wav_dirs = [Path(bank).expanduser()] if bank else []
            try:
                window = self._preview_window
                if window is not None and window.winfo_exists():
                    window.load_oto(oto, wav_dirs)  # reuse: show the latest result in the open window
                    window.deiconify()
                    window.lift()
                    window.focus_force()
                else:
                    self._preview_window = launch_oto_preview(oto, wav_dirs, master=self)
            except Exception as error:
                messagebox.showerror(t("미리보기 실패"), f"{type(error).__name__}: {error}", parent=self)

        def _open_report(self) -> None:
            report = Path(self.output_var.get().strip()).expanduser() / "generation_report.json"
            if self.last_summary.get("oto") and self.last_summary.get("report_path") is None:
                messagebox.showwarning(
                    t("보고서 없음"),
                    t("생성 보고서 저장이 꺼져 있었습니다. 고급 출력 옵션에서 켜고 다시 생성하세요."),
                    parent=self,
                )
                return
            if not report.is_file():
                messagebox.showwarning(t("보고서 없음"), t("먼저 OTO를 생성하세요."), parent=self)
                return
            os.startfile(str(report))

        def _copy_log(self) -> None:
            text = "\n".join(self._log_lines)
            self.clipboard_clear()
            self.clipboard_append(text)
            self.status_var.set(t("로그 복사 완료"))

        def _poll_events(self) -> None:
            try:
                while True:
                    kind, value = self._events.get_nowait()
                    if kind == "progress":
                        prefix, stage, done, total = value
                        fraction = progress_fraction(stage, done, total)
                        if str(self.progress.cget("mode")) != "determinate":
                            self.progress.stop()
                            self.progress.configure(mode="determinate")
                        self.progress.set(fraction)
                        if not self._cancel.is_set():
                            elapsed = time.monotonic() - self._started_at
                            self.status_var.set(progress_text(stage, done, total, elapsed, fraction, prefix))
                        continue
                    if kind == "cancelled":
                        self._set_busy(False)
                        self.status_var.set(t("{title} 취소됨", title=value))
                        self._append_log(t("[취소] {title}: 사용자가 중단했습니다. 출력 폴더의 이전 결과는 그대로입니다.",
                                           title=value))
                        continue
                    if kind == "log":
                        self._append_log(str(value))
                        self.status_var.set(str(value)[:60])
                        continue
                    self._set_busy(False)
                    if kind == "success":
                        result = value
                        assert isinstance(result, UIOperationResult)
                        self.last_summary = dict(result.payload)
                        self.status_var.set(t("{title} 완료", title=result.title))
                        self._append_log(
                            json.dumps(dict(result.payload), ensure_ascii=False, indent=2, default=str)
                        )
                        for line in (*tempo_log_lines(result.payload), *alias_generation_log_lines(result.payload),
                                     *encoding_log_lines(result.payload)):
                            self._append_log(line)
                        for item in result.payload.get("results", []) or []:
                            state = t("완료") if item.get("ok") else t("실패: {error}", error=item.get("error"))
                            self._append_log(t("[일괄] {folder}: {state}", folder=item.get("folder"), state=state))
                        excluded = excluded_rows_log_lines(result.payload)
                        for line in excluded:
                            self._append_log(line)
                        if excluded:
                            self.status_var.set(
                                t("{title} 완료 (생성되지 않은 행 {count}개, 로그 참고)", title=result.title,
                                  count=result.payload.get("excluded_alias_rows")))
                        self._show_result(result.title, result.payload)
                    else:
                        self.status_var.set(t("오류"))
                        self._append_log(t("[오류] {error}", error=value))
                        self.tabview.set(t("로그"))
                        messagebox.showerror(t("작업 실패"), str(value), parent=self)
            except queue.Empty:
                pass
            self.after(100, self._poll_events)

    app = OtoGeneratorApp()
    app.mainloop()
    return app.last_summary


def main(argv: list[str] | None = None) -> int:
    import multiprocessing

    multiprocessing.freeze_support()  # parallel alignment workers in a frozen (exe) build
    import argparse

    parser = argparse.ArgumentParser(description=f"{APP_NAME} desktop UI")
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
