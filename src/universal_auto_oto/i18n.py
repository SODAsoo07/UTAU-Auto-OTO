"""UI language (Korean / English / Japanese) for the product GUI.

Keys are the Korean source strings, so Korean needs no table and a missing translation
falls back to Korean. Templates use ``str.format`` fields: ``t("WAV {count}개", count=3)``.
The English and Japanese tables are machine-translated; the GUI says so at the top.

The language is read once at startup (AUTOOTO_UI_LANGUAGE, else the settings file) and
a change in the GUI applies after a restart, as in the previous AutoOTO.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

DEFAULT_LANGUAGE = "ko"
LANGUAGE_NAMES: Mapping[str, str] = {"ko": "한국어", "en": "English", "ja": "日本語"}
LANGUAGE_ENV = "AUTOOTO_UI_LANGUAGE"
SETTINGS_FILE_NAME = "settings.json"

_language = DEFAULT_LANGUAGE


def normalize_language(value: object) -> str:
    text = str(value or "").strip().casefold()
    if text in {"en", "english"}:
        return "en"
    if text in {"ja", "jp", "japanese", "日本語"}:
        return "ja"
    return DEFAULT_LANGUAGE


def set_language(language: object) -> str:
    global _language
    _language = normalize_language(language)
    return _language


def get_language() -> str:
    return _language


def translate(key: str, language: str, **kwargs: object) -> str:
    table = _TABLES.get(language)
    text = table.get(key, key) if table else key
    return text.format(**kwargs) if kwargs else text


def t(key: str, **kwargs: object) -> str:
    """`key` (Korean) in the active UI language."""

    return translate(key, _language, **kwargs)


def machine_translation_notice(language: str | None = None) -> str:
    """Small notice shown at the top of the GUI for the machine-translated languages."""

    return _MACHINE_TRANSLATION_NOTICES.get(language or _language, "")


def settings_path() -> Path:
    base = os.environ.get("APPDATA") or os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "AutoOTO" / SETTINGS_FILE_NAME


def _read_settings(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_ui_language(path: Path | None = None) -> str:
    """Set and return the saved UI language (the environment variable wins)."""

    configured = os.environ.get(LANGUAGE_ENV) or _read_settings(path or settings_path()).get("ui_language")
    return set_language(configured)


def save_ui_language(language: str, path: Path | None = None) -> None:
    """Remember `language` for the next start; it does not change the running UI."""

    path = path or settings_path()
    settings = _read_settings(path)
    settings["ui_language"] = normalize_language(language)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


_MACHINE_TRANSLATION_NOTICES = {
    "en": "This English UI was machine-translated and may contain errors.",
    "ja": "この日本語UIは機械翻訳のため、不自然な表現や誤りが含まれる場合があります。",
}


_EN: dict[str, str] = {
    # Language selector
    "UI 언어": "UI language",
    "재시작 후 적용됩니다.": "Applies after restart.",
    # Header, tabs, bottom bar
    "WAV를 분석해 새 oto.ini를 생성합니다.": "Analyzes WAVs and generates a new oto.ini.",
    "다크 모드": "Dark mode",
    "라이트 모드": "Light mode",
    "생성": "Generate",
    "결과": "Result",
    "로그": "Log",
    "크레딧": "Credits",
    "취소": "Cancel",
    "대기 중": "Idle",
    # Choices
    "한국어": "Korean",
    "일본어": "Japanese",
    "자동 감지": "Auto-detect",
    "다음 자음 안에서 끝냄 (권장)": "End inside the next consonant (recommended)",
    "템플릿 기본값": "Template default",
    "모음 끝 (권장)": "Vowel end (recommended)",
    "높음": "high",
    "보통": "medium",
    "낮음": "low",
    "없음": "none",
    "기존 oto.ini 별칭": "existing oto.ini aliases",
    "기존 oto.ini + 폴더 이름": "existing oto.ini + folder name",
    "폴더 이름": "folder name",
    "WAV 파일 이름": "WAV file names",
    "다른 폴더": "other folders",
    "근거 없음": "no evidence",
    # Advanced style fields
    "CV 오버랩 비율": "CV overlap ratio",
    "0~1, 오버랩 = 비율 × 선행발성": "0-1, overlap = ratio × preutterance",
    "VC 오버랩 비율": "VC overlap ratio",
    "VCV 오버랩 비율": "VCV overlap ratio",
    "CV 선행발성 이동": "CV preutterance shift",
    "ms, + = 늦게": "ms, + = later",
    "VC 선행발성 이동": "VC preutterance shift",
    "VCV 선행발성 이동": "VCV preutterance shift",
    "고정 범위 연장": "Fixed range extension",
    "ms, 0 이상": "ms, 0 or more",
    "오른쪽 블랭크 여유": "Right blank margin",
    # Diagnostic files
    "unmapped_aliases.csv (생성되지 않은 행 목록)": "unmapped_aliases.csv (rows that were not generated)",
    "generation_report.json (생성 보고서)": "generation_report.json (generation report)",
    "row_report.jsonl (행별 근거)": "row_report.jsonl (per-row evidence)",
    "timelines/ (WAV별 음소 경계)": "timelines/ (phoneme boundaries per WAV)",
    "project_manifest.json (입력 해석 결과)": "project_manifest.json (how the input was read)",
    # Option validation
    "WAV 보이스뱅크 폴더를 지정하세요.": "Choose the WAV voicebank folder.",
    "출력 폴더를 지정하세요.": "Choose the output folder.",
    "검수 임계값은 0.0~1.0 범위여야 합니다.": "The review threshold must be between 0.0 and 1.0.",
    "최대 WAV 수는 1 이상이어야 합니다.": "The maximum number of WAVs must be 1 or more.",
    "알 수 없는 진단 파일 옵션입니다: {names}": "Unknown diagnostic file option: {names}",
    "지원하지 않는 언어 선택입니다: {choice}": "Unsupported language choice: {choice}",
    "지원하지 않는 형식 선택입니다: {choice}": "Unsupported format choice: {choice}",
    "형식을 자동으로 알아내지 못했습니다. 포맷을 직접 선택하세요.": (
        "Could not detect the format automatically. Choose the format yourself."),
    "지원하지 않는 출력 인코딩입니다: {encoding}": "Unsupported output encoding: {encoding}",
    "BPM은 숫자로 입력하세요 (예: 120).": "Enter the BPM as a number (e.g. 120).",
    "BPM은 60~240 범위로 입력하세요 (보통 100~160).": "Enter a BPM between 60 and 240 (usually 100-160).",
    "{label}: 숫자를 입력하세요.": "{label}: enter a number.",
    "고급 설정 값이 범위를 벗어났습니다: {error}": "An advanced setting is out of range: {error}",
    # Format detection
    "형식 감지: {detail}": "Format detection: {detail}",
    "알 수 없음 - 포맷을 직접 선택하세요.": "unknown - choose the format yourself.",
    "{format} (확신 {confidence}, 근거: {source})": "{format} (confidence {confidence}, based on: {source})",
    "형식 감지: 보이스뱅크 폴더를 고르면 표시됩니다.": "Format detection: shown once you choose a voicebank folder.",
    "형식 감지: 일괄 생성은 폴더마다 따로 감지합니다.": "Format detection: batch generation detects each folder separately.",
    "형식 감지 실패: {error}": "Format detection failed: {error}",
    # Log lines
    "[BPM] 입력한 {bpm:g} BPM 사용 (음절 간격 {period:.0f} ms)": (
        "[BPM] Using the entered {bpm:g} BPM (syllable interval {period:.0f} ms)"),
    "[BPM] 자동 추정 {bpm} BPM 사용 ({reason})": "[BPM] Using the estimated {bpm} BPM ({reason})",
    "[BPM] 추정 안 함: {reason} BPM 없이 생성했습니다.": "[BPM] Not estimated: {reason} Generated without BPM.",
    "[별칭 생성] 기존 oto.ini 없이 WAV {count}개의 파일 이름에서 형식에 맞는 별칭을 만들었습니다.": (
        "[Alias generation] No existing oto.ini: made aliases for the format from the names of {count} WAVs."),
    "[주의] 이름을 읽지 못한 WAV {count}개는 빠졌습니다.": "[Note] {count} WAVs whose names could not be read were left out.",
    "review 행은 자동 검수 보류이며 오류 확정이 아닙니다.": (
        "Review rows are held for checking, not confirmed errors."),
    "정렬 {step}단계 {done}/{total}": "Alignment pass {step} {done}/{total}",
    "OTO 작성 {done}/{total}": "Writing OTO {done}/{total}",
    " · 남은 시간 약 {minutes}분": " · about {minutes} min left",
    " · 남은 시간 약 {seconds}초": " · about {seconds} s left",
    "[주의] 생성되지 않은 행 {count}개: 원본 OTO의 별칭을 녹음 파일 이름과 연결하지 못했거나 "
    "녹음을 분석하지 못해 oto.ini에서 뺐습니다.": (
        "[Note] {count} rows were not generated: their aliases in the source OTO could not be matched to a "
        "recording file name, or the recording could not be analyzed, so they were left out of oto.ini."),
    "  분석하지 못한 녹음 {count}개(무음·잘린 녹음일 수 있음): {names}": (
        "  {count} recordings could not be analyzed (they may be silent or cut off): {names}"),
    "  전체 목록: {path}": "  Full list: {path}",
    "  전체 목록을 파일로 받으려면 고급 출력 옵션에서 unmapped_aliases.csv 저장을 켜세요.": (
        "  To get the full list as a file, turn on unmapped_aliases.csv in the advanced output options."),
    "  … 외 {count}개": "  … and {count} more",
    # Result text
    "생성 행: {count}": "Generated rows: {count}",
    "자동 clean: {count}": "Auto clean: {count}",
    "검수 보류: {count}": "Held for review: {count}",
    "먼저 확인: {count}행": "Check first: {count} rows",
    "연결 확인: {count}행": "Check connections: {count} rows",
    "표본 확인: {count}행": "Spot check: {count} rows",
    "검수 순서: {path}": "Review order: {path}",
    "주요 검수 사유": "Main review reasons",
    "구조 검증: {state}": "Structure check: {state}",
    "통과": "passed",
    "실패": "failed",
    "OTO 행: {count}": "OTO rows: {count}",
    "오류: {count}": "Errors: {count}",
    "발견한 WAV: {count}": "WAVs found: {count}",
    "기존 OTO 매핑: {count}": "Mapped to existing OTO: {count}",
    "미매핑 WAV: {count}": "Unmapped WAVs: {count}",
    "상세": "Details",
    "이 UI에는 Tkinter와 customtkinter가 필요합니다. 프로젝트를 다시 설치하세요.": (
        "This UI needs Tkinter and customtkinter. Reinstall the project."),
    # Generate tab
    "입력과 출력": "Input and output",
    "기본은 별도 폴더 저장입니다. 직접 저장은 아래 덮어쓰기 토글을 켠 경우에만 허용됩니다.": (
        "Output goes to a separate folder by default. Saving into the voicebank is allowed only with the "
        "overwrite toggle below."),
    "보이스뱅크 폴더": "Voicebank folder",
    "기존 oto.ini (선택)": "Existing oto.ini (optional)",
    "기존 OTO는 에일리어스·인코딩 참고용이며 timing을 정답으로 재사용하지 않습니다.\n"
    "없으면 형식에 맞는 별칭(CV·VC·VCV)을 WAV 파일 이름에서 만듭니다.": (
        "The existing OTO is used only for aliases and encoding; its timing is not reused as ground truth.\n"
        "Without one, aliases for the format (CV, VC, VCV) are made from the WAV file names."),
    "출력 폴더": "Output folder",
    "기존 oto.ini 덮어쓰기 허용 (보이스뱅크 폴더 직접 저장)": (
        "Allow overwriting the existing oto.ini (save into the voicebank folder)"),
    "켜면 기존 oto.ini는 타임스탬프 .bak으로 백업된 뒤 교체됩니다.": (
        "When on, the existing oto.ini is backed up as a timestamped .bak and then replaced."),
    "하위 폴더 포함 (한 번에 처리)": "Include subfolders (process together)",
    "하위 폴더별 일괄 생성 (WAV가 있는 폴더마다 oto.ini를 따로 만듦)": (
        "Batch per subfolder (a separate oto.ini for every folder with WAVs)"),
    "일괄 생성 때 각 폴더의 oto.ini를 별칭 참고용으로 사용": (
        "In batch generation, use each folder's oto.ini as the alias reference"),
    "생성 프로필": "Generation profile",
    "언어는 직접 고르세요. 포맷은 기존 oto.ini 별칭·파일 이름·폴더 이름으로 자동 감지하거나 직접 고를 수 있습니다.": (
        "Choose the language yourself. The format can be detected from the existing oto.ini aliases, file "
        "names and folder name, or chosen by hand."),
    "언어": "Language",
    "포맷": "Format",
    "녹음 BPM (선택, 실험적)": "Recording BPM (optional, experimental)",
    "예: 120": "e.g. 120",
    "BPM 자동 추정": "Estimate BPM",
    "BPM을 넣으면 음절 위치를 박자 격자 쪽으로 당깁니다. 비우면 사용하지 않습니다(기본).\n"
    "뱅크에 따라 좋아지기도 나빠지기도 하니 결과를 들어 보고 쓰세요. 쉼 표기 파일과 박자가 안 맞는 파일은 자동 제외합니다.\n"
    "자동 추정은 박자가 일정한 파일이 충분할 때만 값을 내며, 가이드 없이 녹음했다면 끄세요.": (
        "A BPM pulls syllable positions toward the beat grid. Leave it blank to not use it (default).\n"
        "It helps some banks and hurts others, so listen to the result. Files with rest marks or off-beat "
        "files are skipped automatically.\n"
        "Estimation gives a value only when enough files keep a steady beat; turn it off if you recorded "
        "without a guide."),
    "별칭 접미사 (선택)": "Alias suffix (optional)",
    "예: C4": "e.g. C4",
    "생성된 oto에 적용": "Apply to generated oto",
    "모든 별칭 끝에 _C4 형태로 붙입니다. 생성할 때 적용되고, 이미 만든 oto.ini에는 버튼으로 붙일 수 있습니다.\n"
    "이미 같은 접미사로 끝나는 별칭은 그대로 둡니다.": (
        "Added to the end of every alias as _C4. It is applied when generating; use the button to add it to an "
        "oto.ini you already made.\nAliases that already end with the same suffix are left as they are."),
    "접미사에는 쉼표(,), = 기호, 줄바꿈을 쓸 수 없습니다.": "The suffix cannot contain a comma (,), '=' or a line break.",
    "적용할 접미사를 입력하세요.": "Enter the suffix to apply.",
    "접미사 적용": "Apply suffix",
    "접미사 {suffix}: 별칭 {changed}/{total}개에 붙였습니다 (이미 붙은 별칭은 그대로).": (
        "Suffix {suffix}: added to {changed}/{total} aliases (aliases that already had it are unchanged)."),
    "기존 oto 값 유지 (권장)": "Keep the existing oto values (recommended)",
    "oto.ini에서 빼기": "Leave them out of oto.ini",
    "생성하지 못한 기존 oto 행": "Existing oto rows that could not be generated",
    "[주의] 생성하지 못한 행 {count}개: 원본 OTO의 별칭을 녹음 파일 이름과 연결하지 못했거나 녹음을 분석하지 못해 "
    "기존 oto.ini 값을 그대로 넣었습니다 (oto.review.ini에도 포함).": (
        "[Note] {count} rows could not be generated: their aliases in the source OTO could not be matched to a "
        "recording file name, or the recording could not be analyzed, so their existing oto.ini values were kept "
        "unchanged (also in oto.review.ini)."),
    "[주의] 파일 이름이나 별칭을 {requested}로 저장할 수 없어 oto.ini를 UTF-8로 저장했습니다.": (
        "[Note] File names or aliases could not be saved as {requested}, so oto.ini was saved as UTF-8."),
    "▶ 고급 설정": "▶ Advanced settings",
    "▼ 고급 설정": "▼ Advanced settings",
    "고급 설정": "Advanced settings",
    "비워 둔 칸은 자동(템플릿) 값을 씁니다. 음소 위치는 그대로 두고 OTO 행 모양만 바꿉니다.": (
        "Blank fields use the automatic (template) value. Only the shape of the OTO rows changes, not the "
        "phoneme positions."),
    "인코딩": "Encoding",
    "VC 끝 위치": "VC end",
    "'-' 선행발성": "'-' preutterance",
    "자동": "auto",
    "프리셋 불러오기": "Load preset",
    "프리셋 저장": "Save preset",
    "기본값으로": "Reset",
    "진단 파일 저장 (기본 꺼짐, 켜도 oto.ini 결과는 같습니다)": (
        "Save diagnostic files (off by default; oto.ini is the same either way)"),
    "입력 검사": "Check input",
    "출력 검증": "Validate output",
    "출력 폴더 열기": "Open output folder",
    "OTO 미리보기": "OTO preview",
    "OTO 생성": "Generate OTO",
    # Result, log and credits tabs
    "review는 자동 검수 보류이며 오류 확정이 아닙니다.": "Review means held for checking, not a confirmed error.",
    "생성 보고서 열기": "Open generation report",
    "아직 실행 결과가 없습니다.": "No results yet.",
    "로그 복사": "Copy log",
    "보정 모델 데이터 제공에 도움 주신 분들": "Contributors who provided correction model data",
    "감사합니다!": "Thank you!",
    "찾아보기": "Browse",
    # Dialogs and status
    "WAV 보이스뱅크 폴더 선택": "Choose the WAV voicebank folder",
    "기존 oto.ini 선택": "Choose the existing oto.ini",
    "출력 폴더 선택": "Choose the output folder",
    "취소하는 중... (진행 중인 파일을 마치고 멈춥니다)": "Cancelling... (stops after the current file)",
    "[취소 요청]": "[Cancel requested]",
    "{title} 실행 중...": "{title} running...",
    "[시작] {title}": "[Start] {title}",
    "고급 설정 프리셋 불러오기": "Load advanced settings preset",
    "프리셋 오류": "Preset error",
    "[고급 설정] 프리셋을 불러왔습니다: {path}": "[Advanced settings] Loaded preset: {path}",
    "입력 오류": "Input error",
    "고급 설정 프리셋 저장": "Save advanced settings preset",
    "[고급 설정] 프리셋을 저장했습니다: {path}": "[Advanced settings] Saved preset: {path}",
    "[언어 확인] {warning}": "[Language check] {warning}",
    "언어 확인": "Language check",
    "{warning}\n\n이 언어로 계속할까요?": "{warning}\n\nContinue with this language?",
    "[입력 오류] {error}": "[Input error] {error}",
    "입력 검사 (일괄)": "Check input (batch)",
    "직접 저장 차단": "Saving into the voicebank blocked",
    "출력 폴더가 보이스뱅크와 같습니다. 기존 oto.ini 덮어쓰기 허용 토글을 켜세요.": (
        "The output folder is the voicebank folder. Turn on the toggle that allows overwriting the existing "
        "oto.ini."),
    "출력 폴더 확인": "Output folder check",
    "출력 폴더가 비어 있지 않습니다. 기존 생성 산출물은 백업 후 교체될 수 있습니다. 계속할까요?": (
        "The output folder is not empty. Earlier generated files may be backed up and replaced. Continue?"),
    "WAV가 있는 폴더를 찾지 못했습니다.": "No folder with WAVs was found.",
    "형식을 알 수 없습니다. 포맷을 직접 고르고 다시 실행하세요.": (
        "The format is unknown. Choose the format yourself and run again."),
    "[일괄 {index}/{total}] {folder} ({format})": "[Batch {index}/{total}] {folder} ({format})",
    "일괄 OTO 생성": "Batch OTO generation",
    "일괄 생성할 폴더 선택": "Choose folders for batch generation",
    "폴더마다 따로 oto.ini를 만듭니다. 형식을 알 수 없는 폴더는 기본으로 꺼져 있습니다.": (
        "A separate oto.ini is made for each folder. Folders with an unknown format are unchecked by default."),
    "직접 선택": "chosen by hand",
    "oto.ini 있음": "has oto.ini",
    "oto.ini 없음": "no oto.ini",
    "생성된 oto.ini가 없습니다: {path}": "No generated oto.ini: {path}",
    "출력 폴더 없음": "No output folder",
    "먼저 OTO를 생성하세요.": "Generate the OTO first.",
    "미리볼 oto.ini 선택": "Choose the oto.ini to preview",
    "모든 파일": "All files",
    "미리보기 실패": "Preview failed",
    "보고서 없음": "No report",
    "생성 보고서 저장이 꺼져 있었습니다. 고급 출력 옵션에서 켜고 다시 생성하세요.": (
        "Saving the generation report was off. Turn it on in the advanced output options and generate again."),
    "로그 복사 완료": "Log copied",
    "{title} 취소됨": "{title} cancelled",
    "[취소] {title}: 사용자가 중단했습니다. 출력 폴더의 이전 결과는 그대로입니다.": (
        "[Cancelled] {title}: stopped by the user. Earlier results in the output folder are unchanged."),
    "{title} 완료": "{title} done",
    "완료": "done",
    "실패: {error}": "failed: {error}",
    "[일괄] {folder}: {state}": "[Batch] {folder}: {state}",
    "{title} 완료 (생성되지 않은 행 {count}개, 로그 참고)": "{title} done ({count} rows not generated, see the log)",
    "오류": "Error",
    "[오류] {error}": "[Error] {error}",
    "작업 실패": "Operation failed",
    # Backend messages shown in the GUI
    "보이스뱅크 폴더가 없습니다: {path}": "The voicebank folder does not exist: {path}",
    "출력 폴더가 비어 있지 않습니다. --force가 필요합니다: {path}": (
        "The output folder is not empty; --force is needed: {path}"),
    "WAV를 찾지 못했습니다: {path}": "No WAVs found: {path}",
    "원본 보호를 위해 보이스뱅크 직접 저장 허용이 필요합니다": (
        "To protect the original, saving into the voicebank must be allowed explicitly"),
    "원본 보호를 위해 output은 보이스뱅크 하위 폴더가 될 수 없습니다": (
        "To protect the original, the output cannot be a subfolder of the voicebank"),
    "다른 폴더 {votes}개가 {format}로 감지되어 같은 형식을 씁니다.": (
        "{votes} other folders were detected as {format}, so the same format is used."),
    "여러 음절 목록 파일이 없어 BPM을 추정할 수 없습니다.": (
        "There are no multi-syllable list files, so the BPM cannot be estimated."),
    "박자가 일정한 파일이 {count}개뿐이라 BPM을 추정하지 않았습니다(가이드 없이 녹음한 경우 정상).": (
        "Only {count} files keep a steady beat, so the BPM was not estimated (normal when recorded without a "
        "guide)."),
    "파일마다 박자가 달라(편차 {spread:.0%}) BPM을 추정하지 않았습니다.": (
        "The beat differs between files (spread {spread:.0%}), so the BPM was not estimated."),
    "추정 BPM {bpm:.0f}이 범위를 벗어났습니다.": "The estimated BPM {bpm:.0f} is out of range.",
    "박자가 일정한 파일 {count}/{total}개에서 추정했습니다.": "Estimated from {count}/{total} files with a steady beat.",
    "별칭 {count}개 분석": "Analyzed {count} aliases",
    " (접미사 '{suffix}' 제외)": " (suffix '{suffix}' ignored)",
    "CV {cv} · VCV {vcv} · VC {vc} · 받침 음절(예: ang) {vc_syllable} · 머리(- CV) {head}": (
        "CV {cv} · VCV {vcv} · VC {vc} · coda syllables (e.g. ang) {vc_syllable} · heads (- CV) {head}"),
    "판정에 쓸 별칭이 너무 적습니다.": "Too few aliases to decide.",
    "모음+CV 별칭(a ka)이 대부분입니다.": "Most aliases are vowel + CV (a ka).",
    "VC 별칭과 받침 음절 별칭(ang, ot)이 함께 있습니다.": "There are both VC aliases and coda-syllable aliases (ang, ot).",
    "CV와 VC(a k) 별칭이 함께 있습니다.": "There are both CV and VC (a k) aliases.",
    "단독 CV 별칭이 대부분입니다.": "Most aliases are plain CV.",
    "별칭 패턴이 어느 형식과도 뚜렷하게 맞지 않습니다.": "The alias pattern does not clearly match any format.",
    "폴더 이름 '{name}'에 {format}가 들어 있습니다.": "The folder name '{name}' contains {format}.",
    "WAV {count}개 중 {share:.0%}가 한 음절 파일입니다.": "{share:.0%} of {count} WAVs are single-syllable files.",
    "여러 음절 목록 파일입니다(중앙값 {median:g}음절). 파일 이름만으로는 CVVC/VCV/CVC를 구분할 수 없습니다.": (
        "These are multi-syllable list files (median {median:g} syllables). File names alone cannot tell "
        "CVVC, VCV and CVC apart."),
    "여러 형식 별칭이 섞인 OTO라 폴더 이름의 {format}를 따릅니다.": (
        "The OTO mixes aliases of several formats, so the folder name's {format} is used."),
    "폴더 이름은 {format}를 가리켜 결과가 엇갈립니다. 형식을 확인하세요.": (
        "The folder name points to {format}, which disagrees. Check the format."),
    "형식을 직접 선택하세요.": "Choose the format yourself.",
    "언어를 한국어로 골랐지만 파일 이름/별칭의 {share:.0%}에 일본어 가나가 있습니다. "
    "일본어 보이스뱅크가 아닌지 확인하세요.": (
        "Korean is selected, but {share:.0%} of the file names/aliases contain Japanese kana. Check that this "
        "is not a Japanese voicebank."),
    "언어를 일본어로 골랐지만 파일 이름/별칭의 {share:.0%}에 한글이 있습니다. "
    "한국어 보이스뱅크가 아닌지 확인하세요.": (
        "Japanese is selected, but {share:.0%} of the file names/aliases contain Hangul. Check that this is "
        "not a Korean voicebank."),
    "언어를 일본어로 골랐지만 로마자 이름의 {share:.0%}에 한국어 표기(eo, eu, ng, ss 등)가 있습니다. "
    "한국어 보이스뱅크가 아닌지 확인하세요.": (
        "Japanese is selected, but {share:.0%} of the romanized names use Korean spellings (eo, eu, ng, ss, "
        "...). Check that this is not a Korean voicebank."),
    # OTO preview
    "위험도 모델이 없는 형식(한국어 VCV·CVC·CVVC 외)이라 자동 판정하지 않았습니다": (
        "Not judged automatically: there is no risk model for this format (only Korean VCV/CVC/CVVC have one)"),
    "미판정": "unrated",
    "{alias}  ({wav}, {line}행)": "{alias}  ({wav}, line {line})",
    "고정": "fixed",
    "선행발성": "preutterance",
    "오버랩": "overlap",
    "상태 {label}": "state {label}",
    "  |  절대 위치: 선행발성 {pre:.1f} ms, 끝 {end:.1f} ms": "  |  absolute: preutterance {pre:.1f} ms, end {end:.1f} ms",
    "근거: {reasons}": "Reasons: {reasons}",
    "{count}행 모두 미판정  |  {note}": "all {count} rows unrated  |  {note}",
    "{count}행  clean {clean}  review {review} (그중 risk>{threshold:.1f} {high})": (
        "{count} rows  clean {clean}  review {review} ({high} with risk>{threshold:.1f})"),
    "  미판정 {count}": "  unrated {count}",
    "  |  clean/review 파일이 없어 위험도 표시 없음": "  |  no clean/review files, so no risk shown",
    "전체": "All",
    "review만": "Review only",
    "clean만": "Clean only",
    "행을 선택하세요.": "Select a row.",
    "위험도순": "By risk",
    "별칭": "Alias",
    "상태": "State",
    "WAV를 찾을 수 없습니다: {wav}": "WAV not found: {wav}",
    "WAV를 읽을 수 없습니다: {error}": "Cannot read the WAV: {error}",
    "재생 실패": "Playback failed",
    "행 재생 (Space)": "Play row (Space)",
    "전체 재생": "Play all",
    "정지": "Stop",
    "행에 맞춤 (F)": "Fit row (F)",
    "전체 보기": "Show all",
    "확대 +": "Zoom in +",
    "축소 -": "Zoom out -",
    "블랭크": "blank",
    "OTO 미리보기 - {path}": "OTO preview - {path}",
}


_JA: dict[str, str] = {
    # Language selector
    "UI 언어": "UI言語",
    "재시작 후 적용됩니다.": "再起動後に適用されます。",
    # Header, tabs, bottom bar
    "WAV를 분석해 새 oto.ini를 생성합니다.": "WAVを解析して新しいoto.iniを生成します。",
    "다크 모드": "ダークモード",
    "라이트 모드": "ライトモード",
    "생성": "生成",
    "결과": "結果",
    "로그": "ログ",
    "크레딧": "クレジット",
    "취소": "キャンセル",
    "대기 중": "待機中",
    # Choices
    "한국어": "韓国語",
    "일본어": "日本語",
    "자동 감지": "自動検出",
    "다음 자음 안에서 끝냄 (권장)": "次の子音の中で終える (推奨)",
    "템플릿 기본값": "テンプレートの既定値",
    "모음 끝 (권장)": "母音の終わり (推奨)",
    "높음": "高",
    "보통": "中",
    "낮음": "低",
    "없음": "なし",
    "기존 oto.ini 별칭": "既存oto.iniのエイリアス",
    "기존 oto.ini + 폴더 이름": "既存oto.ini + フォルダ名",
    "폴더 이름": "フォルダ名",
    "WAV 파일 이름": "WAVファイル名",
    "다른 폴더": "他のフォルダ",
    "근거 없음": "根拠なし",
    # Advanced style fields
    "CV 오버랩 비율": "CV オーバーラップ比",
    "0~1, 오버랩 = 비율 × 선행발성": "0~1、オーバーラップ = 比 × 先行発声",
    "VC 오버랩 비율": "VC オーバーラップ比",
    "VCV 오버랩 비율": "VCV オーバーラップ比",
    "CV 선행발성 이동": "CV 先行発声の移動",
    "ms, + = 늦게": "ms、+ = 遅く",
    "VC 선행발성 이동": "VC 先行発声の移動",
    "VCV 선행발성 이동": "VCV 先行発声の移動",
    "고정 범위 연장": "固定範囲の延長",
    "ms, 0 이상": "ms、0以上",
    "오른쪽 블랭크 여유": "右ブランクの余裕",
    # Diagnostic files
    "unmapped_aliases.csv (생성되지 않은 행 목록)": "unmapped_aliases.csv (生成されなかった行の一覧)",
    "generation_report.json (생성 보고서)": "generation_report.json (生成レポート)",
    "row_report.jsonl (행별 근거)": "row_report.jsonl (行ごとの根拠)",
    "timelines/ (WAV별 음소 경계)": "timelines/ (WAVごとの音素境界)",
    "project_manifest.json (입력 해석 결과)": "project_manifest.json (入力の解釈結果)",
    # Option validation
    "WAV 보이스뱅크 폴더를 지정하세요.": "WAV音源フォルダを指定してください。",
    "출력 폴더를 지정하세요.": "出力フォルダを指定してください。",
    "검수 임계값은 0.0~1.0 범위여야 합니다.": "検品しきい値は0.0~1.0の範囲で指定してください。",
    "최대 WAV 수는 1 이상이어야 합니다.": "最大WAV数は1以上で指定してください。",
    "알 수 없는 진단 파일 옵션입니다: {names}": "不明な診断ファイルオプションです: {names}",
    "지원하지 않는 언어 선택입니다: {choice}": "対応していない言語です: {choice}",
    "지원하지 않는 형식 선택입니다: {choice}": "対応していない形式です: {choice}",
    "형식을 자동으로 알아내지 못했습니다. 포맷을 직접 선택하세요.": (
        "形式を自動で判別できませんでした。フォーマットを手動で選択してください。"),
    "지원하지 않는 출력 인코딩입니다: {encoding}": "対応していない出力エンコーディングです: {encoding}",
    "BPM은 숫자로 입력하세요 (예: 120).": "BPMは数値で入力してください (例: 120)。",
    "BPM은 60~240 범위로 입력하세요 (보통 100~160).": "BPMは60~240の範囲で入力してください (通常100~160)。",
    "{label}: 숫자를 입력하세요.": "{label}: 数値を入力してください。",
    "고급 설정 값이 범위를 벗어났습니다: {error}": "詳細設定の値が範囲外です: {error}",
    # Format detection
    "형식 감지: {detail}": "形式検出: {detail}",
    "알 수 없음 - 포맷을 직접 선택하세요.": "不明 - フォーマットを手動で選択してください。",
    "{format} (확신 {confidence}, 근거: {source})": "{format} (確度 {confidence}、根拠: {source})",
    "형식 감지: 보이스뱅크 폴더를 고르면 표시됩니다.": "形式検出: 音源フォルダを選ぶと表示されます。",
    "형식 감지: 일괄 생성은 폴더마다 따로 감지합니다.": "形式検出: 一括生成ではフォルダごとに検出します。",
    "형식 감지 실패: {error}": "形式検出に失敗しました: {error}",
    # Log lines
    "[BPM] 입력한 {bpm:g} BPM 사용 (음절 간격 {period:.0f} ms)": (
        "[BPM] 入力された {bpm:g} BPM を使用 (音節間隔 {period:.0f} ms)"),
    "[BPM] 자동 추정 {bpm} BPM 사용 ({reason})": "[BPM] 自動推定した {bpm} BPM を使用 ({reason})",
    "[BPM] 추정 안 함: {reason} BPM 없이 생성했습니다.": "[BPM] 推定なし: {reason} BPMなしで生成しました。",
    "[별칭 생성] 기존 oto.ini 없이 WAV {count}개의 파일 이름에서 형식에 맞는 별칭을 만들었습니다.": (
        "[エイリアス生成] 既存のoto.iniがないため、WAV {count}個のファイル名から形式に合うエイリアスを作成しました。"),
    "[주의] 이름을 읽지 못한 WAV {count}개는 빠졌습니다.": "[注意] 名前を読み取れなかったWAV {count}個は除外しました。",
    "review 행은 자동 검수 보류이며 오류 확정이 아닙니다.": "review行は自動検品の保留であり、誤りと確定したわけではありません。",
    "정렬 {step}단계 {done}/{total}": "アライメント 段階{step} {done}/{total}",
    "OTO 작성 {done}/{total}": "OTO書き出し {done}/{total}",
    " · 남은 시간 약 {minutes}분": " · 残り約{minutes}分",
    " · 남은 시간 약 {seconds}초": " · 残り約{seconds}秒",
    "[주의] 생성되지 않은 행 {count}개: 원본 OTO의 별칭을 녹음 파일 이름과 연결하지 못했거나 "
    "녹음을 분석하지 못해 oto.ini에서 뺐습니다.": (
        "[注意] 生成されなかった行 {count}個: 元のOTOのエイリアスを録音ファイル名と対応付けられなかったか、"
        "録音を解析できなかったため、oto.iniから除外しました。"),
    "  분석하지 못한 녹음 {count}개(무음·잘린 녹음일 수 있음): {names}": (
        "  解析できなかった録音 {count}個 (無音・途切れた録音の可能性があります): {names}"),
    "  전체 목록: {path}": "  全一覧: {path}",
    "  전체 목록을 파일로 받으려면 고급 출력 옵션에서 unmapped_aliases.csv 저장을 켜세요.": (
        "  全一覧をファイルで受け取るには、詳細出力オプションでunmapped_aliases.csvの保存をオンにしてください。"),
    "  … 외 {count}개": "  … ほか{count}個",
    # Result text
    "생성 행: {count}": "生成行: {count}",
    "자동 clean: {count}": "自動clean: {count}",
    "검수 보류: {count}": "検品保留: {count}",
    "먼저 확인: {count}행": "優先確認: {count}行",
    "연결 확인: {count}행": "接続確認: {count}行",
    "표본 확인: {count}행": "抜き取り確認: {count}行",
    "검수 순서: {path}": "検品順: {path}",
    "주요 검수 사유": "主な検品理由",
    "구조 검증: {state}": "構造検証: {state}",
    "통과": "合格",
    "실패": "不合格",
    "OTO 행: {count}": "OTO行: {count}",
    "오류: {count}": "エラー: {count}",
    "발견한 WAV: {count}": "見つかったWAV: {count}",
    "기존 OTO 매핑: {count}": "既存OTOと対応: {count}",
    "미매핑 WAV: {count}": "未対応WAV: {count}",
    "상세": "詳細",
    "이 UI에는 Tkinter와 customtkinter가 필요합니다. 프로젝트를 다시 설치하세요.": (
        "このUIにはTkinterとcustomtkinterが必要です。プロジェクトを再インストールしてください。"),
    # Generate tab
    "입력과 출력": "入力と出力",
    "기본은 별도 폴더 저장입니다. 직접 저장은 아래 덮어쓰기 토글을 켠 경우에만 허용됩니다.": (
        "既定では別フォルダに保存します。音源フォルダへの直接保存は、下の上書きトグルをオンにした場合のみ可能です。"),
    "보이스뱅크 폴더": "音源フォルダ",
    "기존 oto.ini (선택)": "既存のoto.ini (任意)",
    "기존 OTO는 에일리어스·인코딩 참고용이며 timing을 정답으로 재사용하지 않습니다.\n"
    "없으면 형식에 맞는 별칭(CV·VC·VCV)을 WAV 파일 이름에서 만듭니다.": (
        "既存のOTOはエイリアスとエンコーディングの参考にのみ使い、タイミングを正解として再利用しません。\n"
        "ない場合は、形式に合うエイリアス (CV・VC・VCV) をWAVファイル名から作成します。"),
    "출력 폴더": "出力フォルダ",
    "기존 oto.ini 덮어쓰기 허용 (보이스뱅크 폴더 직접 저장)": "既存のoto.iniの上書きを許可 (音源フォルダに直接保存)",
    "켜면 기존 oto.ini는 타임스탬프 .bak으로 백업된 뒤 교체됩니다.": (
        "オンにすると、既存のoto.iniはタイムスタンプ付きの.bakにバックアップされてから置き換えられます。"),
    "하위 폴더 포함 (한 번에 처리)": "サブフォルダを含める (まとめて処理)",
    "하위 폴더별 일괄 생성 (WAV가 있는 폴더마다 oto.ini를 따로 만듦)": (
        "サブフォルダごとに一括生成 (WAVのあるフォルダごとにoto.iniを作成)"),
    "일괄 생성 때 각 폴더의 oto.ini를 별칭 참고용으로 사용": "一括生成時、各フォルダのoto.iniをエイリアスの参考に使う",
    "생성 프로필": "生成プロファイル",
    "언어는 직접 고르세요. 포맷은 기존 oto.ini 별칭·파일 이름·폴더 이름으로 자동 감지하거나 직접 고를 수 있습니다.": (
        "言語は手動で選んでください。フォーマットは既存oto.iniのエイリアス・ファイル名・フォルダ名から自動検出するか、"
        "手動で選べます。"),
    "언어": "言語",
    "포맷": "フォーマット",
    "녹음 BPM (선택, 실험적)": "収録BPM (任意・実験的)",
    "예: 120": "例: 120",
    "BPM 자동 추정": "BPMを自動推定",
    "BPM을 넣으면 음절 위치를 박자 격자 쪽으로 당깁니다. 비우면 사용하지 않습니다(기본).\n"
    "뱅크에 따라 좋아지기도 나빠지기도 하니 결과를 들어 보고 쓰세요. 쉼 표기 파일과 박자가 안 맞는 파일은 자동 제외합니다.\n"
    "자동 추정은 박자가 일정한 파일이 충분할 때만 값을 내며, 가이드 없이 녹음했다면 끄세요.": (
        "BPMを入力すると、音節位置を拍のグリッド側へ寄せます。空欄なら使いません (既定)。\n"
        "音源によって良くも悪くもなるので、結果を聴いてから使ってください。休符表記のファイルや拍が合わないファイルは"
        "自動で除外します。\n"
        "自動推定は拍が一定のファイルが十分ある場合のみ値を出します。ガイドなしで収録した場合はオフにしてください。"),
    "별칭 접미사 (선택)": "エイリアス接尾辞 (任意)",
    "예: C4": "例: C4",
    "생성된 oto에 적용": "生成済みotoに適用",
    "모든 별칭 끝에 _C4 형태로 붙입니다. 생성할 때 적용되고, 이미 만든 oto.ini에는 버튼으로 붙일 수 있습니다.\n"
    "이미 같은 접미사로 끝나는 별칭은 그대로 둡니다.": (
        "すべてのエイリアスの末尾に _C4 の形で付けます。生成時に適用され、作成済みのoto.iniにはボタンで付けられます。\n"
        "すでに同じ接尾辞で終わるエイリアスはそのままにします。"),
    "접미사에는 쉼표(,), = 기호, 줄바꿈을 쓸 수 없습니다.": "接尾辞にはカンマ (,)、= 記号、改行は使えません。",
    "적용할 접미사를 입력하세요.": "適用する接尾辞を入力してください。",
    "접미사 적용": "接尾辞の適用",
    "접미사 {suffix}: 별칭 {changed}/{total}개에 붙였습니다 (이미 붙은 별칭은 그대로).": (
        "接尾辞 {suffix}: エイリアス {changed}/{total}個に付けました (すでに付いているものはそのまま)。"),
    "기존 oto 값 유지 (권장)": "既存のotoの値を維持 (推奨)",
    "oto.ini에서 빼기": "oto.iniから除外",
    "생성하지 못한 기존 oto 행": "生成できなかった既存のoto行",
    "[주의] 생성하지 못한 행 {count}개: 원본 OTO의 별칭을 녹음 파일 이름과 연결하지 못했거나 녹음을 분석하지 못해 "
    "기존 oto.ini 값을 그대로 넣었습니다 (oto.review.ini에도 포함).": (
        "[注意] 生成できなかった行 {count}個: 元のOTOのエイリアスを録音ファイル名と対応付けられなかったか、録音を解析できなかったため、"
        "既存のoto.iniの値をそのまま入れました (oto.review.iniにも含みます)。"),
    "[주의] 파일 이름이나 별칭을 {requested}로 저장할 수 없어 oto.ini를 UTF-8로 저장했습니다.": (
        "[注意] ファイル名やエイリアスを{requested}で保存できないため、oto.iniをUTF-8で保存しました。"),
    "▶ 고급 설정": "▶ 詳細設定",
    "▼ 고급 설정": "▼ 詳細設定",
    "고급 설정": "詳細設定",
    "비워 둔 칸은 자동(템플릿) 값을 씁니다. 음소 위치는 그대로 두고 OTO 행 모양만 바꿉니다.": (
        "空欄の項目は自動 (テンプレート) の値を使います。音素位置はそのままで、OTO行の形だけを変えます。"),
    "인코딩": "エンコーディング",
    "VC 끝 위치": "VCの終了位置",
    "'-' 선행발성": "'-' の先行発声",
    "자동": "自動",
    "프리셋 불러오기": "プリセット読込",
    "프리셋 저장": "プリセット保存",
    "기본값으로": "既定値に戻す",
    "진단 파일 저장 (기본 꺼짐, 켜도 oto.ini 결과는 같습니다)": (
        "診断ファイルを保存 (既定はオフ。オンにしてもoto.iniの結果は同じです)"),
    "입력 검사": "入力チェック",
    "출력 검증": "出力検証",
    "출력 폴더 열기": "出力フォルダを開く",
    "OTO 미리보기": "OTOプレビュー",
    "OTO 생성": "OTO生成",
    # Result, log and credits tabs
    "review는 자동 검수 보류이며 오류 확정이 아닙니다.": "reviewは自動検品の保留であり、誤りと確定したわけではありません。",
    "생성 보고서 열기": "生成レポートを開く",
    "아직 실행 결과가 없습니다.": "まだ実行結果がありません。",
    "로그 복사": "ログをコピー",
    "보정 모델 데이터 제공에 도움 주신 분들": "補正モデルのデータ提供にご協力いただいた方々",
    "감사합니다!": "ありがとうございました！",
    "찾아보기": "参照",
    # Dialogs and status
    "WAV 보이스뱅크 폴더 선택": "WAV音源フォルダを選択",
    "기존 oto.ini 선택": "既存のoto.iniを選択",
    "출력 폴더 선택": "出力フォルダを選択",
    "취소하는 중... (진행 중인 파일을 마치고 멈춥니다)": "キャンセル中... (処理中のファイルを終えてから停止します)",
    "[취소 요청]": "[キャンセル要求]",
    "{title} 실행 중...": "{title} 実行中...",
    "[시작] {title}": "[開始] {title}",
    "고급 설정 프리셋 불러오기": "詳細設定プリセットの読込",
    "프리셋 오류": "プリセットエラー",
    "[고급 설정] 프리셋을 불러왔습니다: {path}": "[詳細設定] プリセットを読み込みました: {path}",
    "입력 오류": "入力エラー",
    "고급 설정 프리셋 저장": "詳細設定プリセットの保存",
    "[고급 설정] 프리셋을 저장했습니다: {path}": "[詳細設定] プリセットを保存しました: {path}",
    "[언어 확인] {warning}": "[言語確認] {warning}",
    "언어 확인": "言語確認",
    "{warning}\n\n이 언어로 계속할까요?": "{warning}\n\nこの言語で続けますか？",
    "[입력 오류] {error}": "[入力エラー] {error}",
    "입력 검사 (일괄)": "入力チェック (一括)",
    "직접 저장 차단": "直接保存をブロック",
    "출력 폴더가 보이스뱅크와 같습니다. 기존 oto.ini 덮어쓰기 허용 토글을 켜세요.": (
        "出力フォルダが音源フォルダと同じです。既存のoto.iniの上書きを許可するトグルをオンにしてください。"),
    "출력 폴더 확인": "出力フォルダの確認",
    "출력 폴더가 비어 있지 않습니다. 기존 생성 산출물은 백업 후 교체될 수 있습니다. 계속할까요?": (
        "出力フォルダが空ではありません。以前の生成物はバックアップ後に置き換えられる場合があります。続けますか？"),
    "WAV가 있는 폴더를 찾지 못했습니다.": "WAVのあるフォルダが見つかりませんでした。",
    "형식을 알 수 없습니다. 포맷을 직접 고르고 다시 실행하세요.": (
        "形式が不明です。フォーマットを手動で選んでから再実行してください。"),
    "[일괄 {index}/{total}] {folder} ({format})": "[一括 {index}/{total}] {folder} ({format})",
    "일괄 OTO 생성": "一括OTO生成",
    "일괄 생성할 폴더 선택": "一括生成するフォルダを選択",
    "폴더마다 따로 oto.ini를 만듭니다. 형식을 알 수 없는 폴더는 기본으로 꺼져 있습니다.": (
        "フォルダごとにoto.iniを作成します。形式が不明なフォルダは既定でオフになっています。"),
    "직접 선택": "手動選択",
    "oto.ini 있음": "oto.iniあり",
    "oto.ini 없음": "oto.iniなし",
    "생성된 oto.ini가 없습니다: {path}": "生成されたoto.iniがありません: {path}",
    "출력 폴더 없음": "出力フォルダなし",
    "먼저 OTO를 생성하세요.": "先にOTOを生成してください。",
    "미리볼 oto.ini 선택": "プレビューするoto.iniを選択",
    "모든 파일": "すべてのファイル",
    "미리보기 실패": "プレビューに失敗しました",
    "보고서 없음": "レポートなし",
    "생성 보고서 저장이 꺼져 있었습니다. 고급 출력 옵션에서 켜고 다시 생성하세요.": (
        "生成レポートの保存がオフでした。詳細出力オプションでオンにしてから再生成してください。"),
    "로그 복사 완료": "ログをコピーしました",
    "{title} 취소됨": "{title} キャンセル済み",
    "[취소] {title}: 사용자가 중단했습니다. 출력 폴더의 이전 결과는 그대로입니다.": (
        "[キャンセル] {title}: ユーザーが中断しました。出力フォルダの以前の結果はそのままです。"),
    "{title} 완료": "{title} 完了",
    "완료": "完了",
    "실패: {error}": "失敗: {error}",
    "[일괄] {folder}: {state}": "[一括] {folder}: {state}",
    "{title} 완료 (생성되지 않은 행 {count}개, 로그 참고)": "{title} 完了 (生成されなかった行 {count}個、ログ参照)",
    "오류": "エラー",
    "[오류] {error}": "[エラー] {error}",
    "작업 실패": "処理に失敗しました",
    # Backend messages shown in the GUI
    "보이스뱅크 폴더가 없습니다: {path}": "音源フォルダがありません: {path}",
    "출력 폴더가 비어 있지 않습니다. --force가 필요합니다: {path}": "出力フォルダが空ではありません。--forceが必要です: {path}",
    "WAV를 찾지 못했습니다: {path}": "WAVが見つかりませんでした: {path}",
    "원본 보호를 위해 보이스뱅크 직접 저장 허용이 필요합니다": "元データ保護のため、音源フォルダへの直接保存の許可が必要です",
    "원본 보호를 위해 output은 보이스뱅크 하위 폴더가 될 수 없습니다": (
        "元データ保護のため、出力先を音源フォルダのサブフォルダにはできません"),
    "다른 폴더 {votes}개가 {format}로 감지되어 같은 형식을 씁니다.": (
        "他のフォルダ{votes}個が{format}と検出されたため、同じ形式を使います。"),
    "여러 음절 목록 파일이 없어 BPM을 추정할 수 없습니다.": "複数音節のリストファイルがないため、BPMを推定できません。",
    "박자가 일정한 파일이 {count}개뿐이라 BPM을 추정하지 않았습니다(가이드 없이 녹음한 경우 정상).": (
        "拍が一定のファイルが{count}個しかないため、BPMを推定しませんでした (ガイドなしで収録した場合は正常です)。"),
    "파일마다 박자가 달라(편차 {spread:.0%}) BPM을 추정하지 않았습니다.": (
        "ファイルごとに拍が異なる (ばらつき {spread:.0%}) ため、BPMを推定しませんでした。"),
    "추정 BPM {bpm:.0f}이 범위를 벗어났습니다.": "推定BPM {bpm:.0f} が範囲外です。",
    "박자가 일정한 파일 {count}/{total}개에서 추정했습니다.": "拍が一定のファイル {count}/{total}個から推定しました。",
    "별칭 {count}개 분석": "エイリアス{count}個を解析",
    " (접미사 '{suffix}' 제외)": " (接尾辞 '{suffix}' を除外)",
    "CV {cv} · VCV {vcv} · VC {vc} · 받침 음절(예: ang) {vc_syllable} · 머리(- CV) {head}": (
        "CV {cv} · VCV {vcv} · VC {vc} · パッチム音節 (例: ang) {vc_syllable} · 語頭 (- CV) {head}"),
    "판정에 쓸 별칭이 너무 적습니다.": "判定に使えるエイリアスが少なすぎます。",
    "모음+CV 별칭(a ka)이 대부분입니다.": "母音+CVのエイリアス (a ka) が大半です。",
    "VC 별칭과 받침 음절 별칭(ang, ot)이 함께 있습니다.": "VCエイリアスとパッチム音節エイリアス (ang, ot) が両方あります。",
    "CV와 VC(a k) 별칭이 함께 있습니다.": "CVとVC (a k) のエイリアスが両方あります。",
    "단독 CV 별칭이 대부분입니다.": "単独CVのエイリアスが大半です。",
    "별칭 패턴이 어느 형식과도 뚜렷하게 맞지 않습니다.": "エイリアスのパターンがどの形式にもはっきり当てはまりません。",
    "폴더 이름 '{name}'에 {format}가 들어 있습니다.": "フォルダ名 '{name}' に {format} が含まれています。",
    "WAV {count}개 중 {share:.0%}가 한 음절 파일입니다.": "WAV {count}個のうち {share:.0%} が単音節ファイルです。",
    "여러 음절 목록 파일입니다(중앙값 {median:g}음절). 파일 이름만으로는 CVVC/VCV/CVC를 구분할 수 없습니다.": (
        "複数音節のリストファイルです (中央値 {median:g}音節)。ファイル名だけではCVVC/VCV/CVCを区別できません。"),
    "여러 형식 별칭이 섞인 OTO라 폴더 이름의 {format}를 따릅니다.": (
        "複数形式のエイリアスが混在するOTOのため、フォルダ名の{format}に従います。"),
    "폴더 이름은 {format}를 가리켜 결과가 엇갈립니다. 형식을 확인하세요.": (
        "フォルダ名は{format}を示しており、結果が食い違います。形式を確認してください。"),
    "형식을 직접 선택하세요.": "形式を手動で選択してください。",
    "언어를 한국어로 골랐지만 파일 이름/별칭의 {share:.0%}에 일본어 가나가 있습니다. "
    "일본어 보이스뱅크가 아닌지 확인하세요.": (
        "言語に韓国語を選びましたが、ファイル名/エイリアスの{share:.0%}に日本語の仮名があります。"
        "日本語音源ではないか確認してください。"),
    "언어를 일본어로 골랐지만 파일 이름/별칭의 {share:.0%}에 한글이 있습니다. "
    "한국어 보이스뱅크가 아닌지 확인하세요.": (
        "言語に日本語を選びましたが、ファイル名/エイリアスの{share:.0%}にハングルがあります。"
        "韓国語音源ではないか確認してください。"),
    "언어를 일본어로 골랐지만 로마자 이름의 {share:.0%}에 한국어 표기(eo, eu, ng, ss 등)가 있습니다. "
    "한국어 보이스뱅크가 아닌지 확인하세요.": (
        "言語に日本語を選びましたが、ローマ字名の{share:.0%}に韓国語の表記 (eo, eu, ng, ss など) があります。"
        "韓国語音源ではないか確認してください。"),
    # OTO preview
    "위험도 모델이 없는 형식(한국어 VCV·CVC·CVVC 외)이라 자동 판정하지 않았습니다": (
        "リスクモデルのない形式 (韓国語VCV・CVC・CVVC以外) のため、自動判定していません"),
    "미판정": "未判定",
    "{alias}  ({wav}, {line}행)": "{alias}  ({wav}, {line}行目)",
    "고정": "固定範囲",
    "선행발성": "先行発声",
    "오버랩": "オーバーラップ",
    "상태 {label}": "状態 {label}",
    "  |  절대 위치: 선행발성 {pre:.1f} ms, 끝 {end:.1f} ms": "  |  絶対位置: 先行発声 {pre:.1f} ms、終端 {end:.1f} ms",
    "근거: {reasons}": "根拠: {reasons}",
    "{count}행 모두 미판정  |  {note}": "{count}行すべて未判定  |  {note}",
    "{count}행  clean {clean}  review {review} (그중 risk>{threshold:.1f} {high})": (
        "{count}行  clean {clean}  review {review} (うち risk>{threshold:.1f} {high})"),
    "  미판정 {count}": "  未判定 {count}",
    "  |  clean/review 파일이 없어 위험도 표시 없음": "  |  clean/reviewファイルがないためリスク表示なし",
    "전체": "すべて",
    "review만": "reviewのみ",
    "clean만": "cleanのみ",
    "행을 선택하세요.": "行を選択してください。",
    "위험도순": "リスク順",
    "별칭": "エイリアス",
    "상태": "状態",
    "WAV를 찾을 수 없습니다: {wav}": "WAVが見つかりません: {wav}",
    "WAV를 읽을 수 없습니다: {error}": "WAVを読み込めません: {error}",
    "재생 실패": "再生に失敗しました",
    "행 재생 (Space)": "行を再生 (Space)",
    "전체 재생": "全体を再生",
    "정지": "停止",
    "행에 맞춤 (F)": "行に合わせる (F)",
    "전체 보기": "全体表示",
    "확대 +": "拡大 +",
    "축소 -": "縮小 -",
    "블랭크": "ブランク",
    "OTO 미리보기 - {path}": "OTOプレビュー - {path}",
}

_TABLES: Mapping[str, Mapping[str, str]] = {"en": _EN, "ja": _JA}
