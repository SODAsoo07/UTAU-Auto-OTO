# -*- coding: utf-8 -*-
"""
UI Internationalization (i18n) module for Auto_OTO.

Supported languages:
  ko  Korean (default)
  en  English
  ja  Japanese

Usage:
    from ui.i18n import t, set_language, get_language

    label = ctk.CTkLabel(..., text=t("파이프라인"))
    msg = t("wav_overwrite_body").format(folder=path)

Language is initialised at app startup from config.json ("ui_language" key)
or the UTOA_UI_LANGUAGE environment variable.
A restart is required after changing the language setting.
"""

from __future__ import annotations

import os

_LANG: str = "ko"


def set_language(lang: str) -> None:
    """Set the active UI language.  Recognised values: 'ko', 'en', 'ja'."""
    global _LANG
    _LANG = str(lang or "ko").strip().lower()
    if _LANG not in {"ko", "en", "ja"}:
        _LANG = "ko"


def get_language() -> str:
    """Return the currently active UI language code."""
    return _LANG


def t(key: str, **kwargs) -> str:
    """Return the translated string for *key* in the current language.

    Falls back to *key* itself (Korean) when the translation is missing.
    Keyword arguments are forwarded to ``str.format``.
    """
    if not key:
        return key
    if _LANG == "en":
        translated = _EN.get(key, key)
    elif _LANG == "ja":
        translated = _JA.get(key, key)
    else:
        translated = key
    if kwargs:
        try:
            translated = translated.format(**kwargs)
        except Exception:
            pass
    return translated


# ---------------------------------------------------------------------------
# English translation table
# Keys = Korean source strings (or short symbolic keys for long/dynamic text)
# ---------------------------------------------------------------------------

_EN: dict[str, str] = {

    # ── UI language control ─────────────────────────────────────────────────
    "UI 언어": "UI Language",
    "재시작 후 적용됩니다.": "Changes will apply after restart.",

    # ── Tab names ───────────────────────────────────────────────────────────
    "파이프라인": "Pipeline",
    "파라미터 조정": "Parameter Tuning",
    "로그": "Log",
    "상세 로그": "Detailed Log",
    "크레딧": "Credits",
    "고급 설정": "Advanced Settings",

    # ── Layout / header ─────────────────────────────────────────────────────
    "언어": "UTAU Language",
    "WAV 폴더:": "WAV Folder:",
    "WAV 파일이 있는 폴더 경로": "Path to folder containing WAV files",
    "하위 폴더 자동 탐색(배치 처리)": "Auto-scan subfolders (batch processing)",
    "찾아보기": "Browse",
    "템플릿 OTO:": "Template OTO:",
    "선택 사항 (없을 시 파일명/라벨 기반 자동 생성)": "Optional (auto-generated from filename/label if absent)",
    "템플릿 OTO 없음 (OpenUtau 호환 에일리어스 자동 생성)": "No template OTO (auto-generate OpenUtau-compatible aliases)",
    "출력 경로:": "Output Path:",
    "생성된 oto.ini 저장 경로": "Path to save generated oto.ini",
    "저장": "Save",
    "형식 지정:": "Format:",
    "(템플릿 유무와 무관하게 우선 적용)": "(Takes priority regardless of template)",
    "JP 에일리어스:": "JP Alias:",
    "(일본어 OTO 적용)": "(Applied to Japanese OTO)",
    "원본 그대로": "As-is",
    "히라가나": "Hiragana",
    "로마자": "Romaji",
    "대기 중": "Waiting",

    # ── Pipeline tab · run section ──────────────────────────────────────────
    "실행": "Run",
    "순서대로 실행": "Run in Order",
    "OpenUtau 호환 별도 에일리어스 자동 생성": "Auto-generate OpenUtau-compatible separate aliases",
    "누락된 모음/모음열(VV) 에일리어스 보완 생성": "Generate missing vowel / VV aliases",
    "어두/어미 '-' 에일리어스 생성": "Generate head/tail '-' aliases",
    "ML 보정은 자동으로 적용되며 모델이 없으면 자동으로 건너뜁니다.": (
        "ML correction is applied automatically; skipped if no model is found."
    ),
    "파라미터 조정 탭에서 OTO 보정 계수를 조정할 수 있습니다. 기본값 사용을 권장합니다.": (
        "OTO correction parameters can be adjusted in the Parameter Tuning tab. "
        "Default values are recommended."
    ),

    # ── MFA setup hint ──────────────────────────────────────────────────────
    "처음 설치나 'MFA 진단/복구'는 환경 구성과 현재 언어 모델 다운로드 때문에 10~20분 이상 걸릴 수 있습니다.\n문제가 생기면 먼저 'MFA 진단/복구'를 눌러 자동 복구를 시도한 뒤 정렬을 다시 실행하세요.": (
        "First-time setup or 'MFA Diagnose/Repair' may take 10–20+ minutes "
        "due to environment setup and language model downloads.\n"
        "If problems occur, click 'MFA Diagnose/Repair' first to attempt "
        "auto-repair, then retry alignment."
    ),

    # ── Log tab ─────────────────────────────────────────────────────────────
    "로그 지우기": "Clear Log",
    "로그 파일 열기": "Open Log File",
    "CMD/서브프로세스 원본 로그를 시간순으로 표시합니다. (저부하 버퍼 플러시)": (
        "Displays raw CMD/subprocess logs in chronological order. "
        "(Low-overhead buffer flush)"
    ),
    "상세 로그 지우기": "Clear Detailed Log",

    # ── Credits tab ─────────────────────────────────────────────────────────
    "-보정 모델 데이터 제공에 도움 주신 분들": "Contributors who provided correction model data",
    "감사합니다!": "Thank you!",

    # ── Advanced Settings tab · core options ────────────────────────────────
    "고급 설정은 개발/튜닝용 항목입니다. 실사용에서는 기본값 유지 후 필요한 경우만 조정하세요.": (
        "Advanced settings are for development and tuning. "
        "Keep defaults in production; adjust only when necessary."
    ),
    "핵심 옵션": "Core Options",
    "ML 보정 사용": "Enable ML Correction",
    "연속성/파일 일관성 보정 사용": "Enable Continuity / File-consistency Correction",
    "형식 변경 시 추천값이 자동 적용됩니다. 필요한 경우 아래 개발자 슬라이더에서 세부값만 미세 조정하세요.": (
        "Recommended values are applied automatically on format change. "
        "Fine-tune only via the developer sliders below if needed."
    ),

    # ── Post-processing options ──────────────────────────────────────────────
    "후처리(연속성) 옵션": "Post-processing (Continuity) Options",
    "연속성 offset 보정 상한 (ms)": "Continuity offset correction limit (ms)",
    "기본값(180)": "Default (180)",
    "크게 잡을수록 연속성 보정이 강해지고, 작게 잡을수록 보정이 약해집니다.": (
        "Larger values strengthen continuity correction; smaller values weaken it."
    ),

    # ── Voicebank tuning presets ─────────────────────────────────────────────
    "보이스뱅크 튜닝 프리셋": "Voicebank Tuning Presets",
    "soft/일반 뱅크용 프리셋을 한 번에 적용합니다. 민감형은 보정을 강하게, 일반은 안정적으로 동작합니다.": (
        "Applies presets for soft/normal voicebanks at once. "
        "Aggressive applies stronger correction; conservative is more stable."
    ),
    "soft 뱅크 · 민감형": "Soft Bank · Aggressive",
    "soft 뱅크 · 일반": "Soft Bank · Conservative",
    "일반 뱅크 · 민감형": "Normal Bank · Aggressive",
    "일반 뱅크 · 일반": "Normal Bank · Conservative",

    # ── Phoneme dropout / weak-voice options ────────────────────────────────
    "발음 누락 줄이기 추천": "Reduce Phoneme Dropout (Recommended)",
    "숨소리가 많이 섞이거나 발성이 약한 음원에서 발음 누락과 공백 오검출을 줄일 때 켜는 옵션입니다.": (
        "Enable this for sources with heavy breath noise or weak voicing "
        "to reduce phoneme dropout and silence misdetection."
    ),
    "저볼륨 WAV 자동 증폭/정규화 (RMS 기반, 원본 보존)": (
        "Auto-amplify / normalize low-volume WAV (RMS-based, originals preserved)"
    ),
    "정렬 단계에서만 임시 작업 복사본에 볼륨 보정을 적용합니다. 원본 WAV 파일은 절대 변경하지 않습니다.": (
        "Volume correction is applied only to a temporary working copy during alignment. "
        "Original WAV files are never modified."
    ),
    "발음이 흐린 음원에 추천 (자모 구분 강화)": (
        "Recommended for unclear pronunciation sources (enhances consonant/vowel distinction)"
    ),
    "웅얼거림처럼 들리거나 자음/모음 구분이 흐린 음원에서 정렬 정확도를 높이기 위한 추천 옵션입니다. (원본 WAV 보존)": (
        "Recommended to improve alignment accuracy for mumbled or indistinct "
        "consonant/vowel sources. (Original WAV preserved)"
    ),
    "자모 대비 강도": "Jamo Contrast Strength",

    # ── Mis-mapping block ────────────────────────────────────────────────────
    "음절 오매핑 차단": "Block Syllable Mis-mapping",
    "완전 엄격": "Fully Strict",
    "적당히 엄격(누락 행은 폴백)": "Moderate (fallback for missing rows)",
    "음절을 틀릴 가능성을 줄이는 대신, 자동 설정에서 스킵되는 에일리어스가 늘어날 수 있습니다.": (
        "Reduces the chance of wrong syllable assignment, but may increase "
        "the number of skipped aliases in auto mode."
    ),
    "저신뢰 구간 강제 고정 모드": "Force-lock Low-confidence Intervals",
    "정렬 신뢰가 낮은 구간에서는 후보 점프를 막고 예상 위치에 고정합니다.": (
        "In low-confidence alignment intervals, blocks candidate jumps "
        "and locks to the expected position."
    ),

    # ── Inline quick options ─────────────────────────────────────────────────
    "발음 누락 줄이기": "Reduce Phoneme Dropout",
    "오매핑 차단": "Block Mis-mapping",

    # ── CVN / mapping options ────────────────────────────────────────────────
    "자음 구분기(C/V) 보정 사용": "Enable C/V Consonant Classifier Correction",
    "정렬 저신뢰 샘플에만 C/V 후보정 적용": (
        "Apply C/V post-correction only to low-confidence alignment samples"
    ),
    "매핑 모델(단조 디코딩 리스코어) 사용": "Enable Mapping Model (monotonic decoding rescore)",
    "매핑 모델 모드": "Mapping Model Mode",
    "자동(권장)": "Auto (Recommended)",
    "포맷 우선": "Format-first",
    "전역 우선": "Global-first",
    "CV 파일 순서 기반 매핑 결합 사용": "Enable CV file-order-based mapping combination",
    "강도": "Strength",
    "기본값": "Default",
    "(0.0~1.0, 높을수록 순서 prior 강화)": "(0.0–1.0; higher = stronger order prior)",

    # ── ML correction mode ───────────────────────────────────────────────────
    "ML 보정 모드": "ML Correction Mode",
    "상대적 보정": "Relative Correction",
    "+셀렉터": "+ Selector",
    "ML 방식": "ML Method",
    "자동(자동 라우팅)": "Auto (auto-routing)",
    "E2E 하이브리드(실험)": "E2E Hybrid (experimental)",
    "자동=환경 기반 라우팅, No-MFA=정렬 없이 보정, v1=기존, v2=확장, E2E=실험": (
        "Auto=env-based routing  |  No-MFA=correct without alignment  |  "
        "v1=legacy  |  v2=extended  |  E2E=experimental"
    ),

    # ── Coupled ML options ───────────────────────────────────────────────────
    "멜+OTO 보정 사용": "Enable Mel+OTO Correction",
    "Coupled 최소 신뢰도": "Coupled Min Confidence",
    "Strict 제약": "Strict Constraint",
    "Model-aware min_conf (model_meta) 사용": "Use model-aware min_conf (model_meta)",
    "(range: -0.30 ~ +0.30)": "(range: −0.30 to +0.30)",
    "Batch inference (coupled v2) 사용": "Enable Batch Inference (coupled v2)",
    "(권장 128~512, 최소 32)": "(recommended 128–512, min 32)",
    "v1 fallback(lightgbm) 체인 사용": "Enable v1 fallback (LightGBM) chain",
    "하이브리드 라우팅(가중치 게이트) 사용": "Enable hybrid routing (weighted gate)",
    "E2E 하이브리드(실험) 활성화": "Enable E2E Hybrid (experimental)",
    "E2E 모드": "E2E Mode",
    "(권장: hybrid)": "(recommended: hybrid)",
    "(&gt;1.0: stronger shrink on low mel reliability)": (
        "(>1.0: stronger shrink on low mel reliability)"
    ),

    # ── Model paths ──────────────────────────────────────────────────────────
    "자세히 보기 (경로/버전/생성일)": "Show details (path / version / created)",
    "모델 경로 (한국어)": "Model Path (Korean)",
    "모델 경로 (일본어)": "Model Path (Japanese)",
    "경로는 모델 폴더(내부에 model_meta.json) 또는 상위 루트를 지정할 수 있습니다.": (
        "Specify the model folder (containing model_meta.json) or its parent root."
    ),
    "열기": "Open",

    # ── VC correction section ────────────────────────────────────────────────
    "VC 보정": "VC Correction",
    "기본 UI에서는 보정 사용 여부만 선택합니다. 세부 슬라이더는 개발자 설정 ON일 때만 활성화됩니다.": (
        "In the basic UI, only the enable/disable toggle is shown. "
        "Detailed sliders activate only when developer settings are ON."
    ),
    "한국어 (KR)": "Korean (KR)",
    "VC 이웃 보정 사용": "Enable VC Neighbor Correction",
    "보정 강도(Blend)": "Correction Strength (Blend)",
    "최대 이동(ms)": "Max Shift (ms)",
    "이전 컷오프 기준 여유(ms)": "Margin from previous cutoff (ms)",
    "다음 오프셋 기준 여유(ms)": "Margin from next offset (ms)",
    "최소 VC 길이(ms)": "Minimum VC length (ms)",
    "일본어 (JA)": "Japanese (JA)",
    "고급 정렬 엔진 옵션은 현재 제공하지 않습니다.": (
        "Advanced alignment engine options are not currently available."
    ),
    "개발자 설정 초기화": "Reset Developer Settings",

    # ── Execution backend ────────────────────────────────────────────────────
    "실행 백엔드": "Execution Backend",
    "(auto=ensemble 우선)": "(auto = ensemble preferred)",

    # ── Alignment profile options ────────────────────────────────────────────
    "멜 앵커 중심": "Mel anchor-centered",
    "빠름": "Fast",
    "빠름 (저사양 추천)": "Fast (recommended for low-spec)",
    "고역대 안정": "High-freq Stable",
    "고역대": "High-freq",
    "정밀": "Precise",
    "정확도 우선": "Accuracy-first",
    "정확도 우선 (정밀)": "Accuracy-first (Precise)",
    "정확도 우선 (기본)": "Accuracy-first (Default)",
    "기본": "Default",

    # ── app_mixins dialogs / messages ────────────────────────────────────────
    "모델 경로 확인": "Model Path Check",
    "모델 경로 일부 적용": "Model Path Partially Applied",
    "복사": "Copy",
    "링크 복사": "Copy Link",
    "닫기": "Close",
    "언어 설정 확인": "Language Setting Check",
    "완료": "Done",
    "오류": "Error",
    "경고": "Warning",
    "성공": "Success",
    "알림": "Notice",
    "안내": "Info",
    "작업이 이미 실행 중입니다. 완료 후 다시 시도해 주세요.": (
        "A task is already running. Please wait for it to finish before trying again."
    ),
    "빠른 오류 리포트": "Quick Error Report",
    "오류 리포트": "Error Report",
    "한국어 의존성 설치": "Install Korean Dependencies",
    "전체 초기화": "Full Reset",
    "경로 유지": "Keep Paths",
    "끄기": "Off",
    "비활성": "Disabled",
    "미사용": "Unused",
    "하이브리드": "Hybrid",
    "한국어": "Korean",
    "일본어": "Japanese",
    "영어": "English",
    "자동 감지 (권장)": "Auto-detect (recommended)",
    "일관성 보정 사용": "Enable consistency correction",
    "연속성 보정 사용": "Enable continuity correction",

    # ── Pipeline action messages ─────────────────────────────────────────────
    "기존 TextGrid 덮어쓰기 확인": "Confirm Overwrite Existing TextGrids",
    "정렬 입력 파일 누락": "Missing Alignment Input Files",
    "정렬": "Alignment",
    "정렬 취소됨": "Alignment Cancelled",
    "다른 작업 진행 중": "Another task is running",
    "사전 생성": "Dictionary Generation",
    "전체 파이프라인": "Full Pipeline",
    "미확인": "Unknown",
    "복구 필요": "Repair Required",
    "준비 완료": "Ready",
    "추가 복구 필요": "Further Repair Required",
    "다운로드 필요": "Download Required",

    # ── OTO action status messages ───────────────────────────────────────────
    "생성 진행": "Generating",
    "검증 진행": "Validating",
    "입력 검사": "Input Check",
    "입력 경로 확인 완료": "Input path verified",
    "생성 준비 완료": "Ready to generate",
    "생성 결과 정리": "Organising results",
    "검증 준비": "Preparing validation",
    "검증 완료": "Validation complete",

    # ── ui_layout_defs.py ────────────────────────────────────────────────────
    "ML 고급 옵션": "ML Advanced Options",
    "OFF면 ML 보정이 비활성화됩니다. No-MFA/v1/v2는 설치된 모델 상태에 따라 동작합니다.": (
        "When OFF, ML correction is disabled. "
        "No-MFA/v1/v2 behaviour depends on the installed model state."
    ),
    "모델이 이미 다운로드되어 있다면 경로를 지정해 주세요. KR/JA 모델이 자동 인식됩니다.": (
        "If models are already downloaded, specify the path. "
        "KR/JA models are auto-detected."
    ),
    "발음 누락 줄이기: 경계가 약한 음원에서 자음/모음 누락을 줄이기 위한 보정입니다.\n오매핑 차단: 경계가 불분명한 구간에서 잘못된 음절 연결을 줄입니다.": (
        "Reduce Phoneme Dropout: correction to reduce consonant/vowel dropout "
        "in sources with weak phoneme boundaries.\n"
        "Block Mis-mapping: reduces incorrect syllable connections in "
        "ambiguous boundary segments."
    ),
}


# ---------------------------------------------------------------------------
# Japanese translation table  (機械翻訳 — 誤りが含まれる場合があります)
# ---------------------------------------------------------------------------

_JA: dict[str, str] = {

    # ── UI language control ─────────────────────────────────────────────────
    "UI 언어": "UI言語",
    "재시작 후 적용됩니다.": "再起動後に適用されます。",

    # ── Tab names ───────────────────────────────────────────────────────────
    "파이프라인": "パイプライン",
    "파라미터 조정": "パラメータ調整",
    "로그": "ログ",
    "상세 로그": "詳細ログ",
    "크레딧": "クレジット",
    "고급 설정": "詳細設定",

    # ── Layout / header ─────────────────────────────────────────────────────
    "언어": "言語",
    "WAV 폴더:": "WAVフォルダ:",
    "WAV 파일이 있는 폴더 경로": "WAVファイルが入ったフォルダのパス",
    "하위 폴더 자동 탐색(배치 처리)": "サブフォルダ自動検索（バッチ処理）",
    "찾아보기": "参照",
    "템플릿 OTO:": "テンプレートOTO:",
    "선택 사항 (없을 시 파일명/라벨 기반 자동 생성)": "任意（未指定時はファイル名/ラベルから自動生成）",
    "템플릿 OTO 없음 (OpenUtau 호환 에일리어스 자동 생성)": "テンプレートOTOなし（OpenUtau互換エイリアス自動生成）",
    "출력 경로:": "出力パス:",
    "생성된 oto.ini 저장 경로": "生成したoto.iniの保存先パス",
    "저장": "保存",
    "형식 지정:": "フォーマット指定:",
    "(템플릿 유무와 무관하게 우선 적용)": "（テンプレートの有無に関わらず優先適用）",
    "JP 에일리어스:": "JPエイリアス:",
    "(일본어 OTO 적용)": "（日本語OTOに適用）",
    "원본 그대로": "そのまま",
    "히라가나": "ひらがな",
    "로마자": "ローマ字",
    "대기 중": "待機中",

    # ── Pipeline tab · run section ──────────────────────────────────────────
    "실행": "実行",
    "순서대로 실행": "順番に実行",
    "OpenUtau 호환 별도 에일리어스 자동 생성": "OpenUtau互換エイリアスを別途自動生成",
    "누락된 모음/모음열(VV) 에일리어스 보완 생성": "不足している母音/VVエイリアスを補完生成",
    "어두/어미 '-' 에일리어스 생성": "語頭/語末 '-' エイリアスを生成",
    "ML 보정은 자동으로 적용되며 모델이 없으면 자동으로 건너뜁니다.": (
        "ML補正は自動適用されます。モデルが見つからない場合は自動的にスキップされます。"
    ),
    "파라미터 조정 탭에서 OTO 보정 계수를 조정할 수 있습니다. 기본값 사용을 권장합니다.": (
        "OTO補正係数は「パラメータ調整」タブで変更できます。デフォルト値の使用を推奨します。"
    ),

    # ── MFA setup hint ──────────────────────────────────────────────────────
    "처음 설치나 'MFA 진단/복구'는 환경 구성과 현재 언어 모델 다운로드 때문에 10~20분 이상 걸릴 수 있습니다.\n문제가 생기면 먼저 'MFA 진단/복구'를 눌러 자동 복구를 시도한 뒤 정렬을 다시 실행하세요.": (
        "初回インストールや「MFA診断/修復」は、環境構築と言語モデルのダウンロードのため"
        "10〜20分以上かかる場合があります。\n"
        "問題が発生した場合は、まず「MFA診断/修復」を押して自動修復を試みてから"
        "アライメントを再実行してください。"
    ),

    # ── Log tab ─────────────────────────────────────────────────────────────
    "로그 지우기": "ログをクリア",
    "로그 파일 열기": "ログファイルを開く",
    "CMD/서브프로세스 원본 로그를 시간순으로 표시합니다. (저부하 버퍼 플러시)": (
        "CMD/サブプロセスの生ログを時系列で表示します。（低負荷バッファフラッシュ）"
    ),
    "상세 로그 지우기": "詳細ログをクリア",

    # ── Credits tab ─────────────────────────────────────────────────────────
    "-보정 모델 데이터 제공에 도움 주신 분들": "補正モデルデータ提供にご協力いただいた方々",
    "감사합니다!": "ありがとうございます！",

    # ── Advanced Settings tab · core options ────────────────────────────────
    "고급 설정은 개발/튜닝용 항목입니다. 실사용에서는 기본값 유지 후 필요한 경우만 조정하세요.": (
        "詳細設定は開発・チューニング用の項目です。"
        "実運用ではデフォルト値を維持し、必要な場合のみ調整してください。"
    ),
    "핵심 옵션": "コアオプション",
    "ML 보정 사용": "ML補正を使用",
    "연속성/파일 일관성 보정 사용": "連続性/ファイル一貫性補正を使用",
    "형식 변경 시 추천값이 자동 적용됩니다. 필요한 경우 아래 개발자 슬라이더에서 세부값만 미세 조정하세요.": (
        "フォーマット変更時に推奨値が自動適用されます。"
        "必要な場合は、下の開発者スライダーで詳細値のみ微調整してください。"
    ),

    # ── Post-processing options ──────────────────────────────────────────────
    "후처리(연속성) 옵션": "後処理（連続性）オプション",
    "연속성 offset 보정 상한 (ms)": "連続性オフセット補正の上限 (ms)",
    "기본값(180)": "デフォルト(180)",
    "크게 잡을수록 연속성 보정이 강해지고, 작게 잡을수록 보정이 약해집니다.": (
        "値が大きいほど連続性補正が強くなり、小さいほど弱くなります。"
    ),

    # ── Voicebank tuning presets ─────────────────────────────────────────────
    "보이스뱅크 튜닝 프리셋": "音声ライブラリ調整プリセット",
    "soft/일반 뱅크용 프리셋을 한 번에 적용합니다. 민감형은 보정을 강하게, 일반은 안정적으로 동작합니다.": (
        "ソフト/通常ライブラリ用プリセットを一括適用します。"
        "強補正は補正が強く、標準は安定動作します。"
    ),
    "soft 뱅크 · 민감형": "Softbank · 強補正",
    "soft 뱅크 · 일반": "Softbank · 標準",
    "일반 뱅크 · 민감형": "通常バンク · 強補正",
    "일반 뱅크 · 일반": "通常バンク · 標準",

    # ── Phoneme dropout / weak-voice options ────────────────────────────────
    "발음 누락 줄이기 추천": "音素抜け軽減（推奨）",
    "숨소리가 많이 섞이거나 발성이 약한 음원에서 발음 누락과 공백 오검출을 줄일 때 켜는 옵션입니다.": (
        "息漏れが多い、または発声が弱い音源で音素抜けと無音誤検出を軽減するオプションです。"
    ),
    "저볼륨 WAV 자동 증폭/정규화 (RMS 기반, 원본 보존)": (
        "低音量WAVを自動増幅/正規化（RMSベース、原音保持）"
    ),
    "정렬 단계에서만 임시 작업 복사본에 볼륨 보정을 적용합니다. 원본 WAV 파일은 절대 변경하지 않습니다.": (
        "アライメント段階でのみ一時作業コピーに音量補正を適用します。原音WAVファイルは変更しません。"
    ),
    "발음이 흐린 음원에 추천 (자모 구분 강화)": (
        "発声が不明瞭な音源に推奨（子音/母音識別強化）"
    ),
    "웅얼거림처럼 들리거나 자음/모음 구분이 흐린 음원에서 정렬 정확도를 높이기 위한 추천 옵션입니다. (원본 WAV 보존)": (
        "こもった声や子音/母音の区別が不明瞭な音源でアライメント精度を向上させる推奨オプションです。（原音WAV保持）"
    ),
    "자모 대비 강도": "子音/母音コントラスト強度",

    # ── Mis-mapping block ────────────────────────────────────────────────────
    "음절 오매핑 차단": "音節ミスマッピング防止",
    "완전 엄격": "完全厳密",
    "적당히 엄격(누락 행은 폴백)": "中程度の厳密（欠損行はフォールバック）",
    "음절을 틀릴 가능성을 줄이는 대신, 자동 설정에서 스킵되는 에일리어스가 늘어날 수 있습니다.": (
        "音節の誤り可能性を減らす代わりに、自動設定でスキップされるエイリアスが増える場合があります。"
    ),
    "저신뢰 구간 강제 고정 모드": "低信頼区間強制固定モード",
    "정렬 신뢰가 낮은 구간에서는 후보 점프를 막고 예상 위치에 고정합니다.": (
        "アライメント信頼度が低い区間では候補ジャンプを防ぎ、予想位置に固定します。"
    ),

    # ── Inline quick options ─────────────────────────────────────────────────
    "발음 누락 줄이기": "音素抜け軽減",
    "오매핑 차단": "ミスマッピング防止",

    # ── CVN / mapping options ────────────────────────────────────────────────
    "자음 구분기(C/V) 보정 사용": "子音判別器(C/V)補正を使用",
    "정렬 저신뢰 샘플에만 C/V 후보정 적용": (
        "アライメント低信頼サンプルにのみC/V後補正を適用"
    ),
    "매핑 모델(단조 디코딩 리스코어) 사용": "マッピングモデル（単調デコードリスコア）を使用",
    "매핑 모델 모드": "マッピングモデルモード",
    "자동(권장)": "自動（推奨）",
    "포맷 우선": "フォーマット優先",
    "전역 우선": "グローバル優先",
    "CV 파일 순서 기반 매핑 결합 사용": "CVファイル順序ベースのマッピング結合を使用",
    "강도": "強度",
    "기본값": "デフォルト",
    "(0.0~1.0, 높을수록 순서 prior 강화)": "（0.0〜1.0、高いほど順序優先が強まる）",

    # ── ML correction mode ───────────────────────────────────────────────────
    "ML 보정 모드": "ML補正モード",
    "상대적 보정": "相対補正",
    "+셀렉터": "+セレクタ",
    "ML 방식": "ML方式",
    "자동(자동 라우팅)": "自動（自動ルーティング）",
    "E2E 하이브리드(실험)": "E2Eハイブリッド（実験的）",
    "자동=환경 기반 라우팅, No-MFA=정렬 없이 보정, v1=기존, v2=확장, E2E=실험": (
        "自動=環境ベースルーティング  |  No-MFA=アライメントなし補正  |  "
        "v1=従来  |  v2=拡張  |  E2E=実験的"
    ),

    # ── Coupled ML options ───────────────────────────────────────────────────
    "멜+OTO 보정 사용": "Mel+OTO補正を使用",
    "Coupled 최소 신뢰도": "Coupled最小信頼度",
    "Strict 제약": "Strict制約",
    "Model-aware min_conf (model_meta) 사용": "モデル対応min_conf (model_meta)を使用",
    "(range: -0.30 ~ +0.30)": "（range: −0.30〜+0.30）",
    "Batch inference (coupled v2) 사용": "バッチ推論（coupled v2）を使用",
    "(권장 128~512, 최소 32)": "（推奨128〜512、最小32）",
    "v1 fallback(lightgbm) 체인 사용": "v1フォールバック（LightGBM）チェーンを使用",
    "하이브리드 라우팅(가중치 게이트) 사용": "ハイブリッドルーティング（重みゲート）を使用",
    "E2E 하이브리드(실험) 활성화": "E2Eハイブリッド（実験的）を有効化",
    "E2E 모드": "E2Eモード",
    "(권장: hybrid)": "（推奨: hybrid）",
    "(&gt;1.0: stronger shrink on low mel reliability)": (
        "（>1.0: mel信頼度低時の圧縮を強化）"
    ),

    # ── Model paths ──────────────────────────────────────────────────────────
    "자세히 보기 (경로/버전/생성일)": "詳細を表示（パス/バージョン/作成日）",
    "모델 경로 (한국어)": "モデルパス（韓国語）",
    "모델 경로 (일본어)": "モデルパス（日本語）",
    "경로는 모델 폴더(내부에 model_meta.json) 또는 상위 루트를 지정할 수 있습니다.": (
        "パスはモデルフォルダ（内部にmodel_meta.json）またはその親ルートを指定できます。"
    ),
    "열기": "開く",

    # ── VC correction section ────────────────────────────────────────────────
    "VC 보정": "VC補正",
    "기본 UI에서는 보정 사용 여부만 선택합니다. 세부 슬라이더는 개발자 설정 ON일 때만 활성화됩니다.": (
        "基本UIでは補正の有効/無効のみ選択できます。詳細スライダーは開発者設定ONの時のみ有効です。"
    ),
    "한국어 (KR)": "韓国語 (KR)",
    "VC 이웃 보정 사용": "VC隣接補正を使用",
    "보정 강도(Blend)": "補正強度（Blend）",
    "최대 이동(ms)": "最大移動量 (ms)",
    "이전 컷오프 기준 여유(ms)": "前のカットオフ基準マージン (ms)",
    "다음 오프셋 기준 여유(ms)": "次のオフセット基準マージン (ms)",
    "최소 VC 길이(ms)": "最小VC長 (ms)",
    "일본어 (JA)": "日本語 (JA)",
    "고급 정렬 엔진 옵션은 현재 제공하지 않습니다.": (
        "高度なアライメントエンジンオプションは現在提供していません。"
    ),
    "개발자 설정 초기화": "開発者設定をリセット",

    # ── Execution backend ────────────────────────────────────────────────────
    "실행 백엔드": "実行バックエンド",
    "(auto=ensemble 우선)": "（auto=アンサンブル優先）",

    # ── Alignment profile options ────────────────────────────────────────────
    "멜 앵커 중심": "メルアンカー中心",
    "빠름": "高速",
    "빠름 (저사양 추천)": "高速（低スペック推奨）",
    "고역대 안정": "高周波数安定",
    "고역대": "高周波数",
    "정밀": "精密",
    "정확도 우선": "精度優先",
    "정확도 우선 (정밀)": "精度優先（精密）",
    "정확도 우선 (기본)": "精度優先（デフォルト）",
    "기본": "デフォルト",

    # ── app_mixins dialogs / messages ────────────────────────────────────────
    "모델 경로 확인": "モデルパス確認",
    "모델 경로 일부 적용": "モデルパス一部適用",
    "복사": "コピー",
    "링크 복사": "リンクをコピー",
    "닫기": "閉じる",
    "언어 설정 확인": "言語設定確認",
    "완료": "完了",
    "오류": "エラー",
    "경고": "警告",
    "성공": "成功",
    "알림": "通知",
    "안내": "案内",
    "작업이 이미 실행 중입니다. 완료 후 다시 시도해 주세요.": (
        "タスクが既に実行中です。完了後に再試行してください。"
    ),
    "빠른 오류 리포트": "クイックエラーレポート",
    "오류 리포트": "エラーレポート",
    "한국어 의존성 설치": "韓国語依存関係のインストール",
    "전체 초기화": "完全リセット",
    "경로 유지": "パスを保持",
    "끄기": "オフ",
    "비활성": "無効",
    "미사용": "未使用",
    "하이브리드": "ハイブリッド",
    "한국어": "韓国語",
    "일본어": "日本語",
    "영어": "英語",
    "자동 감지 (권장)": "自動検出（推奨）",
    "일관성 보정 사용": "一貫性補正を使用",
    "연속성 보정 사용": "連続性補正を使用",

    # ── Pipeline action messages ─────────────────────────────────────────────
    "기존 TextGrid 덮어쓰기 확인": "既存のTextGridを上書き確認",
    "정렬 입력 파일 누락": "アライメント入力ファイルが見つかりません",
    "정렬": "アライメント",
    "정렬 취소됨": "アライメントがキャンセルされました",
    "다른 작업 진행 중": "別のタスクが実行中",
    "사전 생성": "辞書生成",
    "전체 파이프라인": "フルパイプライン",
    "미확인": "未確認",
    "복구 필요": "修復が必要",
    "준비 완료": "準備完了",
    "추가 복구 필요": "追加修復が必要",
    "다운로드 필요": "ダウンロードが必要",

    # ── OTO action status messages ───────────────────────────────────────────
    "생성 진행": "生成中",
    "검증 진행": "検証中",
    "입력 검사": "入力チェック",
    "입력 경로 확인 완료": "入力パス確認完了",
    "생성 준비 완료": "生成準備完了",
    "생성 결과 정리": "生成結果を整理中",
    "검증 준비": "検証準備中",
    "검증 완료": "検証完了",

    # ── ui_layout_defs.py ────────────────────────────────────────────────────
    "ML 고급 옵션": "ML詳細オプション",
    "OFF면 ML 보정이 비활성화됩니다. No-MFA/v1/v2는 설치된 모델 상태에 따라 동작합니다.": (
        "OFFにするとML補正が無効になります。No-MFA/v1/v2はインストール済みモデルの状態により動作します。"
    ),
    "모델이 이미 다운로드되어 있다면 경로를 지정해 주세요. KR/JA 모델이 자동 인식됩니다.": (
        "モデルがすでにダウンロード済みの場合はパスを指定してください。KR/JAモデルが自動認識されます。"
    ),
    "발음 누락 줄이기: 경계가 약한 음원에서 자음/모음 누락을 줄이기 위한 보정입니다.\n오매핑 차단: 경계가 불분명한 구간에서 잘못된 음절 연결을 줄입니다.": (
        "音素抜け軽減: 境界が弱い音源で子音/母音の抜けを軽減するための補正です。\n"
        "ミスマッピング防止: 境界が不明瞭な区間で誤った音節接続を減らします。"
    ),
}
