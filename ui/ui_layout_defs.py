# -*- coding: utf-8 -*-

ADVANCED_WEAK_BOUNDARY_OPTIONS = [
    {
        "text": "발음 누락 줄이기",
        "var": "weak_boundary_reduce_missing_var",
        "padx": (10, 6),
    },
    {
        "text": "오매핑 차단",
        "var": "weak_boundary_block_mismap_var",
        "padx": (4, 0),
    },
]

ADVANCED_WEAK_BOUNDARY_HELP = (
    "발음 누락 줄이기: 경계가 약한 음원에서 자음/모음 누락을 줄이기 위한 보정입니다.\n"
    "오매핑 차단: 경계가 불분명한 구간에서 잘못된 음절 연결을 줄입니다."
)

ADVANCED_ML_SECTION_TITLE = "ML 고급 옵션"
ADVANCED_ML_SECTION_SUBTITLE = (
    "OFF면 ML 보정이 비활성화됩니다. No-MFA/v1/v2는 설치된 모델 상태에 따라 동작합니다."
)

PIPELINE_MODEL_QUICK_HELP = (
    "모델이 이미 다운로드되어 있다면 경로를 지정해 주세요. KR/JA 모델이 자동 인식됩니다."
)
