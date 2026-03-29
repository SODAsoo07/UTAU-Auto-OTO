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
    "발음 누락 줄이기: 발음 경계가 흐릿한 음원에서 자음/모음 누락을 줄이기 위한 보정입니다.\n"
    "오매핑 차단: 음절 경계가 섞여 잘못 매핑되는 경우를 줄이기 위한 보정입니다."
)

ADVANCED_ML_SECTION_TITLE = "ML 보정"
ADVANCED_ML_SECTION_SUBTITLE = "OFF 시 ML 보정은 자동으로 생략되며, No-MFA/v1/v2 라우팅 설정은 유지됩니다."

PIPELINE_MODEL_QUICK_HELP = (
    "모델이 설치된 폴더 또는 상위 루트를 지정하면 KR/JA 모델을 자동 탐색합니다."
)
