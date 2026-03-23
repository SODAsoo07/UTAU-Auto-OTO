"""
UTAU Auto OTO - 한국어/일본어 UTAU 음원 자동 OTO 생성기
윈도우 독립 실행형 GUI 프로그램 (CustomTkinter 기반)
"""

import os
import sys
import datetime
import logging
import traceback
import json


def _suppress_windows_loader_popup():
    """
    DLL 누락/로드 실패 시 Windows 시스템 팝업(오류 대화상자)을 억제합니다.
    실행 자체는 계속되며, 오류는 로그/예외로 처리됩니다.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        SEM_FAILCRITICALERRORS = 0x0001
        SEM_NOOPENFILEERRORBOX = 0x8000
        SEM_NOGPFAULTERRORBOX = 0x0002
        flags = SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX | SEM_NOGPFAULTERRORBOX
        prev = kernel32.SetErrorMode(flags)
        try:
            # 지원되는 환경에서는 기존 모드와 병합해 현재 스레드에도 동일 모드 적용
            set_thread_error_mode = kernel32.SetThreadErrorMode
            old_mode = ctypes.c_uint(0)
            set_thread_error_mode(ctypes.c_uint(prev | flags), ctypes.byref(old_mode))
        except Exception:
            # 구형 환경에서는 SetErrorMode만으로 충분
            pass
    except Exception:
        pass


_suppress_windows_loader_popup()

try:
    import customtkinter as ctk
except ImportError:
    if getattr(sys, "frozen", False):
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            None,
            "Startup failed: customtkinter is missing in the bundled executable.\nPlease rebuild the app.",
            "UTAU Auto OTO - Startup Error",
            0x10,
        )
        raise
    print("customtkinter is missing. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "customtkinter"])
    import customtkinter as ctk

from core.mfa_runner import find_mfa_executable
from ui.app_mixins import AppRuntimeMixin, ConfigMixin, FileDialogMixin
from ui.align_actions_mixin import AlignActionsMixin
from ui.oto_actions_mixin import OtoActionsMixin
from ui.pipeline_actions_mixin import PipelineActionsMixin
from ui.tab_builders_mixin import TabBuildersMixin
from ui.layout_mixin import LayoutMixin

# ==============================================================================
# 로깅 설정
# ==============================================================================

if getattr(sys, 'frozen', False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(APP_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = datetime.datetime.now().strftime('log_%Y%m%d_%H%M%S.txt')
LOG_PATH = os.path.join(LOG_DIR, log_filename)
event_log_filename = datetime.datetime.now().strftime('events_%Y%m%d_%H%M%S.jsonl')
EVENT_LOG_PATH = os.path.join(LOG_DIR, event_log_filename)

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 앱 상수
# ==============================================================================

APP_NAME = "UTAU Auto OTO Generator"
APP_VERSION = "2.0.0"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 760

LANGUAGE_OPTIONS = [
    "한국어 (CV/연단음/CVVC/연속음 자동 매핑)",
    "日本語 (CV/연단음/CVVC/연속음 자동 매핑)"
]


def _read_release_channel(app_dir: str) -> str:
    """
    Read packaged release channel metadata.
    Default to stable so preview builds inherit stable feature set unless
    explicitly enabled.
    """
    meta_path = os.path.join(str(app_dir or ""), "release_channel.json")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        channel = str(payload.get("channel", "") or "").strip().lower()
        if channel in {"stable", "preview"}:
            return channel
    except Exception:
        pass
    return "stable"


class App(
    LayoutMixin,
    FileDialogMixin,
    AppRuntimeMixin,
    ConfigMixin,
    TabBuildersMixin,
    AlignActionsMixin,
    OtoActionsMixin,
    PipelineActionsMixin,
    ctk.CTk,
):
    def __init__(self):
        super().__init__()

        self.title(f"{APP_NAME} v{APP_VERSION}")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.minsize(860, 700)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # MFA 경로
        self.mfa_path = find_mfa_executable()
        
        # OpenUtau 호환 에일리어스 생성 여부
        self.openutau_var = ctk.BooleanVar(value=False)
        self.gen_missing_vowels_var = ctk.BooleanVar(value=True)
        self.gen_dash_alias_var = ctk.BooleanVar(value=True)
        self.no_base_oto_var = ctk.BooleanVar(value=False)
        self.enable_ml_correction_var = ctk.BooleanVar(value=True)
        self.ml_route_var = ctk.StringVar(value="legacy")
        self.ml_selector_mode_var = ctk.StringVar(value="기본 정책")
        self.ml_coupled_enable_var = ctk.BooleanVar(value=True)
        self.ml_two_stage_model_enable_var = ctk.BooleanVar(value=True)
        self.ml_coupled_min_conf_var = ctk.StringVar(value="")
        self.ml_coupled_min_conf_use_model_meta_var = ctk.BooleanVar(value=True)
        self.ml_coupled_min_conf_model_offset_var = ctk.StringVar(value="")
        self.ml_coupled_min_conf_kr_cv_var = ctk.StringVar(value="")
        self.ml_coupled_min_conf_kr_cvc_var = ctk.StringVar(value="")
        self.ml_coupled_min_conf_kr_cvvc_var = ctk.StringVar(value="")
        self.ml_coupled_min_conf_kr_vcv_var = ctk.StringVar(value="")
        self.ml_coupled_min_conf_ja_cv_var = ctk.StringVar(value="")
        self.ml_coupled_min_conf_ja_cvc_var = ctk.StringVar(value="")
        self.ml_coupled_min_conf_ja_cvvc_var = ctk.StringVar(value="")
        self.ml_coupled_min_conf_ja_vcv_var = ctk.StringVar(value="")
        self.ml_coupled_device_var = ctk.StringVar(value="auto")
        self.ml_coupled_backend_var = ctk.StringVar(value="auto")
        self.ml_coupled_strict_constraint_var = ctk.BooleanVar(value=False)
        self.ml_batch_inference_enable_var = ctk.BooleanVar(value=True)
        self.ml_batch_inference_size_var = ctk.StringVar(value="256")
        self.ml_legacy_fallback_enable_var = ctk.BooleanVar(value=False)
        self.ml_anchor_mel_gamma_var = ctk.StringVar(value="1.2")
        self.ml_mel_weight_mode_var = ctk.StringVar(value="기본(auto)")
        self.ml_runtime_preset_var = ctk.StringVar(value="권장(균형)")
        self.ml_model_root_kr_var = ctk.StringVar(value="")
        self.ml_model_root_ja_var = ctk.StringVar(value="")
        self.kr_vc_neighbor_enable_var = ctk.BooleanVar(value=True)
        self.kr_vc_neighbor_blend_var = ctk.StringVar(value="")
        self.kr_vc_neighbor_max_shift_var = ctk.StringVar(value="")
        self.kr_vc_neighbor_lead_ms_var = ctk.StringVar(value="")
        self.kr_vc_neighbor_tail_ms_var = ctk.StringVar(value="")
        self.kr_vc_neighbor_min_len_var = ctk.StringVar(value="")
        self.ja_vc_neighbor_enable_var = ctk.BooleanVar(value=True)
        self.ja_vc_neighbor_blend_var = ctk.StringVar(value="")
        self.ja_vc_neighbor_max_shift_var = ctk.StringVar(value="")
        self.ja_vc_neighbor_lead_ms_var = ctk.StringVar(value="")
        self.ja_vc_neighbor_tail_ms_var = ctk.StringVar(value="")
        self.ja_vc_neighbor_min_len_var = ctk.StringVar(value="")
        self.ja_mapping_words_fallback_enabled_var = ctk.BooleanVar(value=True)
        self.ja_mapping_spn_ratio_threshold_var = ctk.DoubleVar(value=0.35)
        self.ja_mapping_min_vowel_phone_ratio_var = ctk.DoubleVar(value=0.5)
        self.ja_mapping_debug_reason_logging_var = ctk.BooleanVar(value=True)
        self.kr_anchor_profile_path_var = ctk.StringVar(value="")
        self.kr_mapping_confidence_threshold_var = ctk.StringVar(value="")
        self.kr_mapping_max_index_jump_default_var = ctk.IntVar(value=1)
        self.kr_mapping_max_index_jump_high_conf_var = ctk.IntVar(value=2)
        self.kr_continuity_max_offset_adj_var = ctk.StringVar(value="")
        self.kr_uncommon_reclist_stable_mode_var = ctk.BooleanVar(value=False)
        self.ml_same_language_borrow_only_var = ctk.BooleanVar(value=True)
        self.mapping_strict_mode_var = ctk.StringVar(value="off")
        self.advanced_options_expanded = False
        
        # 언어 선택
        self.language_var = ctk.StringVar(value=LANGUAGE_OPTIONS[0])

        # 작업 상태
        self.is_running = False
        self._is_closing = False
        self._shown_alert_keys = set()

        # 사용자 옵션
        self.custom_phoneme_var = ctk.StringVar(value="")  # 커스텀 매핑 파일 경로
        self.alias_suffix_var = ctk.StringVar(value="")    # 에일리어스 접미사 (예: C4)
        self.ja_alias_style_var = ctk.StringVar(value="원본 그대로")
        self.aligner_var = ctk.StringVar(value="MFA")
        self.mfa_align_profile_var = ctk.StringVar(value="기본")
        # WhisperX 런타임 옵션(고급): UI에서 직접 노출하지 않아도 config.json으로 제어 가능
        self.whisperx_profile_var = ctk.StringVar(value="balanced")
        self.whisperx_device_var = ctk.StringVar(value="auto")
        self.whisperx_compute_type_var = ctk.StringVar(value="int8")
        self.whisperx_batch_size_var = ctk.IntVar(value=8)
        self.whisperx_align_model_var = ctk.StringVar(value="")
        self.whisperx_cleanup_intermediate_var = ctk.BooleanVar(value=True)
        self.whisperx_save_debug_json_var = ctk.BooleanVar(value=False)

        self.app_dir = APP_DIR
        self.release_channel = _read_release_channel(APP_DIR)
        self.log_path = LOG_PATH
        self.event_log_path = EVENT_LOG_PATH
        self.logger = logger
        self.app_version = APP_VERSION
        self._build_ui()
        self._load_config()
        self.protocol("WM_DELETE_WINDOW", self._on_close_request)

        logger.info(f"{APP_NAME} v{APP_VERSION} 시작")
        logger.info(f"release_channel={self.release_channel}")
        logger.info(f"구조화 이벤트 로그: {EVENT_LOG_PATH}")
        if self.mfa_path:
            logger.info(f"MFA 경로: {self.mfa_path}")
        else:
            logger.warning("MFA를 찾을 수 없습니다.")


def _write_startup_crash_log(exc):
    try:
        crash_name = datetime.datetime.now().strftime("startup_crash_%Y%m%d_%H%M%S.txt")
        crash_path = os.path.join(LOG_DIR, crash_name)
        with open(crash_path, "w", encoding="utf-8") as f:
            f.write(f"APP: {APP_NAME} v{APP_VERSION}\n")
            f.write(f"TIME: {datetime.datetime.now().isoformat()}\n\n")
            f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        return crash_path
    except Exception:
        return ""


if __name__ == "__main__":
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        crash_log = _write_startup_crash_log(e)
        msg = "Application failed to start."
        if crash_log:
            msg += f"\n\nCrash log:\n{crash_log}"
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, msg, "UTAU Auto OTO - Fatal Error", 0x10)
        except Exception:
            pass
        raise
