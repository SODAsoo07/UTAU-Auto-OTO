"""
UTAU Auto OTO - 한국어/일본어 UTAU 음원 자동 OTO 생성기
윈도우 독립 실행형 GUI 프로그램 (CustomTkinter 기반)
"""

import os
import sys
import datetime
import logging
import traceback


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
from core.sofa_runner import get_sofa_env_python
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
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 750

LANGUAGE_OPTIONS = [
    "한국어 (CVC/CVVC/연속음/연단음 자동 매핑)",
    "日本語 (CVVC/연속음/연단음 자동 매핑)"
]


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
        self.minsize(800, 600)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # MFA 경로
        self.mfa_path = find_mfa_executable()
        
        # OpenUtau 호환 에일리어스 생성 여부
        self.openutau_var = ctk.BooleanVar(value=False)
        self.gen_missing_vowels_var = ctk.BooleanVar(value=True)
        self.no_base_oto_var = ctk.BooleanVar(value=False)
        self.enable_ml_correction_var = ctk.BooleanVar(value=True)
        self.enable_pytorch_bridge_var = ctk.BooleanVar(value=False)
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
        self.sofa_ckpt_var = ctk.StringVar(value="")
        self.sofa_dict_var = ctk.StringVar(value="")
        self.sofa_python_var = ctk.StringVar(value=get_sofa_env_python())

        self.app_dir = APP_DIR
        self.log_path = LOG_PATH
        self.logger = logger
        self.app_version = APP_VERSION
        self._build_ui()
        self._load_config()
        self.protocol("WM_DELETE_WINDOW", self._on_close_request)

        logger.info(f"{APP_NAME} v{APP_VERSION} 시작")
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
