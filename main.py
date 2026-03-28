"""
UTAU Auto OTO - 한국어/일본어 UTAU 음원 자동 OTO 생성기
윈도우 독립 실행형 GUI 프로그램 (CustomTkinter 기반)
"""

import os
import sys
import datetime
import json
import logging
import tempfile
import threading
import traceback
import tkinter as tk


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

def _sanitize_python_env_for_children() -> None:
    """
    Prevent host Python (e.g., 3.11) env leakage into subprocesses.
    This avoids mixed DLL/site-packages conflicts when MFA env is Python 3.10.
    """
    for key in (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONEXECUTABLE",
        "__PYVENV_LAUNCHER__",
        "VIRTUAL_ENV",
        "CONDA_DEFAULT_ENV",
        "CONDA_PROMPT_MODIFIER",
    ):
        os.environ.pop(key, None)
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


_sanitize_python_env_for_children()

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
from ui.theme_tokens import (
    DEFAULT_THEME_PROFILE,
    THEME_PROFILE_OPTIONS,
    apply_theme_profile,
    get_theme_appearance_mode,
    normalize_theme_profile_name,
)

# ==============================================================================
# 로깅 설정
# ==============================================================================

if getattr(sys, 'frozen', False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _contains_model_meta(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    try:
        for root, _dirs, files in os.walk(path):
            if "model_meta.json" in files:
                return True
    except Exception:
        return False
    return False


def _set_env_if_unset(env_key: str, candidates):
    if str(os.environ.get(env_key, "") or "").strip():
        return
    for candidate in candidates:
        if _contains_model_meta(candidate):
            os.environ[env_key] = candidate
            return


def _apply_external_model_env_defaults(app_dir: str) -> None:
    """
    If a standalone runtime model folder exists next to the app (./models),
    use it as default without requiring manual env setup.
    """
    models_root = os.path.join(app_dir, "models")
    if not os.path.isdir(models_root):
        return

    _set_env_if_unset(
        "UTOA_KR_OTO_ML_DIR",
        [
            os.path.join(models_root, "korean"),
            os.path.join(models_root, "oto_ml", "korean"),
        ],
    )
    _set_env_if_unset(
        "UTOA_JA_OTO_ML_DIR",
        [
            os.path.join(models_root, "japanese"),
            os.path.join(models_root, "oto_ml", "japanese"),
        ],
    )
    _set_env_if_unset(
        "UTOA_KR_AUTOFREE_ML_DIR",
        [
            os.path.join(models_root, "autofree", "korean"),
            os.path.join(models_root, "korean", "autofree"),
        ],
    )
    _set_env_if_unset(
        "UTOA_JA_AUTOFREE_ML_DIR",
        [
            os.path.join(models_root, "autofree", "japanese"),
            os.path.join(models_root, "japanese", "autofree"),
        ],
    )


_apply_external_model_env_defaults(APP_DIR)


def _resolve_writable_data_dir(app_dir):
    candidates = []
    env_override = str(os.environ.get("UTOA_APP_DATA_DIR", "") or "").strip()
    if env_override:
        candidates.append(env_override)
    local_app_data = str(os.environ.get("LOCALAPPDATA", "") or "").strip()
    if local_app_data:
        candidates.append(os.path.join(local_app_data, "UTAU_Auto_OTO_v3"))
    else:
        candidates.append(os.path.join(os.path.expanduser("~"), "AppData", "Local", "UTAU_Auto_OTO_v3"))
    candidates.append(app_dir)

    seen = set()
    for candidate in candidates:
        norm = os.path.normcase(os.path.abspath(str(candidate or "")))
        if not norm or norm in seen:
            continue
        seen.add(norm)
        try:
            os.makedirs(candidate, exist_ok=True)
            fd, probe = tempfile.mkstemp(prefix=".utoa_write_probe_", dir=candidate)
            os.close(fd)
            os.remove(probe)
            return candidate
        except Exception:
            continue
    return app_dir


def _config_path_candidates(app_dir, data_dir):
    candidates = [os.path.join(data_dir, "config.json"), os.path.join(app_dir, "config.json")]
    dedup = []
    seen = set()
    for path in candidates:
        norm = os.path.normcase(os.path.abspath(str(path or "")))
        if not norm or norm in seen:
            continue
        seen.add(norm)
        dedup.append(path)
    return dedup


APP_DATA_DIR = _resolve_writable_data_dir(APP_DIR)
LOG_DIR = os.path.join(APP_DATA_DIR, 'logs')
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
APP_VERSION = "2.1.3"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 760
SUPPORTED_RELEASE_CHANNELS = {"stable", "preview"}


def _load_release_channel_name(app_dir):
    channel_path = os.path.join(app_dir, "release_channel.json")
    if not os.path.exists(channel_path):
        return "stable"
    try:
        # Windows PowerShell Set-Content 기본 UTF-8(BOM)도 허용합니다.
        with open(channel_path, "r", encoding="utf-8-sig") as f:
            payload = json.load(f)
        channel = str(payload.get("channel", "stable")).strip().lower()
        if channel in SUPPORTED_RELEASE_CHANNELS:
            return channel
    except Exception:
        pass
    return "stable"


RELEASE_CHANNEL = _load_release_channel_name(APP_DIR)
CHANNEL_TITLE_SUFFIX = " [Preview]" if RELEASE_CHANNEL == "preview" else ""


def _load_startup_theme_profile(app_dir, data_dir):
    for config_path in _config_path_candidates(app_dir, data_dir):
        if not os.path.exists(config_path):
            continue
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            candidate = normalize_theme_profile_name(payload.get("ui_theme_profile", DEFAULT_THEME_PROFILE))
            if candidate in THEME_PROFILE_OPTIONS:
                return candidate
        except Exception:
            continue
    return DEFAULT_THEME_PROFILE


STARTUP_THEME_PROFILE = _load_startup_theme_profile(APP_DIR, APP_DATA_DIR)

LANGUAGE_OPTIONS = [
    "한국어 (CV/연단음/CVVC/연속음 자동 매핑)",
    "日本語 (CV/연단음/CVVC/연속음 자동 매핑)",
    "English (Preview CVVC)",
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

        self.title(f"{APP_NAME} v{APP_VERSION}{CHANNEL_TITLE_SUFFIX}")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.minsize(760, 560)

        self.ui_theme_var = ctk.StringVar(value=STARTUP_THEME_PROFILE)
        selected_profile = apply_theme_profile(self.ui_theme_var.get())
        ctk.set_appearance_mode(get_theme_appearance_mode(selected_profile))
        ctk.set_default_color_theme("blue")

        # MFA 경로는 UI 표시를 막지 않도록 백그라운드에서 탐색합니다.
        self.mfa_path = ""
        self._mfa_path_probe_pending = True
        self._mfa_path_probe_started = False
        self._mfa_install_in_progress = False
        self._startup_loading_win = None
        self._startup_loading_label = None
        self._show_startup_loading_window()
        
        # OpenUtau 호환 에일리어스 생성 여부
        self.openutau_var = ctk.BooleanVar(value=False)
        self.gen_missing_vowels_var = ctk.BooleanVar(value=True)
        self.gen_dash_alias_var = ctk.BooleanVar(value=True)
        self.no_base_oto_var = ctk.BooleanVar(value=False)
        self.recursive_voicebank_scan_var = ctk.BooleanVar(value=False)
        self.enable_ml_correction_var = ctk.BooleanVar(value=True)
        self.ml_route_var = ctk.StringVar(value="자동(자동 라우팅)")
        self.ml_selector_mode_var = ctk.StringVar(value="델타+셀렉터")
        self.ml_coupled_enable_var = ctk.BooleanVar(value=True)
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
        self.ml_coupled_strict_constraint_var = ctk.BooleanVar(value=True)
        self.ml_batch_inference_enable_var = ctk.BooleanVar(value=True)
        self.ml_batch_inference_size_var = ctk.StringVar(value="256")
        self.ml_legacy_fallback_enable_var = ctk.BooleanVar(value=False)
        self.ml_hybrid_routing_enable_var = ctk.BooleanVar(value=True)
        self.ml_anchor_mel_gamma_var = ctk.StringVar(value="1.2")
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
        self.kr_continuity_enable_var = ctk.BooleanVar(value=True)
        self.kr_continuity_max_offset_adj_var = ctk.StringVar(value="")
        self.vc_correction_enable_var = ctk.BooleanVar(value=True)
        self.cvn_correction_enable_var = ctk.BooleanVar(value=True)
        self.cvn_low_conf_only_var = ctk.BooleanVar(value=False)
        self.soft_bank_mode_var = ctk.BooleanVar(value=False)
        self.low_rms_gain_enable_var = ctk.BooleanVar(value=True)
        self.weak_voice_assist_enable_var = ctk.BooleanVar(value=True)
        self.weak_voice_assist_strength_var = ctk.StringVar(value="")
        self.mapping_strict_mode_var = ctk.StringVar(value="적당히 엄격 모드(누락 행은 폴백)")
        self.weak_boundary_reduce_missing_var = ctk.BooleanVar(value=False)
        self.weak_boundary_block_mismap_var = ctk.BooleanVar(value=False)
        self.ml_same_language_borrow_only_var = ctk.BooleanVar(value=True)
        self.developer_mode_enabled_var = ctk.BooleanVar(value=False)
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
        self.en_cvvc_pack_var = ctk.StringVar(value="LITE")
        self.en_cvvc_beat_var = ctk.StringVar(value="8-beat")
        self.en_cvvc_preset_var = ctk.StringVar(value="Core")
        self.en_cvvc_list_fallback_var = ctk.BooleanVar(value=True)
        self.aligner_var = ctk.StringVar(value="MFA")
        self.no_mfa_oto_mode_var = ctk.StringVar(value="베이스 OTO 재매핑 + 보정")
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
        self.app_data_dir = APP_DATA_DIR
        self.writable_data_dir = APP_DATA_DIR
        self.log_path = LOG_PATH
        self.event_log_path = EVENT_LOG_PATH
        self.logger = logger
        self.app_version = APP_VERSION
        self.release_channel = RELEASE_CHANNEL
        self.protocol("WM_DELETE_WINDOW", self._on_close_request)
        self._post_ui_startup_done = False
        self._ui_bootstrap_done = False
        self.after(0, self._bootstrap_ui)

        logger.info(f"{APP_NAME} v{APP_VERSION} 시작")
        logger.info(f"릴리스 채널: {RELEASE_CHANNEL}")
        logger.info(f"앱 데이터 경로: {APP_DATA_DIR}")
        logger.info(f"구조화 이벤트 로그: {EVENT_LOG_PATH}")
        if getattr(self, "_mfa_path_probe_pending", False):
            logger.info("MFA 경로 확인 중 (백그라운드).")
        elif self.mfa_path:
            logger.info(f"MFA 경로: {self.mfa_path}")
        else:
            logger.warning("MFA를 찾을 수 없습니다.")

    def _bootstrap_ui(self):
        if bool(getattr(self, "_ui_bootstrap_done", False)):
            return
        self._ui_bootstrap_done = True

        try:
            self._build_ui()
            self._start_async_mfa_path_probe()
            if hasattr(self, "_install_adaptive_ui_scaling"):
                self._install_adaptive_ui_scaling()
            if hasattr(self, "_install_global_exception_hooks"):
                self._install_global_exception_hooks()
            self.after(0, self._run_post_ui_startup_tasks)
        finally:
            self._hide_startup_loading_window()

    def _show_startup_loading_window(self):
        if getattr(self, "_startup_loading_win", None) is not None:
            return
        try:
            self.withdraw()
        except Exception:
            pass
        win = tk.Toplevel(self)
        win.title(f"{APP_NAME} 로딩 중")
        win.resizable(False, False)
        width, height = 420, 220
        win.geometry(f"{width}x{height}")
        try:
            win.attributes("-topmost", True)
        except Exception:
            pass
        try:
            win.update_idletasks()
            screen_w = win.winfo_screenwidth()
            screen_h = win.winfo_screenheight()
            x = max(0, int((screen_w - width) / 2))
            y = max(0, int((screen_h - height) / 2))
            win.geometry(f"{width}x{height}+{x}+{y}")
        except Exception:
            pass

        frame = tk.Frame(win, bg="white")
        frame.pack(fill="both", expand=True)
        label = tk.Label(
            frame,
            text="로딩 중... 잠시만 기다려 주세요.",
            font=("Segoe UI", 12, "bold"),
            bg="white",
        )
        label.pack(expand=True)

        self._startup_loading_win = win
        self._startup_loading_label = label

    def _hide_startup_loading_window(self):
        win = getattr(self, "_startup_loading_win", None)
        if win is not None:
            try:
                win.destroy()
            except Exception:
                pass
        self._startup_loading_win = None
        self._startup_loading_label = None
        try:
            self.deiconify()
            self.lift()
        except Exception:
            pass

    def _run_post_ui_startup_tasks(self):
        if bool(getattr(self, "_post_ui_startup_done", False)):
            return
        self._post_ui_startup_done = True

        try:
            self._load_config()
        except Exception:
            logger.exception("초기 설정 로드 중 예외가 발생했습니다.")

        try:
            self._schedule_startup_mfa_auto_repair()
        except Exception:
            logger.exception("초기 MFA 자동 복구 예약 중 예외가 발생했습니다.")

        try:
            self._schedule_startup_cuda_runtime_check()
        except Exception:
            logger.exception("초기 CUDA 런타임 점검 예약 중 예외가 발생했습니다.")

    def _start_async_mfa_path_probe(self):
        if bool(getattr(self, "_mfa_path_probe_started", False)):
            return
        self._mfa_path_probe_started = True

        def _worker():
            resolved = ""
            try:
                resolved = find_mfa_executable() or ""
            except Exception:
                resolved = ""

            def _apply():
                self.mfa_path = resolved
                self._mfa_path_probe_pending = False
                try:
                    self._update_mfa_status(bool(self.mfa_path))
                except Exception:
                    pass
                try:
                    self._sync_aligner_ui()
                except Exception:
                    pass
                if self.mfa_path:
                    logger.info(f"MFA 경로: {self.mfa_path}")
                else:
                    logger.warning("MFA를 찾을 수 없습니다.")

            try:
                self.after(0, _apply)
            except Exception:
                pass

        thread = threading.Thread(target=_worker, name="mfa-path-probe", daemon=True)
        thread.start()


def _write_startup_crash_log(exc):
    try:
        crash_name = datetime.datetime.now().strftime("startup_crash_%Y%m%d_%H%M%S.txt")
        crash_path = os.path.join(LOG_DIR, crash_name)
        with open(crash_path, "w", encoding="utf-8") as f:
            f.write(f"APP: {APP_NAME} v{APP_VERSION}\n")
            f.write(f"CHANNEL: {RELEASE_CHANNEL}\n")
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
