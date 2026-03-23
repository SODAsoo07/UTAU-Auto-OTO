import os
import sys
import locale
import subprocess as sp
import json
import re
import time
import threading

from core.alignment_pipeline import run_alignment_with_fallback
from core.en_cvvc_builder import generate_en_cvvc_oto
from core.ja_lab_generator import generate_ja_dictionary, generate_ja_labs
from core.kr_cmpx_preview_builder import (
    generate_kr_cmpx_preview_oto,
    resolve_kr_cmpx_preview_source_oto,
)
from core.ja_oto_generator import generate_ja_oto
from core.lab_generator import generate_dictionary, generate_labs
from core.no_mfa_oto_builder import (
    generate_no_mfa_auto_oto,
    resolve_no_mfa_source_oto,
)
from core.mfa_runner import (
    ALERT_MSVC_REQUIRED,
    MFA_PORTABLE_PYTHON_VERSION,
    check_mfa_model,
    diagnose_mfa_runtime,
    download_mfa_model,
    ensure_mfa_python_packaging_stack,
    ensure_japanese_support,
    ensure_korean_support,
    get_default_mfa_conda_root,
    get_default_mfa_env_dir,
    get_default_mfa_micromamba_exe,
    get_default_mfa_micromamba_root,
    get_mfa_env_python_version,
    find_mfa_executable,
    mfa_env_requires_python_downgrade,
)
from core.oto_generator import generate_oto
from core.pipeline_status import normalize_aligner_name
from core.preflight_common import collect_runtime_preflight_issues
from core.error_codes import format_error_with_recovery
from core.format_type_utils import normalize_auto_format_value
from core.distribution_guard import is_training_paths_enabled
from core.cuda_runtime_bootstrap import (
    collect_cuda_runtime_diagnosis,
    install_torch_cuda_runtime,
)
from core.generation.mapping_runtime import (
    format_mapping_reason_schema_summary,
    format_mapping_summary,
)
from core.pipeline_status import normalize_aligner_name
from core.preflight_common import collect_runtime_preflight_issues
from core.error_codes import format_error_with_recovery
from core.format_type_utils import normalize_auto_format_value
from core.distribution_guard import is_training_paths_enabled
from core.cuda_runtime_bootstrap import collect_cuda_runtime_diagnosis
from core.generation.mapping_runtime import (
    format_mapping_reason_schema_summary,
    format_mapping_summary,
)



class PipelineActionsMixin:
    def _notify_korean_dependency_degraded(self):
        self._append_log(ALERT_MSVC_REQUIRED)
        self._append_log(
            "⚠ 일부 의존성 설치에 실패했습니다. 사용은 가능하지만 정확도에 영향이 있을 수 있으니 C++ 툴을 설치해주세요."
        )
        self._append_log("설치 링크: https://visualstudio.microsoft.com/visual-cpp-build-tools/")

    def _notify_long_install_time(self, target="MFA"):
        title = "MFA 초기 설치 안내"
        message = (
            "처음 MFA를 설치하거나 진단/복구를 실행할 때는 환경 구성, Python 패키지 복구, 현재 언어 모델 다운로드 때문에 시간이 오래 걸릴 수 있습니다.\n\n"
            "환경과 네트워크 속도에 따라 보통 10~20분, 경우에 따라 그 이상 걸릴 수 있습니다.\n"
            "설치 중에는 프로그램을 종료하지 말고 기다려 주세요.\n\n"
            "권장 순서:\n"
            "1) 'MFA 진단/복구' 버튼으로 현재 상태 점검\n"
            "2) 필요한 복구와 모델 다운로드 자동 진행\n"
            "3) 완료 후 정렬 다시 시도"
        )
        alert_key = "install_time_mfa"
        self._append_log(f"ℹ {title}: 설치에 시간이 오래 걸릴 수 있습니다.")
        self._after_safe(
            lambda: self._show_copyable_alert(
                title=title,
                message=message,
                alert_key=alert_key,
            )
        )

    def _resolve_setup_mfa_script_path(self):
        candidates = []
        app_dir = getattr(self, "app_dir", "") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if app_dir:
            candidates.append(os.path.join(app_dir, "setup_mfa.bat"))
        exe_dir = os.path.dirname(os.path.abspath(getattr(sys, "executable", ""))) if getattr(sys, "frozen", False) else ""
        if exe_dir:
            candidates.append(os.path.join(exe_dir, "setup_mfa.bat"))
            candidates.append(os.path.join(os.path.dirname(exe_dir), "setup_mfa.bat"))
        local_app_data = str(os.environ.get("LOCALAPPDATA", "") or "").strip()
        if local_app_data:
            candidates.append(os.path.join(local_app_data, "UTAU_Auto_OTO", "setup_mfa.bat"))
        app_data_dir = str(getattr(self, "app_data_dir", "") or "").strip()
        if app_data_dir:
            candidates.append(os.path.join(app_data_dir, "setup_mfa.bat"))
        writable_data_dir = str(getattr(self, "writable_data_dir", "") or "").strip()
        if writable_data_dir:
            candidates.append(os.path.join(writable_data_dir, "setup_mfa.bat"))
        try:
            candidates.append(os.path.join(os.getcwd(), "setup_mfa.bat"))
        except Exception:
            pass

        seen = set()
        for candidate in candidates:
            if not candidate:
                continue
            norm = os.path.normcase(os.path.abspath(candidate))
            if norm in seen:
                continue
            seen.add(norm)
            if os.path.isfile(candidate):
                return candidate
        return ""

    def _run_setup_mfa_script_fallback(self, language="korean", reason="") -> bool:
        script_path = self._resolve_setup_mfa_script_path()
        if not script_path:
            self._append_log("⚠ setup_mfa.bat을 찾지 못해 배치 폴백 복구를 건너뜁니다.")
            return False

        lang = str(language or "korean").strip().lower()
        self._append_log("⚠ 내부 MFA 복구가 실패해 setup_mfa.bat 폴백 복구를 시도합니다.")
        if reason:
            self._append_log(f"   사유: {reason}")

        cmd = [script_path, "--recovery", "--non-interactive"]
        env = os.environ.copy()
        try:
            env_dir = get_default_mfa_env_dir()
            runtime_root = os.path.dirname(env_dir)
            if runtime_root:
                env["UTOA_MFA_SHARED_ROOT"] = runtime_root
        except Exception:
            pass
        self._append_log(f"   실행: {' '.join(cmd)}")

        try:
            process = self._popen_subprocess_hidden(
                cmd,
                cwd=os.path.dirname(script_path) or None,
                stdout=sp.PIPE,
                stderr=sp.STDOUT,
                text=True,
                encoding=self._preferred_subprocess_encoding(),
                errors="replace",
                env=env,
            )
        except Exception as e:
            self._append_log(f"❌ setup_mfa.bat 실행 실패: {e}")
            return False

        if process.stdout is not None:
            for line in process.stdout:
                text = str(line or "").strip()
                if text:
                    self._append_log(text)
        process.wait()
        if process.returncode != 0:
            self._append_log(f"❌ setup_mfa.bat 폴백 복구 실패 (code={process.returncode})")
            return False

        resolved = find_mfa_executable() or ""
        if resolved and os.path.exists(resolved):
            self.mfa_path = resolved

        report = diagnose_mfa_runtime(self.mfa_path or "", language=lang)
        if report.get("ready"):
            self._append_log("✅ setup_mfa.bat 폴백 복구 완료")
            return True
        self._append_log("❌ setup_mfa.bat 실행은 완료됐지만 MFA 상태가 아직 준비되지 않았습니다.")
        return False

    def _build_setup_mfa_recovery_guide(self):
        script_path = self._resolve_setup_mfa_script_path()
        if script_path:
            command = f'cmd /c ""{script_path}" --recovery --non-interactive"'
            return (
                "설치 프로그램에 동봉된 setup_mfa.bat을 직접 실행해 추가 복구를 진행해 주세요.\n"
                f"- 파일: {script_path}\n"
                "- 실행 명령:\n"
                f"{command}"
            )
        return (
            "설치 프로그램에 동봉된 setup_mfa.bat을 직접 실행해 추가 복구를 진행해 주세요.\n"
            "- 실행 예시:\n"
            "cmd /c \"\"%LOCALAPPDATA%\\UTAU_Auto_OTO\\setup_mfa.bat\" --recovery --non-interactive\""
        )

    def _read_runtime_var(self, var_name, default=None):
        var = getattr(self, var_name, None)
        if var is None:
            return default
        try:
            return var.get()
        except Exception:
            return default

    def _to_bool(self, value, default=False):
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on", "y"}:
            return True
        if text in {"0", "false", "no", "off", "n"}:
            return False
        return default

    def _subprocess_no_window_kwargs(self):
        if os.name != "nt":
            return {}
        kwargs = {}
        try:
            startupinfo = sp.STARTUPINFO()
            startupinfo.dwFlags |= sp.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0
            kwargs["startupinfo"] = startupinfo
        except Exception:
            pass
        try:
            kwargs["creationflags"] = int(kwargs.get("creationflags", 0)) | int(
                getattr(sp, "CREATE_NO_WINDOW", 0)
            )
        except Exception:
            pass
        return kwargs

    def _run_subprocess_hidden(self, args, **kwargs):
        for key, value in self._subprocess_no_window_kwargs().items():
            kwargs.setdefault(key, value)
        return sp.run(args, **kwargs)

    def _popen_subprocess_hidden(self, args, **kwargs):
        for key, value in self._subprocess_no_window_kwargs().items():
            kwargs.setdefault(key, value)
        return sp.Popen(args, **kwargs)

    def _preferred_subprocess_encoding(self):
        # Windows 배포 환경에서 utf-8 고정 디코딩 시 콘솔 로그가 깨지는 경우가 있어,
        # 시스템 기본 인코딩을 우선 사용하고 실패 시 utf-8로 폴백한다.
        return locale.getpreferredencoding(False) or "utf-8"

    def _install_mfa_runtime(self, language="korean"):
        import shutil

        lang = str(language or "korean").strip().lower()
        app_dir = getattr(self, "app_dir", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env_dir = get_default_mfa_env_dir()
        legacy_conda_root = get_default_mfa_conda_root()
        micromamba_root = get_default_mfa_micromamba_root()
        micromamba_exe = get_default_mfa_micromamba_exe()
        runtime_root = os.path.dirname(env_dir)
        if any(ord(ch) > 127 for ch in app_dir):
            self._append_log("⚠ 앱 경로에 비ASCII 문자가 있어도 MFA 환경은 공용 폴더를 사용합니다.")
        self._append_log(f"ℹ MFA 공용 환경 경로: {env_dir}")
        self._append_log(f"ℹ MFA Micromamba 경로: {micromamba_root}")
        mfa_exe = os.path.join(env_dir, 'Scripts', 'mfa.exe')
        micromamba_archive = os.path.join(runtime_root, 'micromamba-win-64-latest.tar.bz2')
        vc_redist_exe = os.path.join(runtime_root, 'vc_redist.x64.exe')
        vc_redist_url = 'https://aka.ms/vs/17/release/vc_redist.x64.exe'
        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        vc_runtime_markers = [
            os.path.join(system_root, "System32", "msvcp140.dll"),
            os.path.join(system_root, "System32", "vcruntime140_1.dll"),
        ]
        archive_url = 'https://micro.mamba.pm/api/micromamba/win-64/latest'
        direct_exe_url = 'https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64'

        def _looks_like_tls_error(text: str) -> bool:
            msg = str(text or "").lower()
            return (
                "trust relationship for the ssl/tls secure channel" in msg
                or "ssl/tls secure channel" in msg
                or "certificate" in msg
                or "tls" in msg
            )

        def _looks_like_vc_runtime_missing(text: str) -> bool:
            msg = str(text or "").lower()
            return (
                "msvcp140.dll" in msg
                or "vcruntime140.dll" in msg
                or "vcruntime140_1.dll" in msg
                or "side-by-side configuration is incorrect" in msg
            )

        def _attempt_powershell_download(url: str, out_path: str) -> tuple[bool, str]:
            ps_cmd = (
                f"[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; "
                f"Invoke-WebRequest -Uri '{url}' -OutFile '{out_path}'"
            )
            try:
                result = self._run_subprocess_hidden(
                    ['powershell', '-NoProfile', '-Command', ps_cmd],
                    capture_output=True,
                    text=True,
                )
            except Exception as e:
                return False, str(e)
            ok = result.returncode == 0 and os.path.exists(out_path)
            msg = (result.stderr or result.stdout or "").strip()
            return ok, msg

        def _attempt_curl_download(url: str, out_path: str) -> tuple[bool, str]:
            try:
                result = self._run_subprocess_hidden(
                    ['curl.exe', '-L', '--fail', '--retry', '2', '--retry-delay', '2', '-o', out_path, url],
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError:
                return False, "curl.exe not found"
            except Exception as e:
                return False, str(e)
            ok = result.returncode == 0 and os.path.exists(out_path)
            msg = (result.stderr or result.stdout or "").strip()
            return ok, msg

        def _vc_runtime_ready() -> bool:
            return all(os.path.exists(path) for path in vc_runtime_markers)

        def _ensure_vc_runtime() -> bool:
            if _vc_runtime_ready():
                return True
            self._append_log("🔧 Microsoft VC++ 런타임 점검/복구 중...")
            if not os.path.exists(vc_redist_exe):
                self._append_log("ℹ VC++ 재배포 패키지 다운로드 중...")
                ok, msg = _attempt_powershell_download(vc_redist_url, vc_redist_exe)
                if not ok:
                    if msg:
                        self._append_log(f"⚠ VC++ 런타임 다운로드 1차 실패(PowerShell): {msg}")
                    ok, msg = _attempt_curl_download(vc_redist_url, vc_redist_exe)
                if not ok:
                    self._append_log(f"❌ VC++ 런타임 다운로드 실패: {msg}")
                    if _looks_like_tls_error(msg):
                        self._append_log("   TLS/인증서 문제로 보입니다. 네트워크 정책 확인 후 다시 시도하세요.")
                    self._append_log("   수동 설치 링크: https://aka.ms/vs/17/release/vc_redist.x64.exe")
                    return False
            for args, label in (
                (['/install', '/quiet', '/norestart'], "silent"),
                (['/install', '/passive', '/norestart'], "passive"),
            ):
                try:
                    result = self._run_subprocess_hidden(
                        [vc_redist_exe, *args],
                        capture_output=True,
                        text=True,
                        timeout=600,
                    )
                except Exception as e:
                    self._append_log(f"⚠ VC++ 런타임 설치 실행 실패({label}): {e}")
                    continue
                if result.returncode in {0, 1638, 3010} and _vc_runtime_ready():
                    self._append_log("✅ VC++ 런타임 준비 완료")
                    return True
                out = (result.stderr or result.stdout or "").strip()
                if out:
                    self._append_log(f"⚠ VC++ 런타임 설치 로그({label}): {out[:500]}")
            if _vc_runtime_ready():
                self._append_log("✅ VC++ 런타임 준비 완료")
                return True
            self._append_log("⚠ VC++ 런타임 자동 복구에 실패했습니다.")
            self._append_log("   수동 설치 후 다시 시도: https://aka.ms/vs/17/release/vc_redist.x64.exe")
            return False

        def _remove_env_dir():
            if not os.path.isdir(env_dir):
                return True
            self._append_log(f"🧹 기존 MFA 환경 정리 중: {env_dir}")
            try:
                shutil.rmtree(env_dir)
                return True
            except Exception as e:
                self._append_log(f"❌ 기존 MFA 환경 정리 실패: {e}")
                return False

        def _remove_legacy_conda_root():
            if not os.path.isdir(legacy_conda_root):
                return True
            try:
                shutil.rmtree(legacy_conda_root)
                self._append_log(f"🧹 기존 Miniconda 흔적 정리: {legacy_conda_root}")
                return True
            except Exception as e:
                self._append_log(f"⚠ 기존 Miniconda 폴더 정리 실패: {e}")
                return False

        def _download_micromamba():
            if os.path.exists(micromamba_archive):
                self._append_log("ℹ Micromamba 아카이브가 이미 있어 재사용합니다.")
                return True
            if os.path.exists(micromamba_exe):
                self._append_log("ℹ Micromamba 실행 파일이 이미 있어 재사용합니다.")
                return True
            self._set_status("⬇ Micromamba 다운로드 중...")
            self._append_log("[1/3] ⬇ Micromamba 다운로드 중... (약 15MB)")

            ok, msg = _attempt_powershell_download(archive_url, micromamba_archive)
            if ok:
                return True
            if msg:
                self._append_log(f"⚠ Micromamba 다운로드 1차 실패(PowerShell): {msg}")

            ok, msg = _attempt_curl_download(archive_url, micromamba_archive)
            if ok:
                return True
            if msg:
                self._append_log(f"⚠ Micromamba 다운로드 2차 실패(curl): {msg}")

            self._append_log("ℹ 기본 URL 실패. GitHub 미러로 재시도합니다.")
            os.makedirs(os.path.dirname(micromamba_exe), exist_ok=True)

            ok, msg = _attempt_powershell_download(direct_exe_url, micromamba_exe)
            if ok:
                return True
            if msg:
                self._append_log(f"⚠ Micromamba 미러 다운로드 1차 실패(PowerShell): {msg}")

            ok, msg = _attempt_curl_download(direct_exe_url, micromamba_exe)
            if ok:
                return True
            self._append_log(f"❌ Micromamba 다운로드 실패: {msg}")
            if _looks_like_tls_error(msg):
                self._append_log("   TLS/인증서 문제로 보입니다. 네트워크 정책 확인 또는 수동 파일 배치를 진행하세요.")
            return False

        def _extract_micromamba():
            if os.path.exists(micromamba_exe):
                self._append_log("✅ Micromamba 실행 파일이 이미 있습니다.")
                return True
            self._set_status("📦 Micromamba 압축 해제 중...")
            self._append_log("[2/3] 📦 Micromamba 압축 해제 중...")
            try:
                if os.path.isdir(micromamba_root):
                    shutil.rmtree(micromamba_root)
                os.makedirs(micromamba_root, exist_ok=True)
                result = self._run_subprocess_hidden(
                    ['tar', '-xjf', micromamba_archive, '-C', micromamba_root],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode != 0:
                    self._append_log(f"⚠ Micromamba 압축 해제 실패: {result.stderr or result.stdout}")
                    self._append_log("ℹ 압축 해제 없이 단일 micromamba.exe 경로로 재시도합니다.")
                    os.makedirs(os.path.dirname(micromamba_exe), exist_ok=True)
                    ok, _msg = _attempt_powershell_download(direct_exe_url, micromamba_exe)
                    if not ok:
                        ok, _msg = _attempt_curl_download(direct_exe_url, micromamba_exe)
                    if not ok:
                        self._append_log(f"❌ Micromamba 실행 파일 준비 실패: {_msg}")
                        return False
                    return True
                resolved = get_default_mfa_micromamba_exe()
                if not os.path.exists(resolved):
                    self._append_log("❌ Micromamba 실행 파일을 찾지 못했습니다.")
                    return False
                return True
            except Exception as e:
                self._append_log(f"❌ Micromamba 준비 실패: {e}")
                return False

        def _run_micromamba(cmd, step_label, retry_on_vc_runtime=True):
            env = os.environ.copy()
            env["MAMBA_ROOT_PREFIX"] = micromamba_root
            try:
                process = self._popen_subprocess_hidden(
                    [micromamba_exe, *cmd],
                    stdout=sp.PIPE,
                    stderr=sp.STDOUT,
                    text=True,
                    encoding=self._preferred_subprocess_encoding(),
                    errors='replace',
                    env=env,
                )
            except OSError as e:
                err = str(e)
                if retry_on_vc_runtime and _looks_like_vc_runtime_missing(err):
                    self._append_log("⚠ VC++ 런타임 누락으로 Micromamba 실행에 실패했습니다. 자동 복구 후 재시도합니다.")
                    if _ensure_vc_runtime():
                        return _run_micromamba(cmd, step_label, retry_on_vc_runtime=False)
                self._append_log(f"❌ Micromamba 실행 실패: {err}")
                return False
            self._append_log(step_label)
            for line in process.stdout:
                stripped = line.strip()
                if stripped:
                    self._append_log(stripped)
            process.wait()
            return process.returncode == 0

        def _refresh_mfa_path():
            nonlocal mfa_exe
            resolved = find_mfa_executable()
            if resolved and os.path.exists(resolved):
                mfa_exe = resolved
                self.mfa_path = resolved
                self._append_log(f"ℹ 감지된 MFA 실행 파일: {resolved}")
                return True
            fallback_candidates = [
                os.path.join(env_dir, 'python.exe'),
            ]
            python_exe = fallback_candidates[0]
            if os.path.exists(micromamba_exe):
                wrapper_path = os.path.join(env_dir, 'Scripts', 'mfa.bat')
                try:
                    os.makedirs(os.path.dirname(wrapper_path), exist_ok=True)
                    with open(wrapper_path, 'w', encoding='utf-8') as wf:
                        wf.write('@echo off\r\n')
                        wf.write(f'set "CONDA_PREFIX={env_dir}"\r\n')
                        wf.write(
                            f'set "PATH={env_dir};{env_dir}\\Library\\mingw-w64\\bin;'
                            f'{env_dir}\\Library\\usr\\bin;{env_dir}\\Library\\bin;'
                            f'{env_dir}\\Scripts;{env_dir}\\bin;%PATH%"\r\n'
                        )
                        wf.write(f'"{env_dir}\\python.exe" -m montreal_forced_aligner.command_line.mfa %*\r\n')
                    mfa_exe = wrapper_path
                    self.mfa_path = wrapper_path
                    self._append_log(f"ℹ MFA 배치 래퍼를 생성했습니다: {wrapper_path}")
                    return True
                except Exception as e:
                    self._append_log(f"❌ MFA 배치 래퍼 생성 실패: {e}")
            if os.path.exists(python_exe):
                probe = self._run_subprocess_hidden(
                    [
                        python_exe,
                        '-c',
                        "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('montreal_forced_aligner.command_line.mfa') else 1)",
                    ],
                    capture_output=True,
                    text=True,
                )
                if probe.returncode != 0:
                    return False
                wrapper_path = os.path.join(env_dir, 'Scripts', 'mfa.bat')
                try:
                    os.makedirs(os.path.dirname(wrapper_path), exist_ok=True)
                    with open(wrapper_path, 'w', encoding='utf-8') as wf:
                        wf.write('@echo off\r\n')
                        wf.write(f'"{python_exe}" -m montreal_forced_aligner.command_line.mfa %*\r\n')
                    mfa_exe = wrapper_path
                    self.mfa_path = wrapper_path
                    self._append_log(f"ℹ MFA 배치 래퍼를 생성했습니다: {wrapper_path}")
                    return True
                except Exception as e:
                    self._append_log(f"❌ MFA 래퍼 생성 실패: {e}")
            return False

        if not os.path.exists(mfa_exe):
            _refresh_mfa_path()

        def _remove_env_dir():
            if not os.path.isdir(env_dir):
                return True
            self._append_log(f"🧹 기존 MFA 환경 정리 중: {env_dir}")
            try:
                shutil.rmtree(env_dir)
                return True
            except Exception as e:
                self._append_log(f"❌ 기존 MFA 환경 정리 실패: {e}")
                return False

        def _remove_legacy_conda_root():
            if not os.path.isdir(legacy_conda_root):
                return True
            try:
                shutil.rmtree(legacy_conda_root)
                self._append_log(f"🧹 기존 Miniconda 흔적 정리: {legacy_conda_root}")
                return True
            except Exception as e:
                self._append_log(f"⚠ 기존 Miniconda 폴더 정리 실패: {e}")
                return False

        def _download_micromamba():
            if os.path.exists(micromamba_archive):
                self._append_log("ℹ Micromamba 아카이브가 이미 있어 재사용합니다.")
                return True
            if os.path.exists(micromamba_exe):
                self._append_log("ℹ Micromamba 실행 파일이 이미 있어 재사용합니다.")
                return True
            self._set_status("⬇ Micromamba 다운로드 중...")
            self._append_log("[1/3] ⬇ Micromamba 다운로드 중... (약 15MB)")

            ok, msg = _attempt_powershell_download(archive_url, micromamba_archive)
            if ok:
                return True
            if msg:
                self._append_log(f"⚠ Micromamba 다운로드 1차 실패(PowerShell): {msg}")

            ok, msg = _attempt_curl_download(archive_url, micromamba_archive)
            if ok:
                return True
            if msg:
                self._append_log(f"⚠ Micromamba 다운로드 2차 실패(curl): {msg}")

            self._append_log("ℹ 기본 URL 실패. GitHub 미러로 재시도합니다.")
            os.makedirs(os.path.dirname(micromamba_exe), exist_ok=True)

            ok, msg = _attempt_powershell_download(direct_exe_url, micromamba_exe)
            if ok:
                return True
            if msg:
                self._append_log(f"⚠ Micromamba 미러 다운로드 1차 실패(PowerShell): {msg}")

            ok, msg = _attempt_curl_download(direct_exe_url, micromamba_exe)
            if ok:
                return True
            self._append_log(f"❌ Micromamba 다운로드 실패: {msg}")
            if _looks_like_tls_error(msg):
                self._append_log("   TLS/인증서 문제로 보입니다. 네트워크 정책 확인 또는 수동 파일 배치를 진행하세요.")
            return False

        def _extract_micromamba():
            if os.path.exists(micromamba_exe):
                self._append_log("✅ Micromamba 실행 파일이 이미 있습니다.")
                return True
            self._set_status("📦 Micromamba 압축 해제 중...")
            self._append_log("[2/3] 📦 Micromamba 압축 해제 중...")
            try:
                if os.path.isdir(micromamba_root):
                    shutil.rmtree(micromamba_root)
                os.makedirs(micromamba_root, exist_ok=True)
                result = sp.run(
                    ['tar', '-xjf', micromamba_archive, '-C', micromamba_root],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode != 0:
                    self._append_log(f"⚠ Micromamba 압축 해제 실패: {result.stderr or result.stdout}")
                    self._append_log("ℹ 압축 해제 없이 단일 micromamba.exe 경로로 재시도합니다.")
                    os.makedirs(os.path.dirname(micromamba_exe), exist_ok=True)
                    ok, _msg = _attempt_powershell_download(direct_exe_url, micromamba_exe)
                    if not ok:
                        ok, _msg = _attempt_curl_download(direct_exe_url, micromamba_exe)
                    if not ok:
                        self._append_log(f"❌ Micromamba 실행 파일 준비 실패: {_msg}")
                        return False
                    return True
                resolved = get_default_mfa_micromamba_exe()
                if not os.path.exists(resolved):
                    self._append_log("❌ Micromamba 실행 파일을 찾지 못했습니다.")
                    return False
                return True
            except Exception as e:
                self._append_log(f"❌ Micromamba 준비 실패: {e}")
                return False

        def _run_micromamba(cmd, step_label, retry_on_vc_runtime=True):
            env = os.environ.copy()
            env["MAMBA_ROOT_PREFIX"] = micromamba_root
            try:
                process = self._popen_subprocess_hidden(
                    [micromamba_exe, *cmd],
                    stdout=sp.PIPE,
                    stderr=sp.STDOUT,
                    text=True,
                    encoding=self._preferred_subprocess_encoding(),
                    errors='replace',
                    env=env,
                )
            except OSError as e:
                err = str(e)
                if retry_on_vc_runtime and _looks_like_vc_runtime_missing(err):
                    self._append_log("⚠ VC++ 런타임 누락으로 Micromamba 실행에 실패했습니다. 자동 복구 후 재시도합니다.")
                    if _ensure_vc_runtime():
                        return _run_micromamba(cmd, step_label, retry_on_vc_runtime=False)
                self._append_log(f"❌ Micromamba 실행 실패: {err}")
                return False
            self._append_log(step_label)
            for line in process.stdout:
                stripped = line.strip()
                if stripped:
                    self._append_log(stripped)
            process.wait()
            return process.returncode == 0

        def _refresh_mfa_path():
            nonlocal mfa_exe
            resolved = find_mfa_executable()
            if resolved and os.path.exists(resolved):
                mfa_exe = resolved
                self.mfa_path = resolved
                self._append_log(f"ℹ 감지된 MFA 실행 파일: {resolved}")
                return True
            fallback_candidates = [
                os.path.join(env_dir, 'python.exe'),
            ]
            python_exe = fallback_candidates[0]
            if os.path.exists(micromamba_exe):
                wrapper_path = os.path.join(env_dir, 'Scripts', 'mfa.bat')
                try:
                    os.makedirs(os.path.dirname(wrapper_path), exist_ok=True)
                    with open(wrapper_path, 'w', encoding='utf-8') as wf:
                        wf.write('@echo off\r\n')
                        wf.write(f'set "CONDA_PREFIX={env_dir}"\r\n')
                        wf.write(
                            f'set "PATH={env_dir};{env_dir}\\Library\\mingw-w64\\bin;'
                            f'{env_dir}\\Library\\usr\\bin;{env_dir}\\Library\\bin;'
                            f'{env_dir}\\Scripts;{env_dir}\\bin;%PATH%"\r\n'
                        )
                        wf.write(f'"{env_dir}\\python.exe" -m montreal_forced_aligner.command_line.mfa %*\r\n')
                    mfa_exe = wrapper_path
                    self.mfa_path = wrapper_path
                    self._append_log(f"ℹ MFA 배치 래퍼를 생성했습니다: {wrapper_path}")
                    return True
                except Exception as e:
                    self._append_log(f"❌ MFA 배치 래퍼 생성 실패: {e}")
            if os.path.exists(python_exe):
                probe = self._run_subprocess_hidden(
                    [
                        python_exe,
                        '-c',
                        "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('montreal_forced_aligner.command_line.mfa') else 1)",
                    ],
                    capture_output=True,
                    text=True,
                )
                if probe.returncode != 0:
                    return False
                wrapper_path = os.path.join(env_dir, 'Scripts', 'mfa.bat')
                try:
                    os.makedirs(os.path.dirname(wrapper_path), exist_ok=True)
                    with open(wrapper_path, 'w', encoding='utf-8') as wf:
                        wf.write('@echo off\r\n')
                        wf.write(f'"{python_exe}" -m montreal_forced_aligner.command_line.mfa %*\r\n')
                    mfa_exe = wrapper_path
                    self.mfa_path = wrapper_path
                    self._append_log(f"ℹ MFA 배치 래퍼를 생성했습니다: {wrapper_path}")
                    return True
                except Exception as e:
                    self._append_log(f"❌ MFA 래퍼 생성 실패: {e}")
            return False

        if not os.path.exists(mfa_exe):
            _refresh_mfa_path()

        if os.path.exists(mfa_exe):
            py_ver = get_mfa_env_python_version(mfa_exe)
            if mfa_env_requires_python_downgrade(mfa_exe):
                self._append_log(
                    f"⚠ 현재 MFA 환경 Python {py_ver or '(unknown)'} 은/는 "
                    f"Windows MFA 의존성과 호환되지 않아 Python {MFA_PORTABLE_PYTHON_VERSION} 기준으로 다시 구성합니다."
                )
                if not _remove_env_dir():
                    return False
                self.mfa_path = ""
            else:
                self._append_log("✅ MFA 실행 환경이 이미 있습니다.")
                if py_ver:
                    self._append_log(f"ℹ MFA Python 버전: {py_ver}")
                self.mfa_path = mfa_exe

        if not os.path.exists(mfa_exe):
            self._append_log("🔍 경량 Micromamba 기반으로 MFA 환경을 준비합니다.")
            if not _ensure_vc_runtime():
                self._append_log("⚠ VC++ 런타임 자동 복구가 완료되지 않았습니다. 설치가 실패하면 수동 설치 후 다시 시도하세요.")
            _remove_legacy_conda_root()
            if not _download_micromamba():
                return False
            if not _extract_micromamba():
                return False
            if os.path.isdir(env_dir) and not os.path.exists(mfa_exe):
                if not _remove_env_dir():
                    return False
            self._set_status("🔧 MFA 환경 설치 중...")
            if not _run_micromamba(
                [
                    'create',
                    '-y',
                    '-r',
                    micromamba_root,
                    '-p',
                    env_dir,
                    '-c',
                    'conda-forge',
                    f'python={MFA_PORTABLE_PYTHON_VERSION}',
                    'montreal-forced-aligner',
                    'colorama',
                ],
                "[3/3] 🔧 MFA 설치 중... (3~10분)",
            ):
                self._append_log("❌ MFA 설치 실패")
                return False
            self._append_log("✅ Micromamba 기반 MFA 설치 완료!")
            if not _refresh_mfa_path():
                self._append_log("❌ MFA 실행 파일을 찾지 못했습니다.")
                return False

        self._set_status("🔧 MFA Python 도구 점검 중...")
        if not ensure_mfa_python_packaging_stack(self.mfa_path or mfa_exe, callback=self._append_log):
            self._append_log("❌ MFA Python 패키지 도구 준비 실패")
            return False

        if lang == "korean":
            self._set_status("🔧 한국어 MFA 의존성 점검 중...")
            if not ensure_korean_support(self.mfa_path, callback=self._append_log):
                self._append_log("❌ 한국어 MFA 의존성 준비 실패")
                return False
            dep_report = diagnose_mfa_runtime(self.mfa_path or "", language=lang)
            if "korean_deps_degraded" in list(dep_report.get("issues", []) or []):
                self._notify_korean_dependency_degraded()
        elif lang == "japanese":
            self._set_status("🔧 일본어 MFA 의존성 점검 중...")
            if not ensure_japanese_support(self.mfa_path, callback=self._append_log):
                self._append_log("❌ 일본어 MFA 의존성 준비 실패")
                return False

        self._set_status("⬇ MFA 모델 다운로드 중...")
        has_model, msg = check_mfa_model(self.mfa_path, language=lang)
        if msg:
            self._append_log(msg)
        if not has_model:
            self._append_log("[추가 단계] ⬇ 현재 언어용 MFA 음향 모델 다운로드 중...")
            if not download_mfa_model(self.mfa_path, language=lang, callback=self._append_log):
                self._append_log("❌ MFA 모델 다운로드 실패")
                return False

        if os.path.exists(micromamba_archive):
            try:
                os.remove(micromamba_archive)
            except OSError:
                pass

        self._update_mfa_status(True)
        self._append_log("")
        self._append_log("✅ MFA 설치가 완료되었습니다!")
        self._append_log("   이제 정렬을 바로 계속 진행합니다.")
        return True

    def _format_mfa_diagnosis_summary(self, report):
        checks = dict((report or {}).get("checks", {}) or {})
        issues = list((report or {}).get("issues", []) or [])
        lang = "일본어" if str((report or {}).get("language", "")).lower() == "japanese" else "한국어"
        lang_support_ok = checks.get("language_support")
        lang_support_mode = str(checks.get("language_support_mode", "") or "").strip().lower()
        if lang_support_ok is None:
            lang_support_text = "미확인"
        elif lang_support_ok and lang_support_mode == "fallback":
            lang_support_text = "OK (degraded fallback)"
        else:
            lang_support_text = "OK" if lang_support_ok else "복구 필요"
        return "\n".join([
            f"언어: {lang}",
            f"MFA 실행 파일: {'OK' if checks.get('mfa_executable') else '없음'}",
            f"Python 실행 파일: {'OK' if checks.get('python_exe') else '없음'}",
            f"pip 실행 파일: {'OK' if checks.get('pip_exe') else '없음'}",
            f"Python 버전: {checks.get('python_version') or '(확인 불가)'}",
            f"재구성 필요: {'예' if checks.get('python_rebuild_required') else '아니오'}",
            f"패키지 도구(pip/setuptools/wheel): {('OK' if checks.get('packaging_stack') else '복구 필요') if checks.get('packaging_stack') is not None else '미확인'}",
            f"{lang} 의존성: {lang_support_text}",
            f"{lang} MFA 모델: {'OK' if checks.get('model_ready') else '다운로드 필요'}",
            f"이슈 코드: {', '.join(issues) if issues else '(없음)'}",
            f"최종 준비 상태: {'준비 완료' if (report or {}).get('ready') else '추가 복구 필요'}",
        ])

    def _confirm_mfa_install_action(self, language="korean", reason="install_required") -> bool:
        lang = str(language or "korean").strip().lower()
        lang_label = "일본어" if lang == "japanese" else "한국어"
        reason_map = {
            "missing_runtime": "MFA 실행 환경이 없습니다.",
            "python_rebuild": "MFA Python 버전 재구성이 필요합니다.",
            "model_download": f"{lang_label} MFA 모델 다운로드가 필요합니다.",
            "startup_auto_repair": "초기 자동 점검에서 복구 필요 상태가 감지되었습니다.",
            "install_required": "MFA 설치/복구가 필요합니다.",
        }
        reason_text = reason_map.get(str(reason or "").strip().lower(), reason_map["install_required"])
        title = "MFA 설치/복구 확인"
        message = (
            f"{reason_text}\n\n"
            "진단은 완료되었고, 지금부터 실제 설치/복구를 진행하려고 합니다.\n"
            "설치를 시작할까요?"
        )
        self._append_log(f"ℹ MFA 진단 결과: {reason_text}")
        self._append_log("ℹ 설치 전 사용자 확인 팝업을 표시합니다.")
        approved = False
        if hasattr(self, "_ask_yes_no_dialog_sync"):
            approved = bool(
                self._ask_yes_no_dialog_sync(
                    title=title,
                    message=message,
                    default=False,
                )
            )
        if approved:
            self._append_log("ℹ 사용자 확인: MFA 설치/복구를 진행합니다.")
        else:
            self._append_log("ℹ 사용자 선택: MFA 설치/복구를 진행하지 않고 진단만 유지합니다.")
        return approved

    def _repair_existing_mfa_runtime(self, language="korean"):
        lang = str(language or "korean").strip().lower()
        resolved = self.mfa_path or find_mfa_executable() or ""
        if not resolved or not os.path.exists(resolved):
            self._append_log("❌ 복구할 MFA 환경을 찾지 못했습니다.")
            return False
        self.mfa_path = resolved

        self._set_status("🔧 MFA Python 도구 점검 중...")
        if not ensure_mfa_python_packaging_stack(self.mfa_path, callback=self._append_log):
            self._append_log("❌ MFA Python 패키지 도구 복구 실패")
            return False

        if lang == "korean":
            self._set_status("🔧 한국어 MFA 의존성 점검 중...")
            if not ensure_korean_support(self.mfa_path, callback=self._append_log):
                self._append_log("❌ 한국어 MFA 의존성 준비 실패")
                return False
            dep_report = diagnose_mfa_runtime(self.mfa_path or "", language=lang)
            if "korean_deps_degraded" in list(dep_report.get("issues", []) or []):
                self._notify_korean_dependency_degraded()
        elif lang == "japanese":
            self._set_status("🔧 일본어 MFA 의존성 점검 중...")
            if not ensure_japanese_support(self.mfa_path, callback=self._append_log):
                self._append_log("❌ 일본어 MFA 의존성 준비 실패")
                return False

        self._set_status("🔧 MFA 모델 점검 중...")
        has_model, msg = check_mfa_model(self.mfa_path, language=lang)
        if msg:
            self._append_log(msg)
        if not has_model:
            self._append_log("ℹ 현재 언어용 MFA 모델이 없어 자동 다운로드를 시작합니다.")
            if not download_mfa_model(self.mfa_path, language=lang, callback=self._append_log):
                self._append_log("❌ MFA 모델 다운로드 실패")
                return False
        return True

    def _ensure_mfa_ready_for_language(self, language="korean"):
        lang = str(language or "korean").strip().lower()
        self._last_mfa_install_declined = False
        resolved = self.mfa_path if (self.mfa_path and os.path.exists(self.mfa_path)) else (find_mfa_executable() or "")
        if resolved and os.path.exists(resolved):
            self.mfa_path = resolved
        if not self.mfa_path or not os.path.exists(self.mfa_path):
            self._notify_long_install_time("MFA")
            self._append_log("ℹ MFA가 없어 지금 자동 설치를 시작합니다.")
            if not self._confirm_mfa_install_action(language=lang, reason="missing_runtime"):
                self._mfa_ready_cache_ok = False
                self._last_mfa_install_declined = True
                return False
            if self._install_mfa_runtime(language=lang):
                cache_key = f"{lang}|{os.path.normcase(os.path.abspath(str(self.mfa_path or '')))}"
                self._mfa_ready_cache_key = cache_key
                self._mfa_ready_cache_ok = True
                return True
            self._mfa_ready_cache_ok = False
            return self._run_setup_mfa_script_fallback(language=lang, reason="auto_install_failed")
        if mfa_env_requires_python_downgrade(self.mfa_path):
            py_ver = get_mfa_env_python_version(self.mfa_path)
            self._append_log(
                f"⚠ 현재 MFA 환경 Python {py_ver or '(unknown)'} 은/는 "
                f"Windows MFA 의존성과 호환되지 않아 Python {MFA_PORTABLE_PYTHON_VERSION} 기준으로 다시 구성합니다."
            )
            self.mfa_path = ""
            self._mfa_ready_cache_ok = False
            if not self._confirm_mfa_install_action(language=lang, reason="python_rebuild"):
                self._last_mfa_install_declined = True
                return False
            if self._install_mfa_runtime(language=lang):
                cache_key = f"{lang}|{os.path.normcase(os.path.abspath(str(self.mfa_path or '')))}"
                self._mfa_ready_cache_key = cache_key
                self._mfa_ready_cache_ok = True
                return True
            return self._run_setup_mfa_script_fallback(language=lang, reason="python_rebuild_install_failed")

        # Runtime path exists: keep per-run check lightweight.
        cache_key = f"{lang}|{os.path.normcase(os.path.abspath(str(self.mfa_path or '')))}"
        if (
            bool(getattr(self, "_mfa_ready_cache_ok", False))
            and str(getattr(self, "_mfa_ready_cache_key", "") or "") == cache_key
        ):
            return True
        has_model, msg = check_mfa_model(self.mfa_path, language=lang)
        if msg:
            self._append_log(msg)
        if not has_model:
            self._append_log("ℹ 현재 언어용 MFA 모델이 없어 자동 다운로드를 시작합니다.")
            if not self._confirm_mfa_install_action(language=lang, reason="model_download"):
                self._mfa_ready_cache_ok = False
                self._last_mfa_install_declined = True
                return False
            if not download_mfa_model(self.mfa_path, language=lang, callback=self._append_log):
                self._append_log("❌ MFA 모델 다운로드 실패")
                self._mfa_ready_cache_ok = False
                return False
        self._mfa_ready_cache_key = cache_key
        self._mfa_ready_cache_ok = True
        return True

    def _mfa_startup_repair_state_path(self):
        base_dir = (
            getattr(self, "writable_data_dir", "")
            or getattr(self, "app_data_dir", "")
            or getattr(self, "app_dir", "")
            or os.getcwd()
        )
        return os.path.join(base_dir, ".mfa_startup_repair_state.json")

    def _load_mfa_startup_repair_state(self):
        path = self._mfa_startup_repair_state_path()
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_mfa_startup_repair_state(self, **updates):
        state = self._load_mfa_startup_repair_state()
        state.update(updates or {})
        path = self._mfa_startup_repair_state_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            return

    def _schedule_startup_mfa_auto_repair(self):
        if str(os.environ.get("UTOA_DISABLE_STARTUP_MFA_AUTO_REPAIR", "")).strip().lower() in {"1", "true", "yes", "on"}:
            return
        if getattr(self, "_startup_mfa_auto_repair_scheduled", False):
            return
        try:
            if normalize_aligner_name(self.aligner_var.get(), default="mfa") == "none":
                return
        except Exception:
            pass

        state = self._load_mfa_startup_repair_state()
        # Run startup MFA check only once per install/runtime.
        if bool(state.get("first_check_done", False)):
            return
        if float(state.get("last_attempt_ts", 0.0) or 0.0) > 0:
            return

        def _kickoff():
            if self.is_running:
                return
            self._run_in_thread(self._run_startup_mfa_auto_repair_task)

        self._startup_mfa_auto_repair_scheduled = True
        self._after_safe(_kickoff, delay_ms=1200)

    def _cuda_startup_check_state_path(self):
        base_dir = (
            getattr(self, "writable_data_dir", "")
            or getattr(self, "app_data_dir", "")
            or getattr(self, "app_dir", "")
            or os.getcwd()
        )
        return os.path.join(base_dir, ".cuda_startup_check_state.json")

    def _load_cuda_startup_check_state(self):
        path = self._cuda_startup_check_state_path()
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _save_cuda_startup_check_state(self, **updates):
        state = self._load_cuda_startup_check_state()
        state.update(updates or {})
        path = self._cuda_startup_check_state_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            return

    def _schedule_startup_cuda_runtime_check(self):
        if str(os.environ.get("UTOA_DISABLE_STARTUP_CUDA_RUNTIME_CHECK", "")).strip().lower() in {"1", "true", "yes", "on"}:
            return
        if getattr(self, "_startup_cuda_runtime_check_scheduled", False):
            return
        state = self._load_cuda_startup_check_state()
        if bool(state.get("first_check_done", False)):
            return

        def _kickoff():
            if getattr(self, "_is_closing", False):
                return
            thread = threading.Thread(
                target=self._run_startup_cuda_runtime_check_task,
                daemon=True,
            )
            thread.start()

        self._startup_cuda_runtime_check_scheduled = True
        self._after_safe(_kickoff, delay_ms=1800)

    def _run_startup_cuda_runtime_check_task(self):
        now = time.time()
        self._save_cuda_startup_check_state(
            last_attempt_ts=now,
            last_result="running",
        )
        try:
            diagnosis = collect_cuda_runtime_diagnosis()
            has_nvidia = bool(diagnosis.get("nvidia_gpu_present"))
            has_cuda = bool(diagnosis.get("torch_cuda_available"))
            gpu_name = str((diagnosis.get("nvidia_gpu_names") or [""])[0] or "").strip()
            nvidia_cuda_ver = str(diagnosis.get("nvidia_cuda_version", "") or "").strip()
            torch_ver = str(diagnosis.get("torch_version", "") or "").strip()
            torch_cuda_ver = str(diagnosis.get("torch_cuda_version", "") or "").strip()
            direct_cmd = str(diagnosis.get("recommended_install_command_text", "") or "").strip()

            if (not has_nvidia) or has_cuda:
                self._save_cuda_startup_check_state(
                    first_check_done=True,
                    last_result="ready" if has_cuda else "skip_no_nvidia",
                    last_checked_ts=time.time(),
                    nvidia_gpu_present=has_nvidia,
                    torch_cuda_available=has_cuda,
                )
                return

            app_dir = getattr(self, "app_dir", "") or os.getcwd()
            setup_script_path = os.path.join(app_dir, "setup_mfa.bat")
            setup_cmd = "setup_mfa.bat --with-ml --with-cuda"
            setup_cmd_line = setup_cmd if os.path.isfile(setup_script_path) else "(setup_mfa.bat 파일을 찾지 못함)"
            title = "CUDA 런타임 자동 설치"
            message = (
                "NVIDIA GPU가 감지되었지만 현재 Python 런타임에서 CUDA를 사용할 수 없습니다.\n\n"
                f"- GPU: {gpu_name or '(미확인)'}\n"
                f"- nvidia-smi CUDA 버전: {nvidia_cuda_ver or '(미확인)'}\n"
                f"- torch 버전: {torch_ver or '(미설치 또는 import 실패)'}\n"
                f"- torch CUDA 빌드: {torch_cuda_ver or '(없음)'}\n\n"
                "지금 네트워크로 CUDA용 PyTorch를 자동 설치할까요?\n\n"
                f"- 자동 설치 실패 시 수동 명령: {direct_cmd or '(권장 pip 명령 생성 실패)'}\n"
                f"- 대체 경로: {setup_cmd_line}"
            )
            self._append_log(
                "⚠ NVIDIA GPU는 감지되었지만 torch CUDA 런타임이 비활성 상태입니다. "
                "초기 1회 자동 설치 선택 팝업을 표시합니다."
            )

            wants_auto_install = False
            if hasattr(self, "_ask_yes_no_dialog_sync"):
                wants_auto_install = bool(
                    self._ask_yes_no_dialog_sync(
                        title=title,
                        message=message,
                        default=False,
                    )
                )

            post_diag = diagnosis
            install_result = {}
            state_result = "needs_cuda_runtime"
            if wants_auto_install:
                self._append_log("ℹ 사용자 선택: CUDA 런타임 자동 설치를 시작합니다.")
                command_args = diagnosis.get("recommended_install_command") or []
                python_exe = ""
                if isinstance(command_args, (list, tuple)) and command_args:
                    python_exe = str(command_args[0] or "").strip()
                install_result = install_torch_cuda_runtime(
                    python_exe=python_exe,
                    cuda_version=nvidia_cuda_ver,
                    timeout_sec=3600,
                )
                post_diag = collect_cuda_runtime_diagnosis(python_exe=python_exe)
                install_ok = bool(install_result.get("success", False))
                runtime_ok = bool(post_diag.get("torch_cuda_available", False))
                if install_ok and runtime_ok:
                    state_result = "auto_install_success"
                    self._append_log("✅ CUDA 런타임 자동 설치가 완료되었습니다.")
                else:
                    state_result = "auto_install_failed"
                    self._append_log("❌ CUDA 런타임 자동 설치에 실패했습니다.")
                    tail = str(install_result.get("stderr_tail", "") or install_result.get("stdout_tail", "") or "").strip()
                    if tail:
                        self._append_log(f"   설치 로그 요약: {tail[-240:]}")
                    self._after_safe(
                        lambda cmd=direct_cmd, setup=setup_cmd_line, t=tail: self._show_copyable_alert(
                            title="CUDA 런타임 자동 설치 실패",
                            message=(
                                "CUDA 자동 설치가 실패했습니다.\n\n"
                                "아래 방법으로 수동 설치 후 앱을 다시 실행해 주세요.\n"
                                f"1) {cmd or '(권장 pip 명령 생성 실패)'}\n"
                                f"2) {setup}\n\n"
                                f"마지막 오류:\n{(t or '(로그 없음)')[-600:]}"
                            ),
                            alert_key="cuda_runtime_auto_install_failed",
                            link_url="https://pytorch.org/get-started/locally/",
                            link_label="PyTorch 설치 가이드",
                        )
                    )
            else:
                state_result = "user_declined_install"
                self._append_log("ℹ 사용자 선택: CUDA 런타임 자동 설치를 건너뛰었습니다.")

            self._save_cuda_startup_check_state(
                first_check_done=True,
                last_result=state_result,
                last_checked_ts=time.time(),
                nvidia_gpu_present=has_nvidia,
                torch_cuda_available=bool(post_diag.get("torch_cuda_available", False)),
                nvidia_gpu_name=gpu_name,
                nvidia_cuda_version=nvidia_cuda_ver,
                torch_version=torch_ver,
                torch_cuda_version=str(post_diag.get("torch_cuda_version", "") or torch_cuda_ver),
                recommended_command=direct_cmd,
                setup_command=setup_cmd if os.path.isfile(setup_script_path) else "",
                auto_install_attempted=bool(wants_auto_install),
                auto_install_success=bool(state_result == "auto_install_success"),
                install_returncode=int(install_result.get("returncode", -1)) if install_result else -1,
            )
        except Exception as exc:
            self._save_cuda_startup_check_state(
                first_check_done=True,
                last_result="error",
                last_error=str(exc),
                last_error_ts=time.time(),
            )

    def _run_startup_mfa_auto_repair_task(self):
        self._set_running(True)
        lang = self._get_language()
        now = time.time()
        self._save_mfa_startup_repair_state(
            last_attempt_ts=now,
            last_result="running",
            last_language=lang,
        )
        try:
            self._set_status("⏳ 조금만 기다려주세요, 프로그램 필수 요소들을 설치하고 점검하는 중입니다.")
            self._append_log("🔍 초기 설치 점검: MFA 상태를 자동 진단합니다.")
            resolved = self.mfa_path or find_mfa_executable() or ""
            if resolved and os.path.exists(resolved):
                self.mfa_path = resolved

            before = diagnose_mfa_runtime(self.mfa_path or "", language=lang)
            if before.get("ready"):
                if "korean_deps_degraded" in list(before.get("issues", []) or []):
                    self._notify_korean_dependency_degraded()
                self._append_log("✅ 초기 MFA 상태 점검 완료 (복구 불필요)")
                self._update_mfa_status(True)
                self._set_status("✅ 초기 MFA 점검 완료")
                self._save_mfa_startup_repair_state(
                    last_result="success",
                    last_success_ts=time.time(),
                    last_success_version=str(getattr(self, "app_version", "") or ""),
                    last_issues=[],
                    first_check_done=True,
                )
                return

            self._append_log("⚠ 초기 설치 점검에서 MFA 이상을 감지했습니다.")
            self._append_log("ℹ 자동복구는 진단 후 사용자 확인을 받은 경우에만 설치를 진행합니다.")
            self._append_log(self._format_mfa_diagnosis_summary(before))
            recovered = self._ensure_mfa_ready_for_language(lang)
            after = diagnose_mfa_runtime(self.mfa_path or "", language=lang)

            if recovered and after.get("ready"):
                self._append_log("✅ 초기 MFA 자동 복구 완료")
                self._update_mfa_status(True)
                self._set_status("✅ 초기 MFA 복구 완료")
                self._save_mfa_startup_repair_state(
                    last_result="success",
                    last_success_ts=time.time(),
                    last_success_version=str(getattr(self, "app_version", "") or ""),
                    last_issues=[],
                    first_check_done=True,
                )
            else:
                issues = list(after.get("issues", []) or [])
                if bool(getattr(self, "_last_mfa_install_declined", False)):
                    self._append_log("ℹ 사용자 선택으로 MFA 설치/복구를 건너뛰었습니다.")
                self._append_log("⚠ 초기 MFA 자동 복구가 완료되지 않았습니다. 'MFA 진단/복구' 버튼으로 재시도하세요.")
                recovery_guide = self._build_setup_mfa_recovery_guide()
                self._after_safe(
                    lambda guide=recovery_guide: self._show_copyable_alert(
                        title="MFA 자동 복구 추가 안내",
                        message=(
                            "초기 자동 복구가 완료되지 않았습니다.\n"
                            "앱 내부의 'MFA 진단/복구'를 다시 시도하거나 아래 방법으로 수동 복구를 진행해 주세요.\n\n"
                            f"{guide}"
                        ),
                        alert_key="mfa_startup_repair_needs_attention",
                    )
                )
                self._set_status("⚠ MFA 추가 복구 필요")
                declined = bool(getattr(self, "_last_mfa_install_declined", False))
                self._save_mfa_startup_repair_state(
                    last_result="user_declined" if declined else "failed",
                    last_issues=issues,
                    last_failure_ts=time.time(),
                    first_check_done=True,
                )
        except Exception as e:
            self._save_mfa_startup_repair_state(
                last_result="error",
                last_error=str(e),
                last_error_ts=time.time(),
                first_check_done=True,
            )
            self._handle_error("초기 MFA 자동 복구", e)
        finally:
            self._save_mfa_startup_repair_state(
                first_check_done=True,
                last_checked_ts=time.time(),
            )
            self._set_running(False)

    def _schedule_alignment_failure_mfa_followup(self, language="korean", align_code="", align_message=""):
        if getattr(self, "_mfa_alignment_failure_followup_running", False):
            return
        self._mfa_alignment_failure_followup_running = True
        thread = threading.Thread(
            target=self._run_alignment_failure_mfa_followup_task,
            args=(language, align_code, align_message),
            daemon=True,
        )
        thread.start()

    def _run_alignment_failure_mfa_followup_task(self, language="korean", align_code="", align_message=""):
        lang = str(language or "korean").strip().lower()
        try:
            resolved = self.mfa_path or find_mfa_executable() or ""
            if resolved and os.path.exists(resolved):
                self.mfa_path = resolved
            report = diagnose_mfa_runtime(self.mfa_path or "", language=lang)
            summary = self._format_mfa_diagnosis_summary(report)
            issues = list(report.get("issues", []) or [])
            self._append_log("🔎 정렬 실패 후 MFA 자동 점검 결과")
            self._append_log(summary)
            if "korean_deps_degraded" in issues:
                self._notify_korean_dependency_degraded()

            ready = bool(report.get("ready", False))
            issue_token = "_".join(sorted(str(i) for i in issues))[:80] or "none"
            if ready:
                message = (
                    "정렬 실패 후 MFA 자동 점검 결과, 런타임 자체는 준비 상태입니다.\n"
                    "이번 실패는 입력 데이터(발음/사전/오디오) 또는 현재 정렬 프로필 영향일 가능성이 큽니다.\n\n"
                    f"- 실패 코드: {align_code or '(없음)'}\n"
                    f"- 실패 메시지: {align_message or '(없음)'}\n\n"
                    "권장:\n"
                    "1) 사전/라벨 파일 재생성\n"
                    "2) MFA 프로필을 '기본' 또는 '빠름'으로 변경 후 재시도\n"
                    "3) 계속 실패하면 'MFA 진단/복구' 버튼 실행\n\n"
                    "[자동 점검 요약]\n"
                    f"{summary}"
                )
                alert_key = f"mfa_after_align_fail_ready_{lang}_{issue_token}"
            else:
                recovery_guide = self._build_setup_mfa_recovery_guide()
                message = (
                    "정렬 실패 후 MFA 자동 점검에서 추가 복구가 필요하다고 판단되었습니다.\n"
                    "앱의 'MFA 진단/복구' 버튼을 눌러 복구를 진행해 주세요.\n\n"
                    f"- 실패 코드: {align_code or '(없음)'}\n"
                    f"- 실패 메시지: {align_message or '(없음)'}\n\n"
                    "[자동 점검 요약]\n"
                    f"{summary}\n\n"
                    "[수동 복구]\n"
                    f"{recovery_guide}"
                )
                alert_key = f"mfa_after_align_fail_repair_{lang}_{issue_token}"

            self._after_safe(
                lambda msg=message, key=alert_key: self._show_copyable_alert(
                    title="MFA 정렬 실패 후 점검 안내",
                    message=msg,
                    alert_key=key,
                )
            )
        except Exception as exc:
            self._append_log(f"⚠ 정렬 실패 후 MFA 자동 점검 중 오류: {exc}")
        finally:
            self._mfa_alignment_failure_followup_running = False

    def _mfa_startup_repair_state_path(self):
        base_dir = (
            getattr(self, "writable_data_dir", "")
            or getattr(self, "app_data_dir", "")
            or getattr(self, "app_dir", "")
            or os.getcwd()
        )
        return os.path.join(base_dir, ".mfa_startup_repair_state.json")

    def _load_mfa_startup_repair_state(self):
        path = self._mfa_startup_repair_state_path()
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_mfa_startup_repair_state(self, **updates):
        state = self._load_mfa_startup_repair_state()
        state.update(updates or {})
        path = self._mfa_startup_repair_state_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            return

    def _schedule_startup_mfa_auto_repair(self):
        if str(os.environ.get("UTOA_DISABLE_STARTUP_MFA_AUTO_REPAIR", "")).strip().lower() in {"1", "true", "yes", "on"}:
            return
        if getattr(self, "_startup_mfa_auto_repair_scheduled", False):
            return
        try:
            if normalize_aligner_name(self.aligner_var.get(), default="mfa") == "none":
                return
        except Exception:
            pass

        state = self._load_mfa_startup_repair_state()
        # Run startup MFA check only once per install/runtime.
        if bool(state.get("first_check_done", False)):
            return
        if float(state.get("last_attempt_ts", 0.0) or 0.0) > 0:
            return

        def _kickoff():
            if self.is_running:
                return
            self._run_in_thread(self._run_startup_mfa_auto_repair_task)

        self._startup_mfa_auto_repair_scheduled = True
        self._after_safe(_kickoff, delay_ms=1200)

    def _cuda_startup_check_state_path(self):
        base_dir = (
            getattr(self, "writable_data_dir", "")
            or getattr(self, "app_data_dir", "")
            or getattr(self, "app_dir", "")
            or os.getcwd()
        )
        return os.path.join(base_dir, ".cuda_startup_check_state.json")

    def _load_cuda_startup_check_state(self):
        path = self._cuda_startup_check_state_path()
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _save_cuda_startup_check_state(self, **updates):
        state = self._load_cuda_startup_check_state()
        state.update(updates or {})
        path = self._cuda_startup_check_state_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            return

    def _schedule_startup_cuda_runtime_check(self):
        if str(os.environ.get("UTOA_DISABLE_STARTUP_CUDA_RUNTIME_CHECK", "")).strip().lower() in {"1", "true", "yes", "on"}:
            return
        if getattr(self, "_startup_cuda_runtime_check_scheduled", False):
            return
        state = self._load_cuda_startup_check_state()
        if bool(state.get("first_check_done", False)):
            return

        def _kickoff():
            if getattr(self, "_is_closing", False):
                return
            thread = threading.Thread(
                target=self._run_startup_cuda_runtime_check_task,
                daemon=True,
            )
            thread.start()

        self._startup_cuda_runtime_check_scheduled = True
        self._after_safe(_kickoff, delay_ms=1800)

    def _run_startup_cuda_runtime_check_task(self):
        now = time.time()
        self._save_cuda_startup_check_state(
            last_attempt_ts=now,
            last_result="running",
        )
        try:
            diagnosis = collect_cuda_runtime_diagnosis()
            has_nvidia = bool(diagnosis.get("nvidia_gpu_present"))
            has_cuda = bool(diagnosis.get("torch_cuda_available"))
            gpu_name = str((diagnosis.get("nvidia_gpu_names") or [""])[0] or "").strip()
            nvidia_cuda_ver = str(diagnosis.get("nvidia_cuda_version", "") or "").strip()
            torch_ver = str(diagnosis.get("torch_version", "") or "").strip()
            torch_cuda_ver = str(diagnosis.get("torch_cuda_version", "") or "").strip()
            direct_cmd = str(diagnosis.get("recommended_install_command_text", "") or "").strip()

            if (not has_nvidia) or has_cuda:
                self._save_cuda_startup_check_state(
                    first_check_done=True,
                    last_result="ready" if has_cuda else "skip_no_nvidia",
                    last_checked_ts=time.time(),
                    nvidia_gpu_present=has_nvidia,
                    torch_cuda_available=has_cuda,
                )
                return

            app_dir = getattr(self, "app_dir", "") or os.getcwd()
            setup_script_path = os.path.join(app_dir, "setup_mfa.bat")
            setup_cmd = "setup_mfa.bat --with-ml --with-cuda"
            setup_cmd_line = setup_cmd if os.path.isfile(setup_script_path) else "(setup_mfa.bat 파일을 찾지 못함)"
            title = "CUDA 런타임 자동 설치"
            message = (
                "NVIDIA GPU가 감지되었지만 현재 Python 런타임에서 CUDA를 사용할 수 없습니다.\n\n"
                f"- GPU: {gpu_name or '(미확인)'}\n"
                f"- nvidia-smi CUDA 버전: {nvidia_cuda_ver or '(미확인)'}\n"
                f"- torch 버전: {torch_ver or '(미설치 또는 import 실패)'}\n"
                f"- torch CUDA 빌드: {torch_cuda_ver or '(없음)'}\n\n"
                "지금 네트워크로 CUDA용 PyTorch를 자동 설치할까요?\n\n"
                f"- 자동 설치 실패 시 수동 명령: {direct_cmd or '(권장 pip 명령 생성 실패)'}\n"
                f"- 대체 경로: {setup_cmd_line}"
            )
            self._append_log(
                "⚠ NVIDIA GPU는 감지되었지만 torch CUDA 런타임이 비활성 상태입니다. "
                "초기 1회 자동 설치 선택 팝업을 표시합니다."
            )

            wants_auto_install = False
            if hasattr(self, "_ask_yes_no_dialog_sync"):
                wants_auto_install = bool(
                    self._ask_yes_no_dialog_sync(
                        title=title,
                        message=message,
                        default=False,
                    )
                )

            post_diag = diagnosis
            install_result = {}
            state_result = "needs_cuda_runtime"
            if wants_auto_install:
                self._append_log("ℹ 사용자 선택: CUDA 런타임 자동 설치를 시작합니다.")
                command_args = diagnosis.get("recommended_install_command") or []
                python_exe = ""
                if isinstance(command_args, (list, tuple)) and command_args:
                    python_exe = str(command_args[0] or "").strip()
                install_result = install_torch_cuda_runtime(
                    python_exe=python_exe,
                    cuda_version=nvidia_cuda_ver,
                    timeout_sec=3600,
                )
                post_diag = collect_cuda_runtime_diagnosis(python_exe=python_exe)
                install_ok = bool(install_result.get("success", False))
                runtime_ok = bool(post_diag.get("torch_cuda_available", False))
                if install_ok and runtime_ok:
                    state_result = "auto_install_success"
                    self._append_log("✅ CUDA 런타임 자동 설치가 완료되었습니다.")
                else:
                    state_result = "auto_install_failed"
                    self._append_log("❌ CUDA 런타임 자동 설치에 실패했습니다.")
                    tail = str(install_result.get("stderr_tail", "") or install_result.get("stdout_tail", "") or "").strip()
                    if tail:
                        self._append_log(f"   설치 로그 요약: {tail[-240:]}")
                    self._after_safe(
                        lambda cmd=direct_cmd, setup=setup_cmd_line, t=tail: self._show_copyable_alert(
                            title="CUDA 런타임 자동 설치 실패",
                            message=(
                                "CUDA 자동 설치가 실패했습니다.\n\n"
                                "아래 방법으로 수동 설치 후 앱을 다시 실행해 주세요.\n"
                                f"1) {cmd or '(권장 pip 명령 생성 실패)'}\n"
                                f"2) {setup}\n\n"
                                f"마지막 오류:\n{(t or '(로그 없음)')[-600:]}"
                            ),
                            alert_key="cuda_runtime_auto_install_failed",
                            link_url="https://pytorch.org/get-started/locally/",
                            link_label="PyTorch 설치 가이드",
                        )
                    )
            else:
                state_result = "user_declined_install"
                self._append_log("ℹ 사용자 선택: CUDA 런타임 자동 설치를 건너뛰었습니다.")

            self._save_cuda_startup_check_state(
                first_check_done=True,
                last_result=state_result,
                last_checked_ts=time.time(),
                nvidia_gpu_present=has_nvidia,
                torch_cuda_available=bool(post_diag.get("torch_cuda_available", False)),
                nvidia_gpu_name=gpu_name,
                nvidia_cuda_version=nvidia_cuda_ver,
                torch_version=torch_ver,
                torch_cuda_version=str(post_diag.get("torch_cuda_version", "") or torch_cuda_ver),
                recommended_command=direct_cmd,
                setup_command=setup_cmd if os.path.isfile(setup_script_path) else "",
                auto_install_attempted=bool(wants_auto_install),
                auto_install_success=bool(state_result == "auto_install_success"),
                install_returncode=int(install_result.get("returncode", -1)) if install_result else -1,
            )
        except Exception as exc:
            self._save_cuda_startup_check_state(
                first_check_done=True,
                last_result="error",
                last_error=str(exc),
                last_error_ts=time.time(),
            )

    def _run_startup_mfa_auto_repair_task(self):
        self._set_running(True)
        lang = self._get_language()
        now = time.time()
        self._save_mfa_startup_repair_state(
            last_attempt_ts=now,
            last_result="running",
            last_language=lang,
        )
        try:
            self._set_status("⏳ 조금만 기다려주세요, 프로그램 필수 요소들을 설치하고 점검하는 중입니다.")
            self._append_log("🔍 초기 설치 점검: MFA 상태를 자동 진단합니다.")
            resolved = self.mfa_path or find_mfa_executable() or ""
            if resolved and os.path.exists(resolved):
                self.mfa_path = resolved

            before = diagnose_mfa_runtime(self.mfa_path or "", language=lang)
            if before.get("ready"):
                if "korean_deps_degraded" in list(before.get("issues", []) or []):
                    self._notify_korean_dependency_degraded()
                self._append_log("✅ 초기 MFA 상태 점검 완료 (복구 불필요)")
                self._update_mfa_status(True)
                self._set_status("✅ 초기 MFA 점검 완료")
                self._save_mfa_startup_repair_state(
                    last_result="success",
                    last_success_ts=time.time(),
                    last_success_version=str(getattr(self, "app_version", "") or ""),
                    last_issues=[],
                    first_check_done=True,
                )
                return

            self._append_log("⚠ 초기 설치 점검에서 MFA 이상을 감지해 자동 복구를 시작합니다.")
            self._append_log(self._format_mfa_diagnosis_summary(before))
            recovered = self._ensure_mfa_ready_for_language(lang)
            after = diagnose_mfa_runtime(self.mfa_path or "", language=lang)

            if recovered and after.get("ready"):
                self._append_log("✅ 초기 MFA 자동 복구 완료")
                self._update_mfa_status(True)
                self._set_status("✅ 초기 MFA 복구 완료")
                self._save_mfa_startup_repair_state(
                    last_result="success",
                    last_success_ts=time.time(),
                    last_success_version=str(getattr(self, "app_version", "") or ""),
                    last_issues=[],
                    first_check_done=True,
                )
            else:
                issues = list(after.get("issues", []) or [])
                if bool(getattr(self, "_last_mfa_install_declined", False)):
                    self._append_log("ℹ 사용자 선택으로 MFA 설치/복구를 건너뛰었습니다.")
                self._append_log("⚠ 초기 MFA 자동 복구가 완료되지 않았습니다. 'MFA 진단/복구' 버튼으로 재시도하세요.")
                recovery_guide = self._build_setup_mfa_recovery_guide()
                self._after_safe(
                    lambda guide=recovery_guide: self._show_copyable_alert(
                        title="MFA 자동 복구 추가 안내",
                        message=(
                            "초기 자동 복구가 완료되지 않았습니다.\n"
                            "앱 내부의 'MFA 진단/복구'를 다시 시도하거나 아래 방법으로 수동 복구를 진행해 주세요.\n\n"
                            f"{guide}"
                        ),
                        alert_key="mfa_startup_repair_needs_attention",
                    )
                )
                self._set_status("⚠ MFA 추가 복구 필요")
                declined = bool(getattr(self, "_last_mfa_install_declined", False))
                self._save_mfa_startup_repair_state(
                    last_result="user_declined" if declined else "failed",
                    last_issues=issues,
                    last_failure_ts=time.time(),
                    first_check_done=True,
                )
        except Exception as e:
            self._save_mfa_startup_repair_state(
                last_result="error",
                last_error=str(e),
                last_error_ts=time.time(),
                first_check_done=True,
            )
            self._handle_error("초기 MFA 자동 복구", e)
        finally:
            self._save_mfa_startup_repair_state(
                first_check_done=True,
                last_checked_ts=time.time(),
            )
            self._set_running(False)

    def _schedule_alignment_failure_mfa_followup(self, language="korean", align_code="", align_message=""):
        if getattr(self, "_mfa_alignment_failure_followup_running", False):
            return
        self._mfa_alignment_failure_followup_running = True
        thread = threading.Thread(
            target=self._run_alignment_failure_mfa_followup_task,
            args=(language, align_code, align_message),
            daemon=True,
        )
        thread.start()

    def _run_alignment_failure_mfa_followup_task(self, language="korean", align_code="", align_message=""):
        lang = str(language or "korean").strip().lower()
        try:
            resolved = self.mfa_path or find_mfa_executable() or ""
            if resolved and os.path.exists(resolved):
                self.mfa_path = resolved
            report = diagnose_mfa_runtime(self.mfa_path or "", language=lang)
            summary = self._format_mfa_diagnosis_summary(report)
            issues = list(report.get("issues", []) or [])
            self._append_log("🔎 정렬 실패 후 MFA 자동 점검 결과")
            self._append_log(summary)
            if "korean_deps_degraded" in issues:
                self._notify_korean_dependency_degraded()

            ready = bool(report.get("ready", False))
            issue_token = "_".join(sorted(str(i) for i in issues))[:80] or "none"
            if ready:
                message = (
                    "정렬 실패 후 MFA 자동 점검 결과, 런타임 자체는 준비 상태입니다.\n"
                    "이번 실패는 입력 데이터(발음/사전/오디오) 또는 현재 정렬 프로필 영향일 가능성이 큽니다.\n\n"
                    f"- 실패 코드: {align_code or '(없음)'}\n"
                    f"- 실패 메시지: {align_message or '(없음)'}\n\n"
                    "권장:\n"
                    "1) 사전/라벨 파일 재생성\n"
                    "2) MFA 프로필을 '기본' 또는 '빠름'으로 변경 후 재시도\n"
                    "3) 계속 실패하면 'MFA 진단/복구' 버튼 실행\n\n"
                    "[자동 점검 요약]\n"
                    f"{summary}"
                )
                alert_key = f"mfa_after_align_fail_ready_{lang}_{issue_token}"
            else:
                recovery_guide = self._build_setup_mfa_recovery_guide()
                message = (
                    "정렬 실패 후 MFA 자동 점검에서 추가 복구가 필요하다고 판단되었습니다.\n"
                    "앱의 'MFA 진단/복구' 버튼을 눌러 복구를 진행해 주세요.\n\n"
                    f"- 실패 코드: {align_code or '(없음)'}\n"
                    f"- 실패 메시지: {align_message or '(없음)'}\n\n"
                    "[자동 점검 요약]\n"
                    f"{summary}\n\n"
                    "[수동 복구]\n"
                    f"{recovery_guide}"
                )
                alert_key = f"mfa_after_align_fail_repair_{lang}_{issue_token}"

            self._after_safe(
                lambda msg=message, key=alert_key: self._show_copyable_alert(
                    title="MFA 정렬 실패 후 점검 안내",
                    message=msg,
                    alert_key=key,
                )
            )
        except Exception as exc:
            self._append_log(f"⚠ 정렬 실패 후 MFA 자동 점검 중 오류: {exc}")
        finally:
            self._mfa_alignment_failure_followup_running = False

    def _run_mfa_diagnose_repair(self):
        self._notify_long_install_time("MFA")

        def task():
            self._set_running(True)
            lang = self._get_language()
            lang_label = "일본어" if lang == "japanese" else "한국어"
            self._set_status("🔍 MFA 진단/복구 중... (시간이 걸릴 수 있습니다)")
            try:
                resolved = self.mfa_path or find_mfa_executable() or ""
                if resolved and os.path.exists(resolved):
                    self.mfa_path = resolved
                before = diagnose_mfa_runtime(self.mfa_path or "", language=lang)
                self._append_log("🔎 MFA 진단 시작")
                self._append_log(self._format_mfa_diagnosis_summary(before))

                ok = False
                if not before.get("checks", {}).get("mfa_executable") or before.get("checks", {}).get("python_rebuild_required"):
                    self._append_log("ℹ MFA 환경이 없거나 재구성이 필요해 새로 설치합니다.")
                    ok = self._install_mfa_runtime(language=lang)
                else:
                    ok = self._repair_existing_mfa_runtime(language=lang)

                after = diagnose_mfa_runtime(self.mfa_path or "", language=lang)
                needs_attention = not bool(after.get("ready", False))
                recovery_guide = self._build_setup_mfa_recovery_guide() if needs_attention else ""
                summary = (
                    "[진단 전]\n"
                    f"{self._format_mfa_diagnosis_summary(before)}\n\n"
                    "[진단 후]\n"
                    f"{self._format_mfa_diagnosis_summary(after)}"
                )
                if needs_attention:
                    summary = (
                        f"{summary}\n\n"
                        "[추가 수동 복구]\n"
                        f"{recovery_guide}"
                    )
                self._after_safe(lambda s=summary, ready=after.get("ready", False), ll=lang_label: self._show_copyable_alert(
                    title=f"MFA 진단/복구 결과 ({ll})",
                    message=s,
                    alert_key=f"mfa_diag_repair_{lang}_{'ok' if ready else 'needs_attention'}",
                ))
                if ok and after.get("ready"):
                    self._update_mfa_status(True)
                    self._set_status("✅ MFA 진단/복구 완료")
                else:
                    self._set_status("❌ MFA 진단/복구 미완료")
            except Exception as e:
                self._handle_error("MFA 진단/복구", e)
            finally:
                self._set_running(False)

        self._run_in_thread(task)

    def _run_mfa_setup(self):
        """GUI 안에서 MFA 포터블 환경을 자동 설치합니다."""
        self._notify_long_install_time("MFA")
        def task():
            self._set_running(True)
            self._set_status("⏳ 조금만 기다려주세요, 프로그램 필수 요소들을 설치하고 점검하는 중입니다.")
            try:
                if self._install_mfa_runtime(language=self._get_language()):
                    self._set_status("✅ MFA 설치 완료")
                else:
                    self._set_status("❌ MFA 설치 실패")

            except Exception as e:
                self._handle_error("MFA 설치", e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    def _update_mfa_status(self, installed):
        """MFA 설치 상태를 UI에 반영합니다."""
        def _do():
            status_label = getattr(self, "mfa_status_label", None)
            install_btn = getattr(self, "mfa_install_btn", None)
            if status_label is None or install_btn is None:
                return
            if installed:
                status_label.configure(text="✅ MFA 설치됨", text_color="#4F8F61")
                install_btn.configure(text="✅ 설치 완료", state="disabled", fg_color="#388E3C")
            else:
                status_label.configure(text="❌ MFA 미설치", text_color="#B45A63")
                install_btn.configure(text="⬇ MFA 원클릭 설치", state="normal", fg_color="#FFA726")
        self._after_safe(_do)

    # --------------------------------------------------------------------------

    def _run_lab_dict_gen(self):
        def task():
            self._set_running(True)
            self._set_status("1~2단계 - Lab+사전 파일 생성 중...")
            try:
                wav_dir = self.wav_entry.get()
                if not wav_dir:
                    self._append_log("❌ WAV 경로를 먼저 지정해 주세요.")
                    return

                lang = self._get_language()
                if lang == "english":
                    self._append_log("ℹ 영어 Preview CVVC 모드에서는 Lab/사전 생성 단계를 사용하지 않습니다.")
                    self._set_status("✅ Lab+사전 단계 건너뜀 (영어 Preview)")
                    return

                custom_phonemes_path = self.custom_phoneme_var.get().strip()

                if lang == "japanese":
                    lab_count, lab_total, lab_errors = generate_ja_labs(
                        wav_dir,
                        custom_phonemes_path=custom_phonemes_path,
                        callback=self._append_log,
                    )
                else:
                    lab_count, lab_total, lab_errors = generate_labs(
                        wav_dir,
                        custom_phonemes_path=custom_phonemes_path,
                        callback=self._append_log,
                    )
                if lab_errors:
                    for err in lab_errors:
                        self._append_log(f"  ⚠ {err}")
                self._append_log(f"🧪 Lab 생성 완료 ({lab_count}/{lab_total})")

                dict_filename = "japanese_dict.txt" if lang == "japanese" else "korean_dict.txt"
                dict_path = os.path.join(wav_dir, dict_filename)
                if lang == "japanese":
                    _count, entries, dict_errors = generate_ja_dictionary(
                        wav_dir,
                        dict_path,
                        custom_phonemes_path=custom_phonemes_path,
                        callback=self._append_log,
                    )
                else:
                    _count, entries, dict_errors = generate_dictionary(
                        wav_dir,
                        dict_path,
                        custom_phonemes_path=custom_phonemes_path,
                        callback=self._append_log,
                    )
                if dict_errors:
                    for err in dict_errors:
                        self._append_log(f"  ⚠ {err}")
                self._append_log(f"📝 사전 저장 경로: {dict_path}")
                self._set_status(
                    f"✅ Lab+사전 생성 완료 (Lab {lab_count}/{lab_total}, Dict {entries}개 항목)"
                )
            except Exception as e:
                self._handle_error("Lab+사전 생성", e)
            finally:
                self._set_running(False)

        self._run_in_thread(task)

    def _run_lab_gen(self):
        def task():
            self._set_running(True)
            self._set_status("1단계 - Lab 파일 생성 중...")
            try:
                wav_dir = self.wav_entry.get()
                if not wav_dir:
                    self._append_log("❌ WAV 경로를 먼저 지정해 주세요.")
                    return

                lang = self._get_language()
                if lang == "english":
                    self._append_log("ℹ 영어 Preview CVVC 모드에서는 Lab 생성 단계를 사용하지 않습니다.")
                    self._set_status("✅ Lab 단계 건너뜀 (영어 Preview)")
                    return

                custom_phonemes_path = self.custom_phoneme_var.get().strip()

                if lang == "japanese":
                    count, total, errors = generate_ja_labs(wav_dir, custom_phonemes_path=custom_phonemes_path, callback=self._append_log)
                else:
                    count, total, errors = generate_labs(wav_dir, custom_phonemes_path=custom_phonemes_path, callback=self._append_log)
                if errors:
                    for e in errors:
                        self._append_log(f"  ⚠ {e}")
                self._set_status(f"✅ Lab 생성 완료 ({count}/{total})")
            except Exception as e:
                self._handle_error("Lab 생성", e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    def _run_dict_gen(self):
        def task():
            self._set_running(True)
            self._set_status("2단계 - 사전 파일 생성 중...")
            try:
                wav_dir = self.wav_entry.get()
                if not wav_dir:
                    self._append_log("❌ WAV 경로를 먼저 지정해 주세요.")
                    return

                custom_phonemes_path = self.custom_phoneme_var.get().strip()

                lang = self._get_language()
                if lang == "english":
                    self._append_log("ℹ 영어 Preview CVVC 모드에서는 사전 생성 단계를 사용하지 않습니다.")
                    self._set_status("✅ 사전 단계 건너뜀 (영어 Preview)")
                    return
                if lang == 'japanese':
                    dict_filename = "japanese_dict.txt"
                else:
                    dict_filename = "korean_dict.txt"
                dict_path = os.path.join(wav_dir, dict_filename)
                
                if lang == 'japanese':
                    count, entries, errors = generate_ja_dictionary(wav_dir, dict_path, custom_phonemes_path=custom_phonemes_path, callback=self._append_log)
                else:
                    count, entries, errors = generate_dictionary(wav_dir, dict_path, custom_phonemes_path=custom_phonemes_path, callback=self._append_log)
                if errors:
                    for e in errors:
                        self._append_log(f"  ⚠ {e}")
                self._append_log(f"📝 사전 저장 경로: {dict_path}")
                self._set_status(f"✅ 사전 생성 완료 ({entries}개 항목)")
            except Exception as e:
                self._handle_error("사전 생성", e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    def _run_profile_finetune(self):
        def task():
            self._set_running(True)
            self._set_status("⚙ 프로파일 미세 조정 중...")
            try:
                if not is_training_paths_enabled():
                    self._append_log("⚠ 배포 빌드에서는 프로파일 훈련/미세 조정 기능이 비활성화되어 있습니다.")
                    self._set_status("⚠ 배포 빌드에서는 프로파일 훈련 기능을 사용할 수 없습니다.")
                    return

                auto_oto = self.tune_auto_oto_var.get().strip()
                manual_oto = self.tune_manual_oto_var.get().strip()
                profile_out = self.tune_profile_out_var.get().strip()
                apply_target = self.tune_apply_target_var.get().strip()
                custom_phonemes_path = self.custom_phoneme_var.get().strip()

                if not auto_oto or not os.path.exists(auto_oto):
                    self._append_log("❌ 자동 생성 OTO 파일 경로가 올바르지 않습니다.")
                    return
                if not manual_oto or not os.path.exists(manual_oto):
                    self._append_log("❌ 수동 보정 OTO 파일 경로가 올바르지 않습니다.")
                    return

                if not profile_out:
                    base_dir = os.path.dirname(os.path.abspath(auto_oto))
                    if self._get_language() == "japanese":
                        profile_out = os.path.join(base_dir, ".ja_user_autotune_profile.json")
                    else:
                        profile_out = os.path.join(base_dir, ".kr_user_autotune_profile.json")
                    self.tune_profile_out_var.set(profile_out)

                if not apply_target:
                    apply_target = auto_oto
                    self.tune_apply_target_var.set(apply_target)
                if not os.path.exists(apply_target):
                    self._append_log(f"❌ 적용 대상 OTO 파일이 존재하지 않습니다: {apply_target}")
                    return

                lang = self._get_language()
                self._append_log(f"📝 프로파일 학습 시작 ({'일본어' if lang == 'japanese' else '한국어'})")
                self._append_log(f"   자동 OTO: {auto_oto}")
                self._append_log(f"   수동 OTO: {manual_oto}")

                if lang == "japanese":
                    from core.ja_oto_autotune import (
                        apply_ja_autotune_profile_to_oto,
                        save_ja_autotune_profile,
                        train_ja_autotune_profile,
                    )
                    profile = train_ja_autotune_profile(auto_oto, manual_oto, custom_phonemes_path=custom_phonemes_path)
                    if not profile:
                        self._append_log("⚠ 학습 가능한 매칭 데이터가 부족합니다. (최소 8개 이상 권장)")
                        return
                    if not save_ja_autotune_profile(profile_out, profile):
                        self._append_log(f"❌ 프로파일 저장 실패: {profile_out}")
                        return
                    changed = apply_ja_autotune_profile_to_oto(
                        apply_target, profile, custom_phonemes_path=custom_phonemes_path
                    )
                else:
                    from core.kr_oto_file_ops import (
                        apply_kr_autotune_profile_to_oto,
                        save_kr_autotune_profile,
                        train_kr_autotune_profile,
                    )
                    profile = train_kr_autotune_profile(auto_oto, manual_oto, custom_phonemes_path=custom_phonemes_path)
                    if not profile:
                        self._append_log("⚠ 학습 가능한 매칭 데이터가 부족합니다. (최소 8개 이상 권장)")
                        return
                    if not save_kr_autotune_profile(profile_out, profile):
                        self._append_log(f"❌ 프로파일 저장 실패: {profile_out}")
                        return
                    changed = apply_kr_autotune_profile_to_oto(
                        apply_target, profile, custom_phonemes_path=custom_phonemes_path
                    )

                pairs = int(profile.get("matched_pairs", 0))
                buckets = len((profile.get("buckets") or {}))
                self._append_log(f"✅ 프로파일 저장 완료: {profile_out}")
                self._append_log(f"✅ 학습 통계: matched_pairs={pairs}, buckets={buckets}")
                self._append_log(f"✅ 적용 완료: {changed} lines adjusted ({apply_target})")
                self._set_status(f"✅ 미세 조정 완료 ({changed} lines)")
            except Exception as e:
                self._handle_error("프로파일 기반 미세 조정", e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    def _run_full_pipeline(self):
        """Lab 생성 → 정렬 → OTO 생성 → 검증 순서로 전체 파이프라인을 실행합니다."""
        def task():
            self._set_running(True)
            stage_order = ["lab", "dict", "align", "oto", "validate"]
            stage_weights = {
                "lab": 0.22,
                "dict": 0.12,
                "align": 0.28,
                "oto": 0.28,
                "validate": 0.10,
            }
            stage_progress = {name: 0.0 for name in stage_order}

            def _safe_ratio(value):
                try:
                    return max(0.0, min(1.0, float(value)))
                except Exception:
                    return 0.0

            def _overall_progress():
                return sum(stage_weights[name] * stage_progress.get(name, 0.0) for name in stage_order)

            def _set_stage_progress(stage, ratio):
                ratio_clamped = _safe_ratio(ratio)
                prev = stage_progress.get(stage, 0.0)
                if ratio_clamped < prev:
                    ratio_clamped = prev
                stage_progress[stage] = ratio_clamped
                self._set_progress(_overall_progress())

            def _extract_progress_ratio(msg):
                text = str(msg or "")
                ratio = self._parse_progress_ratio_from_status(text) if hasattr(self, "_parse_progress_ratio_from_status") else None
                if ratio is not None:
                    return ratio
                m = re.search(r"(?:files?|rows?|items?)\s*=\s*(\d+)\s*/\s*(\d+)", text, flags=re.IGNORECASE)
                if m:
                    total = int(m.group(2))
                    if total > 0:
                        return max(0.0, min(1.0, float(int(m.group(1))) / float(total)))
                return None

            def _make_stage_callback(stage):
                def _cb(msg):
                    self._append_log(msg)
                    ratio = _extract_progress_ratio(msg)
                    if ratio is not None:
                        _set_stage_progress(stage, ratio)
                return _cb

            try:
                # Step 1: Lab
                _set_stage_progress("lab", 0.02)
                self._set_status("1/5 - Lab 파일 생성 중...")
                wav_dir = self.wav_entry.get()
                if not wav_dir:
                    self._append_log("❌ WAV 경로를 먼저 지정해 주세요.")
                    return
                out_path = self.out_entry.get().strip()
                cleanup_snapshot = self._snapshot_output_tree_for_cleanup(out_path) if out_path else None
                lang = self._get_language()
                selected_format = normalize_auto_format_value(
                    lang,
                    self.auto_format_var.get() if hasattr(self, "auto_format_var") else "",
                )
                textgrid_dir = os.path.join(wav_dir, "textgrids")
                has_textgrid = False
                if os.path.isdir(textgrid_dir):
                    try:
                        has_textgrid = any(str(name).lower().endswith(".textgrid") for name in os.listdir(textgrid_dir))
                    except Exception:
                        has_textgrid = False
                tpl_path_preflight = "" if self.no_base_oto_var.get() else self.tpl_entry.get().strip()
                custom_phonemes_path = self.custom_phoneme_var.get().strip()
                aligner_engine = normalize_aligner_name(
                    self.aligner_var.get() if hasattr(self, "aligner_var") else "mfa",
                    default="mfa",
                )
                no_mfa_auto_mode = (
                    aligner_engine == "none"
                    and lang != "english"
                    and not (lang == "korean" and selected_format == "cmpx")
                )
                no_mfa_mode_code = (
                    self._get_no_mfa_oto_mode_code()
                    if hasattr(self, "_get_no_mfa_oto_mode_code")
                    else "remap"
                )
                no_mfa_mode_text = (
                    "에일리어스 기반 자동 생성(빈 OTO 기준)"
                    if no_mfa_mode_code == "alias_auto"
                    else "베이스 OTO 재매핑 + 보정"
                )
                no_mfa_source_oto = ""
                if no_mfa_auto_mode:
                    if bool(self.no_base_oto_var.get()):
                        self._append_log("❌ No-MFA 자동설정 모드에서는 베이스 OTO(템플릿 ini)가 필요합니다.")
                        self._set_status("❌ 베이스 OTO 필요")
                        return
                    no_mfa_source_oto = resolve_no_mfa_source_oto(
                        wav_dir=wav_dir,
                        source_hint=tpl_path_preflight,
                    )
                    if not no_mfa_source_oto:
                        self._append_log("❌ No-MFA 자동설정용 베이스 OTO를 찾지 못했습니다.")
                        self._append_log("   템플릿 OTO 경로에 baseoto.ini 또는 oto.ini를 지정해 주세요.")
                        self._set_status("❌ 베이스 OTO 필요")
                        return
                    tpl_path_preflight = no_mfa_source_oto

                if lang == "english":
                    if not getattr(self, "_is_preview_channel", lambda: False)():
                        self._append_log("❌ 영어 CVVC 모드는 Preview 채널에서만 사용할 수 있습니다.")
                        self._set_status("❌ Preview 전용 기능")
                        return
                    if not out_path:
                        self._append_log("❌ 출력 경로를 먼저 지정해 주세요.")
                        self._set_status("❌ 출력 경로 누락")
                        return
                    if self.no_base_oto_var.get() or not tpl_path_preflight or not os.path.isfile(tpl_path_preflight):
                        self._append_log("❌ 영어 Preview CVVC 모드는 베이스 OTO(템플릿 ini)가 필수입니다.")
                        self._append_log("   템플릿 OTO 경로에 base oto.ini 또는 oto.ini를 지정해 주세요.")
                        self._set_status("❌ 베이스 OTO 필요")
                        return

                    self._append_log("ℹ 영어 Preview CVVC 모드: Lab/사전/정렬 단계를 건너뜁니다.")
                    self._append_log(f"[EN-CVVC] base oto: {tpl_path_preflight}")
                    _set_stage_progress("lab", 1.0)
                    _set_stage_progress("dict", 1.0)
                    _set_stage_progress("align", 1.0)

                    en_pack = self.en_cvvc_pack_var.get() if hasattr(self, "en_cvvc_pack_var") else "LITE"
                    en_beat = self.en_cvvc_beat_var.get() if hasattr(self, "en_cvvc_beat_var") else "8-beat"
                    en_preset = self.en_cvvc_preset_var.get() if hasattr(self, "en_cvvc_preset_var") else "Core"
                    en_list_fallback = (
                        bool(self.en_cvvc_list_fallback_var.get())
                        if hasattr(self, "en_cvvc_list_fallback_var")
                        else True
                    )
                    self._append_log(
                        f"[EN-CVVC] pack={en_pack} beat={en_beat} preset={en_preset} "
                        f"list_only_synth={'ON' if en_list_fallback else 'OFF'}"
                    )

                    _set_stage_progress("oto", 0.03)
                    self._set_status("4/5 - EN CVVC OTO 생성 중...")
                    _processed, _total, oto_errors = generate_en_cvvc_oto(
                        wav_dir=wav_dir,
                        out_path=out_path,
                        pack=en_pack,
                        beat=en_beat,
                        preset=en_preset,
                        alias_suffix=self.alias_suffix_var.get().strip(),
                        include_list_only_synthesis=en_list_fallback,
                        callback=_make_stage_callback("oto"),
                    )
                    if _total:
                        _set_stage_progress("oto", float(_processed) / float(_total))
                    _set_stage_progress("oto", 1.0)

                    _set_stage_progress("validate", 0.05)
                    self._set_status("5/5 - OTO 자동 검증 중...")
                    self._run_auto_validation(wav_dir, textgrid_dir, out_path, callback=_make_stage_callback("validate"))
                    _set_stage_progress("validate", 1.0)
                    if oto_errors:
                        self._append_log(f"⚠ OTO 생성 중 오류가 있습니다. ({len(oto_errors)}건)")
                        for err in oto_errors:
                            self._append_log(f"  - {err}")
                    else:
                        self._cleanup_generated_output_artifacts(out_path, snapshot=cleanup_snapshot)

                    self._set_status("✅ 전체 파이프라인 완료")
                    self._append_log("\n" + "=" * 50)
                    self._append_log("✅ 모든 작업이 정상적으로 완료되었습니다!")
                    self._append_log("=" * 50)
                    return

                if lang == "korean" and selected_format == "cmpx":
                    if not getattr(self, "_is_preview_channel", lambda: False)():
                        self._append_log("❌ 한국어 CMPX 모드는 Preview 채널에서만 사용할 수 있습니다.")
                        self._set_status("❌ Preview 전용 기능")
                        return
                    if hasattr(self, "ml_route_var"):
                        try:
                            current_route = (
                                self._get_ml_route_code()
                                if hasattr(self, "_get_ml_route_code")
                                else str(self.ml_route_var.get() or "").strip().lower()
                            )
                        except Exception:
                            current_route = ""
                        if current_route != "nomfa":
                            try:
                                if hasattr(self, "_set_ml_route_from_code"):
                                    self._set_ml_route_from_code("nomfa")
                                else:
                                    self.ml_route_var.set("No-MFA")
                            except Exception:
                                pass
                            self._append_log("ℹ CMPX 모드 제한: ML route를 No-MFA로 고정합니다.")
                    os.environ["UTOA_ML_ROUTE"] = "autofree_v1"
                    os.environ["UTOA_ML_AUTOFREE_AUX_ENABLE"] = "1"
                    os.environ["UTOA_ML_LEGACY_FALLBACK_ENABLE"] = "0"
                    if not out_path:
                        self._append_log("❌ 출력 경로를 먼저 지정해 주세요.")
                        self._set_status("❌ 출력 경로 누락")
                        return
                    if self.no_base_oto_var.get():
                        self._append_log("❌ 한국어 CMPX Preview 모드는 베이스 OTO(템플릿 ini)가 필수입니다.")
                        self._set_status("❌ 베이스 OTO 필요")
                        return
                    source_oto = resolve_kr_cmpx_preview_source_oto(
                        wav_dir=wav_dir,
                        source_hint=tpl_path_preflight,
                    )
                    if not source_oto:
                        self._append_log("❌ 한국어 CMPX Preview용 베이스 OTO를 찾지 못했습니다.")
                        self._append_log("   템플릿 OTO에 비교용 oto.ini 또는 baseoto.ini를 지정해 주세요.")
                        self._set_status("❌ 베이스 OTO 필요")
                        return

                    self._append_log("ℹ 한국어 CMPX Preview 모드: Lab/사전/정렬 단계를 건너뜁니다.")
                    self._append_log(f"[KR-CMPX] base oto: {source_oto}")
                    _set_stage_progress("lab", 1.0)
                    _set_stage_progress("dict", 1.0)
                    _set_stage_progress("align", 1.0)

                    _set_stage_progress("oto", 0.03)
                    self._set_status("4/5 - KR CMPX Preview OTO 생성 중...")
                    _processed, _total, oto_errors = generate_kr_cmpx_preview_oto(
                        wav_dir=wav_dir,
                        out_path=out_path,
                        source_oto_path=source_oto,
                        alias_suffix=self.alias_suffix_var.get().strip(),
                        callback=_make_stage_callback("oto"),
                    )
                    if _total:
                        _set_stage_progress("oto", float(_processed) / float(_total))
                    _set_stage_progress("oto", 1.0)

                    _set_stage_progress("validate", 0.05)
                    self._set_status("5/5 - OTO 자동 검증 중...")
                    self._run_auto_validation(wav_dir, textgrid_dir, out_path, callback=_make_stage_callback("validate"))
                    _set_stage_progress("validate", 1.0)
                    if oto_errors:
                        self._append_log(f"⚠ OTO 생성 중 오류가 있습니다. ({len(oto_errors)}건)")
                        for err in oto_errors:
                            self._append_log(f"  - {err}")
                    else:
                        self._cleanup_generated_output_artifacts(out_path, snapshot=cleanup_snapshot)

                    self._set_status("✅ 전체 파이프라인 완료")
                    self._append_log("\n" + "=" * 50)
                    self._append_log("✅ 모든 작업이 정상적으로 완료되었습니다!")
                    self._append_log("=" * 50)
                    return

                preflight = collect_runtime_preflight_issues(
                    language=lang,
                    wav_dir=wav_dir,
                    out_path=out_path,
                    aligner=self.aligner_var.get(),
                    textgrid_dir=textgrid_dir,
                    tpl_path=tpl_path_preflight,
                    no_base_oto=bool(self.no_base_oto_var.get()),
                    custom_phonemes_path=custom_phonemes_path,
                    require_output=False,
                )
                warning_records = list(preflight.get("warning_records") or [])
                error_records = list(preflight.get("error_records") or [])
                for item in warning_records:
                    self._append_log(f"⚠ {item.get('display')}")
                if error_records:
                    for item in error_records:
                        self._append_log(f"❌ {item.get('display')}")
                    first_code = str((error_records or [{}])[0].get("code", "PRECHECK_FAILED"))
                    self._set_status(f"❌ 사전 점검 실패 ({first_code})")
                    return

                if no_mfa_auto_mode:
                    self._append_log("ℹ No-MFA 모드: Lab/사전/정렬 단계를 건너뜁니다.")
                    if has_textgrid:
                        self._append_log("ℹ TextGrid가 있어도 No-MFA 선택 시에는 선택한 No-MFA 생성 방식으로 진행합니다.")
                    self._append_log(f"ℹ No-MFA 생성 방식: {no_mfa_mode_text}")
                    self._append_log(f"[No-MFA] base oto: {no_mfa_source_oto}")
                    _set_stage_progress("lab", 1.0)
                    _set_stage_progress("dict", 1.0)
                    _set_stage_progress("align", 1.0)

                    _set_stage_progress("oto", 0.03)
                    self._set_status("4/5 - No-MFA 자동설정 OTO 생성 중...")
                    _processed, _total, oto_errors = generate_no_mfa_auto_oto(
                        wav_dir=wav_dir,
                        out_path=out_path,
                        source_oto_path=no_mfa_source_oto,
                        alias_suffix=self.alias_suffix_var.get().strip(),
                        language=lang,
                        stats_oto_path=os.environ.get("UTOA_NO_MFA_STATS_OTO", ""),
                        generation_mode=no_mfa_mode_code,
                        callback=_make_stage_callback("oto"),
                    )
                    if _total:
                        _set_stage_progress("oto", float(_processed) / float(_total))
                    _set_stage_progress("oto", 1.0)

                    _set_stage_progress("validate", 0.05)
                    self._set_status("5/5 - OTO 자동 검증 중...")
                    self._run_auto_validation(wav_dir, textgrid_dir, out_path, callback=_make_stage_callback("validate"))
                    _set_stage_progress("validate", 1.0)
                    if oto_errors:
                        self._append_log(f"⚠ OTO 생성 중 오류가 있습니다. ({len(oto_errors)}건)")
                        for err in oto_errors:
                            self._append_log(f"  - {err}")
                    else:
                        self._cleanup_generated_output_artifacts(out_path, snapshot=cleanup_snapshot)

                    self._set_status("✅ 전체 파이프라인 완료")
                    self._append_log("\n" + "=" * 50)
                    self._append_log("✅ 모든 작업이 정상적으로 완료되었습니다!")
                    self._append_log("=" * 50)
                    return

                if lang == 'japanese':
                    lab_count, lab_total, _lab_errors = generate_ja_labs(
                        wav_dir,
                        custom_phonemes_path=custom_phonemes_path,
                        callback=_make_stage_callback("lab"),
                    )
                else:
                    lab_count, lab_total, _lab_errors = generate_labs(
                        wav_dir,
                        custom_phonemes_path=custom_phonemes_path,
                        callback=_make_stage_callback("lab"),
                    )
                if lab_total:
                    _set_stage_progress("lab", float(lab_count) / float(lab_total))
                _set_stage_progress("lab", 1.0)

                # Step 2: Dictionary
                _set_stage_progress("dict", 0.05)
                self._set_status("2/5 - 사전 파일 생성 중...")
                dict_filename = "japanese_dict.txt" if lang == 'japanese' else "korean_dict.txt"
                dict_path = os.path.join(wav_dir, dict_filename)
                if lang == 'japanese':
                    generate_ja_dictionary(
                        wav_dir,
                        dict_path,
                        custom_phonemes_path=custom_phonemes_path,
                        callback=_make_stage_callback("dict"),
                    )
                else:
                    generate_dictionary(
                        wav_dir,
                        dict_path,
                        custom_phonemes_path=custom_phonemes_path,
                        callback=_make_stage_callback("dict"),
                    )
                _set_stage_progress("dict", 1.0)

                # Step 3: Alignment
                output_dir = os.path.join(wav_dir, "textgrids")
                align_ok = False
                align_err = ""
                align_engine = self.aligner_var.get()
                primary_engine = normalize_aligner_name(align_engine, default="mfa")
                fallback_engine = ""
                _set_stage_progress("align", 0.05)
                if primary_engine == "none":
                    self._set_status("3/5 - 정렬 건너뛰기(no-MFA)")
                else:
                    self._set_status("3/5 - MFA 정렬 준비 중...")
                    if not self._ensure_mfa_ready_for_language(lang):
                        self._append_log("❌ MFA 설치/모델 준비 실패")
                        self._set_status("❌ MFA 설치/모델 준비 실패")
                        return
                mfa_profile = (
                    self._get_mfa_align_profile_code()
                    if hasattr(self, "_get_mfa_align_profile_code")
                    else "accurate"
                )
                if primary_engine == "mfa":
                    self._append_log(f"ℹ MFA 정렬 프로필: {mfa_profile}")
                else:
                    self._append_log("ℹ 정렬 엔진: none (MFA 비사용)")
                if hasattr(self, "_apply_advanced_tuning_envs"):
                    self._apply_advanced_tuning_envs()

                align_result = run_alignment_with_fallback(
                    language=lang,
                    wav_folder=wav_dir,
                    dictionary_path=dict_path,
                    output_folder=output_dir,
                    primary_aligner=primary_engine,
                    fallback_aligner=fallback_engine,
                    mfa_path=self.mfa_path or "",
                    mfa_align_profile=(
                        self._get_mfa_align_profile_code()
                        if hasattr(self, "_get_mfa_align_profile_code")
                        else "accurate"
                    ),
                    callback=_make_stage_callback("align"),
                )
                align_ok = bool(align_result.get("ok", False))
                align_err = str(align_result.get("message", "") or "")
                if align_result.get("fallback_used"):
                    self._append_log(f"ℹ 정렬 fallback 경로: {align_result.get('fallback_path', '')}")
                if primary_engine == "none":
                    has_textgrid = False
                    if os.path.isdir(output_dir):
                        for _name in os.listdir(output_dir):
                            if str(_name).lower().endswith(".textgrid"):
                                has_textgrid = True
                                break
                    if not has_textgrid:
                        self._append_log("❌ No-MFA 모드는 OTO 생성을 위해 TextGrid 입력이 필요합니다.")
                        self._append_log("   대안: 정렬 엔진을 MFA로 변경하거나 textgrids 폴더에 TextGrid를 미리 준비하세요.")
                        self._set_status("❌ No-MFA 입력 부족 (TextGrid 필요)")
                        return

                if not align_ok:
                    align_code = str(align_result.get("code", "ALIGN_RUN_FAILED"))
                    self._append_log(f"⚠ {format_error_with_recovery(align_code, align_err)}")
                    if primary_engine == "mfa":
                        self._schedule_alignment_failure_mfa_followup(
                            language=lang,
                            align_code=align_code,
                            align_message=align_err,
                        )
                    self._append_log("⚠ 정렬 실패 상태로 다음 단계를 진행합니다.")
                _set_stage_progress("align", 1.0)

                # Step 4: OTO
                _set_stage_progress("oto", 0.03)
                self._set_status("4/5 - OTO.ini 생성 중...")
                tpl_path = "" if self.no_base_oto_var.get() else self.tpl_entry.get()
                if out_path:
                    tg_folder = os.path.join(wav_dir, "textgrids")
                    params = self._get_params()
                    gen_ou = self.openutau_var.get()
                    gen_missing = self.gen_missing_vowels_var.get()
                    gen_dash_alias = (
                        self.gen_dash_alias_var.get()
                        if hasattr(self, "gen_dash_alias_var")
                        else True
                    )
                    enable_ml_correction = self.enable_ml_correction_var.get()
                    # 공개 배포에서는 복잡한 ML 파라미터 대신 자동 정책을 사용합니다.
                    kr_continuity_max_offset_adj_raw = (
                        self.kr_continuity_max_offset_adj_var.get()
                        if hasattr(self, "kr_continuity_max_offset_adj_var")
                        else ""
                    )
                    kr_vc_neighbor_enable = (
                        bool(self.kr_vc_neighbor_enable_var.get())
                        if hasattr(self, "kr_vc_neighbor_enable_var")
                        else True
                    )
                    ja_vc_neighbor_enable = (
                        bool(self.ja_vc_neighbor_enable_var.get())
                        if hasattr(self, "ja_vc_neighbor_enable_var")
                        else True
                    )
                    kr_vc_neighbor_blend_raw = (
                        self.kr_vc_neighbor_blend_var.get()
                        if hasattr(self, "kr_vc_neighbor_blend_var")
                        else ""
                    )
                    kr_vc_neighbor_max_shift_raw = (
                        self.kr_vc_neighbor_max_shift_var.get()
                        if hasattr(self, "kr_vc_neighbor_max_shift_var")
                        else ""
                    )
                    kr_vc_neighbor_lead_ms_raw = (
                        self.kr_vc_neighbor_lead_ms_var.get()
                        if hasattr(self, "kr_vc_neighbor_lead_ms_var")
                        else ""
                    )
                    kr_vc_neighbor_tail_ms_raw = (
                        self.kr_vc_neighbor_tail_ms_var.get()
                        if hasattr(self, "kr_vc_neighbor_tail_ms_var")
                        else ""
                    )
                    kr_vc_neighbor_min_len_raw = (
                        self.kr_vc_neighbor_min_len_var.get()
                        if hasattr(self, "kr_vc_neighbor_min_len_var")
                        else ""
                    )
                    ja_vc_neighbor_blend_raw = (
                        self.ja_vc_neighbor_blend_var.get()
                        if hasattr(self, "ja_vc_neighbor_blend_var")
                        else ""
                    )
                    ja_vc_neighbor_max_shift_raw = (
                        self.ja_vc_neighbor_max_shift_var.get()
                        if hasattr(self, "ja_vc_neighbor_max_shift_var")
                        else ""
                    )
                    ja_vc_neighbor_lead_ms_raw = (
                        self.ja_vc_neighbor_lead_ms_var.get()
                        if hasattr(self, "ja_vc_neighbor_lead_ms_var")
                        else ""
                    )
                    ja_vc_neighbor_tail_ms_raw = (
                        self.ja_vc_neighbor_tail_ms_var.get()
                        if hasattr(self, "ja_vc_neighbor_tail_ms_var")
                        else ""
                    )
                    ja_vc_neighbor_min_len_raw = (
                        self.ja_vc_neighbor_min_len_var.get()
                        if hasattr(self, "ja_vc_neighbor_min_len_var")
                        else ""
                    )
                    ml_anchor_mel_gamma_raw = (
                        self.ml_anchor_mel_gamma_var.get()
                        if hasattr(self, "ml_anchor_mel_gamma_var")
                        else ""
                    )
                    ml_route = (
                        str(self.ml_route_var.get()).strip().lower()
                        if hasattr(self, "ml_route_var")
                        else "autofree_v1"
                    )
                    kr_continuity_max_offset_adj_raw = (
                        self.kr_continuity_max_offset_adj_var.get()
                        if hasattr(self, "kr_continuity_max_offset_adj_var")
                        else ""
                    )
                    kr_vc_neighbor_enable = (
                        bool(self.kr_vc_neighbor_enable_var.get())
                        if hasattr(self, "kr_vc_neighbor_enable_var")
                        else True
                    )
                    ja_vc_neighbor_enable = (
                        bool(self.ja_vc_neighbor_enable_var.get())
                        if hasattr(self, "ja_vc_neighbor_enable_var")
                        else True
                    )
                    kr_vc_neighbor_blend_raw = (
                        self.kr_vc_neighbor_blend_var.get()
                        if hasattr(self, "kr_vc_neighbor_blend_var")
                        else ""
                    )
                    kr_vc_neighbor_max_shift_raw = (
                        self.kr_vc_neighbor_max_shift_var.get()
                        if hasattr(self, "kr_vc_neighbor_max_shift_var")
                        else ""
                    )
                    kr_vc_neighbor_lead_ms_raw = (
                        self.kr_vc_neighbor_lead_ms_var.get()
                        if hasattr(self, "kr_vc_neighbor_lead_ms_var")
                        else ""
                    )
                    kr_vc_neighbor_tail_ms_raw = (
                        self.kr_vc_neighbor_tail_ms_var.get()
                        if hasattr(self, "kr_vc_neighbor_tail_ms_var")
                        else ""
                    )
                    kr_vc_neighbor_min_len_raw = (
                        self.kr_vc_neighbor_min_len_var.get()
                        if hasattr(self, "kr_vc_neighbor_min_len_var")
                        else ""
                    )
                    ja_vc_neighbor_blend_raw = (
                        self.ja_vc_neighbor_blend_var.get()
                        if hasattr(self, "ja_vc_neighbor_blend_var")
                        else ""
                    )
                    ja_vc_neighbor_max_shift_raw = (
                        self.ja_vc_neighbor_max_shift_var.get()
                        if hasattr(self, "ja_vc_neighbor_max_shift_var")
                        else ""
                    )
                    ja_vc_neighbor_lead_ms_raw = (
                        self.ja_vc_neighbor_lead_ms_var.get()
                        if hasattr(self, "ja_vc_neighbor_lead_ms_var")
                        else ""
                    )
                    ja_vc_neighbor_tail_ms_raw = (
                        self.ja_vc_neighbor_tail_ms_var.get()
                        if hasattr(self, "ja_vc_neighbor_tail_ms_var")
                        else ""
                    )
                    ja_vc_neighbor_min_len_raw = (
                        self.ja_vc_neighbor_min_len_var.get()
                        if hasattr(self, "ja_vc_neighbor_min_len_var")
                        else ""
                    )
                    ml_anchor_mel_gamma_raw = (
                        self.ml_anchor_mel_gamma_var.get()
                        if hasattr(self, "ml_anchor_mel_gamma_var")
                        else ""
                    )
                    ml_model_root = ""
                    if lang == "japanese" and hasattr(self, "ml_model_root_ja_var"):
                        ml_model_root = str(self.ml_model_root_ja_var.get() or "").strip()
                    if lang == "korean" and hasattr(self, "ml_model_root_kr_var"):
                        ml_model_root = str(self.ml_model_root_kr_var.get() or "").strip()
                    # (legacy tuning values removed)
                    auto_format = self.auto_format_var.get()
                    custom_phonemes_path = self.custom_phoneme_var.get().strip()
                    alias_suffix = self.alias_suffix_var.get().strip()
                    ja_alias_style = self._get_ja_alias_style_code()
                    ja_words_fallback = self.ja_mapping_words_fallback_enabled_var.get() if hasattr(self, "ja_mapping_words_fallback_enabled_var") else True
                    ja_spn_threshold = self.ja_mapping_spn_ratio_threshold_var.get() if hasattr(self, "ja_mapping_spn_ratio_threshold_var") else 0.35
                    ja_min_vowel_ratio = self.ja_mapping_min_vowel_phone_ratio_var.get() if hasattr(self, "ja_mapping_min_vowel_phone_ratio_var") else 0.5
                    ja_debug_reason = self.ja_mapping_debug_reason_logging_var.get() if hasattr(self, "ja_mapping_debug_reason_logging_var") else True
                    auto_policy = self._apply_auto_ml_policy_env(
                        lang,
                        auto_format,
                        enable_ml_default=enable_ml_correction,
                        gen_dash_alias=gen_dash_alias,
                    )
                    enable_ml_correction = bool(auto_policy.get("enable_ml"))
                    self._apply_advanced_tuning_envs()
                    self._append_log(
                        f"[OTO-ML] auto policy: ml={'ON' if enable_ml_correction else 'OFF'}, "
                        f"route={auto_policy.get('route')}, coupled={'ON' if auto_policy.get('has_coupled') else 'OFF'}, "
                        f"hybrid={'ON' if auto_policy.get('hybrid_routing') else 'OFF'}"
                    )
                    if not enable_ml_correction:
                        self._append_log("[OTO-ML] ML 모델이 없어 보정을 건너뜁니다.")
                    route_code = str(auto_policy.get("route", "") or "").strip().lower()
                    if route_code in {"nomfa", "v2"}:
                        self._append_log(f"[OTO-ML] route policy: {route_code} (coupled primary + autofree auxiliary)")
                    if self.no_base_oto_var.get():
                        self._append_log("ℹ '베이스 OTO 없이 생성'이 활성화되어 OpenUtau 스타일로 생성합니다.")

                    oto_errors = []
                    generation_runtime_report = {}
                    if lang == 'japanese':
                        self._append_log(f"ℹ 일본어 별칭 스타일: {self.ja_alias_style_var.get()}")
                        _processed, _total, oto_errors = generate_ja_oto(
                            tg_folder, tpl_path, out_path,
                            params=None,
                            generate_openutau=gen_ou,
                            gen_missing_vowels=gen_missing,
                            enable_ml_correction=enable_ml_correction,
                            alias_style=ja_alias_style,
                            ja_mapping_words_fallback_enabled=bool(ja_words_fallback),
                            ja_mapping_spn_ratio_threshold=float(ja_spn_threshold),
                            ja_mapping_min_vowel_phone_ratio=float(ja_min_vowel_ratio),
                            ja_mapping_debug_reason_logging=bool(ja_debug_reason),
                            auto_format=auto_format,
                            custom_phonemes_path=custom_phonemes_path,
                            alias_suffix=alias_suffix,
                            callback=_make_stage_callback("oto"),
                            runtime_report=generation_runtime_report,
                        )
                    else:
                        _processed, _total, oto_errors = generate_oto(
                            tg_folder, tpl_path, out_path,
                            params,
                            gen_ou,
                            gen_missing,
                            enable_ml_correction=enable_ml_correction,
                            auto_format=auto_format,
                            custom_phonemes_path=custom_phonemes_path,
                            alias_suffix=alias_suffix,
                            callback=_make_stage_callback("oto"),
                            runtime_report=generation_runtime_report,
                        )
                    if isinstance(generation_runtime_report, dict):
                        gen_code = str(generation_runtime_report.get("code", "") or "")
                        gen_msg = str(generation_runtime_report.get("message", "") or "")
                        gen_processed = int(generation_runtime_report.get("processed", _processed) or _processed)
                        gen_total = int(generation_runtime_report.get("total", _total) or _total)
                        self._append_log(
                            f"[Generate] code={gen_code or '-'} "
                            f"processed={gen_processed}/{gen_total} "
                            f"message={gen_msg or '-'}"
                        )
                        gen_mapping = (
                            generation_runtime_report.get("mapping")
                            if isinstance(generation_runtime_report.get("mapping"), dict)
                            else {}
                        )
                        if gen_mapping:
                            self._append_log(
                                "[Generate][Mapping] " + format_mapping_summary(gen_mapping)
                            )
                            self._append_log(
                                "[Generate][MappingReason] "
                                + format_mapping_reason_schema_summary(gen_mapping)
                            )
                    if _total:
                        _set_stage_progress("oto", float(_processed) / float(_total))
                    _set_stage_progress("oto", 1.0)

                    _set_stage_progress("validate", 0.05)
                    self._set_status("5/5 - OTO 자동 검증 중...")
                    self._run_auto_validation(wav_dir, tg_folder, out_path, callback=_make_stage_callback("validate"))
                    _set_stage_progress("validate", 1.0)
                    if oto_errors:
                        self._append_log(f"⚠ OTO 생성 중 오류가 있어 자동 정리를 건너뜁니다. ({len(oto_errors)}건)")
                    else:
                        self._cleanup_generated_output_artifacts(out_path, snapshot=cleanup_snapshot)
                else:
                    self._append_log("⚠ 출력 경로가 없어 OTO 생성을 건너뜁니다.")
                    _set_stage_progress("oto", 1.0)
                    _set_stage_progress("validate", 1.0)

                self._set_status("✅ 전체 파이프라인 완료")
                self._append_log("\n" + "=" * 50)
                self._append_log("✅ 모든 작업이 정상적으로 완료되었습니다!")
                self._append_log("=" * 50)

            except Exception as e:
                self._handle_error("전체 파이프라인", e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)



