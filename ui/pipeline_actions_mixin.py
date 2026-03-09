import os
import locale
import subprocess as sp

from core.alignment_pipeline import run_alignment_with_fallback
from core.ja_lab_generator import generate_ja_dictionary, generate_ja_labs
from core.ja_oto_generator import (
    apply_ja_autotune_profile_to_oto,
    generate_ja_oto,
    save_ja_autotune_profile,
    train_ja_autotune_profile,
)
from core.lab_generator import generate_dictionary, generate_labs
from core.mfa_runner import (
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
    patch_mfa_korean_support,
)
from core.oto_generator import (
    apply_kr_autotune_profile_to_oto,
    generate_oto,
    save_kr_autotune_profile,
    train_kr_autotune_profile,
)
from core.sofa_runner import (
    download_default_sofa_model,
    ensure_sofa_support,
    get_default_sofa_repo_dir,
    find_sofa_ckpt,
    get_default_sofa_model_root,
    get_sofa_env_python,
    get_sofa_release_link,
    is_sofa_ready,
)


class PipelineActionsMixin:
    def _notify_long_install_time(self, target="MFA"):
        target_text = str(target or "MFA").upper()
        if target_text == "SOFA":
            title = "SOFA 초기 설치 안내"
            message = (
                "처음 SOFA를 설치할 때는 저장소 다운로드와 Python 패키지 설치 때문에 시간이 꽤 걸릴 수 있습니다.\n\n"
                "환경과 네트워크 속도에 따라 보통 수 분에서 10분 이상 걸릴 수 있습니다.\n"
                "설치 중에는 프로그램을 종료하지 말고 기다려 주세요."
            )
            alert_key = "install_time_sofa"
        else:
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

    def _preferred_subprocess_encoding(self):
        # Windows 배포 환경에서 utf-8 고정 디코딩 시 콘솔 로그가 깨지는 경우가 있어,
        # 시스템 기본 인코딩을 우선 사용하고 실패 시 utf-8로 폴백한다.
        return locale.getpreferredencoding(False) or "utf-8"

    def _get_sofa_repo_dir_for_language(self, language):
        override = self._read_runtime_var("sofa_repo_dir_var", "") or ""
        override = str(override).strip()
        if override:
            return override
        if (language or "").lower() == "korean":
            return get_default_sofa_repo_dir("utau_kr_v1")
        return get_default_sofa_repo_dir()

    def _get_sofa_runtime_kwargs(self, language):
        lang = (language or "").lower()
        out_formats = self._read_runtime_var("sofa_out_formats_var", "TextGrid")
        save_conf_default = lang == "korean"
        two_pass_default = lang == "korean"
        return {
            "sofa_repo_dir": self._get_sofa_repo_dir_for_language(lang),
            "mode": str(self._read_runtime_var("sofa_mode_var", "force") or "force"),
            "g2p": str(self._read_runtime_var("sofa_g2p_var", "Dictionary") or "Dictionary"),
            "ap_detector": str(
                self._read_runtime_var(
                    "sofa_ap_detector_var",
                    "LoudnessSpectralcentroidAPDetector",
                )
                or "LoudnessSpectralcentroidAPDetector"
            ),
            "ap_detector_config": str(
                self._read_runtime_var("sofa_ap_detector_config_var", "") or ""
            ),
            "save_confidence": self._to_bool(
                self._read_runtime_var("sofa_save_confidence_var", save_conf_default),
                default=save_conf_default,
            ),
            "out_formats": out_formats,
            "extra_infer_args": self._read_runtime_var("sofa_extra_infer_args_var", ""),
            "two_pass_retry": self._to_bool(
                self._read_runtime_var("sofa_two_pass_retry_var", two_pass_default),
                default=two_pass_default,
            ),
            "two_pass_retry_mode": str(
                self._read_runtime_var("sofa_two_pass_retry_mode_var", "match") or "match"
            ),
            "confidence_threshold": self._read_runtime_var("sofa_confidence_threshold_var", 0.55),
            "low_confidence_max_files": self._read_runtime_var("sofa_low_confidence_max_files_var", 0),
        }

    def _download_sofa_model_for_current_language(self):
        """현재 언어 기준 SOFA 모델을 GitHub 릴리즈에서 자동 다운로드합니다."""
        def task():
            self._set_running(True)
            try:
                lang = self._get_language()
                self._set_status("SOFA 모델 다운로드 중...")
                self._append_log(f"⬇ SOFA 모델 자동 다운로드 시작 ({'일본어' if lang == 'japanese' else '한국어'})")
                ok, model_path, err = download_default_sofa_model(
                    language=lang,
                    target_root=get_default_sofa_model_root(),
                    callback=self._append_log,
                )
                if ok and model_path:
                    self._after_safe(lambda p=model_path: self.sofa_ckpt_var.set(p))
                    self._append_log(f"✅ SOFA 모델 다운로드 완료: {model_path}")
                    self._set_status("✅ SOFA 모델 다운로드 완료")
                else:
                    self._append_log(f"❌ SOFA 모델 다운로드 실패: {err}")
                    self._set_status("❌ SOFA 모델 다운로드 실패")
            except Exception as e:
                self._append_log(f"❌ SOFA 모델 자동 다운로드 중 예외: {e}")
                self._set_status("❌ SOFA 모델 다운로드 실패")
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    def _ensure_sofa_model_ready(self, language):
        """SOFA ckpt 경로를 확보합니다. 없으면 사용자 폴더 탐색 후 자동 다운로드를 시도합니다."""
        ckpt = (self.sofa_ckpt_var.get() or "").strip()
        if ckpt and os.path.exists(ckpt):
            return ckpt

        found = find_sofa_ckpt(language, search_root=get_default_sofa_model_root())
        if found and os.path.exists(found):
            self._append_log(f"ℹ 사용자 폴더에서 SOFA 모델 자동 감지: {found}")
            self._after_safe(lambda p=found: self.sofa_ckpt_var.set(p))
            return found

        self._append_log("ℹ SOFA 모델이 지정되지 않아 자동 다운로드를 시도합니다...")
        ok, model_path, err = download_default_sofa_model(
            language=language,
            target_root=get_default_sofa_model_root(),
            callback=self._append_log,
        )
        if ok and model_path and os.path.exists(model_path):
            self._after_safe(lambda p=model_path: self.sofa_ckpt_var.set(p))
            self._append_log(f"✅ SOFA 모델 자동 준비 완료: {model_path}")
            return model_path

        release_link = get_sofa_release_link(language)
        model_root = get_default_sofa_model_root()
        self._after_safe(lambda: self._show_copyable_alert(
            title="SOFA 모델 다운로드 실패",
            message=(
                f"SOFA 모델 자동 다운로드에 실패했습니다.\n\n"
                f"언어: {'일본어' if language == 'japanese' else '한국어'}\n"
                f"기본 모델 폴더:\n{model_root}\n\n"
                f"수동 다운로드 링크:\n{release_link}\n\n"
                f"다운로드한 `.ckpt` 파일을 모델 폴더에 넣은 뒤 다시 시도해 주세요.\n\n"
                f"오류 요약:\n{err or '알 수 없는 오류'}"
            ),
            alert_key=f"sofa_model_download_fail_{language}",
        ))
        return ""

    def _notify_mfa_failure_suggest_sofa(self, language, err_msg=""):
        """MFA 실패 시 SOFA 정렬로 전환하는 방법을 안내합니다."""
        model_root = get_default_sofa_model_root()
        release_link = get_sofa_release_link(language)
        self._append_log("⚠ MFA 정렬에 실패했습니다. SOFA 정렬로 전환할 수 있습니다.")
        self._append_log(f"   SOFA 기본 모델 폴더: {model_root}")
        self._append_log(f"   SOFA 모델 다운로드 링크: {release_link}")
        self._after_safe(lambda: self._show_copyable_alert(
            title="MFA 실패 - SOFA 전환 안내",
            message=(
                "MFA 정렬이 실패했습니다.\n\n"
                "대안으로 SOFA 정렬을 사용할 수 있습니다.\n"
                "먼저 SOFA 모델(.ckpt)을 준비해 주세요.\n\n"
                f"SOFA 모델 폴더(기본):\n{model_root}\n\n"
                f"SOFA 모델 다운로드 링크:\n{release_link}\n\n"
                f"MFA 오류:\n{err_msg or '(없음)'}"
            ),
            alert_key=f"mfa_fail_sofa_hint_{language}",
        ))

    def _install_mfa_runtime(self, language="korean"):
        import shutil

        lang = str(language or "korean").strip().lower()
        app_dir = getattr(self, "app_dir", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env_dir = get_default_mfa_env_dir()
        legacy_conda_root = get_default_mfa_conda_root()
        micromamba_root = get_default_mfa_micromamba_root()
        micromamba_exe = get_default_mfa_micromamba_exe()
        if any(ord(ch) > 127 for ch in app_dir):
            self._append_log("⚠ 앱 경로에 비ASCII 문자가 있어도 MFA 환경은 공용 폴더를 사용합니다.")
        self._append_log(f"ℹ MFA 공용 환경 경로: {env_dir}")
        self._append_log(f"ℹ MFA Micromamba 경로: {micromamba_root}")
        mfa_exe = os.path.join(env_dir, 'Scripts', 'mfa.exe')
        micromamba_archive = os.path.join(app_dir, 'micromamba-win-64-latest.tar.bz2')

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
            self._set_status("⬇ Micromamba 다운로드 중...")
            self._append_log("[1/3] ⬇ Micromamba 다운로드 중... (약 15MB)")
            url = 'https://micro.mamba.pm/api/micromamba/win-64/latest'
            ps_cmd = (
                f'[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; '
                f"Invoke-WebRequest -Uri '{url}' -OutFile '{micromamba_archive}'"
            )
            result = sp.run(['powershell', '-Command', ps_cmd], capture_output=True, text=True)
            if result.returncode != 0:
                self._append_log(f"❌ Micromamba 다운로드 실패: {result.stderr or result.stdout}")
                return False
            return True

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
                    self._append_log(f"❌ Micromamba 압축 해제 실패: {result.stderr or result.stdout}")
                    return False
                resolved = get_default_mfa_micromamba_exe()
                if not os.path.exists(resolved):
                    self._append_log("❌ Micromamba 실행 파일을 찾지 못했습니다.")
                    return False
                return True
            except Exception as e:
                self._append_log(f"❌ Micromamba 준비 실패: {e}")
                return False

        def _run_micromamba(cmd, step_label):
            env = os.environ.copy()
            env["MAMBA_ROOT_PREFIX"] = micromamba_root
            process = sp.Popen(
                [micromamba_exe, *cmd],
                stdout=sp.PIPE,
                stderr=sp.STDOUT,
                text=True,
                encoding=self._preferred_subprocess_encoding(),
                errors='replace',
                env=env,
            )
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
                probe = sp.run(
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
            self._append_log("[Patch] 윈도우용 한국어 파서(eunjeon) 연동 처리 중...")
            patch_mfa_korean_support(self.mfa_path, callback=self._append_log)
            if not ensure_korean_support(self.mfa_path, callback=self._append_log):
                self._append_log("❌ 한국어 MFA 의존성 준비 실패")
                return False
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
        return "\n".join([
            f"언어: {lang}",
            f"MFA 실행 파일: {'OK' if checks.get('mfa_executable') else '없음'}",
            f"Python 실행 파일: {'OK' if checks.get('python_exe') else '없음'}",
            f"pip 실행 파일: {'OK' if checks.get('pip_exe') else '없음'}",
            f"Python 버전: {checks.get('python_version') or '(확인 불가)'}",
            f"재구성 필요: {'예' if checks.get('python_rebuild_required') else '아니오'}",
            f"패키지 도구(pip/setuptools/wheel): {('OK' if checks.get('packaging_stack') else '복구 필요') if checks.get('packaging_stack') is not None else '미확인'}",
            f"{lang} 의존성: {('OK' if checks.get('language_support') else '복구 필요') if checks.get('language_support') is not None else '미확인'}",
            f"{lang} MFA 모델: {'OK' if checks.get('model_ready') else '다운로드 필요'}",
            f"이슈 코드: {', '.join(issues) if issues else '(없음)'}",
            f"최종 준비 상태: {'준비 완료' if (report or {}).get('ready') else '추가 복구 필요'}",
        ])

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
        if not self.mfa_path or not os.path.exists(self.mfa_path):
            self._notify_long_install_time("MFA")
            self._append_log("ℹ MFA가 없어 지금 자동 설치를 시작합니다.")
            return self._install_mfa_runtime(language=lang)
        if mfa_env_requires_python_downgrade(self.mfa_path):
            py_ver = get_mfa_env_python_version(self.mfa_path)
            self._append_log(
                f"⚠ 현재 MFA 환경 Python {py_ver or '(unknown)'} 은/는 "
                f"Windows MFA 의존성과 호환되지 않아 Python {MFA_PORTABLE_PYTHON_VERSION} 기준으로 다시 구성합니다."
            )
            self.mfa_path = ""
            return self._install_mfa_runtime(language=lang)
        return self._repair_existing_mfa_runtime(language=lang)

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
                summary = (
                    "[진단 전]\n"
                    f"{self._format_mfa_diagnosis_summary(before)}\n\n"
                    "[진단 후]\n"
                    f"{self._format_mfa_diagnosis_summary(after)}"
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
            self._set_status("⬇ MFA 자동 설치 중... (10~20분 소요)")
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

    def _is_sofa_installed(self):
        lang = self._get_language()
        repo_dir = self._get_sofa_repo_dir_for_language(lang)
        ok, _ = is_sofa_ready(
            sofa_python=self.sofa_python_var.get().strip(),
            mfa_path=self.mfa_path or "",
            sofa_repo_dir=repo_dir,
        )
        return ok

    def _run_sofa_setup(self):
        """SOFA 지원 환경을 자동 설치/점검합니다."""
        self._notify_long_install_time("SOFA")
        def task():
            self._set_running(True)
            self._set_status("⬇ SOFA 설치 준비 중... (수 분 소요)")
            try:
                lang = self._get_language()
                repo_dir = self._get_sofa_repo_dir_for_language(lang)
                self._append_log("🔧 SOFA 지원 환경 점검을 시작합니다.")
                self._append_log(f"ℹ SOFA repo 경로: {repo_dir}")
                ok, err = ensure_sofa_support(
                    mfa_path=self.mfa_path or "",
                    sofa_python=self.sofa_python_var.get().strip(),
                    sofa_repo_dir=repo_dir,
                    callback=self._append_log,
                )
                if ok:
                    if not self.sofa_python_var.get().strip():
                        self.sofa_python_var.set(get_sofa_env_python())
                    self._update_sofa_status(True)
                    self._append_log("✅ SOFA 설치 완료")
                    self._set_status("✅ SOFA 준비 완료")
                else:
                    self._append_log(f"❌ SOFA 설치 실패: {err}")
                    self._update_sofa_status(False)
                    self._set_status("❌ SOFA 설치 실패")
            except Exception as e:
                self._append_log(f"❌ SOFA 설치 중 예외: {e}")
                self._update_sofa_status(False)
                self._set_status("❌ SOFA 설치 실패")
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    def _update_mfa_status(self, installed):
        """MFA 설치 상태를 UI에 반영합니다."""
        def _do():
            if installed:
                self.mfa_status_label.configure(text="✅ MFA 설치됨", text_color="#66BB6A")
                self.mfa_install_btn.configure(text="✅ 설치 완료", state="disabled", fg_color="#388E3C")
            else:
                self.mfa_status_label.configure(text="❌ MFA 미설치", text_color="#FF6B6B")
                self.mfa_install_btn.configure(text="⬇ MFA 원클릭 설치", state="normal", fg_color="#FFA726")
        self._after_safe(_do)

    def _update_sofa_status(self, installed):
        def _do():
            if hasattr(self, "sofa_status_label"):
                if installed:
                    self.sofa_status_label.configure(text="✅ SOFA 설치됨", text_color="#66BB6A")
                else:
                    self.sofa_status_label.configure(text="❌ SOFA 미설치", text_color="#FF6B6B")
            if not hasattr(self, "sofa_install_btn"):
                return
            if installed:
                self.sofa_install_btn.configure(text="✅ 설치 완료", state="disabled", fg_color="#388E3C")
            else:
                self.sofa_install_btn.configure(text="⬇ SOFA 자동 설치", state="normal", fg_color="#42A5F5")
        self._after_safe(_do)

    # --------------------------------------------------------------------------

    def _run_lab_gen(self):
        def task():
            self._set_running(True)
            self._set_status("1단계 - Lab 파일 생성 중...")
            try:
                wav_dir = self.wav_entry.get()
                if not wav_dir:
                    self._append_log("❌ WAV 경로를 먼저 지정해 주세요.")
                    return

                custom_phonemes_path = self.custom_phoneme_var.get().strip()
                
                if self._get_language() == 'japanese':
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
            try:
                # Step 1: Lab
                self._set_status("1/4 - Lab 파일 생성 중...")
                wav_dir = self.wav_entry.get()
                if not wav_dir:
                    self._append_log("❌ WAV 경로를 먼저 지정해 주세요.")
                    return
                out_path = self.out_entry.get().strip()
                cleanup_snapshot = self._snapshot_output_tree_for_cleanup(out_path) if out_path else None

                custom_phonemes_path = self.custom_phoneme_var.get().strip()
                
                lang = self._get_language()
                if lang == 'japanese':
                    generate_ja_labs(wav_dir, custom_phonemes_path=custom_phonemes_path, callback=self._append_log)
                else:
                    generate_labs(wav_dir, custom_phonemes_path=custom_phonemes_path, callback=self._append_log)

                # Step 2: Dictionary
                self._set_status("2/4 - 사전 파일 생성 중...")
                dict_filename = "japanese_dict.txt" if lang == 'japanese' else "korean_dict.txt"
                dict_path = os.path.join(wav_dir, dict_filename)
                if lang == 'japanese':
                    generate_ja_dictionary(wav_dir, dict_path, custom_phonemes_path=custom_phonemes_path, callback=self._append_log)
                else:
                    generate_dictionary(wav_dir, dict_path, custom_phonemes_path=custom_phonemes_path, callback=self._append_log)

                # Step 3: Alignment
                output_dir = os.path.join(wav_dir, "textgrids")
                align_ok = False
                align_err = ""
                align_engine = self.aligner_var.get()
                primary_engine = "sofa" if align_engine == "SOFA" else "mfa"
                fallback_engine = "mfa" if primary_engine == "sofa" else ""
                sofa_kwargs = {}
                ckpt = ""
                sdic = dict_path
                if primary_engine == "sofa":
                    sofa_kwargs = self._get_sofa_runtime_kwargs(lang)
                    ckpt = self.sofa_ckpt_var.get().strip()
                    self._set_status("3/4 - SOFA 정렬 준비 중...")
                    self._append_log(f"ℹ SOFA 실행 Python: {self.sofa_python_var.get().strip()}")
                    self._append_log(
                        "ℹ SOFA 실행 설정: "
                        f"repo={sofa_kwargs.get('sofa_repo_dir')}, "
                        f"mode={sofa_kwargs.get('mode')}, "
                        f"g2p={sofa_kwargs.get('g2p')}, "
                        f"ap={sofa_kwargs.get('ap_detector')}, "
                        f"2-pass={'ON' if sofa_kwargs.get('two_pass_retry') else 'OFF'}"
                    )
                    ckpt = self._ensure_sofa_model_ready(lang)
                    sdic = self.sofa_dict_var.get().strip() or dict_path
                    if not self.sofa_dict_var.get().strip():
                        self._after_safe(lambda p=sdic: self.sofa_dict_var.set(p))
                        self._append_log(f"ℹ SOFA 사전이 비어 있어 현재 생성 사전을 사용합니다: {sdic}")
                else:
                    self._set_status("3/4 - MFA 정렬 준비 중...")
                    if not self._ensure_mfa_ready_for_language(lang):
                        self._append_log("❌ MFA 설치/모델 준비 실패")
                        self._set_status("❌ MFA 설치/모델 준비 실패")
                        return
                    mfa_profile = (
                        self._get_mfa_align_profile_code()
                        if hasattr(self, "_get_mfa_align_profile_code")
                        else "accurate"
                    )
                    self._append_log(f"ℹ MFA 정렬 프로필: {mfa_profile}")

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
                    sofa_kwargs={
                        **sofa_kwargs,
                        "ckpt_path": ckpt,
                        "dictionary_path": sdic,
                        "sofa_python": self.sofa_python_var.get().strip(),
                    },
                    callback=self._append_log,
                )
                align_ok = bool(align_result.get("ok", False))
                align_err = str(align_result.get("message", "") or "")
                if align_result.get("fallback_used"):
                    self._append_log(f"ℹ 정렬 fallback 경로: {align_result.get('fallback_path', '')}")

                if not align_ok:
                    show_advanced = bool(self.show_advanced_aligner_var.get()) if hasattr(self, "show_advanced_aligner_var") else False
                    if align_engine == "MFA" and align_err and show_advanced:
                        self._notify_mfa_failure_suggest_sofa(lang, align_err)
                    self._append_log("⚠ 정렬 실패 상태로 다음 단계를 진행합니다.")

                # Step 4: OTO
                self._set_status("4/4 - OTO.ini 생성 중...")
                tpl_path = "" if self.no_base_oto_var.get() else self.tpl_entry.get()
                if out_path:
                    tg_folder = os.path.join(wav_dir, "textgrids")
                    params = self._get_params()
                    gen_ou = self.openutau_var.get()
                    gen_missing = self.gen_missing_vowels_var.get()
                    enable_ml_correction = self.enable_ml_correction_var.get()
                    ml_selector_mode = self.ml_selector_mode_var.get() if hasattr(self, "ml_selector_mode_var") else "기본 정책"
                    auto_format = self.auto_format_var.get()
                    custom_phonemes_path = self.custom_phoneme_var.get().strip()
                    alias_suffix = self.alias_suffix_var.get().strip()
                    ja_alias_style = self._get_ja_alias_style_code()
                    ja_words_fallback = self.ja_mapping_words_fallback_enabled_var.get() if hasattr(self, "ja_mapping_words_fallback_enabled_var") else True
                    ja_spn_threshold = self.ja_mapping_spn_ratio_threshold_var.get() if hasattr(self, "ja_mapping_spn_ratio_threshold_var") else 0.35
                    ja_min_vowel_ratio = self.ja_mapping_min_vowel_phone_ratio_var.get() if hasattr(self, "ja_mapping_min_vowel_phone_ratio_var") else 0.5
                    ja_debug_reason = self.ja_mapping_debug_reason_logging_var.get() if hasattr(self, "ja_mapping_debug_reason_logging_var") else True
                    selector_mode_code = self._apply_ml_selector_runtime_mode(ml_selector_mode)
                    self._append_log(
                        f"[OTO-ML] 설정: ml={'ON' if enable_ml_correction else 'OFF'}, selector={self._describe_ml_selector_mode(selector_mode_code)}"
                    )
                    if self.no_base_oto_var.get():
                        self._append_log("ℹ '베이스 OTO 없이 생성'이 활성화되어 OpenUtau 스타일로 생성합니다.")

                    oto_errors = []
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
                            callback=self._append_log
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
                            callback=self._append_log
                        )
                    self._run_auto_validation(wav_dir, tg_folder, out_path)
                    if oto_errors:
                        self._append_log(f"⚠ OTO 생성 중 오류가 있어 자동 정리를 건너뜁니다. ({len(oto_errors)}건)")
                    else:
                        self._cleanup_generated_output_artifacts(out_path, snapshot=cleanup_snapshot)
                else:
                    self._append_log("⚠ 출력 경로가 없어 OTO 생성을 건너뜁니다.")

                self._set_status("✅ 전체 파이프라인 완료")
                self._append_log("\n" + "=" * 50)
                self._append_log("✅ 모든 작업이 정상적으로 완료되었습니다!")
                self._append_log("=" * 50)

            except Exception as e:
                self._handle_error("전체 파이프라인", e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)


