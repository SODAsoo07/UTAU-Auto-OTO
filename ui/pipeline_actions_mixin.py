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



class PipelineActionsMixin:
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
                primary_engine = "mfa"
                fallback_engine = ""
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
                    callback=self._append_log,
                )
                align_ok = bool(align_result.get("ok", False))
                align_err = str(align_result.get("message", "") or "")
                if align_result.get("fallback_used"):
                    self._append_log(f"ℹ 정렬 fallback 경로: {align_result.get('fallback_path', '')}")

                if not align_ok:
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
                    ml_coupled_enable = (
                        self.ml_coupled_enable_var.get()
                        if hasattr(self, "ml_coupled_enable_var")
                        else True
                    )
                    ml_coupled_min_conf_raw = (
                        self.ml_coupled_min_conf_var.get()
                        if hasattr(self, "ml_coupled_min_conf_var")
                        else ""
                    )
                    ml_coupled_device = (
                        str(self.ml_coupled_device_var.get()).strip().lower()
                        if hasattr(self, "ml_coupled_device_var")
                        else "auto"
                    )
                    ml_coupled_backend = (
                        str(self.ml_coupled_backend_var.get()).strip().lower()
                        if hasattr(self, "ml_coupled_backend_var")
                        else "auto"
                    )
                    kr_continuity_max_offset_adj_raw = (
                        self.kr_continuity_max_offset_adj_var.get()
                        if hasattr(self, "kr_continuity_max_offset_adj_var")
                        else ""
                    )
                    ml_model_root = ""
                    if lang == "japanese" and hasattr(self, "ml_model_root_ja_var"):
                        ml_model_root = str(self.ml_model_root_ja_var.get() or "").strip()
                    if lang == "korean" and hasattr(self, "ml_model_root_kr_var"):
                        ml_model_root = str(self.ml_model_root_kr_var.get() or "").strip()
                    ml_coupled_strict_constraint = (
                        self.ml_coupled_strict_constraint_var.get()
                        if hasattr(self, "ml_coupled_strict_constraint_var")
                        else False
                    )
                    auto_format = self.auto_format_var.get()
                    custom_phonemes_path = self.custom_phoneme_var.get().strip()
                    alias_suffix = self.alias_suffix_var.get().strip()
                    ja_alias_style = self._get_ja_alias_style_code()
                    ja_words_fallback = self.ja_mapping_words_fallback_enabled_var.get() if hasattr(self, "ja_mapping_words_fallback_enabled_var") else True
                    ja_spn_threshold = self.ja_mapping_spn_ratio_threshold_var.get() if hasattr(self, "ja_mapping_spn_ratio_threshold_var") else 0.35
                    ja_min_vowel_ratio = self.ja_mapping_min_vowel_phone_ratio_var.get() if hasattr(self, "ja_mapping_min_vowel_phone_ratio_var") else 0.5
                    ja_debug_reason = self.ja_mapping_debug_reason_logging_var.get() if hasattr(self, "ja_mapping_debug_reason_logging_var") else True
                    selector_mode_code = self._apply_ml_selector_runtime_mode(ml_selector_mode)
                    if ml_coupled_device not in {"auto", "cpu", "cuda"}:
                        ml_coupled_device = "auto"
                    ml_coupled_backend = {
                        "ensemble_v1": "ensemble",
                        "coupled_nn_v1": "v1",
                        "coupled_nn_v2_rawmel": "v2",
                    }.get(ml_coupled_backend, ml_coupled_backend)
                    if ml_coupled_backend not in {"auto", "ensemble", "v1", "v2"}:
                        ml_coupled_backend = "auto"
                    ensemble_enabled = ml_coupled_backend in {"auto", "ensemble"}

                    def _parse_optional_threshold(raw_value, lo=0.0, hi=1.0):
                        text = str(raw_value or "").strip()
                        if not text:
                            return None
                        try:
                            return max(float(lo), min(float(hi), float(text)))
                        except Exception:
                            return None

                    def _parse_optional_float(raw_value, lo=0.0, hi=None):
                        text = str(raw_value or "").strip()
                        if not text:
                            return None
                        try:
                            value = max(float(lo), float(text))
                            if hi is not None:
                                value = min(float(hi), value)
                            return value
                        except Exception:
                            return None

                    ml_coupled_min_conf = _parse_optional_threshold(ml_coupled_min_conf_raw, lo=0.0, hi=1.0)
                    kr_continuity_max_offset_adj = _parse_optional_float(kr_continuity_max_offset_adj_raw, lo=0.0, hi=2000.0)

                    os.environ["UTOA_ML_COUPLED_ENABLE"] = "1" if ml_coupled_enable else "0"
                    os.environ["UTOA_ML_ENSEMBLE_ENABLE"] = "1" if ensemble_enabled else "0"
                    if ml_coupled_min_conf is None:
                        os.environ.pop("UTOA_ML_COUPLED_MIN_CONF", None)
                    else:
                        os.environ["UTOA_ML_COUPLED_MIN_CONF"] = str(float(ml_coupled_min_conf))
                    os.environ["UTOA_ML_COUPLED_DEVICE"] = str(ml_coupled_device)
                    coupled_backend_env = {"auto": "auto", "ensemble": "auto", "v1": "v1", "v2": "v2"}.get(
                        ml_coupled_backend,
                        "auto",
                    )
                    os.environ["UTOA_ML_COUPLED_BACKEND"] = str(coupled_backend_env)
                    os.environ["UTOA_ML_COUPLED_STRICT_CONSTRAINT"] = "1" if ml_coupled_strict_constraint else "0"
                    if kr_continuity_max_offset_adj is None:
                        os.environ.pop("UTOA_KR_CONTINUITY_MAX_OFFSET_ADJ", None)
                    else:
                        os.environ["UTOA_KR_CONTINUITY_MAX_OFFSET_ADJ"] = str(float(kr_continuity_max_offset_adj))
                    if lang == "japanese":
                        if ml_model_root:
                            os.environ["UTOA_JA_OTO_ML_DIR"] = ml_model_root
                        else:
                            os.environ.pop("UTOA_JA_OTO_ML_DIR", None)
                    else:
                        if ml_model_root:
                            os.environ["UTOA_KR_OTO_ML_DIR"] = ml_model_root
                        else:
                            os.environ.pop("UTOA_KR_OTO_ML_DIR", None)
                    self._append_log(
                        f"[OTO-ML] 설정: ml={'ON' if enable_ml_correction else 'OFF'}, selector={self._describe_ml_selector_mode(selector_mode_code)}"
                    )
                    min_conf_display = f"{float(ml_coupled_min_conf):.2f}" if ml_coupled_min_conf is not None else "default(0.55)"
                    self._append_log(
                        f"[OTO-ML] coupled={'ON' if ml_coupled_enable else 'OFF'}, backend={ml_coupled_backend}, ensemble={'ON' if ensemble_enabled else 'OFF'}, min_conf={min_conf_display}, device={ml_coupled_device}, strict={'ON' if ml_coupled_strict_constraint else 'OFF'}"
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


