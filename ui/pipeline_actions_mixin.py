import os
import locale
import subprocess as sp

from core.ja_lab_generator import generate_ja_dictionary, generate_ja_labs
from core.ja_oto_generator import (
    apply_ja_autotune_profile_to_oto,
    generate_ja_oto,
    save_ja_autotune_profile,
    train_ja_autotune_profile,
)
from core.lab_generator import generate_dictionary, generate_labs
from core.mfa_runner import check_mfa_model, download_mfa_model, patch_mfa_korean_support, run_mfa_align
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
    run_sofa_align,
)


class PipelineActionsMixin:
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

    def _run_mfa_setup(self):
        """GUI 안에서 MFA 포터블 환경을 자동 설치합니다."""
        def task():
            self._set_running(True)
            self._set_status("⬇ MFA 자동 설치 중... (10~20분 소요)")
            try:
                import shutil
                app_dir = getattr(self, "app_dir", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                portable_env_dir = os.path.join(app_dir, '.env')
                public_root = os.environ.get('PUBLIC', r'C:\Users\Public')
                fallback_env_dir = os.path.join(public_root, 'UTAU_Auto_OTO_v3', '.env')
                env_dir = portable_env_dir
                if any(ord(ch) > 127 for ch in portable_env_dir):
                    env_dir = fallback_env_dir
                    self._append_log("⚠ 앱 경로에 비ASCII 문자가 있어 MFA 환경을 공용 폴더에 설치합니다.")
                    self._append_log(f"   대체 설치 경로: {env_dir}")
                mfa_exe = os.path.join(env_dir, 'Scripts', 'mfa.exe')
                installer = os.path.join(app_dir, 'Miniconda3-latest-Windows-x86_64.exe')

                # 이미 설치 확인
                if os.path.exists(mfa_exe):
                    self._append_log("✅ MFA가 이미 설치되어 있습니다!")
                    self.mfa_path = mfa_exe
                    self._update_mfa_status(True)
                    self._set_status("✅ MFA 준비 완료")
                    return

                system_conda = shutil.which('conda')

                if system_conda:
                    self._append_log(f"🔍 시스템에 설치된 Conda 발견: {system_conda}")
                    self._append_log("   Miniconda 다운로드를 건너뛰고 기존 Conda를 활용해 환경을 구성합니다.")
                    self._append_log("[1/2] 🔧 MFA 전용 로컬 환경 생성 및 설치 중... (5~10분)")
                    
                    cmd = [system_conda, 'create', '-y', '-p', env_dir, '-c', 'conda-forge', '--override-channels', 'montreal-forced-aligner', 'colorama']
                    process = sp.Popen(
                        cmd,
                        stdout=sp.PIPE,
                        stderr=sp.STDOUT,
                        text=True,
                        encoding=self._preferred_subprocess_encoding(),
                        errors='replace',
                    )
                    for line in process.stdout:
                        stripped = line.strip()
                        if stripped:
                            self._append_log(stripped)
                    process.wait()
                    
                    if process.returncode != 0:
                        self._append_log("❌ MFA 설치 실패")
                        return

                    self._append_log("[2/2] 📦 추가 의존성 모듈 설치 중...")
                    # conda run으로 해당 환경 내에서 pip 실행을 보장한다.
                    sp.run([system_conda, 'run', '-p', env_dir, 'pip', 'install', 'eunjeon', 'jamo', 'textgrid'], capture_output=True)

                    self._append_log("[Patch] 윈도우용 한국어 파서(eunjeon) 연동 처리 중...")
                    patch_mfa_korean_support(mfa_exe, callback=self._append_log)

                    self._append_log("✅ MFA 시스템 구성 완료!")
                
                else:
                    self._append_log("🔍 시스템 Conda를 찾지 못했습니다. Miniconda 포터블 환경을 구축합니다.")
                    conda_exe = os.path.join(env_dir, 'Scripts', 'conda.exe')
                    # Step 1: Miniconda 다운로드
                    if not os.path.exists(conda_exe):
                        if not os.path.exists(installer):
                            self._append_log("[1/3] ⬇ Miniconda 다운로드 중... (약 80MB)")
                            url = 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe'
                            ps_cmd = (
                                f'[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; '
                                f"Invoke-WebRequest -Uri '{url}' -OutFile '{installer}'"
                            )
                            result = sp.run(['powershell', '-Command', ps_cmd], capture_output=True, text=True)
                            if result.returncode != 0:
                                self._append_log(f"❌ 다운로드 실패: {result.stderr}")
                                return
                        self._append_log("✅ Miniconda 다운로드 완료!")

                        # Step 2: 포터블 설치
                        self._append_log("[2/3] 📦 Miniconda 포터블 설치 중... (2~5분)")
                        self._append_log(f"   설치 경로: {env_dir}")
                        # Miniconda(NSIS)는 /D= 경로를 raw command-line에서 파싱한다.
                        # subprocess(list)는 공백 경로를 자동 인용하면서 /D가 무시될 수 있어,
                        # 직접 command-line 문자열로 실행한다.
                        if os.path.isdir(env_dir) and not os.path.exists(conda_exe):
                            try:
                                shutil.rmtree(env_dir)
                                self._append_log("   이전 실패 흔적(.env 폴더)을 정리하고 재시도합니다.")
                            except Exception as cleanup_error:
                                self._append_log(f"❌ 기존 .env 폴더 정리 실패: {cleanup_error}")
                                return

                        install_cmd = (
                            f'"{installer}" /InstallationType=JustMe /RegisterPython=0 '
                            f'/AddToPath=0 /S /D={env_dir}'
                        )
                        result = sp.run(
                            install_cmd,
                            capture_output=True, text=True, timeout=1200
                        )
                        if result.returncode != 0 or not os.path.exists(conda_exe):
                            # /D 경로가 무시된 경우 기본 경로 설치를 보정 탐지한다.
                            user_home = os.path.expanduser('~')
                            fallback_conda_candidates = [
                                os.path.join(user_home, 'miniconda3', 'Scripts', 'conda.exe'),
                                os.path.join(user_home, 'Miniconda3', 'Scripts', 'conda.exe'),
                                os.path.join(os.environ.get('LOCALAPPDATA', ''), 'miniconda3', 'Scripts', 'conda.exe'),
                                os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Miniconda3', 'Scripts', 'conda.exe'),
                            ]
                            detected_conda = next((p for p in fallback_conda_candidates if p and os.path.exists(p)), None)
                            if detected_conda:
                                conda_exe = detected_conda
                                env_dir = os.path.dirname(os.path.dirname(conda_exe))
                                mfa_exe = os.path.join(env_dir, 'Scripts', 'mfa.exe')
                                self._append_log("⚠ 지정 경로에서 Conda를 찾지 못했지만 기본 설치 경로를 감지했습니다.")
                                self._append_log(f"   감지된 Conda: {conda_exe}")
                            else:
                                self._append_log(f"❌ Miniconda 설치 실패 (code={result.returncode})")
                                if result.stdout and result.stdout.strip():
                                    self._append_log(f"   stdout: {result.stdout.strip()[:500]}")
                                if result.stderr and result.stderr.strip():
                                    self._append_log(f"   stderr: {result.stderr.strip()[:500]}")
                                return
                        self._append_log("✅ Miniconda 설치 완료!")

                    # Step 3: MFA 설치
                    self._append_log("[3/3] 🔧 MFA 설치 중... (5~10분)")
                    # HTTP 000 에러를 줄이기 위해 conda-forge 채널만 강제로 사용한다.
                    process = sp.Popen(
                        [conda_exe, 'install', '-y', '-c', 'conda-forge', '--override-channels', 'montreal-forced-aligner', 'colorama'],
                        stdout=sp.PIPE,
                        stderr=sp.STDOUT,
                        text=True,
                        encoding=self._preferred_subprocess_encoding(),
                        errors='replace'
                    )
                    for line in process.stdout:
                        stripped = line.strip()
                        if stripped:
                            self._append_log(stripped)
                    process.wait()
                    if process.returncode != 0:
                        self._append_log("❌ MFA 설치 실패")
                        return

                    self._append_log("✅ Conda 패키지 설치 완료!")
                    
                    self._append_log("[3.5/4] 📦 추가 의존성 모듈 설치 중...")
                    sp.run([conda_exe, 'run', '-p', env_dir, 'pip', 'install', 'eunjeon', 'jamo', 'textgrid'], capture_output=True)

                    self._append_log("[Patch] 윈도우용 한국어 파서(eunjeon) 연동 처리 중...")
                    patch_mfa_korean_support(mfa_exe, callback=self._append_log)

                # Step 4: 한국어 모델 다운로드
                self._append_log("[추가 단계] ⬇ 한국어 음향 모델 다운로드 중... (1~2분)")
                process = sp.Popen(
                    [mfa_exe, 'model', 'download', 'acoustic', 'korean_mfa', '--ignore_cache'],
                    stdout=sp.PIPE,
                    stderr=sp.STDOUT,
                    text=True,
                    encoding=self._preferred_subprocess_encoding(),
                    errors='replace'
                )
                for line in process.stdout:
                    stripped = line.strip()
                    if stripped:
                        self._append_log(stripped)
                process.wait()

                # 임시 설치 파일 정리
                if os.path.exists(installer):
                    os.remove(installer)

                self.mfa_path = mfa_exe
                self._update_mfa_status(True)
                self._append_log("")
                self._append_log("✅ MFA 설치가 완료되었습니다!")
                self._append_log("   이제 '3단계 MFA 음성 정렬' 버튼으로 정렬을 진행할 수 있습니다.")
                self._set_status("✅ MFA 설치 완료")

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
                self.mfa_install_btn.configure(text="⬇ MFA 자동 설치", state="normal", fg_color="#FFA726")
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

                if align_engine == "SOFA":
                    self._set_status("3/4 - SOFA 음성 정렬 중...")
                    self._append_log(f"ℹ SOFA 전용 Python: {self.sofa_python_var.get().strip()}")
                    sofa_kwargs = self._get_sofa_runtime_kwargs(lang)
                    self._append_log(
                        "ℹ SOFA 실행 옵션: "
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
                    if not ckpt or not sdic:
                        align_err = "SOFA 체크포인트 또는 사전 경로 누락"
                        self._append_log(f"❌ {align_err}")
                    else:
                        align_ok, align_err = run_sofa_align(
                            wav_folder=wav_dir,
                            output_folder=output_dir,
                            ckpt_path=ckpt,
                            dictionary_path=sdic,
                            mfa_path=self.mfa_path or "",
                            sofa_python=self.sofa_python_var.get().strip(),
                            callback=self._append_log,
                            **sofa_kwargs,
                        )
                else:
                    if self.mfa_path:
                        self._set_status("3/4 - MFA 음성 정렬 중...")
                        has_model, _ = check_mfa_model(self.mfa_path, language=lang)
                        if not has_model:
                            download_mfa_model(self.mfa_path, language=lang, callback=self._append_log)
                        mfa_profile = (
                            self._get_mfa_align_profile_code()
                            if hasattr(self, "_get_mfa_align_profile_code")
                            else "accurate"
                        )
                        self._append_log(f"ℹ MFA 정렬 프로필: {mfa_profile}")
                        align_ok, align_err = run_mfa_align(
                            self.mfa_path,
                            wav_dir,
                            dict_path,
                            output_dir,
                            language=lang,
                            callback=self._append_log,
                            align_profile=mfa_profile,
                        )
                    else:
                        align_err = "MFA 실행 파일이 없어 정렬을 건너뜁니다."
                        self._append_log(f"⚠ {align_err}")

                if not align_ok:
                    if align_engine == "MFA" and align_err:
                        self._notify_mfa_failure_suggest_sofa(lang, align_err)
                    self._append_log("⚠ 정렬 실패 상태로 다음 단계를 진행합니다.")

                # Step 4: OTO
                self._set_status("4/4 - OTO.ini 생성 중...")
                tpl_path = "" if self.no_base_oto_var.get() else self.tpl_entry.get()
                out_path = self.out_entry.get()
                if out_path:
                    tg_folder = os.path.join(wav_dir, "textgrids")
                    params = self._get_params()
                    gen_ou = self.openutau_var.get()
                    gen_missing = self.gen_missing_vowels_var.get()
                    enable_ml_correction = self.enable_ml_correction_var.get()
                    auto_format = self.auto_format_var.get()
                    custom_phonemes_path = self.custom_phoneme_var.get().strip()
                    alias_suffix = self.alias_suffix_var.get().strip()
                    ja_alias_style = self._get_ja_alias_style_code()
                    ja_words_fallback = self.ja_mapping_words_fallback_enabled_var.get() if hasattr(self, "ja_mapping_words_fallback_enabled_var") else True
                    ja_spn_threshold = self.ja_mapping_spn_ratio_threshold_var.get() if hasattr(self, "ja_mapping_spn_ratio_threshold_var") else 0.35
                    ja_min_vowel_ratio = self.ja_mapping_min_vowel_phone_ratio_var.get() if hasattr(self, "ja_mapping_min_vowel_phone_ratio_var") else 0.5
                    ja_debug_reason = self.ja_mapping_debug_reason_logging_var.get() if hasattr(self, "ja_mapping_debug_reason_logging_var") else True
                    self._append_log(
                        f"[OTO-ML] 설정: ml={'ON' if enable_ml_correction else 'OFF'}"
                    )
                    if self.no_base_oto_var.get():
                        self._append_log("ℹ '베이스 OTO 없이 생성'이 활성화되어 OpenUtau 스타일로 생성합니다.")

                    if lang == 'japanese':
                        self._append_log(f"ℹ 일본어 별칭 스타일: {self.ja_alias_style_var.get()}")
                        generate_ja_oto(
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
                        generate_oto(
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


