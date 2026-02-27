import os
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
    find_sofa_ckpt,
    get_default_sofa_model_root,
    get_sofa_env_python,
    get_sofa_release_link,
    is_sofa_ready,
    run_sofa_align,
)


class PipelineActionsMixin:
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
            title="SOFA 모델 준비 실패",
            message=(
                f"SOFA 모델 자동 다운로드에 실패했습니다.\n\n"
                f"언어: {'일본어' if language == 'japanese' else '한국어'}\n"
                f"권장 저장 폴더:\n{model_root}\n\n"
                f"수동 다운로드 링크:\n{release_link}\n\n"
                f"다운로드 후 .ckpt 파일 경로를 SOFA 체크포인트에 지정해 주세요.\n\n"
                f"오류 요약:\n{err or '알 수 없는 오류'}"
            ),
            alert_key=f"sofa_model_download_fail_{language}",
        ))
        return ""

    def _notify_mfa_failure_suggest_sofa(self, language, err_msg=""):
        """MFA 실패 시 SOFA 대체 실행을 안내합니다."""
        model_root = get_default_sofa_model_root()
        release_link = get_sofa_release_link(language)
        self._append_log("⚠ MFA 정렬에 실패했습니다. SOFA 정렬 엔진으로 재시도해 보세요.")
        self._append_log(f"   SOFA 모델 자동 다운로드 위치: {model_root}")
        self._append_log(f"   모델 릴리즈: {release_link}")
        self._after_safe(lambda: self._show_copyable_alert(
            title="MFA 실패 - SOFA 재시도 안내",
            message=(
                "MFA 정렬이 실패했습니다.\n\n"
                "대안으로 정렬 엔진을 SOFA로 바꿔 다시 실행할 수 있습니다.\n"
                "SOFA 모델은 체크포인트가 비어 있으면 자동 다운로드를 시도합니다.\n\n"
                f"모델 저장 폴더(기본):\n{model_root}\n"
                f"모델 릴리즈 링크:\n{release_link}\n\n"
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
                portable_env_dir = os.path.join(APP_DIR, '.env')
                public_root = os.environ.get('PUBLIC', r'C:\Users\Public')
                fallback_env_dir = os.path.join(public_root, 'UTAU_Auto_OTO_v3', '.env')
                env_dir = portable_env_dir
                if any(ord(ch) > 127 for ch in portable_env_dir):
                    env_dir = fallback_env_dir
                    self._append_log("⚠ 앱 경로에 비ASCII 문자가 있어 MFA 환경을 공용 폴더에 설치합니다.")
                    self._append_log(f"   대체 설치 경로: {env_dir}")
                mfa_exe = os.path.join(env_dir, 'Scripts', 'mfa.exe')
                installer = os.path.join(APP_DIR, 'Miniconda3-latest-Windows-x86_64.exe')

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
                    self._append_log("   Miniconda 다운로드를 건너뛰고 기존 Conda를 활용하여 환경을 구성합니다.")
                    self._append_log("[1/2] 🔧 MFA 전용 로컬 환경 생성 및 설치 중... (5~10분, 용량이 큽니다)")
                    
                    cmd = [system_conda, 'create', '-y', '-p', env_dir, '-c', 'conda-forge', '--override-channels', 'montreal-forced-aligner', 'colorama']
                    process = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True, encoding='utf-8', errors='replace')
                    for line in process.stdout:
                        stripped = line.strip()
                        if stripped:
                            self._append_log(stripped)
                    process.wait()
                    
                    if process.returncode != 0:
                        self._append_log("❌ MFA 설치 실패")
                        return

                    self._append_log("[2/2] 📦 추가 의존성 모듈 설치 중...")
                    # conda run을 사용하여 해당 환경 내에서 pip 실행 보장
                    sp.run([system_conda, 'run', '-p', env_dir, 'pip', 'install', 'eunjeon', 'jamo', 'textgrid'], capture_output=True)

                    self._append_log("[Patch] 윈도우용 한국어 파서(eunjeon) 연동 처리 중...")
                    patch_mfa_korean_support(mfa_exe, callback=self._append_log)

                    self._append_log("✅ MFA 시스템 구성 완료!")
                
                else:
                    self._append_log("🔍 시스템 Conda를 찾을 수 없습니다. 자체 Miniconda 포터블 환경을 구축합니다.")
                    conda_exe = os.path.join(env_dir, 'Scripts', 'conda.exe')
                    # Step 1: Miniconda 다운로드
                    if not os.path.exists(conda_exe):
                        if not os.path.exists(installer):
                            self._append_log("[1/3] 📥 Miniconda 다운로드 중... (약 80MB)")
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
                            # /D 경로가 무시된 경우 기본 경로에 설치되었는지 보정 탐지
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
                    self._append_log("[3/3] 🔧 MFA 설치 중... (5~10분, 용량이 큽니다)")
                    # HTTP 000 에러를 막기 위해 --override-channels 로 conda-forge 만 강제 사용
                    process = sp.Popen(
                        [conda_exe, 'install', '-y', '-c', 'conda-forge', '--override-channels', 'montreal-forced-aligner', 'colorama'],
                        stdout=sp.PIPE, stderr=sp.STDOUT, text=True, encoding='utf-8', errors='replace'
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

                # Step 4: 한국어 모델
                self._append_log("[마지막] 🌐 한국어 음향 모델 다운로드 중... (1~2분)")
                process = sp.Popen(
                    [mfa_exe, 'model', 'download', 'acoustic', 'korean_mfa', '--ignore_cache'],
                    stdout=sp.PIPE, stderr=sp.STDOUT, text=True, encoding='utf-8', errors='replace'
                )
                for line in process.stdout:
                    stripped = line.strip()
                    if stripped:
                        self._append_log(stripped)
                process.wait()

                # 설치파일 정리
                if os.path.exists(installer):
                    os.remove(installer)

                self.mfa_path = mfa_exe
                self._update_mfa_status(True)
                self._append_log("")
                self._append_log("🎉 MFA 설치가 모두 완료되었습니다!")
                self._append_log("   이제 '3️⃣ MFA 음성 정렬' 버튼을 사용할 수 있습니다.")
                self._set_status("✅ MFA 설치 완료!")

            except Exception as e:
                self._handle_error("MFA 설치", e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    def _is_sofa_installed(self):
        ok, _ = is_sofa_ready(
            sofa_python=self.sofa_python_var.get().strip(),
            mfa_path=self.mfa_path or "",
        )
        return ok

    def _run_sofa_setup(self):
        """SOFA 전용 가상환경과 의존성을 자동 설치합니다."""
        def task():
            self._set_running(True)
            self._set_status("⬇ SOFA 자동 설치 중... (수 분 소요)")
            try:
                self._append_log("🔧 SOFA 자동 설치를 시작합니다.")
                ok, err = ensure_sofa_support(
                    mfa_path=self.mfa_path or "",
                    sofa_python=self.sofa_python_var.get().strip(),
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
                self._append_log(f"❌ SOFA 설치 중 오류: {e}")
                self._update_sofa_status(False)
                self._set_status("❌ SOFA 설치 실패")
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    def _update_mfa_status(self, installed):
        """MFA 상태 UI를 업데이트합니다."""
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

    # ── 개별 실행 ──

    def _run_lab_gen(self):
        def task():
            self._set_running(True)
            self._set_status("1️⃣ Lab 파일 생성 중...")
            try:
                wav_dir = self.wav_entry.get()
                if not wav_dir:
                    self._append_log("❌ WAV 폴더 경로를 입력해 주세요.")
                    return

                custom_phonemes_path = self.custom_phoneme_var.get().strip()
                
                if self._get_language() == 'japanese':
                    count, total, errors = generate_ja_labs(wav_dir, custom_phonemes_path=custom_phonemes_path, callback=self._append_log)
                else:
                    count, total, errors = generate_labs(wav_dir, custom_phonemes_path=custom_phonemes_path, callback=self._append_log)
                if errors:
                    for e in errors:
                        self._append_log(f"  ⚠️ {e}")
                self._set_status(f"✅ Lab 생성 완료 ({count}/{total})")
            except Exception as e:
                self._handle_error("Lab 생성", e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    def _run_dict_gen(self):
        def task():
            self._set_running(True)
            self._set_status("2️⃣ 사전 파일 생성 중...")
            try:
                wav_dir = self.wav_entry.get()
                if not wav_dir:
                    self._append_log("❌ WAV 폴더 경로를 입력해 주세요.")
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
                        self._append_log(f"  ⚠️ {e}")
                self._append_log(f"📘 사전 저장 경로: {dict_path}")
                self._set_status(f"✅ 사전 생성 완료 ({entries}개 항목)")
            except Exception as e:
                self._handle_error("사전 생성", e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    def _run_profile_finetune(self):
        def task():
            self._set_running(True)
            self._set_status("🧩 프로파일 미세조정 진행 중...")
            try:
                auto_oto = self.tune_auto_oto_var.get().strip()
                manual_oto = self.tune_manual_oto_var.get().strip()
                profile_out = self.tune_profile_out_var.get().strip()
                apply_target = self.tune_apply_target_var.get().strip()
                custom_phonemes_path = self.custom_phoneme_var.get().strip()

                if not auto_oto or not os.path.exists(auto_oto):
                    self._append_log("❌ 자동 OTO 입력 파일 경로가 비어있거나 파일이 없습니다.")
                    return
                if not manual_oto or not os.path.exists(manual_oto):
                    self._append_log("❌ 수동 OTO 참조 파일 경로가 비어있거나 파일이 없습니다.")
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
                    self._append_log(f"❌ 적용 대상 OTO 파일을 찾을 수 없습니다: {apply_target}")
                    return

                lang = self._get_language()
                self._append_log(f"🧪 미세조정 학습 시작 ({'일본어' if lang == 'japanese' else '한국어'})")
                self._append_log(f"   자동 OTO: {auto_oto}")
                self._append_log(f"   수동 OTO: {manual_oto}")

                if lang == "japanese":
                    profile = train_ja_autotune_profile(auto_oto, manual_oto, custom_phonemes_path=custom_phonemes_path)
                    if not profile:
                        self._append_log("⚠️ 학습 가능한 매칭 샘플이 부족합니다. (최소 8쌍 이상 권장)")
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
                        self._append_log("⚠️ 학습 가능한 매칭 샘플이 부족합니다. (최소 8쌍 이상 권장)")
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
                self._append_log(f"✅ 학습 결과: matched_pairs={pairs}, buckets={buckets}")
                self._append_log(f"✅ 적용 완료: {changed} lines adjusted ({apply_target})")
                self._set_status(f"✅ 미세조정 완료 ({changed} lines)")
            except Exception as e:
                self._handle_error("프로파일 미세조정", e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    def _run_full_pipeline(self):
        """전체 파이프라인을 순서대로 실행"""
        def task():
            self._set_running(True)
            try:
                # Step 1: Lab
                self._set_status("1/4 - Lab 파일 생성 중...")
                wav_dir = self.wav_entry.get()
                if not wav_dir:
                    self._append_log("❌ WAV 폴더 경로를 입력해 주세요.")
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

                # Step 3: MFA
                output_dir = os.path.join(wav_dir, "textgrids")
                if self.aligner_var.get() == "SOFA":
                    self._set_status("3/4 - SOFA 음성 정렬 중... (시간 소요)")
                    self._append_log(f"ℹ SOFA 전용 Python: {self.sofa_python_var.get().strip()}")
                    ckpt = self._ensure_sofa_model_ready(lang)
                    sdic = self.sofa_dict_var.get().strip() or dict_path
                    if not self.sofa_dict_var.get().strip():
                        self._after_safe(lambda p=sdic: self.sofa_dict_var.set(p))
                        self._append_log(f"ℹ SOFA 사전이 비어 있어 현재 생성 사전을 사용합니다: {sdic}")
                    if not ckpt or not sdic:
                        self._append_log("❌ SOFA 사용 시 체크포인트(.ckpt)와 사전(dictionary) 경로가 필요합니다.")
                        self._append_log("   SOFA 정렬 없이 계속 진행합니다 (기존 TextGrid 사용)")
                    else:
                        success, err = run_sofa_align(
                            wav_folder=wav_dir,
                            output_folder=output_dir,
                            ckpt_path=ckpt,
                            dictionary_path=sdic,
                            mfa_path=self.mfa_path or "",
                            sofa_python=self.sofa_python_var.get().strip(),
                            callback=self._append_log,
                        )
                        if not success:
                            self._append_log(f"❌ SOFA 실패: {err}")
                            self._append_log("   SOFA 없이 계속 진행합니다 (기존 TextGrid 사용)")
                else:
                    if self.mfa_path:
                        self._set_status("3/4 - MFA 음성 정렬 중... (시간 소요)")
                        has_model, _ = check_mfa_model(self.mfa_path, language=lang)
                        if not has_model:
                            download_mfa_model(self.mfa_path, language=lang, callback=self._append_log)
                        success, err = run_mfa_align(self.mfa_path, wav_dir, dict_path, output_dir, language=lang, callback=self._append_log)
                        if not success:
                            self._append_log(f"❌ MFA 실패: {err}")
                            self._notify_mfa_failure_suggest_sofa(lang, err)
                            self._append_log("   MFA 없이 계속 진행합니다 (기존 TextGrid 사용)")
                    else:
                        self._append_log("⚠️ MFA가 설치되어 있지 않습니다. TextGrid를 직접 준비해 주세요.")

                # Step 4: OTO
                self._set_status("4/4 - OTO.ini 생성 중...")
                tpl_path = "" if self.no_base_oto_var.get() else self.tpl_entry.get()
                out_path = self.out_entry.get()
                if out_path: # tpl_path는 이제 필수가 아님
                    tg_folder = os.path.join(wav_dir, "textgrids")
                    params = self._get_params()
                    gen_ou = self.openutau_var.get()
                    gen_missing = self.gen_missing_vowels_var.get()
                    auto_format = self.auto_format_var.get()
                    custom_phonemes_path = self.custom_phoneme_var.get().strip()
                    alias_suffix = self.alias_suffix_var.get().strip()
                    ja_alias_style = self._get_ja_alias_style_code()
                    if self.no_base_oto_var.get():
                        self._append_log("ℹ '베이스 OTO 없음' 선택: 템플릿 없이 OpenUtau 호환 자동 에일리어스 생성 모드로 실행합니다.")

                    if lang == 'japanese':
                        self._append_log(f"ℹ 일본어 에일리어스 형식: {self.ja_alias_style_var.get()}")
                        generate_ja_oto(
                            tg_folder, tpl_path, out_path,
                            params=None,
                            generate_openutau=gen_ou,
                            gen_missing_vowels=gen_missing,
                            alias_style=ja_alias_style,
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
                            auto_format=auto_format,
                            custom_phonemes_path=custom_phonemes_path,
                            alias_suffix=alias_suffix,
                            callback=self._append_log
                        )
                    self._run_auto_validation(wav_dir, tg_folder, out_path)
                else:
                    self._append_log("⚠️ 출력 경로가 비어있어 OTO 생성을 건너뜁니다.")

                self._set_status("🎉 전체 파이프라인 완료!")
                self._append_log("\n" + "=" * 50)
                self._append_log("🎉 모든 작업이 완료되었습니다!")
                self._append_log("=" * 50)

            except Exception as e:
                self._handle_error("전체 파이프라인", e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)

if __name__ == "__main__":
    app = App()
    app.mainloop()

