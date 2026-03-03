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
        """嶸・椪 ・ｸ・ｴ ・ｰ・ SOFA ・ｨ・ｸ・・GitHub ・ｴ・ｬ・溢乱・・・尖徐 ・､・ｴ・罹糖﨑ｩ・壱共."""
        def task():
            self._set_running(True)
            try:
                lang = self._get_language()
                self._set_status("SOFA ・ｨ・ｸ ・､・ｴ・罹糖 ・・..")
                self._append_log(f"筮・SOFA ・ｨ・ｸ ・尖徐 ・､・ｴ・罹糖 ・懍梠 ({'・ｼ・ｸ・ｴ' if lang == 'japanese' else '﨑懋ｵｭ・ｴ'})")
                ok, model_path, err = download_default_sofa_model(
                    language=lang,
                    target_root=get_default_sofa_model_root(),
                    callback=self._append_log,
                )
                if ok and model_path:
                    self._after_safe(lambda p=model_path: self.sofa_ckpt_var.set(p))
                    self._append_log(f"笨・SOFA ・ｨ・ｸ ・､・ｴ・罹糖 ・・｣・ {model_path}")
                    self._set_status("✅ SOFA 모델 다운로드 완료")
                else:
                    self._append_log(f"笶・SOFA ・ｨ・ｸ ・､・ｴ・罹糖 ・､甯ｨ: {err}")
                    self._set_status("笶・SOFA ・ｨ・ｸ ・､・ｴ・罹糖 ・､甯ｨ")
            except Exception as e:
                self._append_log(f"笶・SOFA ・ｨ・ｸ ・尖徐 ・､・ｴ・罹糖 ・・・溢匣: {e}")
                self._set_status("笶・SOFA ・ｨ・ｸ ・､・ｴ・罹糖 ・､甯ｨ")
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    def _ensure_sofa_model_ready(self, language):
        """SOFA ckpt ・ｽ・罹･ｼ 嶹簿ｳｴ﨑ｩ・壱共. ・・愍・ｴ ・ｬ・ｩ・・尞ｴ・・夋川ラ 弡・・尖徐 ・､・ｴ・罹糖・ｼ ・罹巡﨑ｩ・壱共."""
        ckpt = (self.sofa_ckpt_var.get() or "").strip()
        if ckpt and os.path.exists(ckpt):
            return ckpt

        found = find_sofa_ckpt(language, search_root=get_default_sofa_model_root())
        if found and os.path.exists(found):
            self._append_log(f"邃ｹ ・ｬ・ｩ・・尞ｴ・肥乱・・SOFA ・ｨ・ｸ ・尖徐 ・川ｧ: {found}")
            self._after_safe(lambda p=found: self.sofa_ckpt_var.set(p))
            return found

        self._append_log("邃ｹ SOFA ・ｨ・ｸ・ｴ ・・簿据・ ・喜符 ・尖徐 ・､・ｴ・罹糖・ｼ ・罹巡﨑ｩ・壱共...")
        ok, model_path, err = download_default_sofa_model(
            language=language,
            target_root=get_default_sofa_model_root(),
            callback=self._append_log,
        )
        if ok and model_path and os.path.exists(model_path):
            self._after_safe(lambda p=model_path: self.sofa_ckpt_var.set(p))
            self._append_log(f"笨・SOFA ・ｨ・ｸ ・尖徐 ・・・・・｣・ {model_path}")
            return model_path

        release_link = get_sofa_release_link(language)
        model_root = get_default_sofa_model_root()
        self._after_safe(lambda: self._show_copyable_alert(
            title="SOFA ・ｨ・ｸ ・・・・､甯ｨ",
            message=(
                f"SOFA ・ｨ・ｸ ・尖徐 ・､・ｴ・罹糖・・・､甯ｨ嵂溢慣・壱共.\n\n"
                f"・ｸ・ｴ: {'・ｼ・ｸ・ｴ' if language == 'japanese' else '﨑懋ｵｭ・ｴ'}\n"
                f"・護棗 ・・･ 尞ｴ・・\n{model_root}\n\n"
                f"・俯徐 ・､・ｴ・罹糖 ・・〓:\n{release_link}\n\n"
                f"・､・ｴ・罹糖 弡・.ckpt 甯護攵 ・ｽ・罹･ｼ SOFA ・ｴ增ｬ尞ｬ・ｸ孖ｸ・・・・倣紛 ・ｼ・ｸ・・\n\n"
                f"오류 요약:\n{err or '알 수 없는 오류'}"
            ),
            alert_key=f"sofa_model_download_fail_{language}",
        ))
        return ""

    def _notify_mfa_failure_suggest_sofa(self, language, err_msg=""):
        """MFA ・､甯ｨ ・・SOFA ・・ｴ ・､嵂餓揆 ・壱ざ﨑ｩ・壱共."""
        model_root = get_default_sofa_model_root()
        release_link = get_sofa_release_link(language)
        self._append_log("笞 MFA ・簿ｬ・・・､甯ｨ嵂溢慣・壱共. SOFA ・簿ｬ ・肥ｧ・愍・・・ｬ・罹巡﨑ｴ ・ｴ・ｸ・・")
        self._append_log(f"   SOFA ・ｨ・ｸ ・尖徐 ・､・ｴ・罹糖 ・・ｹ・ {model_root}")
        self._append_log(f"   ・ｨ・ｸ ・ｴ・ｬ・・ {release_link}")
        self._after_safe(lambda: self._show_copyable_alert(
            title="MFA ・､甯ｨ - SOFA ・ｬ・罹巡 ・壱ざ",
            message=(
                "MFA ・簿ｬ・ｴ ・､甯ｨ嵂溢慣・壱共.\n\n"
                "・・溢愍・・・簿ｬ ・肥ｧ・揆 SOFA・・・緋ｿ・・､・・・､嵂駕腹 ・・・溢慣・壱共.\n"
                "SOFA ・ｨ・ｸ・ ・ｴ增ｬ尞ｬ・ｸ孖ｸ・ ・・牟 ・溢愍・ｴ ・尖徐 ・､・ｴ・罹糖・ｼ ・罹巡﨑ｩ・壱共.\n\n"
                f"・ｨ・ｸ ・・･ 尞ｴ・・・ｰ・ｸ):\n{model_root}\n"
                f"・ｨ・ｸ ・ｴ・ｬ・・・・〓:\n{release_link}\n\n"
                f"MFA ・､・・\n{err_msg or '(・・搆)'}"
            ),
            alert_key=f"mfa_fail_sofa_hint_{language}",
        ))

    def _run_mfa_setup(self):
        """GUI ・溢乱・・MFA 尞ｬ奓ｰ・・嶹俾ｲｽ・・・尖徐 ・､・倆鮒・壱共."""
        def task():
            self._set_running(True)
            self._set_status("筮・MFA ・尖徐 ・､・・・・.. (10~20・・・護囈)")
            try:
                import shutil
                portable_env_dir = os.path.join(APP_DIR, '.env')
                public_root = os.environ.get('PUBLIC', r'C:\Users\Public')
                fallback_env_dir = os.path.join(public_root, 'UTAU_Auto_OTO_v3', '.env')
                env_dir = portable_env_dir
                if any(ord(ch) > 127 for ch in portable_env_dir):
                    env_dir = fallback_env_dir
                    self._append_log("笞 ・ｱ ・ｽ・懍乱 ・БSCII ・ｸ・専ｰ ・溢牟 MFA 嶹俾ｲｽ・・・ｵ・ｩ 尞ｴ・肥乱 ・､・倆鮒・壱共.")
                    self._append_log(f"   ・・ｴ ・､・・・ｽ・・ {env_dir}")
                mfa_exe = os.path.join(env_dir, 'Scripts', 'mfa.exe')
                installer = os.path.join(APP_DIR, 'Miniconda3-latest-Windows-x86_64.exe')

                # ・ｴ・ｸ ・､・・嶹菩攤
                if os.path.exists(mfa_exe):
                    self._append_log("笨・MFA・ ・ｴ・ｸ ・､・俯据・ｴ ・溢慣・壱共!")
                    self.mfa_path = mfa_exe
                    self._update_mfa_status(True)
                    self._set_status("✅ MFA 준비 완료")
                    return

                system_conda = shutil.which('conda')

                if system_conda:
                    self._append_log(f"剥 ・懍侃奛懍乱 ・､・俯頗 Conda ・懋ｲｬ: {system_conda}")
                    self._append_log("   Miniconda ・､・ｴ・罹糖・ｼ ・ｴ・壱峅・ ・ｰ・ｴ Conda・ｼ 嶹懍圸﨑們流 嶹俾ｲｽ・・・ｬ・ｱ﨑ｩ・壱共.")
                    self._append_log("[1/2] 肌 MFA ・・圸 ・懍ｻｬ 嶹俾ｲｽ ・晧┳ ・・・､・・・・.. (5~10・・ ・ｩ・餓擽 增ｽ・壱共)")
                    
                    cmd = [system_conda, 'create', '-y', '-p', env_dir, '-c', 'conda-forge', '--override-channels', 'montreal-forced-aligner', 'colorama']
                    process = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True, encoding='utf-8', errors='replace')
                    for line in process.stdout:
                        stripped = line.strip()
                        if stripped:
                            self._append_log(stripped)
                    process.wait()
                    
                    if process.returncode != 0:
                        self._append_log("笶・MFA ・､・・・､甯ｨ")
                        return

                    self._append_log("[2/2] 逃 ・緋ｰ ・們｡ｴ・ｱ ・ｨ・・・､・・・・..")
                    # conda run・・・ｬ・ｩ﨑們流 﨑ｴ・ｹ 嶹俾ｲｽ ・ｴ・川・ pip ・､嵂・・ｴ・･
                    sp.run([system_conda, 'run', '-p', env_dir, 'pip', 'install', 'eunjeon', 'jamo', 'textgrid'], capture_output=True)

                    self._append_log("[Patch] ・壱巡・ｰ・ｩ 﨑懋ｵｭ・ｴ 甯護・(eunjeon) ・ｰ・・・俯ｦｬ ・・..")
                    patch_mfa_korean_support(mfa_exe, callback=self._append_log)

                    self._append_log("笨・MFA ・懍侃奛・・ｬ・ｱ ・・｣・")
                
                else:
                    self._append_log("剥 ・懍侃奛・Conda・ｼ ・ｾ・・・・・・慣・壱共. ・川ｲｴ Miniconda 尞ｬ奓ｰ・・嶹俾ｲｽ・・・ｬ・倣鮒・壱共.")
                    conda_exe = os.path.join(env_dir, 'Scripts', 'conda.exe')
                    # Step 1: Miniconda ・､・ｴ・罹糖
                    if not os.path.exists(conda_exe):
                        if not os.path.exists(installer):
                            self._append_log("[1/3] 踏 Miniconda ・､・ｴ・罹糖 ・・.. (・ｽ 80MB)")
                            url = 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe'
                            ps_cmd = (
                                f'[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; '
                                f"Invoke-WebRequest -Uri '{url}' -OutFile '{installer}'"
                            )
                            result = sp.run(['powershell', '-Command', ps_cmd], capture_output=True, text=True)
                            if result.returncode != 0:
                                self._append_log(f"笶・・､・ｴ・罹糖 ・､甯ｨ: {result.stderr}")
                                return
                        self._append_log("笨・Miniconda ・､・ｴ・罹糖 ・・｣・")

                        # Step 2: 尞ｬ奓ｰ・・・､・・
                        self._append_log("[2/3] 逃 Miniconda 尞ｬ奓ｰ・・・､・・・・.. (2~5・・")
                        self._append_log(f"   ・､・・・ｽ・・ {env_dir}")
                        # Miniconda(NSIS)・・/D= ・ｽ・罹･ｼ raw command-line・川・ 甯護恭﨑罹共.
                        # subprocess(list)・・・ｵ・ｱ ・ｽ・罹･ｼ ・尖徐 ・ｸ・ｩ﨑俯ｩｴ・・/D・ ・ｴ・罹摺 ・・・溢牟,
                        # ・・・command-line ・ｸ・川龍・・・､嵂駕復・､.
                        if os.path.isdir(env_dir) and not os.path.exists(conda_exe):
                            try:
                                shutil.rmtree(env_dir)
                                self._append_log("   ・ｴ・・・､甯ｨ 彧肥・.env 尞ｴ・・・・・簿ｦｬ﨑俾ｳ ・ｬ・罹巡﨑ｩ・壱共.")
                            except Exception as cleanup_error:
                                self._append_log(f"笶・・ｰ・ｴ .env 尞ｴ・・・簿ｦｬ ・､甯ｨ: {cleanup_error}")
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
                            # /D ・ｽ・懋ｰ ・ｴ・罹頗 ・ｽ・ｰ ・ｰ・ｸ ・ｽ・懍乱 ・､・俯据・壱株・ ・ｴ・・夋川ｧ
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
                                self._append_log("笞 ・・・・ｽ・懍乱・・Conda・ｼ ・ｾ・ ・ｻ嵂溢ｧ・・・ｰ・ｸ ・､・・・ｽ・罹･ｼ ・川ｧ嵂溢慣・壱共.")
                                self._append_log(f"   ・川ｧ・・Conda: {conda_exe}")
                            else:
                                self._append_log(f"笶・Miniconda ・､・・・､甯ｨ (code={result.returncode})")
                                if result.stdout and result.stdout.strip():
                                    self._append_log(f"   stdout: {result.stdout.strip()[:500]}")
                                if result.stderr and result.stderr.strip():
                                    self._append_log(f"   stderr: {result.stderr.strip()[:500]}")
                                return
                        self._append_log("笨・Miniconda ・､・・・・｣・")

                    # Step 3: MFA ・､・・
                    self._append_log("[3/3] 肌 MFA ・､・・・・.. (5~10・・ ・ｩ・餓擽 增ｽ・壱共)")
                    # HTTP 000 ・尖洳・ｼ ・賀ｸｰ ・・紛 --override-channels ・・conda-forge ・・・菩・・ｬ・ｩ
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
                        self._append_log("笶・MFA ・､・・・､甯ｨ")
                        return

                    self._append_log("笨・Conda 甯ｨ墲､・ ・､・・・・｣・")
                    
                    self._append_log("[3.5/4] 逃 ・緋ｰ ・們｡ｴ・ｱ ・ｨ・・・､・・・・..")
                    sp.run([conda_exe, 'run', '-p', env_dir, 'pip', 'install', 'eunjeon', 'jamo', 'textgrid'], capture_output=True)

                    self._append_log("[Patch] ・壱巡・ｰ・ｩ 﨑懋ｵｭ・ｴ 甯護・(eunjeon) ・ｰ・・・俯ｦｬ ・・..")
                    patch_mfa_korean_support(mfa_exe, callback=self._append_log)

                # Step 4: 﨑懋ｵｭ・ｴ ・ｨ・ｸ
                self._append_log("[・溢ｧ・云 倹 﨑懋ｵｭ・ｴ ・醐箕 ・ｨ・ｸ ・､・ｴ・罹糖 ・・.. (1~2・・")
                process = sp.Popen(
                    [mfa_exe, 'model', 'download', 'acoustic', 'korean_mfa', '--ignore_cache'],
                    stdout=sp.PIPE, stderr=sp.STDOUT, text=True, encoding='utf-8', errors='replace'
                )
                for line in process.stdout:
                    stripped = line.strip()
                    if stripped:
                        self._append_log(stripped)
                process.wait()

                # ・､・倆血・ｼ ・簿ｦｬ
                if os.path.exists(installer):
                    os.remove(installer)

                self.mfa_path = mfa_exe
                self._update_mfa_status(True)
                self._append_log("")
                self._append_log("脂 MFA ・､・俾ｰ ・ｨ・・・・｣誤据・溢慣・壱共!")
                self._append_log("   ・ｴ・・'3・鞘Ε MFA ・護┳ ・簿ｬ' ・・款・・・ｬ・ｩ﨑 ・・・溢慣・壱共.")
                self._set_status("笨・MFA ・､・・・・｣・")

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
        """SOFA ・・圸 ・・・劍・ｽ・ｼ ・們｡ｴ・ｱ・・・尖徐 ・､・倆鮒・壱共."""
        def task():
            self._set_running(True)
            self._set_status("筮・SOFA ・尖徐 ・､・・・・.. (・・・・・護囈)")
            try:
                self._append_log("肌 SOFA ・尖徐 ・､・俯･ｼ ・懍梠﨑ｩ・壱共.")
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
                    self._append_log(f"笶・SOFA ・､・・・､甯ｨ: {err}")
                    self._update_sofa_status(False)
                    self._set_status("笶・SOFA ・､・・・､甯ｨ")
            except Exception as e:
                self._append_log(f"笶・SOFA ・､・・・・・､・・ {e}")
                self._update_sofa_status(False)
                self._set_status("笶・SOFA ・､・・・､甯ｨ")
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    def _update_mfa_status(self, installed):
        """MFA ・・・ UI・ｼ ・・魂・ｴ孖ｸ﨑ｩ・壱共."""
        def _do():
            if installed:
                self.mfa_status_label.configure(text="笨・MFA ・､・俯勢", text_color="#66BB6A")
                self.mfa_install_btn.configure(text="✅ 설치 완료", state="disabled", fg_color="#388E3C")
            else:
                self.mfa_status_label.configure(text="❌ MFA 미설치", text_color="#FF6B6B")
                self.mfa_install_btn.configure(text="⬇ MFA 자동 설치", state="normal", fg_color="#FFA726")
        self._after_safe(_do)

    def _update_sofa_status(self, installed):
        def _do():
            if hasattr(self, "sofa_status_label"):
                if installed:
                    self.sofa_status_label.configure(text="笨・SOFA ・､・俯勢", text_color="#66BB6A")
                else:
                    self.sofa_status_label.configure(text="❌ SOFA 미설치", text_color="#FF6B6B")
            if not hasattr(self, "sofa_install_btn"):
                return
            if installed:
                self.sofa_install_btn.configure(text="✅ 설치 완료", state="disabled", fg_color="#388E3C")
            else:
                self.sofa_install_btn.configure(text="⬇ SOFA 자동 설치", state="normal", fg_color="#42A5F5")
        self._after_safe(_do)

    # 笏笏 ・罹ｳ・・､嵂・笏笏

    def _run_lab_gen(self):
        def task():
            self._set_running(True)
            self._set_status("1・鞘Ε Lab 甯護攵 ・晧┳ ・・..")
            try:
                wav_dir = self.wav_entry.get()
                if not wav_dir:
                    self._append_log("笶・WAV 尞ｴ・・・ｽ・罹･ｼ ・・･﨑ｴ ・ｼ・ｸ・・")
                    return

                custom_phonemes_path = self.custom_phoneme_var.get().strip()
                
                if self._get_language() == 'japanese':
                    count, total, errors = generate_ja_labs(wav_dir, custom_phonemes_path=custom_phonemes_path, callback=self._append_log)
                else:
                    count, total, errors = generate_labs(wav_dir, custom_phonemes_path=custom_phonemes_path, callback=self._append_log)
                if errors:
                    for e in errors:
                        self._append_log(f"  笞・・{e}")
                self._set_status(f"笨・Lab ・晧┳ ・・｣・({count}/{total})")
            except Exception as e:
                self._handle_error("Lab ・晧┳", e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    def _run_dict_gen(self):
        def task():
            self._set_running(True)
            self._set_status("2・鞘Ε ・ｬ・・甯護攵 ・晧┳ ・・..")
            try:
                wav_dir = self.wav_entry.get()
                if not wav_dir:
                    self._append_log("笶・WAV 尞ｴ・・・ｽ・罹･ｼ ・・･﨑ｴ ・ｼ・ｸ・・")
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
                        self._append_log(f"  笞・・{e}")
                self._append_log(f"祷 ・ｬ・・・・･ ・ｽ・・ {dict_path}")
                self._set_status(f"笨・・ｬ・・・晧┳ ・・｣・({entries}・・﨑ｭ・ｩ)")
            except Exception as e:
                self._handle_error("・ｬ・・・晧┳", e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)

    def _run_profile_finetune(self):
        def task():
            self._set_running(True)
            self._set_status("ｧｩ 嵓・｡懦血・ｼ ・ｸ・ｸ・ｰ・・・・哩 ・・..")
            try:
                auto_oto = self.tune_auto_oto_var.get().strip()
                manual_oto = self.tune_manual_oto_var.get().strip()
                profile_out = self.tune_profile_out_var.get().strip()
                apply_target = self.tune_apply_target_var.get().strip()
                custom_phonemes_path = self.custom_phoneme_var.get().strip()

                if not auto_oto or not os.path.exists(auto_oto):
                    self._append_log("笶・・尖徐 OTO ・・･ 甯護攵 ・ｽ・懋ｰ ・・牟・一ｱｰ・・甯護攵・ｴ ・・慣・壱共.")
                    return
                if not manual_oto or not os.path.exists(manual_oto):
                    self._append_log("笶・・俯徐 OTO ・ｸ・ｰ 甯護攵 ・ｽ・懋ｰ ・・牟・一ｱｰ・・甯護攵・ｴ ・・慣・壱共.")
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
                    self._append_log(f"笶・・・圸 ・・・OTO 甯護攵・・・ｾ・・・・・・慣・壱共: {apply_target}")
                    return

                lang = self._get_language()
                self._append_log(f"ｧｪ ・ｸ・ｸ・ｰ・・﨑呷慣 ・懍梠 ({'・ｼ・ｸ・ｴ' if lang == 'japanese' else '﨑懋ｵｭ・ｴ'})")
                self._append_log(f"   ・尖徐 OTO: {auto_oto}")
                self._append_log(f"   ・俯徐 OTO: {manual_oto}")

                if lang == "japanese":
                    profile = train_ja_autotune_profile(auto_oto, manual_oto, custom_phonemes_path=custom_phonemes_path)
                    if not profile:
                        self._append_log("笞・・﨑呷慣 ・・･﨑・・､・ｭ ・倆伯・ｴ ・・ｱ﨑ｩ・壱共. (・懍・ 8・・・ｴ・・・護棗)")
                        return
                    if not save_ja_autotune_profile(profile_out, profile):
                        self._append_log(f"笶・嵓・｡懦血・ｼ ・・･ ・､甯ｨ: {profile_out}")
                        return
                    changed = apply_ja_autotune_profile_to_oto(
                        apply_target, profile, custom_phonemes_path=custom_phonemes_path
                    )
                else:
                    profile = train_kr_autotune_profile(auto_oto, manual_oto, custom_phonemes_path=custom_phonemes_path)
                    if not profile:
                        self._append_log("笞・・﨑呷慣 ・・･﨑・・､・ｭ ・倆伯・ｴ ・・ｱ﨑ｩ・壱共. (・懍・ 8・・・ｴ・・・護棗)")
                        return
                    if not save_kr_autotune_profile(profile_out, profile):
                        self._append_log(f"笶・嵓・｡懦血・ｼ ・・･ ・､甯ｨ: {profile_out}")
                        return
                    changed = apply_kr_autotune_profile_to_oto(
                        apply_target, profile, custom_phonemes_path=custom_phonemes_path
                    )

                pairs = int(profile.get("matched_pairs", 0))
                buckets = len((profile.get("buckets") or {}))
                self._append_log(f"笨・嵓・｡懦血・ｼ ・・･ ・・｣・ {profile_out}")
                self._append_log(f"笨・﨑呷慣 ・ｰ・ｼ: matched_pairs={pairs}, buckets={buckets}")
                self._append_log(f"笨・・・圸 ・・｣・ {changed} lines adjusted ({apply_target})")
                self._set_status(f"笨・・ｸ・ｸ・ｰ・・・・｣・({changed} lines)")
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
                self._set_status("1/4 - Lab 甯護攵 ・晧┳ ・・..")
                wav_dir = self.wav_entry.get()
                if not wav_dir:
                    self._append_log("笶・WAV 尞ｴ・・・ｽ・罹･ｼ ・・･﨑ｴ ・ｼ・ｸ・・")
                    return

                custom_phonemes_path = self.custom_phoneme_var.get().strip()
                
                lang = self._get_language()
                if lang == 'japanese':
                    generate_ja_labs(wav_dir, custom_phonemes_path=custom_phonemes_path, callback=self._append_log)
                else:
                    generate_labs(wav_dir, custom_phonemes_path=custom_phonemes_path, callback=self._append_log)

                # Step 2: Dictionary
                self._set_status("2/4 - ・ｬ・・甯護攵 ・晧┳ ・・..")
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
                        )
                else:
                    if self.mfa_path:
                        self._set_status("3/4 - MFA 음성 정렬 중...")
                        has_model, _ = check_mfa_model(self.mfa_path, language=lang)
                        if not has_model:
                            download_mfa_model(self.mfa_path, language=lang, callback=self._append_log)
                        align_ok, align_err = run_mfa_align(
                            self.mfa_path,
                            wav_dir,
                            dict_path,
                            output_dir,
                            language=lang,
                            callback=self._append_log,
                        )
                    else:
                        align_err = "MFA 실행 파일이 없어 정렬을 건너뜁니다."
                        self._append_log(f"⚠ {align_err}")

                if not align_ok:
                    if align_engine == "MFA" and align_err:
                        self._notify_mfa_failure_suggest_sofa(lang, align_err)
                    self._append_log("⚠ 정렬 실패 상태로 다음 단계를 진행합니다.")

                # Step 4: OTO
                self._set_status("4/4 - OTO.ini ・晧┳ ・・..")
                tpl_path = "" if self.no_base_oto_var.get() else self.tpl_entry.get()
                out_path = self.out_entry.get()
                if out_path: # tpl_path・・・ｴ・・﨑・・・ ・・鋸
                    tg_folder = os.path.join(wav_dir, "textgrids")
                    params = self._get_params()
                    gen_ou = self.openutau_var.get()
                    gen_missing = self.gen_missing_vowels_var.get()
                    enable_ml_correction = self.enable_ml_correction_var.get()
                    enable_pytorch_bridge = self.enable_pytorch_bridge_var.get()
                    auto_format = self.auto_format_var.get()
                    custom_phonemes_path = self.custom_phoneme_var.get().strip()
                    alias_suffix = self.alias_suffix_var.get().strip()
                    ja_alias_style = self._get_ja_alias_style_code()
                    self._append_log(
                        f"[OTO-ML] ・ｰ夋・・・ｵ・・ ml={'ON' if enable_ml_correction else 'OFF'}, "
                        f"pytorch_bridge={'ON' if enable_pytorch_bridge else 'OFF'}"
                    )
                    if self.no_base_oto_var.get():
                        self._append_log("邃ｹ '・・ｴ・､ OTO ・・搆' ・夋・ 奛懦伯・ｿ ・・擽 OpenUtau 嶸ｸ嶹・・尖徐 ・川攵・ｬ・ｴ・､ ・晧┳ ・ｨ・罹｡・・､嵂駕鮒・壱共.")

                    if lang == 'japanese':
                        self._append_log(f"邃ｹ ・ｼ・ｸ・ｴ ・川攵・ｬ・ｴ・､ 嶸菩享: {self.ja_alias_style_var.get()}")
                        generate_ja_oto(
                            tg_folder, tpl_path, out_path,
                            params=None,
                            generate_openutau=gen_ou,
                            gen_missing_vowels=gen_missing,
                            enable_ml_correction=enable_ml_correction,
                            enable_pytorch_bridge=enable_pytorch_bridge,
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
                            enable_ml_correction=enable_ml_correction,
                            enable_pytorch_bridge=enable_pytorch_bridge,
                            auto_format=auto_format,
                            custom_phonemes_path=custom_phonemes_path,
                            alias_suffix=alias_suffix,
                            callback=self._append_log
                        )
                    self._run_auto_validation(wav_dir, tg_folder, out_path)
                else:
                    self._append_log("笞・・・罹･ ・ｽ・懋ｰ ・・牟・溢牟 OTO ・晧┳・・・ｴ・壱怐・壱共.")

                self._set_status("脂 ・・ｲｴ 甯護擽嵓・攵・ｸ ・・｣・")
                self._append_log("\n" + "=" * 50)
                self._append_log("脂 ・ｨ・ ・卓羅・ｴ ・・｣誤据・溢慣・壱共!")
                self._append_log("=" * 50)

            except Exception as e:
                self._handle_error("・・ｲｴ 甯護擽嵓・攵・ｸ", e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)


