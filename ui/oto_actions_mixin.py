import os

from core.ja_oto_generator import generate_ja_oto
from core.oto_generator import generate_oto


class OtoActionsMixin:
    def _run_oto_gen(self):
        def task():
            self._set_running(True)
            self._set_status("4️⃣ OTO.ini 생성 중...")
            try:
                wav_dir = self.wav_entry.get()
                tpl_path = "" if self.no_base_oto_var.get() else self.tpl_entry.get()
                out_path = self.out_entry.get()

                if not wav_dir:
                    self._append_log("❌ WAV 폴더 경로를 입력해 주세요.")
                    return
                if not out_path:
                    self._append_log("❌ 출력 경로를 입력해 주세요.")
                    return

                tg_folder = os.path.join(wav_dir, "textgrids")
                if not os.path.exists(tg_folder):
                    self._append_log("❌ TextGrid 폴더를 찾을 수 없습니다. (3단계 생략 시 오류 발생 가능)")
                
                # 템플릿은 없어도 무방함
                
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
                    f"[OTO-ML] 런타임 옵션: ml={'ON' if enable_ml_correction else 'OFF'}, "
                    f"pytorch_bridge={'ON' if enable_pytorch_bridge else 'OFF'}"
                )
                if self.no_base_oto_var.get():
                    self._append_log("ℹ '베이스 OTO 없음' 선택: 템플릿 없이 OpenUtau 호환 자동 에일리어스 생성 모드로 실행합니다.")

                lang = self._get_language()
                if lang == 'japanese':
                    self._append_log(f"ℹ 일본어 에일리어스 형식: {self.ja_alias_style_var.get()}")
                    processed, total, errors = generate_ja_oto(
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
                    processed, total, errors = generate_oto(
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
                if errors:
                    for e in errors:
                        self._append_log(f"  ⚠️ {e}")
                    self._set_status(f"⚠ OTO 생성 완료(경고 {len(errors)}건) ({processed}/{total})")
                else:
                    self._set_status(f"✅ OTO 생성 완료 ({processed}/{total})")
            except Exception as e:
                self._handle_error("OTO 생성", e)
            finally:
                self._set_running(False)
        self._run_in_thread(task)

