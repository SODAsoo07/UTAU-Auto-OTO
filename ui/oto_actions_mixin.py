import os
import re

from ui.i18n import t
from core.en_cvvc_builder import generate_en_cvvc_oto
from core.format_type_utils import normalize_auto_format_value
from core.ja_oto_generator import generate_ja_oto
from core.kr_cmpx_preview_builder import (
    generate_kr_cmpx_preview_oto,
    resolve_kr_cmpx_preview_source_oto,
)
from core.no_mfa_oto_builder import (
    generate_no_mfa_auto_oto,
    resolve_no_mfa_source_oto,
)
from core.oto_file_utils import (
    apply_alias_suffix_to_oto_file,
    detect_oto_text_encoding,
    normalize_alias_suffix,
    reencode_oto_file,
)
from core.oto_generator import generate_oto
from core.pipeline_status import normalize_aligner_name
from core.preflight_common import collect_runtime_preflight_issues
from ui.voicebank_batch import (
    discover_recursive_voicebank_dirs,
    has_top_level_wavs,
    resolve_recursive_out_path,
    resolve_voicebank_batch_targets,
)


class OtoActionsMixin:
    @staticmethod
    def _recommended_format_env_preset() -> dict[str, str]:
        # Keep as conservative defaults; users can still override by explicitly setting env/UI values.
        return {
            # KR format thresholds
            "UTOA_KR_MAPPING_CONF_THRESHOLD_CV": "0.62",
            "UTOA_KR_MAPPING_CONF_THRESHOLD_CVC": "0.63",
            "UTOA_KR_MAPPING_CONF_THRESHOLD_CVVC": "0.67",
            "UTOA_KR_MAPPING_CONF_THRESHOLD_VCV": "0.66",
            # JA format thresholds
            "UTOA_JA_MAPPING_CONF_THRESHOLD_CV": "0.70",
            "UTOA_JA_MAPPING_CONF_THRESHOLD_CVVC": "0.71",
            "UTOA_JA_MAPPING_CONF_THRESHOLD_VCV": "0.62",
            # Low-tier forward search caps
            "UTOA_KR_LOW_TIER_FORWARD_MAX": "1",
            "UTOA_KR_LOW_TIER_FORWARD_MAX_CV": "1",
            "UTOA_KR_LOW_TIER_FORWARD_MAX_CVC": "1",
            "UTOA_KR_LOW_TIER_FORWARD_MAX_CVVC": "1",
            "UTOA_KR_LOW_TIER_FORWARD_MAX_VCV": "1",
            "UTOA_JA_LOW_TIER_FORWARD_MAX": "1",
            "UTOA_JA_LOW_TIER_FORWARD_MAX_CV": "1",
            "UTOA_JA_LOW_TIER_FORWARD_MAX_CVVC": "1",
            "UTOA_JA_LOW_TIER_FORWARD_MAX_VCV": "1",
            # CV minimum cutoff span guards
            "UTOA_KR_CV_MIN_CUTOFF_SPAN_MS": "96",
            "UTOA_KR_CV_HEAD_MIN_CUTOFF_SPAN_MS": "84",
            "UTOA_JA_CV_MIN_CUTOFF_SPAN_MS": "92",
            "UTOA_JA_CV_HEAD_MIN_CUTOFF_SPAN_MS": "82",
        }

    def _apply_recommended_format_env_preset(self) -> tuple[int, int]:
        preset = self._recommended_format_env_preset()
        applied = 0
        total = len(preset)
        for key, value in preset.items():
            if str(os.environ.get(key, "") or "").strip():
                continue
            os.environ[key] = str(value)
            applied += 1
        return applied, total

    @staticmethod
    def _has_top_level_wavs(folder_path: str) -> bool:
        return has_top_level_wavs(folder_path)

    def _discover_recursive_voicebank_dirs(self, root_dir: str) -> list[str]:
        return discover_recursive_voicebank_dirs(root_dir)

    @staticmethod
    def _resolve_recursive_out_path(
        root_wav_dir: str,
        base_out_path: str,
        target_wav_dir: str,
        target_count: int,
    ) -> str:
        return resolve_recursive_out_path(root_wav_dir, base_out_path, target_wav_dir, target_count)

    def _run_oto_gen(self):
        def task():
            self._set_running(True)
            self._set_status("4단계 OTO.ini 생성 중...")
            progress_state = {"oto": 0.0, "validate": 0.0}
            progress_detail_state = {"oto_last": -1.0, "validate_last": -1.0}

            def _safe_ratio(value):
                try:
                    return max(0.0, min(1.0, float(value)))
                except Exception:
                    return 0.0

            def _set_progress_part(*, oto_ratio=None, validate_ratio=None):
                if oto_ratio is not None:
                    progress_state["oto"] = max(progress_state["oto"], _safe_ratio(oto_ratio))
                if validate_ratio is not None:
                    progress_state["validate"] = max(progress_state["validate"], _safe_ratio(validate_ratio))
                self._set_progress(progress_state["oto"] * 0.75 + progress_state["validate"] * 0.25)

            def _set_stage_status(stage: str, detail: str, ratio: float) -> None:
                r = _safe_ratio(ratio)
                pct = int(round(r * 100.0))
                self._set_status(f"{stage} [{detail}] ({pct}%)")

            def _update_oto_stage(detail: str, ratio: float, *, force: bool = False) -> None:
                rr = _safe_ratio(ratio)
                _set_progress_part(oto_ratio=rr)
                if force or abs(rr - progress_detail_state["oto_last"]) >= 0.03:
                    progress_detail_state["oto_last"] = rr
                    _set_stage_status("4단계 OTO.ini 생성 중...", detail, rr)

            def _update_validate_stage(detail: str, ratio: float, *, force: bool = False) -> None:
                rr = _safe_ratio(ratio)
                _set_progress_part(validate_ratio=rr)
                if force or abs(rr - progress_detail_state["validate_last"]) >= 0.03:
                    progress_detail_state["validate_last"] = rr
                    _set_stage_status("5단계 OTO 자동 검증 중...", detail, rr)

            def _batch_ratio(batch_index: int, batch_total: int, local_ratio: float) -> float:
                r = _safe_ratio(local_ratio)
                if int(batch_total or 0) <= 1:
                    return r
                return _safe_ratio((float(batch_index) + r) / float(batch_total))

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

            def _make_progress_callback(stage, batch_index=0, batch_total=1):
                def _cb(msg):
                    self._append_log(msg)
                    ratio = _extract_progress_ratio(msg)
                    if ratio is None:
                        return
                    scaled = _batch_ratio(batch_index, batch_total, ratio)
                    if stage == "oto":
                        detail = t("생성 진행") if batch_total <= 1 else f"{t('생성 진행')} ({batch_index + 1}/{batch_total})"
                        _update_oto_stage(detail, scaled)
                    else:
                        detail = t("검증 진행") if batch_total <= 1 else f"{t('검증 진행')} ({batch_index + 1}/{batch_total})"
                        _update_validate_stage(detail, scaled)
                return _cb

            def _read_oto_line_count(path):
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as handle:
                        return sum(1 for _ in handle)
                except Exception:
                    return -1

            try:
                _update_oto_stage(t("입력 검사"), 0.02, force=True)
                root_wav_dir = str(self.wav_entry.get() or "").strip()
                base_tpl_path = "" if self.no_base_oto_var.get() else str(self.tpl_entry.get() or "").strip()
                base_out_path = str(self.out_entry.get() or "").strip()
                batch_scan_enabled = (
                    bool(self.recursive_voicebank_scan_var.get())
                    if hasattr(self, "recursive_voicebank_scan_var")
                    else False
                )
                lang = self._get_language()
                self._append_log(
                    f"ℹ 현재 언어: {'일본어' if lang == 'japanese' else '한국어' if lang == 'korean' else '영어'}"
                )
                selected_format = normalize_auto_format_value(
                    lang,
                    self.auto_format_var.get() if hasattr(self, "auto_format_var") else "",
                )
                if lang in {"korean", "japanese"} and hasattr(self, "_confirm_language_script_mismatch"):
                    if not self._confirm_language_script_mismatch(lang, root_wav_dir, stage_name="OTO 생성"):
                        self._set_status("취소됨: 언어 설정 확인")
                        return

                if not root_wav_dir:
                    self._append_log("오류: WAV 폴더를 입력해 주세요.")
                    return
                if not base_out_path and not batch_scan_enabled:
                    self._append_log("오류: 출력 경로를 입력해 주세요.")
                    return

                batch_targets = resolve_voicebank_batch_targets(
                    root_wav_dir,
                    base_out_path,
                    batch_scan_enabled=batch_scan_enabled,
                )
                if not batch_targets:
                    if batch_scan_enabled:
                        self._append_log("오류: 하위 폴더에서 WAV 파일이 있는 보이스 폴더를 찾지 못했습니다.")
                    else:
                        self._append_log("오류: WAV 폴더를 찾지 못했습니다.")
                    self._set_status("오류: 배치 대상 없음")
                    return
                target_wav_dirs = [target.wav_dir for target in batch_targets]

                target_count = len(target_wav_dirs)
                if target_count > 1:
                    self._append_log(f"ℹ 하위 폴더 자동 탐색 활성화: 총 {target_count}개 보이스 폴더를 순차 처리합니다.")
                elif batch_scan_enabled:
                    self._append_log("ℹ 하위 폴더 자동 탐색 활성화: 처리 대상 1개를 찾았습니다.")
                _update_oto_stage(t("입력 경로 확인 완료"), 0.07, force=True)

                # 일본어: 베이스 OTO가 Shift-JIS(cp932)면 생성 결과도 같은 인코딩으로 보존한다.
                preserve_out_encoding = ""
                if lang == "japanese" and base_tpl_path and os.path.isfile(base_tpl_path):
                    try:
                        base_encoding = detect_oto_text_encoding(base_tpl_path)
                    except Exception:
                        base_encoding = "utf-8"
                    if str(base_encoding or "").strip().lower() not in {"utf-8", "utf_8", "utf-8-sig", "utf_8_sig"}:
                        preserve_out_encoding = base_encoding
                        self._append_log(
                            f"ℹ 베이스 OTO 인코딩 감지: {base_encoding} → 생성 OTO 인코딩 보존"
                        )

                # 일반 OTO 경로에서는 파이프라인과 동일한 사전 점검을 먼저 수행합니다.
                # (영어 Preview / 한국어 템플릿 전용 포맷은 별도 전용 분기에서 검사)
                if (not batch_scan_enabled) and not (
                    lang == "english"
                    or (lang == "korean" and selected_format in {"cmpx", "c_plus_v"})
                ):
                    preflight_wav_dir = os.path.abspath(root_wav_dir)
                    preflight_out_path = (
                        os.path.abspath(base_out_path)
                        if str(base_out_path or "").strip()
                        else os.path.join(preflight_wav_dir, "oto.ini")
                    )
                    preflight_tg_folder = os.path.join(preflight_wav_dir, "textgrids")
                    preflight_tpl_path = "" if self.no_base_oto_var.get() else base_tpl_path
                    preflight = collect_runtime_preflight_issues(
                        language=lang,
                        wav_dir=preflight_wav_dir,
                        out_path=preflight_out_path,
                        aligner=self.aligner_var.get() if hasattr(self, "aligner_var") else "HSMM OTO",
                        textgrid_dir=preflight_tg_folder,
                        tpl_path=preflight_tpl_path,
                        no_mfa_oto_mode=(
                            self._get_no_mfa_oto_mode_code()
                            if hasattr(self, "_get_no_mfa_oto_mode_code")
                            else ""
                        ),
                        no_base_oto=bool(self.no_base_oto_var.get()),
                        custom_phonemes_path=self.custom_phoneme_var.get().strip(),
                        require_output=True,
                    )
                    warning_records = list(preflight.get("warning_records") or [])
                    error_records = list(preflight.get("error_records") or [])
                    for item in warning_records:
                        self._append_log(f"⚠ {item.get('display')}")
                    if error_records:
                        for item in error_records:
                            self._append_log(f"❌ {item.get('display')}")
                        first_code = str((error_records or [{}])[0].get("code", "PRECHECK_FAILED"))
                        self._set_status(f"오류: 사전 점검 실패 ({first_code})")
                        return

                params = self._get_params()
                gen_ou = self.openutau_var.get()
                gen_missing = self.gen_missing_vowels_var.get()
                gen_dash_alias = self.gen_dash_alias_var.get() if hasattr(self, "gen_dash_alias_var") else True
                enable_ml_correction = self.enable_ml_correction_var.get()
                auto_format = self.auto_format_var.get()
                custom_phonemes_path = self.custom_phoneme_var.get().strip()
                alias_suffix = self.alias_suffix_var.get().strip()
                ja_alias_style = self._get_ja_alias_style_code()
                ja_words_fallback = (
                    self.ja_mapping_words_fallback_enabled_var.get()
                    if hasattr(self, "ja_mapping_words_fallback_enabled_var")
                    else True
                )
                ja_spn_threshold = (
                    self.ja_mapping_spn_ratio_threshold_var.get()
                    if hasattr(self, "ja_mapping_spn_ratio_threshold_var")
                    else 0.35
                )
                ja_min_vowel_ratio = (
                    self.ja_mapping_min_vowel_phone_ratio_var.get()
                    if hasattr(self, "ja_mapping_min_vowel_phone_ratio_var")
                    else 0.5
                )
                ja_debug_reason = (
                    self.ja_mapping_debug_reason_logging_var.get()
                    if hasattr(self, "ja_mapping_debug_reason_logging_var")
                    else True
                )
                kr_anchor_profile_path = (
                    self.kr_anchor_profile_path_var.get().strip()
                    if hasattr(self, "kr_anchor_profile_path_var")
                    else ""
                )
                kr_conf_raw = (
                    self.kr_mapping_confidence_threshold_var.get()
                    if hasattr(self, "kr_mapping_confidence_threshold_var")
                    else ""
                )
                kr_conf_threshold = None
                try:
                    raw_conf = str(kr_conf_raw or "").strip()
                    if raw_conf:
                        kr_conf_threshold = max(0.0, min(1.0, float(raw_conf)))
                except Exception:
                    kr_conf_threshold = None
                kr_jump_default = (
                    self.kr_mapping_max_index_jump_default_var.get()
                    if hasattr(self, "kr_mapping_max_index_jump_default_var")
                    else 1
                )
                kr_jump_hi = (
                    self.kr_mapping_max_index_jump_high_conf_var.get()
                    if hasattr(self, "kr_mapping_max_index_jump_high_conf_var")
                    else 2
                )

                _update_oto_stage("옵션/정책 적용", 0.12, force=True)
                auto_policy = self._apply_auto_ml_policy_env(
                    lang,
                    auto_format,
                    enable_ml_default=enable_ml_correction,
                    gen_dash_alias=gen_dash_alias,
                )
                enable_ml_correction = bool(auto_policy.get("enable_ml"))
                self._apply_advanced_tuning_envs()
                boundary_model_path = (
                    self._apply_boundary_model_env() if hasattr(self, "_apply_boundary_model_env") else ""
                )
                self._append_log(
                    f"ℹ 경계 인코더: {'학습 모델 ' + os.path.basename(boundary_model_path) if boundary_model_path else '규칙 기반(world_v1)'}"
                )
                preset_applied, preset_total = self._apply_recommended_format_env_preset()
                if preset_applied > 0:
                    self._append_log(
                        f"[Preset] format tuning preset applied ({preset_applied}/{preset_total}, env setdefault)"
                    )
                if lang == "korean":
                    os.environ["UTOA_KR_MAPPING_MAX_INDEX_JUMP_DEFAULT"] = str(int(kr_jump_default))
                    os.environ["UTOA_KR_MAPPING_MAX_INDEX_JUMP_HIGH_CONF"] = str(int(kr_jump_hi))
                    if kr_anchor_profile_path:
                        os.environ["UTOA_KR_ANCHOR_PROFILE_PATH"] = kr_anchor_profile_path
                    else:
                        os.environ.pop("UTOA_KR_ANCHOR_PROFILE_PATH", None)
                else:
                    os.environ.pop("UTOA_KR_ANCHOR_PROFILE_PATH", None)

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
                if lang == "korean":
                    self._append_log("[KR-MAP] confidence_threshold=default(by format)")
                if self.no_base_oto_var.get():
                    self._append_log("설정: '기본 OTO 없이 생성' 사용 중입니다.")

                def _run_single_target(target_wav_dir: str, target_out_path: str, batch_index: int, batch_total: int):
                    prefix = f"[Batch {batch_index + 1}/{batch_total}] " if batch_total > 1 else ""

                    def _update_oto_local(detail: str, ratio: float, *, force: bool = False):
                        _update_oto_stage(detail, _batch_ratio(batch_index, batch_total, ratio), force=force)

                    def _update_validate_local(detail: str, ratio: float, *, force: bool = False):
                        _update_validate_stage(detail, _batch_ratio(batch_index, batch_total, ratio), force=force)

                    _update_oto_local(t("생성 준비 완료"), 0.18, force=True)
                    tpl_path = "" if self.no_base_oto_var.get() else base_tpl_path
                    cleanup_snapshot = self._snapshot_output_tree_for_cleanup(target_out_path)
                    tg_folder = os.path.join(target_wav_dir, "textgrids")
                    has_textgrid = False
                    if os.path.isdir(tg_folder):
                        try:
                            has_textgrid = any(str(name).lower().endswith(".textgrid") for name in os.listdir(tg_folder))
                        except Exception:
                            has_textgrid = False
                    aligner_engine = normalize_aligner_name(
                        self.aligner_var.get() if hasattr(self, "aligner_var") else "HSMM OTO",
                        default="hsmm_oto",
                    )
                    hsmm_oto_mode = aligner_engine == "hsmm_oto"
                    no_mfa_auto_mode = (
                        aligner_engine in {"none", "coarse_crnn"}
                        and lang != "english"
                        and selected_format not in {"cmpx", "c_plus_v"}
                    )
                    no_mfa_mode_code = (
                        self._get_no_mfa_oto_mode_code()
                        if hasattr(self, "_get_no_mfa_oto_mode_code")
                        else "remap"
                    )
                    if aligner_engine == "coarse_crnn":
                        no_mfa_mode_code = "remap"
                    if no_mfa_mode_code != "remap":
                        no_mfa_mode_code = "remap"
                    no_mfa_mode_text = "base OTO remap + correction"
                    no_mfa_source_oto = ""
                    source_oto_required_for_no_mfa = True
                    if no_mfa_auto_mode:
                        if source_oto_required_for_no_mfa and bool(self.no_base_oto_var.get()):
                            self._append_log(f"{prefix}오류: No-MFA 자동설정 모드에서는 베이스 OTO(템플릿 ini)가 필요합니다.")
                            self._set_status("오류: 베이스 OTO 필요")
                            return False, False, 0, 0
                        if source_oto_required_for_no_mfa or not bool(self.no_base_oto_var.get()):
                            no_mfa_source_oto = resolve_no_mfa_source_oto(
                                wav_dir=target_wav_dir,
                                source_hint=tpl_path,
                            )
                        if source_oto_required_for_no_mfa and not no_mfa_source_oto:
                            self._append_log(f"{prefix}오류: No-MFA 자동설정용 베이스 OTO를 찾지 못했습니다.")
                            self._append_log("   템플릿 OTO 경로에 baseoto.ini 또는 oto.ini를 지정해 주세요.")
                            self._set_status("오류: 베이스 OTO 필요")
                            return False, False, 0, 0
                        if no_mfa_source_oto:
                            tpl_path = no_mfa_source_oto
                        self._append_log("ℹ No-MFA 모드: 선택한 생성 방식으로 OTO를 생성합니다.")
                        if has_textgrid:
                            self._append_log(f"{prefix}ℹ TextGrid가 있어도 No-MFA 선택 시에는 선택한 No-MFA 생성 방식으로 진행합니다.")
                        self._append_log(f"ℹ No-MFA 생성 방식: {no_mfa_mode_text}")
                        if no_mfa_source_oto:
                            self._append_log(f"[No-MFA] base oto: {no_mfa_source_oto}")
                        else:
                            self._append_log("[No-MFA] source oto not used.")
                    elif lang != "english" and selected_format not in {"cmpx", "c_plus_v"} and not has_textgrid:
                        self._append_log(f"{prefix}경고: textgrids 폴더가 없습니다. 3단계 정렬/라벨 생성을 먼저 실행하세요.")

                    if not (
                        lang == "english"
                        or (lang == "korean" and selected_format in {"cmpx", "c_plus_v"})
                    ):
                        preflight = collect_runtime_preflight_issues(
                            language=lang,
                            wav_dir=target_wav_dir,
                            out_path=target_out_path,
                            aligner=self.aligner_var.get() if hasattr(self, "aligner_var") else "HSMM OTO",
                            textgrid_dir=tg_folder,
                            tpl_path=tpl_path,
                            no_mfa_oto_mode=no_mfa_mode_code,
                            no_base_oto=bool(self.no_base_oto_var.get()),
                            custom_phonemes_path=custom_phonemes_path,
                            require_output=True,
                        )
                        warning_records = list(preflight.get("warning_records") or [])
                        error_records = list(preflight.get("error_records") or [])
                        for item in warning_records:
                            self._append_log(f"⚠ {item.get('display')}")
                        if error_records:
                            for item in error_records:
                                self._append_log(f"❌ {item.get('display')}")
                            first_code = str((error_records or [{}])[0].get("code", "PRECHECK_FAILED"))
                            self._set_status(f"오류: 사전 점검 실패 ({first_code})")
                            return False, False, 0, 0

                    if lang == "english":
                        if not getattr(self, "_is_preview_channel", lambda: False)():
                            self._append_log(f"{prefix}오류: 영어 CVVC 모드는 Preview 채널에서만 사용할 수 있습니다.")
                            self._set_status("오류: Preview 전용 기능")
                            return False, True, 0, 0
                        if (not self.no_base_oto_var.get()) and str(tpl_path or "").strip():
                            self._append_log("ℹ 영어 Preview CVVC 모드에서는 템플릿 OTO 입력을 사용하지 않습니다.")
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
                        _update_oto_local("영어 CVVC 생성", 0.22, force=True)
                        processed, total, errors = generate_en_cvvc_oto(
                            wav_dir=target_wav_dir,
                            out_path=target_out_path,
                            pack=en_pack,
                            beat=en_beat,
                            preset=en_preset,
                            alias_suffix=alias_suffix,
                            include_list_only_synthesis=en_list_fallback,
                            callback=_make_progress_callback("oto", batch_index=batch_index, batch_total=batch_total),
                        )
                    elif lang == "korean" and selected_format == "cmpx":
                        if not getattr(self, "_is_preview_channel", lambda: False)():
                            self._append_log(f"{prefix}오류: 한국어 CMPX 모드는 Preview 채널에서만 사용할 수 있습니다.")
                            self._set_status("오류: Preview 전용 기능")
                            return False, True, 0, 0
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
                        if self.no_base_oto_var.get():
                            self._append_log(f"{prefix}오류: 한국어 CMPX Preview 모드는 베이스 OTO(템플릿 ini)가 필수입니다.")
                            self._set_status("오류: 베이스 OTO 필요")
                            return False, False, 0, 0
                        source_oto = resolve_kr_cmpx_preview_source_oto(
                            wav_dir=target_wav_dir,
                            source_hint=tpl_path,
                        )
                        if not source_oto:
                            self._append_log(f"{prefix}오류: 한국어 CMPX Preview용 베이스 OTO를 찾지 못했습니다.")
                            self._append_log("   템플릿 OTO에 비교용 oto.ini 또는 baseoto.ini를 지정해 주세요.")
                            self._set_status("오류: 베이스 OTO 필요")
                            return False, False, 0, 0

                        self._append_log("ℹ 한국어 CMPX Preview 모드: 정렬 단계를 건너뜁니다.")
                        self._append_log(f"[KR-CMPX] base oto: {source_oto}")
                        _update_oto_local("CMPX 베이스 매핑 생성", 0.22, force=True)
                        processed, total, errors = generate_kr_cmpx_preview_oto(
                            wav_dir=target_wav_dir,
                            out_path=target_out_path,
                            source_oto_path=source_oto,
                            alias_suffix=alias_suffix,
                            callback=_make_progress_callback("oto", batch_index=batch_index, batch_total=batch_total),
                        )
                    elif lang == "korean" and selected_format == "c_plus_v":
                        if self.no_base_oto_var.get():
                            self._append_log(f"{prefix}오류: 한국어 C+V 모드는 베이스 OTO(템플릿 ini)가 필수입니다.")
                            self._set_status("오류: 베이스 OTO 필요")
                            return False, False, 0, 0
                        source_oto = resolve_no_mfa_source_oto(
                            wav_dir=target_wav_dir,
                            source_hint=tpl_path,
                        )
                        if not source_oto:
                            self._append_log(f"{prefix}오류: 한국어 C+V 모드용 베이스 OTO를 찾지 못했습니다.")
                            self._append_log("   템플릿 OTO에 baseoto.ini 또는 oto.ini를 지정해 주세요.")
                            self._set_status("오류: 베이스 OTO 필요")
                            return False, False, 0, 0

                        self._append_log("ℹ 한국어 C+V 모드: 정렬 없이 템플릿 OTO 재매핑으로 생성합니다.")
                        self._append_log(f"[KR-C+V] base oto: {source_oto}")
                        _update_oto_local("C+V 템플릿 재매핑 생성", 0.22, force=True)
                        processed, total, errors = generate_no_mfa_auto_oto(
                            wav_dir=target_wav_dir,
                            out_path=target_out_path,
                            source_oto_path=source_oto,
                            alias_suffix=alias_suffix,
                            language=lang,
                            stats_oto_path=os.environ.get("UTOA_NO_MFA_STATS_OTO", ""),
                            generation_mode="remap",
                            callback=_make_progress_callback("oto", batch_index=batch_index, batch_total=batch_total),
                        )
                    elif hsmm_oto_mode:
                        hsmm_source_oto = ""
                        if not bool(self.no_base_oto_var.get()):
                            hsmm_source_oto = resolve_no_mfa_source_oto(
                                wav_dir=target_wav_dir,
                                source_hint=tpl_path,
                            )
                        apply_hsmm_lightgbm = (
                            self._should_apply_hsmm_lightgbm(enable_ml_correction)
                            if hasattr(self, "_should_apply_hsmm_lightgbm")
                            else bool(enable_ml_correction)
                        )
                        self._append_log(
                            f"{prefix}HSMM OTO 시작: target={target_wav_dir}, "
                            f"format={selected_format}, lightgbm={'ON' if apply_hsmm_lightgbm else 'OFF'}"
                        )
                        if bool(enable_ml_correction) and not apply_hsmm_lightgbm:
                            self._append_log(f"{prefix}[HSMM] LightGBM postprocess disabled by developer setting.")
                        if hsmm_source_oto:
                            self._append_log(f"{prefix}[HSMM] base oto: {hsmm_source_oto}")
                        else:
                            self._append_log(f"{prefix}[HSMM] no base oto: filename slots will be used.")
                        _update_oto_local("HSMM OTO generation", 0.22, force=True)
                        processed, total, errors = self._run_hsmm_oto_preview_generation(
                            wav_dir=target_wav_dir,
                            out_path=target_out_path,
                            source_oto_path=hsmm_source_oto,
                            language=lang,
                            format_type=selected_format,
                            apply_lightgbm=apply_hsmm_lightgbm,
                            callback=_make_progress_callback("oto", batch_index=batch_index, batch_total=batch_total),
                        )
                        self._append_log(f"{prefix}HSMM OTO 완료: processed={processed}/{total}, errors={len(errors or [])}")
                    elif no_mfa_auto_mode:
                        _update_oto_local("No-MFA OTO generation", 0.22, force=True)
                        processed, total, errors = generate_no_mfa_auto_oto(
                            wav_dir=target_wav_dir,
                            out_path=target_out_path,
                            source_oto_path=no_mfa_source_oto or tpl_path,
                            alias_suffix=alias_suffix,
                            language=lang,
                            stats_oto_path=os.environ.get("UTOA_NO_MFA_STATS_OTO", ""),
                            generation_mode=no_mfa_mode_code,
                            callback=_make_progress_callback("oto", batch_index=batch_index, batch_total=batch_total),
                        )
                    elif lang == "japanese":
                        self._append_log(f"설정: 일본어 에일리어스 스타일 = {self.ja_alias_style_var.get()}")
                        _update_oto_local("일본어 OTO 생성", 0.22, force=True)
                        processed, total, errors = generate_ja_oto(
                            tg_folder,
                            tpl_path,
                            target_out_path,
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
                            callback=_make_progress_callback("oto", batch_index=batch_index, batch_total=batch_total),
                        )
                    else:
                        _update_oto_local("한국어 OTO 생성", 0.22, force=True)
                        processed, total, errors = generate_oto(
                            tg_folder,
                            tpl_path,
                            target_out_path,
                            params,
                            gen_ou,
                            gen_missing,
                            enable_ml_correction=enable_ml_correction,
                            auto_format=auto_format,
                            custom_phonemes_path=custom_phonemes_path,
                            alias_suffix=alias_suffix,
                            kr_anchor_profile_path=kr_anchor_profile_path,
                            kr_mapping_confidence_threshold=kr_conf_threshold,
                            kr_mapping_max_index_jump_default=int(kr_jump_default),
                            kr_mapping_max_index_jump_high_conf=int(kr_jump_hi),
                            callback=_make_progress_callback("oto", batch_index=batch_index, batch_total=batch_total),
                        )

                    if errors:
                        for err in errors:
                            self._append_log(f"  - {err}")
                        self._set_status(f"오류: OTO 생성 실패 {len(errors)}건 ({processed}/{total})")
                        return False, False, int(processed or 0), int(total or 0)

                    if total:
                        _update_oto_local(t("생성 결과 정리"), float(processed) / float(total))
                    _update_oto_local(t("생성 결과 정리"), 0.92, force=True)

                    out_exists = os.path.isfile(target_out_path)
                    out_lines = _read_oto_line_count(target_out_path) if out_exists else -1
                    if out_exists:
                        line_info = f"{out_lines}" if out_lines >= 0 else "unknown"
                        self._append_log(
                            f"{prefix}1차 생성 완료: OTO 파일 저장 -> {target_out_path} "
                            f"(processed={processed}/{total}, lines={line_info})"
                        )
                        _update_oto_local("OTO 파일 저장 확인", 1.0, force=True)
                    else:
                        self._append_log(f"{prefix}오류: OTO 파일 저장 실패 -> {target_out_path}")
                        self._set_status("오류: OTO 저장 실패")
                        return False, False, int(processed or 0), int(total or 0)

                    if int(processed or 0) <= 0 and int(total or 0) <= 0:
                        self._append_log(f"{prefix}[ERROR] 오류: OTO 생성 결과가 비어 있어 자동 검증을 건너뜁니다.")
                        self._set_status("오류: OTO 생성 결과 없음")
                        return False, False, int(processed or 0), int(total or 0)

                    _update_validate_local(t("검증 준비"), 0.05, force=True)
                    self._run_auto_validation(
                        target_wav_dir,
                        tg_folder,
                        target_out_path,
                        callback=_make_progress_callback("validate", batch_index=batch_index, batch_total=batch_total),
                    )
                    _update_validate_local(t("검증 완료"), 1.0, force=True)
                    self._cleanup_generated_output_artifacts(target_out_path, snapshot=cleanup_snapshot)

                    self._set_status(f"완료: OTO 생성 성공 ({processed}/{total})")
                    return True, False, int(processed or 0), int(total or 0)

                success_count = 0
                failed_count = 0
                fatal_stop = False
                last_processed = 0
                last_total = 0
                for idx, target_wav_dir in enumerate(target_wav_dirs):
                    target_out_path = self._resolve_recursive_out_path(
                        root_wav_dir,
                        base_out_path,
                        target_wav_dir,
                        target_count,
                    )
                    if not target_out_path:
                        failed_count += 1
                        self._append_log(f"[Batch {idx + 1}/{target_count}] 오류: 출력 경로 계산 실패")
                        continue
                    if target_count > 1:
                        self._append_log(f"[Batch {idx + 1}/{target_count}] 대상 폴더: {target_wav_dir}")
                        self._append_log(f"[Batch {idx + 1}/{target_count}] 출력 경로: {target_out_path}")
                    ok, fatal, processed, total = _run_single_target(
                        target_wav_dir,
                        target_out_path,
                        idx,
                        target_count,
                    )
                    last_processed = processed
                    last_total = total
                    if ok:
                        success_count += 1
                        if preserve_out_encoding:
                            try:
                                changed, used = reencode_oto_file(target_out_path, preserve_out_encoding)
                                if changed:
                                    self._append_log(
                                        f"ℹ 생성 OTO 인코딩 보존: {os.path.basename(target_out_path)} → {used}"
                                    )
                            except Exception as enc_exc:
                                self._append_log(f"⚠ OTO 인코딩 보존 실패({preserve_out_encoding}): {enc_exc}")
                    else:
                        failed_count += 1
                    if fatal:
                        fatal_stop = True
                        break

                if target_count > 1:
                    if success_count <= 0:
                        if fatal_stop:
                            self._set_status("오류: 배치 OTO 생성 중단")
                        else:
                            self._set_status(f"오류: 배치 OTO 생성 실패 (0/{target_count})")
                    elif failed_count > 0:
                        self._set_status(f"완료(부분 성공): 배치 OTO 생성 {success_count}/{target_count}")
                    else:
                        self._set_status(f"완료: 배치 OTO 생성 성공 ({success_count}/{target_count})")
                elif failed_count > 0 and last_total <= 0:
                    self._set_status("오류: OTO 생성 실패")

            except Exception as e:
                self._handle_error("OTO 생성", e)
            finally:
                self._set_running(False)

        self._run_in_thread(task)

    def _resolve_generated_oto_paths(self) -> list[str]:
        """Resolve the oto.ini path(s) for the current output/batch settings."""
        root_wav_dir = str(self.wav_entry.get() or "").strip()
        base_out_path = str(self.out_entry.get() or "").strip()
        batch_scan_enabled = (
            bool(self.recursive_voicebank_scan_var.get())
            if hasattr(self, "recursive_voicebank_scan_var")
            else False
        )
        out_paths: list[str] = []
        batch_targets = []
        try:
            batch_targets = resolve_voicebank_batch_targets(
                root_wav_dir,
                base_out_path,
                batch_scan_enabled=batch_scan_enabled,
            )
        except Exception:
            batch_targets = []
        target_wav_dirs = [target.wav_dir for target in (batch_targets or [])]
        target_count = len(target_wav_dirs)
        if target_count == 0:
            if base_out_path:
                out_paths.append(base_out_path)
        else:
            for target_wav_dir in target_wav_dirs:
                resolved = self._resolve_recursive_out_path(
                    root_wav_dir,
                    base_out_path,
                    target_wav_dir,
                    target_count,
                )
                if resolved:
                    out_paths.append(resolved)
        seen: set[str] = set()
        unique: list[str] = []
        for path in out_paths:
            key = os.path.abspath(path)
            if key not in seen:
                seen.add(key)
                unique.append(path)
        return unique

    def _apply_suffix_to_generated_oto(self):
        """Append the current alias suffix to already-generated oto.ini file(s).

        Lets the user type a suffix *after* generation and apply it without a full
        regeneration. Idempotent: aliases that already end with the suffix are left
        untouched, so it is safe to run repeatedly.
        """
        def task():
            self._set_running(True)
            self._set_status(t("접미사 적용 중..."))
            try:
                suffix = str(self.alias_suffix_var.get() or "").strip()
                if not normalize_alias_suffix(suffix):
                    self._append_log("오류: 적용할 접미사를 입력해 주세요.")
                    self._set_status("오류: 접미사 없음")
                    return
                out_paths = self._resolve_generated_oto_paths()
                if not out_paths:
                    self._append_log("오류: 출력 경로를 입력해 주세요.")
                    self._set_status("오류: 출력 경로 없음")
                    return
                normalized = normalize_alias_suffix(suffix)
                total_changed = 0
                files_applied = 0
                files_missing = 0
                for path in out_paths:
                    if not os.path.isfile(path):
                        files_missing += 1
                        self._append_log(f"⚠ oto.ini 없음(먼저 OTO를 생성하세요): {path}")
                        continue
                    changed, alias_total = apply_alias_suffix_to_oto_file(path, suffix)
                    files_applied += 1
                    total_changed += changed
                    self._append_log(
                        f"[접미사] {os.path.basename(path)}: "
                        f"alias {changed}/{alias_total}개에 _{normalized} 적용"
                    )
                if files_applied == 0:
                    self._set_status("오류: 적용할 oto.ini를 찾지 못했습니다")
                    return
                self._save_config()
                self._set_status(
                    f"완료: 접미사 _{normalized} 적용 "
                    f"(alias {total_changed}개 / 파일 {files_applied}개)"
                )
            except Exception as e:
                self._handle_error("접미사 적용", e)
            finally:
                self._set_running(False)

        self._run_in_thread(task)
