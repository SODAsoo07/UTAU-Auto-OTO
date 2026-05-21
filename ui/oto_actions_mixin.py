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
from core.oto_generator import generate_oto
from core.pipeline_status import normalize_aligner_name
from core.preflight_common import collect_runtime_preflight_issues


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
        try:
            return any(str(name).lower().endswith(".wav") for name in os.listdir(folder_path))
        except Exception:
            return False

    def _discover_recursive_voicebank_dirs(self, root_dir: str) -> list[str]:
        root = os.path.abspath(str(root_dir or "").strip())
        if not root or not os.path.isdir(root):
            return []

        candidates: list[str] = []
        for cur_root, _dirs, files in os.walk(root):
            if any(str(name).lower().endswith(".wav") for name in files):
                candidates.append(os.path.abspath(cur_root))

        if not candidates:
            return []

        deduped = sorted(set(candidates), key=lambda path: (len(path), path.lower()))
        leaf_only: list[str] = []
        for current in deduped:
            current_norm = os.path.normcase(os.path.abspath(current)).rstrip("\\/")
            child_prefix = current_norm + os.sep
            has_child_candidate = False
            for other in deduped:
                if other == current:
                    continue
                other_norm = os.path.normcase(os.path.abspath(other))
                if other_norm.startswith(child_prefix):
                    has_child_candidate = True
                    break
            if not has_child_candidate:
                leaf_only.append(current)

        return leaf_only or deduped

    @staticmethod
    def _resolve_recursive_out_path(
        root_wav_dir: str,
        base_out_path: str,
        target_wav_dir: str,
        target_count: int,
    ) -> str:
        target_abs = os.path.abspath(str(target_wav_dir or "").strip())
        out_raw = str(base_out_path or "").strip()
        if target_count <= 1:
            if out_raw:
                if os.path.isdir(out_raw) or out_raw.endswith(("\\", "/")):
                    return os.path.join(os.path.abspath(out_raw), "oto.ini")
                return out_raw
            return os.path.join(target_abs, "oto.ini")

        if not out_raw:
            return os.path.join(target_abs, "oto.ini")

        out_abs = os.path.abspath(out_raw)
        out_ext = str(os.path.splitext(out_abs)[1] or "").lower()
        if out_ext == ".ini":
            out_root = os.path.dirname(out_abs)
            out_name = os.path.basename(out_abs) or "oto.ini"
        else:
            out_root = out_abs
            out_name = "oto.ini"
        root_abs = os.path.abspath(str(root_wav_dir or "").strip())
        rel = ""
        try:
            rel = os.path.relpath(target_abs, root_abs)
        except Exception:
            rel = ""
        if not rel or rel.startswith(".."):
            rel = os.path.basename(target_abs)
        if rel in {".", ""}:
            return os.path.join(out_root, out_name)
        return os.path.join(out_root, rel, out_name)

    def _run_oto_gen(self):
        def task():
            self._set_running(True)
            self._set_status("4・ｨ・・OTO.ini ・晧┳ ・・..")
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
                    _set_stage_status("4・ｨ・・OTO.ini ・晧┳ ・・..", detail, rr)

            def _update_validate_stage(detail: str, ratio: float, *, force: bool = False) -> None:
                rr = _safe_ratio(ratio)
                _set_progress_part(validate_ratio=rr)
                if force or abs(rr - progress_detail_state["validate_last"]) >= 0.03:
                    progress_detail_state["validate_last"] = rr
                    _set_stage_status("5・ｨ・・OTO ・尖徐 ・・・・・..", detail, rr)

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
                        detail = t("・晧┳ ・・哩") if batch_total <= 1 else f"{t('・晧┳ ・・哩')} ({batch_index + 1}/{batch_total})"
                        _update_oto_stage(detail, scaled)
                    else:
                        detail = t("・・・・・哩") if batch_total <= 1 else f"{t('・・・・・哩')} ({batch_index + 1}/{batch_total})"
                        _update_validate_stage(detail, scaled)
                return _cb

            def _read_oto_line_count(path):
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as handle:
                        return sum(1 for _ in handle)
                except Exception:
                    return -1

            try:
                _update_oto_stage(t("・・･ ・・ｬ"), 0.02, force=True)
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
                    f"邃ｹ 嶸・椪 ・ｸ・ｴ: {'・ｼ・ｸ・ｴ' if lang == 'japanese' else '﨑懋ｵｭ・ｴ' if lang == 'korean' else '・・牟'}"
                )
                selected_format = normalize_auto_format_value(
                    lang,
                    self.auto_format_var.get() if hasattr(self, "auto_format_var") else "",
                )
                if lang in {"korean", "japanese"} and hasattr(self, "_confirm_language_script_mismatch"):
                    if not self._confirm_language_script_mismatch(lang, root_wav_dir, stage_name="OTO ・晧┳"):
                        self._set_status("・ｨ・誤勢: ・ｸ・ｴ ・､・・嶹菩攤")
                        return

                if not root_wav_dir:
                    self._append_log("・､・・ WAV 尞ｴ・罷･ｼ ・・･﨑ｴ ・ｼ・ｸ・・")
                    return
                if not base_out_path and not batch_scan_enabled:
                    self._append_log("・､・・ ・罹･ ・ｽ・罹･ｼ ・・･﨑ｴ ・ｼ・ｸ・・")
                    return

                if batch_scan_enabled:
                    target_wav_dirs = self._discover_recursive_voicebank_dirs(root_wav_dir)
                    if not target_wav_dirs:
                        self._append_log("・､・・ 﨑們怱 尞ｴ・肥乱・・WAV 甯護攵・ｴ ・壱株 ・ｴ・ｴ・､ 尞ｴ・罷･ｼ ・ｾ・ ・ｻ嵂溢慣・壱共.")
                        self._set_status("・､・・ ・ｰ・・・・・・・搆")
                        return
                else:
                    target_wav_dirs = [os.path.abspath(root_wav_dir)]

                target_count = len(target_wav_dirs)
                if target_count > 1:
                    self._append_log(f"邃ｹ 﨑們怱 尞ｴ・・・尖徐 夋川ラ 嶹懍┳嶹・ ・・{target_count}・・・ｴ・ｴ・､ 尞ｴ・罷･ｼ ・懍ｰｨ ・俯ｦｬ﨑ｩ・壱共.")
                elif batch_scan_enabled:
                    self._append_log("邃ｹ 﨑們怱 尞ｴ・・・尖徐 夋川ラ 嶹懍┳嶹・ ・俯ｦｬ ・・・1・罹･ｼ ・ｾ・們慣・壱共.")
                _update_oto_stage(t("Checking output path"), 0.07, force=True)

                if batch_scan_enabled:
                    target_wav_dirs = self._discover_recursive_voicebank_dirs(root_wav_dir)
                    if not target_wav_dirs:
                        self._append_log("・､・・ 﨑們怱 尞ｴ・肥乱・・WAV 甯護攵・ｴ ・壱株 ・ｴ・ｴ・､ 尞ｴ・罷･ｼ ・ｾ・ ・ｻ嵂溢慣・壱共.")
                        self._set_status("・､・・ ・ｰ・・・・・・・搆")
                        return
                else:
                    target_wav_dirs = [os.path.abspath(root_wav_dir)]

                target_count = len(target_wav_dirs)
                if target_count > 1:
                    self._append_log(f"邃ｹ 﨑們怱 尞ｴ・・・尖徐 夋川ラ 嶹懍┳嶹・ ・・{target_count}・・・ｴ・ｴ・､ 尞ｴ・罷･ｼ ・懍ｰｨ ・俯ｦｬ﨑ｩ・壱共.")
                elif batch_scan_enabled:
                    self._append_log("邃ｹ 﨑們怱 尞ｴ・・・尖徐 夋川ラ 嶹懍┳嶹・ ・俯ｦｬ ・・・1・罹･ｼ ・ｾ・們慣・壱共.")
                _update_oto_stage(t("Checking output path"), 0.07, force=True)

                # ・ｼ・・OTO ・ｽ・懍乱・罹株 甯護擽嵓・攵・ｸ・ｼ ・呷攵﨑・・ｬ・・・専ｲ・・・ｼ・ ・倆哩﨑ｩ・壱共.
                # (・・牟 Preview / 﨑懋ｵｭ・ｴ 奛懦伯・ｿ ・・圸 尞ｬ・ｷ・ ・・巡 ・・圸 ・・ｸｰ・川・ ・・ｬ)
                if not (
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
                        aligner=self.aligner_var.get() if hasattr(self, "aligner_var") else "MFA",
                        textgrid_dir=preflight_tg_folder,
                        tpl_path=preflight_tpl_path,
                        no_base_oto=bool(self.no_base_oto_var.get()),
                        custom_phonemes_path=self.custom_phoneme_var.get().strip(),
                        require_output=True,
                    )
                    warning_records = list(preflight.get("warning_records") or [])
                    error_records = list(preflight.get("error_records") or [])
                    for item in warning_records:
                        self._append_log(f"笞 {item.get('display')}")
                    if error_records:
                        for item in error_records:
                            self._append_log(f"笶・{item.get('display')}")
                        first_code = str((error_records or [{}])[0].get("code", "PRECHECK_FAILED"))
                        self._set_status(f"・､・・ ・ｬ・・・専ｲ ・､甯ｨ ({first_code})")
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

                _update_oto_stage("・ｵ・・・菩ｱ・・・圸", 0.12, force=True)
                auto_policy = self._apply_auto_ml_policy_env(
                    lang,
                    auto_format,
                    enable_ml_default=enable_ml_correction,
                    gen_dash_alias=gen_dash_alias,
                )
                enable_ml_correction = bool(auto_policy.get("enable_ml"))
                self._apply_advanced_tuning_envs()
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
                    self._append_log("[OTO-ML] ML ・ｨ・ｸ・ｴ ・・牟 ・ｴ・菩揆 ・ｴ・壱怐・壱共.")
                route_code = str(auto_policy.get("route", "") or "").strip().lower()
                if route_code in {"nomfa", "v2"}:
                    self._append_log(f"[OTO-ML] route policy: {route_code} (coupled primary + autofree auxiliary)")
                if lang == "korean":
                    self._append_log("[KR-MAP] confidence_threshold=default(by format)")
                if self.no_base_oto_var.get():
                    self._append_log("・､・・ '・ｰ・ｸ OTO ・・擽 ・晧┳' ・ｬ・ｩ ・卓桿・壱共.")

                def _run_single_target(target_wav_dir: str, target_out_path: str, batch_index: int, batch_total: int):
                    prefix = f"[Batch {batch_index + 1}/{batch_total}] " if batch_total > 1 else ""

                    def _update_oto_local(detail: str, ratio: float, *, force: bool = False):
                        _update_oto_stage(detail, _batch_ratio(batch_index, batch_total, ratio), force=force)

                    def _update_validate_local(detail: str, ratio: float, *, force: bool = False):
                        _update_validate_stage(detail, _batch_ratio(batch_index, batch_total, ratio), force=force)

                    _update_oto_local(t("Preparing generation"), 0.18, force=True)
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
                        self.aligner_var.get() if hasattr(self, "aligner_var") else "mfa",
                        default="mfa",
                    )
                    no_mfa_auto_mode = (
                        aligner_engine == "none"
                        and lang != "english"
                        and selected_format not in {"cmpx", "c_plus_v"}
                    )
                    no_mfa_mode_code = (
                        self._get_no_mfa_oto_mode_code()
                        if hasattr(self, "_get_no_mfa_oto_mode_code")
                        else "remap"
                    )
                    no_mfa_mode_text = (
                        "MFA-Free SSL ・ｬ・ｯ ・ｴ・啄┣(・､嵭・"
                        if no_mfa_mode_code == "mfa_free_ssl_slot"
                        else "Base OTO remap"
                    )
                    no_mfa_source_oto = ""
                    if no_mfa_auto_mode:
                        if bool(self.no_base_oto_var.get()):
                            self._append_log(f"{prefix}・､・・ No-MFA ・尖徐・､・・・ｨ・懍乱・罹株 ・・ｴ・､ OTO(奛懦伯・ｿ ini)・ 﨑・囈﨑ｩ・壱共.")
                            self._set_status("・､・・ ・・ｴ・､ OTO 﨑・囈")
                            return False, False, 0, 0
                        no_mfa_source_oto = resolve_no_mfa_source_oto(
                            wav_dir=target_wav_dir,
                            source_hint=tpl_path,
                        )
                        if not no_mfa_source_oto:
                            self._append_log(f"{prefix}・､・・ No-MFA ・尖徐・､・菩圸 ・・ｴ・､ OTO・ｼ ・ｾ・ ・ｻ嵂溢慣・壱共.")
                            self._append_log("   奛懦伯・ｿ OTO ・ｽ・懍乱 baseoto.ini ・尖株 oto.ini・ｼ ・・倣紛 ・ｼ・ｸ・・")
                            self._set_status("・､・・ ・・ｴ・､ OTO 﨑・囈")
                            return False, False, 0, 0
                        tpl_path = no_mfa_source_oto
                        self._append_log("邃ｹ No-MFA ・ｨ・・ ・夋晨復 ・晧┳ ・ｩ・晧愍・・OTO・ｼ ・晧┳﨑ｩ・壱共.")
                        if has_textgrid:
                            self._append_log(f"{prefix}邃ｹ TextGrid・ ・溢牟・・No-MFA ・夋・・懍乱・・・夋晨復 No-MFA ・晧┳ ・ｩ・晧愍・・・・哩﨑ｩ・壱共.")
                        self._append_log(f"邃ｹ No-MFA ・晧┳ ・ｩ・・ {no_mfa_mode_text}")
                        self._append_log(f"[No-MFA] base oto: {no_mfa_source_oto}")
                    elif lang != "english" and selected_format not in {"cmpx", "c_plus_v"} and not has_textgrid:
                        self._append_log(f"{prefix}・ｽ・: textgrids 尞ｴ・緋ｰ ・・慣・壱共. 3・ｨ・・・簿ｬ/・ｼ・ｨ ・晧┳・・・ｼ・ ・､嵂駕葺・ｸ・・")

                    if not (
                        lang == "english"
                        or (lang == "korean" and selected_format in {"cmpx", "c_plus_v"})
                    ):
                        preflight = collect_runtime_preflight_issues(
                            language=lang,
                            wav_dir=target_wav_dir,
                            out_path=target_out_path,
                            aligner=self.aligner_var.get() if hasattr(self, "aligner_var") else "MFA",
                            textgrid_dir=tg_folder,
                            tpl_path=tpl_path,
                            no_base_oto=bool(self.no_base_oto_var.get()),
                            custom_phonemes_path=custom_phonemes_path,
                            require_output=True,
                        )
                        warning_records = list(preflight.get("warning_records") or [])
                        error_records = list(preflight.get("error_records") or [])
                        for item in warning_records:
                            self._append_log(f"笞 {item.get('display')}")
                        if error_records:
                            for item in error_records:
                                self._append_log(f"笶・{item.get('display')}")
                            first_code = str((error_records or [{}])[0].get("code", "PRECHECK_FAILED"))
                            self._set_status(f"・､・・ ・ｬ・・・専ｲ ・､甯ｨ ({first_code})")
                            return False, False, 0, 0

                    if lang == "english":
                        if not getattr(self, "_is_preview_channel", lambda: False)():
                            self._append_log(f"{prefix}・､・・ ・・牟 CVVC ・ｨ・罹株 Preview ・・ю・川・・・・ｬ・ｩ﨑 ・・・溢慣・壱共.")
                            self._set_status("・､・・ Preview ・・圸 ・ｰ・･")
                            return False, True, 0, 0
                        if (not self.no_base_oto_var.get()) and str(tpl_path or "").strip():
                            self._append_log("邃ｹ ・・牟 Preview CVVC ・ｨ・懍乱・罹株 奛懦伯・ｿ OTO ・・･・・・ｬ・ｩ﨑們ｧ ・喜慣・壱共.")
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
                        _update_oto_local("・・牟 CVVC ・晧┳", 0.22, force=True)
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
                            self._append_log(f"{prefix}・､・・ 﨑懋ｵｭ・ｴ CMPX ・ｨ・罹株 Preview ・・ю・川・・・・ｬ・ｩ﨑 ・・・溢慣・壱共.")
                            self._set_status("・､・・ Preview ・・圸 ・ｰ・･")
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
                                self._append_log("邃ｹ CMPX ・ｨ・・・懦復: ML route・ｼ No-MFA・・・・倣鮒・壱共.")
                        os.environ["UTOA_ML_ROUTE"] = "autofree_v1"
                        os.environ["UTOA_ML_AUTOFREE_AUX_ENABLE"] = "1"
                        os.environ["UTOA_ML_LEGACY_FALLBACK_ENABLE"] = "0"
                        if self.no_base_oto_var.get():
                            self._append_log(f"{prefix}・､・・ 﨑懋ｵｭ・ｴ CMPX Preview ・ｨ・罹株 ・・ｴ・､ OTO(奛懦伯・ｿ ini)・ 﨑・・・・笈・､.")
                            self._set_status("・､・・ ・・ｴ・､ OTO 﨑・囈")
                            return False, False, 0, 0
                        source_oto = resolve_kr_cmpx_preview_source_oto(
                            wav_dir=target_wav_dir,
                            source_hint=tpl_path,
                        )
                        if not source_oto:
                            self._append_log(f"{prefix}・､・・ 﨑懋ｵｭ・ｴ CMPX Preview・ｩ ・・ｴ・､ OTO・ｼ ・ｾ・ ・ｻ嵂溢慣・壱共.")
                            self._append_log("   奛懦伯・ｿ OTO・・・・ｵ川圸 oto.ini ・尖株 baseoto.ini・ｼ ・・倣紛 ・ｼ・ｸ・・")
                            self._set_status("・､・・ ・・ｴ・､ OTO 﨑・囈")
                            return False, False, 0, 0

                        self._append_log("邃ｹ 﨑懋ｵｭ・ｴ CMPX Preview ・ｨ・・ ・簿ｬ ・ｨ・・･ｼ ・ｴ・壱怐・壱共.")
                        self._append_log(f"[KR-CMPX] base oto: {source_oto}")
                        _update_oto_local("CMPX ・・ｴ・､ ・､﨑・・晧┳", 0.22, force=True)
                        processed, total, errors = generate_kr_cmpx_preview_oto(
                            wav_dir=target_wav_dir,
                            out_path=target_out_path,
                            source_oto_path=source_oto,
                            alias_suffix=alias_suffix,
                            callback=_make_progress_callback("oto", batch_index=batch_index, batch_total=batch_total),
                        )
                    elif lang == "korean" and selected_format == "c_plus_v":
                        if self.no_base_oto_var.get():
                            self._append_log(f"{prefix}・､・・ 﨑懋ｵｭ・ｴ C+V ・ｨ・罹株 ・・ｴ・､ OTO(奛懦伯・ｿ ini)・ 﨑・・・・笈・､.")
                            self._set_status("・､・・ ・・ｴ・､ OTO 﨑・囈")
                            return False, False, 0, 0
                        source_oto = resolve_no_mfa_source_oto(
                            wav_dir=target_wav_dir,
                            source_hint=tpl_path,
                        )
                        if not source_oto:
                            self._append_log(f"{prefix}・､・・ 﨑懋ｵｭ・ｴ C+V ・ｨ・懍圸 ・・ｴ・､ OTO・ｼ ・ｾ・ ・ｻ嵂溢慣・壱共.")
                            self._append_log("   奛懦伯・ｿ OTO・・baseoto.ini ・尖株 oto.ini・ｼ ・・倣紛 ・ｼ・ｸ・・")
                            self._set_status("・､・・ ・・ｴ・､ OTO 﨑・囈")
                            return False, False, 0, 0

                        self._append_log("邃ｹ 﨑懋ｵｭ・ｴ C+V ・ｨ・・ ・簿ｬ ・・擽 奛懦伯・ｿ OTO ・ｬ・､﨑卓愍・・・晧┳﨑ｩ・壱共.")
                        self._append_log(f"[KR-C+V] base oto: {source_oto}")
                        _update_oto_local("C+V 奛懦伯・ｿ ・ｬ・､﨑・・晧┳", 0.22, force=True)
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
                    elif no_mfa_auto_mode:
                        if no_mfa_mode_code == "mfa_free_ssl_slot":
                            _update_oto_local("MFA-Free SSL ・ｬ・ｯ ・ｴ・啄┣ ・晧┳", 0.22, force=True)
                            processed, total, errors = self._run_mfa_free_oto_preview_generation(
                                wav_dir=target_wav_dir,
                                out_path=target_out_path,
                                source_oto_path=no_mfa_source_oto or tpl_path,
                                language=lang,
                                format_type=selected_format,
                                callback=_make_progress_callback("oto", batch_index=batch_index, batch_total=batch_total),
                            )
                        else:
                            _update_oto_local("No-MFA ・尖徐・､・・・晧┳", 0.22, force=True)
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
                        self._append_log(f"・､・・ ・ｼ・ｸ・ｴ ・川攵・ｬ・ｴ・､ ・､夋・ｼ = {self.ja_alias_style_var.get()}")
                        _update_oto_local("・ｼ・ｸ・ｴ OTO ・晧┳", 0.22, force=True)
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
                        _update_oto_local("﨑懋ｵｭ・ｴ OTO ・晧┳", 0.22, force=True)
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

                    if total:
                        _update_oto_local(t("・晧┳ ・ｰ・ｼ ・簿ｦｬ"), float(processed) / float(total))
                    _update_oto_local(t("・晧┳ ・ｰ・ｼ ・簿ｦｬ"), 0.92, force=True)

                    out_exists = os.path.isfile(target_out_path)
                    out_lines = _read_oto_line_count(target_out_path) if out_exists else -1
                    if out_exists:
                        line_info = f"{out_lines}" if out_lines >= 0 else "unknown"
                        self._append_log(
                            f"{prefix}1・ｨ ・晧┳ ・・｣・ OTO 甯護攵 ・・･ -> {target_out_path} "
                            f"(processed={processed}/{total}, lines={line_info})"
                        )
                        _update_oto_local("OTO 甯護攵 ・・･ 嶹菩攤", 1.0, force=True)
                    else:
                        self._append_log(f"{prefix}・､・・ OTO 甯護攵 ・・･ ・､甯ｨ -> {target_out_path}")
                        self._set_status("・､・・ OTO ・・･ ・､甯ｨ")
                        return False, False, int(processed or 0), int(total or 0)

                    if int(processed or 0) <= 0 and int(total or 0) <= 0:
                        self._append_log(f"{prefix}[ERROR] ・､・・ OTO ・晧┳ ・ｰ・ｼ・ ・・牟 ・溢牟 ・尖徐 ・・晧揆 ・ｴ・壱怐・壱共.")
                        self._set_status("・､・・ OTO ・晧┳ ・ｰ・ｼ ・・搆")
                        return False, False, int(processed or 0), int(total or 0)

                    _update_validate_local(t("Validating"), 0.05, force=True)
                    self._run_auto_validation(
                        target_wav_dir,
                        tg_folder,
                        target_out_path,
                        callback=_make_progress_callback("validate", batch_index=batch_index, batch_total=batch_total),
                    )
                    _update_validate_local(t("Validation complete"), 1.0, force=True)
                    if not errors:
                        self._cleanup_generated_output_artifacts(target_out_path, snapshot=cleanup_snapshot)
                    if errors:
                        for err in errors:
                            self._append_log(f"  - {err}")
                        self._set_status(f"・､・・ OTO ・晧┳ ・､甯ｨ {len(errors)}・ｴ ({processed}/{total})")
                        return False, False, int(processed or 0), int(total or 0)

                    self._set_status(f"・・｣・ OTO ・晧┳ ・ｱ・ｵ ({processed}/{total})")
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
                        self._append_log(f"[Batch {idx + 1}/{target_count}] ・､・・ ・罹･ ・ｽ・・・・げ ・､甯ｨ")
                        continue
                    if target_count > 1:
                        self._append_log(f"[Batch {idx + 1}/{target_count}] ・・・尞ｴ・・ {target_wav_dir}")
                        self._append_log(f"[Batch {idx + 1}/{target_count}] ・罹･ ・ｽ・・ {target_out_path}")
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
                    else:
                        failed_count += 1
                    if fatal:
                        fatal_stop = True
                        break

                if target_count > 1:
                    if success_count <= 0:
                        if fatal_stop:
                            self._set_status("・､・・ ・ｰ・・OTO ・晧┳ ・瀧卿")
                        else:
                            self._set_status(f"・､・・ ・ｰ・・OTO ・晧┳ ・､甯ｨ (0/{target_count})")
                    elif failed_count > 0:
                        self._set_status(f"・・｣・・・・・ｱ・ｵ): ・ｰ・・OTO ・晧┳ {success_count}/{target_count}")
                    else:
                        self._set_status(f"・・｣・ ・ｰ・・OTO ・晧┳ ・ｱ・ｵ ({success_count}/{target_count})")
                elif failed_count > 0 and last_total <= 0:
                    self._set_status("・､・・ OTO ・晧┳ ・､甯ｨ")

            except Exception as e:
                self._handle_error("OTO ・晧┳", e)
            finally:
                self._set_running(False)

        self._run_in_thread(task)
