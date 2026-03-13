import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_lightgbm import evaluate_lightgbm_selector_bundle
from core.oto_ml_refiner import apply_oto_ml_to_oto_file
from core.oto_ml_selector import (
    build_selector_candidates,
    build_selector_dataset_csv_from_delta_dataset,
    build_selector_training_rows_from_delta_rows,
    select_best_candidate,
)


class OtoMlSelectorTests(unittest.TestCase):
    def _base_row(self):
        return {
            "language": "korean",
            "format_type": "cv",
            "alias_type": "cv",
            "alias_group": "cv_other",
            "voicebank_id": "vb1",
            "wav": "a.wav",
            "wav_norm": "a",
            "alias": "ga",
            "alias_norm": "ga",
            "occurrence_index": 0,
            "line_index": 0,
            "mapping_confidence": 0.92,
            "jump_blocked_flag": 0.0,
            "used_alias_based_syllables": 1.0,
            "used_nuclei_fallback": 0.0,
            "words_vs_alias_score_margin": 0.8,
            "wav_duration_ms": 1000.0,
            "expected_anchor_ms": 118.0,
            "base_offset": 80.0,
            "base_cons": 140.0,
            "base_cutoff_abs": 300.0,
            "base_pre": 60.0,
            "base_ovl": 35.0,
            "base_offset_to_expected_ms": -38.0,
            "base_pre_to_expected_ms": 22.0,
            "base_cutoff_to_next_anchor_ms": 18.0,
            "prev_phone_gap_ms": 4.0,
            "next_phone_gap_ms": 8.0,
            "energy_mean": 0.4,
            "db_silence_ratio": 0.1,
            "local_peak_db": -8.0,
            "local_valley_db": -22.0,
            "mel_window_energy_mean": 0.35,
            "mel_window_silence_ratio": 0.08,
            "coda_type": "none",
            "vowel_class": "a",
            "bridge_type": "none",
            "mora_position": "single",
            "onset_class": "plosive",
            "voicing_class": "voiced",
            "mapping_reason_code": "direct",
            "prev_alias_type": "",
            "next_alias_type": "",
            "manual_offset": 80.0,
            "manual_cons": 140.0,
            "manual_pre": 56.0,
            "manual_ovl": 28.0,
            "delta_cutoff": -6.0,
        }

    def test_build_selector_training_rows_marks_best_candidate(self):
        rows = build_selector_training_rows_from_delta_rows([self._base_row()])
        self.assertGreaterEqual(len(rows), 3)
        best = [row for row in rows if int(row.get("selector_is_best", 0)) == 1]
        self.assertEqual(len(best), 1)
        self.assertIn("selector_rank_label", best[0])
        self.assertIn("candidate_mode", best[0])

    def test_selector_group_id_uses_source_row_id_when_present(self):
        row1 = self._base_row()
        row2 = self._base_row()
        row1["source_oto_id"] = "oto_a"
        row1["source_row_id"] = "oto_a:0"
        row2["source_oto_id"] = "oto_b"
        row2["source_row_id"] = "oto_b:0"
        rows = build_selector_training_rows_from_delta_rows([row1, row2])
        groups = {str(row.get("selector_group_id", "")) for row in rows}
        self.assertEqual(len(groups), 2)

    def test_selector_group_id_fallback_keeps_alias_occurrence_separate(self):
        row1 = self._base_row()
        row2 = self._base_row()
        row2["alias"] = "gaS"
        row2["alias_norm"] = "gas"
        rows = build_selector_training_rows_from_delta_rows([row1, row2])
        groups = {str(row.get("selector_group_id", "")) for row in rows}
        self.assertEqual(len(groups), 2)

    def test_selector_dataset_builder_applies_conservative_filters(self):
        with tempfile.TemporaryDirectory() as td:
            delta_csv = os.path.join(td, "delta.csv")
            selector_csv = os.path.join(td, "selector.csv")
            row1 = self._base_row()
            row1["label_source"] = "manual"
            row1["train_keep_default"] = 1
            row1["used_nuclei_fallback"] = 0
            row1["source_row_id"] = "oto_a:0"
            row1["source_oto_id"] = "oto_a"
            row1["mapping_confidence"] = 0.9
            row2 = dict(row1)
            row2["source_row_id"] = "oto_a:1"
            row2["line_index"] = 1
            row2["train_keep_default"] = 0
            with open(delta_csv, "w", encoding="utf-8", newline="") as f:
                import csv
                writer = csv.DictWriter(f, fieldnames=sorted(set(row1) | set(row2)))
                writer.writeheader()
                writer.writerow(row1)
                writer.writerow(row2)
            summary = build_selector_dataset_csv_from_delta_dataset(
                delta_csv,
                selector_csv,
                language="korean",
                format_type="cv",
                require_train_keep=True,
                min_mapping_confidence=0.5,
                exclude_nuclei_fallback=True,
            )
            self.assertGreater(summary["rows"], 0)
            with open(selector_csv, "r", encoding="utf-8") as f:
                text = f.read()
            self.assertIn("oto_a:0", text)
            self.assertNotIn("oto_a:1", text)

    def test_selector_candidates_cover_supported_language_formats(self):
        cases = [
            ("japanese", "cv", "cv", "none", "ja_cv_hold"),
            ("japanese", "cvvc", "cv", "none", "ja_cvvc_cv_hold"),
            ("japanese", "cvvc", "vc", "none", "ja_cvvc_bridge_hold"),
            ("japanese", "vcv", "vcv", "none", "ja_vcv_bridge_hold"),
            ("korean", "cv", "cv", "none", "kr_cv_hold"),
            ("korean", "cvc", "vc", "stop", "kr_cvc_tail_hold"),
            ("korean", "cvvc", "vv", "none", "kr_cvvc_bridge_tight"),
            ("korean", "vcv", "cv", "none", "kr_vcv_body_relaxed"),
        ]
        for language, format_type, alias_type, coda_type, expected_mode in cases:
            row = self._base_row()
            row["language"] = language
            row["format_type"] = format_type
            row["alias_type"] = alias_type
            row["coda_type"] = coda_type
            candidates = build_selector_candidates(language, row)
            modes = {candidate["candidate_mode"] for candidate in candidates}
            self.assertIn("base", modes)
            self.assertIn(expected_mode, modes)

    def test_select_best_candidate_uses_score_callback(self):
        selected = select_best_candidate(
            "korean",
            self._base_row(),
            lambda row: 10.0 if row.get("candidate_mode") == "anchor_tight" else 0.0,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected["candidate"]["candidate_mode"], "anchor_tight")

    def test_apply_oto_ml_to_oto_file_uses_selector_candidate_before_delta(self):
        with tempfile.TemporaryDirectory() as td:
            oto_path = os.path.join(td, "oto.ini")
            with open(oto_path, "w", encoding="utf-8") as f:
                f.write("a.wav=ga,80.00,140.00,-220.00,60.00,35.00\n")

            rows = [
                {
                    "line_index": 0,
                    "wav": "a.wav",
                    "alias": "ga",
                    "offset": 80.0,
                    "cons": 140.0,
                    "cutoff": -220.0,
                    "pre": 60.0,
                    "ovl": 35.0,
                }
            ]
            feats = [self._base_row()]
            report = {}
            with mock.patch("core.oto_ml_refiner.parse_oto_rows", return_value=rows), mock.patch(
                "core.oto_ml_refiner.extract_feature_rows", return_value=feats
            ), mock.patch(
                "core.oto_ml_refiner.check_oto_ml_ready",
                return_value={"code": "OK", "model_dir": td},
            ), mock.patch(
                "core.oto_ml_refiner._resolve_model_dir", return_value=td
            ), mock.patch(
                "core.oto_ml_refiner.load_oto_model_bundle", return_value=SimpleNamespace(backend="lightgbm")
            ), mock.patch(
                "core.oto_ml_refiner.load_lightgbm_selector_bundle", return_value={"dummy": True}
            ), mock.patch(
                "core.oto_ml_refiner.predict_oto_deltas",
                return_value=SimpleNamespace(
                    deltas={
                        "delta_offset": 0.0,
                        "delta_cons": 0.0,
                        "delta_cutoff": 0.0,
                        "delta_pre": 0.0,
                        "delta_ovl": 0.0,
                    }
                ),
            ), mock.patch(
                "core.oto_ml_refiner.predict_lightgbm_selector_score",
                side_effect=lambda payload, feature_row: 10.0 if feature_row.get("candidate_mode") == "cv_guard" else 0.0,
            ):
                changed = apply_oto_ml_to_oto_file(
                    "korean",
                    oto_path,
                    tg_dir=td,
                    wav_dir=td,
                    policy="on",
                    report=report,
                )
            self.assertEqual(changed, 1)
            self.assertEqual(report["selector_rows"], 1)
            self.assertGreaterEqual(report["selector_mode_counts"].get("cv_guard", 0), 1)
            with open(oto_path, "r", encoding="utf-8") as f:
                text = f.read()
            self.assertNotIn(",80.00,140.00,-220.00,60.00,35.00", text)

    def test_apply_oto_ml_to_oto_file_skips_selector_for_japanese_cvvc_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            oto_path = os.path.join(td, "oto.ini")
            with open(oto_path, "w", encoding="utf-8") as f:
                f.write("a.wav=ka,80.00,140.00,-220.00,60.00,35.00\n")

            rows = [
                {
                    "line_index": 0,
                    "wav": "a.wav",
                    "alias": "ka",
                    "offset": 80.0,
                    "cons": 140.0,
                    "cutoff": -220.0,
                    "pre": 60.0,
                    "ovl": 35.0,
                }
            ]
            feat = self._base_row()
            feat["language"] = "japanese"
            feat["format_type"] = "cvvc"
            feat["alias"] = "ka"
            feat["alias_norm"] = "ka"
            feat["alias_type"] = "cv"
            feat["alias_group"] = "cv_plosive"
            feat["mapping_confidence"] = 0.92
            with mock.patch("core.oto_ml_refiner.parse_oto_rows", return_value=rows), mock.patch(
                "core.oto_ml_refiner.extract_feature_rows", return_value=[feat]
            ), mock.patch(
                "core.oto_ml_refiner.check_oto_ml_ready",
                return_value={"code": "OK", "model_dir": td},
            ), mock.patch(
                "core.oto_ml_refiner._resolve_model_dir", return_value=td
            ), mock.patch(
                "core.oto_ml_refiner.load_oto_model_bundle", return_value=SimpleNamespace(backend="lightgbm")
            ), mock.patch(
                "core.oto_ml_refiner.load_lightgbm_selector_bundle"
            ) as selector_loader, mock.patch(
                "core.oto_ml_refiner.predict_oto_deltas",
                return_value=SimpleNamespace(
                    deltas={
                        "delta_offset": 0.0,
                        "delta_cons": 0.0,
                        "delta_cutoff": 0.0,
                        "delta_pre": 0.0,
                        "delta_ovl": 0.0,
                    }
                ),
            ):
                apply_oto_ml_to_oto_file("japanese", oto_path, tg_dir=td, wav_dir=td)
            self.assertTrue(selector_loader.called)

    def test_evaluate_lightgbm_selector_bundle_reports_top1_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            csv_path = os.path.join(td, "selector.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                f.write(
                    "language,format_type,selector_group_id,selector_is_best,selector_quality_score,candidate_mode,candidate_index\n"
                    "korean,cv,g1,1,10.0,anchor_tight,0\n"
                    "korean,cv,g1,0,2.0,base,1\n"
                )
            fake_payload = {
                "meta": {"language": "korean", "selector_objective": "pointwise", "categorical_features": ["language", "format_type", "candidate_mode"]},
                "schema": {
                    "feature_names": ["language", "format_type", "candidate_mode", "candidate_index"],
                    "categorical_features": ["language", "format_type", "candidate_mode"],
                },
                "model": SimpleNamespace(predict=lambda frame: [9.0, 1.0]),
            }
            with mock.patch("core.oto_ml_lightgbm.load_lightgbm_selector_bundle", return_value=fake_payload):
                summary = evaluate_lightgbm_selector_bundle(td, csv_path, language="korean", format_type="cv")
            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["groups"], 1)
            self.assertGreaterEqual(summary["top1_model"], 1.0)


if __name__ == "__main__":
    unittest.main()
