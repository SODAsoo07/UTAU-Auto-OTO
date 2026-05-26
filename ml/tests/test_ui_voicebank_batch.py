import os
from pathlib import Path

from ui.pipeline_actions_mixin import PipelineActionsMixin
from ui.voicebank_batch import resolve_voicebank_batch_targets


class _Entry:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def test_voicebank_batch_targets_use_leaf_wav_folders(tmp_path):
    root = tmp_path / "bank"
    a3 = root / "A3"
    c4 = root / "C4"
    a3.mkdir(parents=True)
    c4.mkdir()
    (a3 / "a_i_u.wav").write_bytes(b"")
    (c4 / "ka_ki_ku.wav").write_bytes(b"")

    targets = resolve_voicebank_batch_targets(str(root), "", batch_scan_enabled=True)

    assert [Path(target.wav_dir).name for target in targets] == ["A3", "C4"]
    assert [Path(target.out_path).parent.name for target in targets] == ["A3", "C4"]
    assert [Path(target.out_path).name for target in targets] == ["oto.ini", "oto.ini"]


def test_full_pipeline_uses_batch_targets_before_root_processing(tmp_path):
    root = tmp_path / "bank"
    for pitch in ("A3", "C4"):
        pitch_dir = root / pitch
        pitch_dir.mkdir(parents=True, exist_ok=True)
        (pitch_dir / f"{pitch.lower()}_sample.wav").write_bytes(b"")

    class Dummy(PipelineActionsMixin):
        def __init__(self):
            self.wav_entry = _Entry(str(root))
            self.out_entry = _Entry("")
            self.tpl_entry = _Entry("")
            self.custom_phoneme_var = _Var("")
            self.auto_format_var = _Var("CVVC")
            self.aligner_var = _Var("HSMM OTO")
            self.no_base_oto_var = _Var(True)
            self.recursive_voicebank_scan_var = _Var(True)
            self.logs = []
            self.status = ""
            self.batch_targets = []

        def _run_in_thread(self, func):
            func()

        def _set_running(self, _running):
            pass

        def _set_status(self, status):
            self.status = str(status)

        def _set_progress(self, _value):
            pass

        def _append_log(self, message):
            self.logs.append(str(message))

        def _get_language(self):
            return "japanese"

        def _get_no_mfa_oto_mode_code(self):
            return "remap"

        def _snapshot_output_tree_for_cleanup(self, _path):
            return None

        def _run_full_pipeline_batch_targets(self, targets, **_kwargs):
            self.batch_targets = list(targets)
            return len(targets), 0

    dummy = Dummy()
    dummy._run_full_pipeline()

    assert [Path(target.wav_dir).name for target in dummy.batch_targets] == ["A3", "C4"]
    assert [Path(target.out_path).name for target in dummy.batch_targets] == ["oto.ini", "oto.ini"]
    assert any("전체 실행" in log and "2개" in log for log in dummy.logs)


def test_hsmm_batch_path_logs_progress(tmp_path):
    root = tmp_path / "bank"
    a3 = root / "A3"
    a3.mkdir(parents=True)
    (a3 / "a_i_u.wav").write_bytes(b"")
    targets = resolve_voicebank_batch_targets(str(root), "", batch_scan_enabled=True)

    class Dummy(PipelineActionsMixin):
        def __init__(self):
            self.gen_dash_alias_var = _Var(True)
            self.enable_ml_correction_var = _Var(False)
            self.auto_format_var = _Var("CVVC")
            self.alias_suffix_var = _Var("")
            self.no_base_oto_var = _Var(True)
            self.openutau_var = _Var(False)
            self.gen_missing_vowels_var = _Var(False)
            self.logs = []
            self.status = ""
            self.progress = 0.0

        def _append_log(self, message):
            self.logs.append(str(message))

        def _set_status(self, status):
            self.status = str(status)

        def _set_progress(self, value):
            self.progress = float(value)

        def _apply_auto_ml_policy_env(self, *_args, **_kwargs):
            return {"enable_ml": False, "route": "nomfa", "has_coupled": False, "hybrid_routing": False}

        def _apply_advanced_tuning_envs(self):
            pass

        def _snapshot_output_tree_for_cleanup(self, _path):
            return {}

        def _run_hsmm_oto_preview_generation(self, *, out_path, callback=None, **_kwargs):
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text("a_i_u.wav=a,0,0,0,0,0\n", encoding="utf-8")
            if callback:
                callback("[HSMM OTO] progress 1/1")
            return 1, 1, []

        def _run_auto_validation(self, *_args, **_kwargs):
            pass

        def _cleanup_generated_output_artifacts(self, *_args, **_kwargs):
            pass

    dummy = Dummy()
    success, failed = dummy._run_full_pipeline_batch_targets(
        targets,
        lang="japanese",
        selected_format="CVVC",
        base_tpl_path="",
        aligner_engine="hsmm_oto",
        no_mfa_mode_code="remap",
        custom_phonemes_path="",
    )

    assert (success, failed) == (1, 0)
    assert any("HSMM OTO 시작" in log for log in dummy.logs)
    assert any("HSMM OTO 완료" in log for log in dummy.logs)
    assert any("[HSMM OTO] progress 1/1" in log for log in dummy.logs)
    assert os.path.isfile(targets[0].out_path)
