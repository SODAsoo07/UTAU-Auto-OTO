from __future__ import annotations

from pathlib import Path

from core.alignment import wfl_runner


def test_bundled_wfl_paths_are_preferred(monkeypatch, tmp_path: Path):
    runtime = tmp_path / "wfl_runtime"
    python_exe = runtime / "python" / "python.exe"
    repo = runtime / "WFL-ASR"
    model = runtime / "models" / "japanese"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_bytes(b"python")
    repo.mkdir(parents=True)
    (repo / "infer.py").write_text("", encoding="utf-8")
    model.mkdir(parents=True)

    monkeypatch.delenv("UTOA_WFL_PYTHON", raising=False)
    monkeypatch.delenv("UTOA_WFL_REPO", raising=False)
    monkeypatch.delenv("UTOA_WFL_MODEL_DIR", raising=False)
    monkeypatch.delenv("UTOA_WFL_MODEL_DIR_JAPANESE", raising=False)
    monkeypatch.setattr(wfl_runner, "_app_root", lambda: str(tmp_path))

    assert wfl_runner._resolve_wfl_python() == str(python_exe)
    assert wfl_runner._resolve_wfl_repo() == str(repo)
    assert wfl_runner._resolve_wfl_model_dir("japanese") == str(model)


def test_release_pruner_allows_only_wfl_model_checkpoints():
    import build

    assert not build._is_forbidden_release_file(
        "UTAU_Auto_OTO/wfl_runtime/models/japanese/last.ckpt"
    )
    assert build._is_forbidden_release_file(
        "UTAU_Auto_OTO/ml_workspace/mfa_free_oto/last.ckpt"
    )
