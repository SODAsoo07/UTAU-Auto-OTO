import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core import mfa_runner


def test_safe_env_subprocess_cwd_returns_env_dir(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()

    assert mfa_runner._safe_env_subprocess_cwd(str(env_dir)) == str(env_dir)
    assert mfa_runner._safe_env_subprocess_cwd(str(env_dir / "missing")) is None


def test_ensure_japanese_support_uses_env_dir_as_safe_cwd(monkeypatch, tmp_path):
    env_dir = tmp_path / ".env"
    scripts_dir = env_dir / "Scripts"
    scripts_dir.mkdir(parents=True)
    python_exe = scripts_dir / "python.exe"
    python_exe.write_text("", encoding="utf-8")
    conda_exe = scripts_dir / "conda.exe"
    conda_exe.write_text("", encoding="utf-8")
    mfa_bat = scripts_dir / "mfa.bat"
    mfa_bat.write_text("", encoding="utf-8")

    calls = []

    class Result:
        def __init__(self, returncode=0, stdout="", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        if args[:2] == [str(python_exe), '-c'] and 'import spacy' in args[2]:
            return Result(returncode=1, stderr='Traceback: import email') if len(calls) == 1 else Result(returncode=0)
        if args and args[0] == str(conda_exe):
            return Result(returncode=0)
        return Result(returncode=0)

    monkeypatch.setattr(mfa_runner, '_get_conda_env', lambda _: {'PATH': 'x'})
    monkeypatch.setattr(mfa_runner, '_run_subprocess_text', fake_run)
    monkeypatch.setattr(mfa_runner, 'ensure_mfa_python_packaging_stack', lambda *a, **k: True)

    assert mfa_runner.ensure_japanese_support(str(mfa_bat)) is True
    assert len(calls) == 3
    for _args, kwargs in calls:
        assert kwargs.get('cwd') == str(env_dir)


def test_run_mfa_align_japanese_dependency_check_is_soft_by_default(monkeypatch, tmp_path):
    env_dir = tmp_path / ".env"
    scripts_dir = env_dir / "Scripts"
    scripts_dir.mkdir(parents=True)
    mfa_exe = scripts_dir / "mfa.exe"
    mfa_exe.write_text("", encoding="utf-8")

    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    out_dir = tmp_path / "textgrids"
    dict_path = tmp_path / "dict.txt"
    dict_path.write_text("a a\n", encoding="utf-8")

    monkeypatch.delenv("UTOA_STRICT_TOKENIZER_GATE", raising=False)
    monkeypatch.setattr(mfa_runner, "ensure_japanese_support", lambda *_a, **_k: False)
    monkeypatch.setattr(mfa_runner, "_get_conda_env", lambda *_a, **_k: {})
    monkeypatch.setattr(mfa_runner, "_preflight_compute_mfcc", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr(mfa_runner, "_contains_non_ascii", lambda *_a, **_k: False)
    monkeypatch.setattr(mfa_runner, "_sanitize_alignment_dictionary_for_mfa", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr(mfa_runner, "_validate_alignment_dictionary", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr(mfa_runner, "_resolve_single_speaker_flag", lambda *_a, **_k: "--single_speaker")
    monkeypatch.setattr(mfa_runner, "_resolve_mfa_align_options", lambda *_a, **_k: ("default", {"beam": 1000, "retry_beam": 4000, "num_jobs": 1, "fine_tune": False}))
    monkeypatch.setattr(
        mfa_runner,
        "_resolve_mfa_runtime_options",
        lambda **_k: {"constrained_mode": "off", "recursive_mfa": False, "recursive_chunk_size": 96, "recursive_max_depth": 8, "beam_scale": 1.0},
    )
    monkeypatch.setattr(mfa_runner, "_run_mfa_align_command", lambda **_k: (True, "", []))

    ok, err = mfa_runner.run_mfa_align(
        str(mfa_exe),
        str(wav_dir),
        str(dict_path),
        str(out_dir),
        language="japanese",
    )

    assert ok is True
    assert err == ""


def test_run_mfa_align_korean_dependency_check_is_soft_by_default(monkeypatch, tmp_path):
    env_dir = tmp_path / ".env"
    scripts_dir = env_dir / "Scripts"
    scripts_dir.mkdir(parents=True)
    mfa_exe = scripts_dir / "mfa.exe"
    mfa_exe.write_text("", encoding="utf-8")

    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    out_dir = tmp_path / "textgrids"
    dict_path = tmp_path / "dict.txt"
    dict_path.write_text("a a\n", encoding="utf-8")

    monkeypatch.delenv("UTOA_STRICT_TOKENIZER_GATE", raising=False)
    monkeypatch.setattr(mfa_runner, "ensure_korean_support", lambda *_a, **_k: False)
    monkeypatch.setattr(mfa_runner, "_get_conda_env", lambda *_a, **_k: {})
    monkeypatch.setattr(mfa_runner, "_preflight_compute_mfcc", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr(mfa_runner, "_contains_non_ascii", lambda *_a, **_k: False)
    monkeypatch.setattr(mfa_runner, "_sanitize_alignment_dictionary_for_mfa", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr(mfa_runner, "_validate_alignment_dictionary", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr(mfa_runner, "_resolve_single_speaker_flag", lambda *_a, **_k: "--single_speaker")
    monkeypatch.setattr(mfa_runner, "_resolve_mfa_align_options", lambda *_a, **_k: ("default", {"beam": 1000, "retry_beam": 4000, "num_jobs": 1, "fine_tune": False}))
    monkeypatch.setattr(
        mfa_runner,
        "_resolve_mfa_runtime_options",
        lambda **_k: {"constrained_mode": "off", "recursive_mfa": False, "recursive_chunk_size": 96, "recursive_max_depth": 8, "beam_scale": 1.0},
    )
    monkeypatch.setattr(mfa_runner, "_run_mfa_align_command", lambda **_k: (True, "", []))

    ok, err = mfa_runner.run_mfa_align(
        str(mfa_exe),
        str(wav_dir),
        str(dict_path),
        str(out_dir),
        language="korean",
    )

    assert ok is True
    assert err == ""


def test_preflight_compute_mfcc_is_soft_by_default(monkeypatch):
    monkeypatch.delenv("UTOA_STRICT_MFA_PREFLIGHT", raising=False)
    monkeypatch.setattr(mfa_runner, "_get_conda_env", lambda *_a, **_k: {})

    def _raise_not_found(*_args, **_kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(mfa_runner.subprocess, "run", _raise_not_found)

    ok, err = mfa_runner._preflight_compute_mfcc("fake_mfa")
    assert ok is True
    assert err == ""
