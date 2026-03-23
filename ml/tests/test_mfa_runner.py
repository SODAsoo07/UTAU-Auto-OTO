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
