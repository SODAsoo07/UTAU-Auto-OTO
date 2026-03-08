import argparse
import json
import os
import sys
import tempfile
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.mfa_runner import diagnose_mfa_runtime, find_mfa_executable, get_default_mfa_env_dir
from core.sofa_runner import _resolve_sofa_repo_dir, get_sofa_env_python


def _norm(path):
    return os.path.normcase(os.path.normpath(str(path or "")))


def probe_runtime_portability(fake_exe_path: str, language: str = "korean"):
    fake_exe_path = os.path.abspath(fake_exe_path)
    with mock.patch("core.mfa_runner.sys.frozen", True, create=True), mock.patch(
        "core.mfa_runner.sys.executable", fake_exe_path, create=True
    ):
        mfa_path = find_mfa_executable() or ""
        mfa_diag = diagnose_mfa_runtime(mfa_path, language=language)

    with mock.patch("core.sofa_runner.sys.frozen", True, create=True), mock.patch(
        "core.sofa_runner.sys.executable", fake_exe_path, create=True
    ):
        sofa_python = get_sofa_env_python() or ""
        sofa_repo = _resolve_sofa_repo_dir("")

    shared_mfa_root = os.path.dirname(os.path.dirname(get_default_mfa_env_dir()))
    return {
        "fake_exe_path": fake_exe_path,
        "language": language,
        "mfa_path": mfa_path,
        "mfa_ready": bool(mfa_diag.get("ready")),
        "mfa_issues": list(mfa_diag.get("issues", []) or []),
        "mfa_uses_shared_root": bool(mfa_path) and _norm(mfa_path).startswith(_norm(shared_mfa_root)),
        "sofa_python": sofa_python,
        "sofa_repo": sofa_repo,
        "sofa_python_exists": bool(sofa_python and os.path.exists(sofa_python)),
        "sofa_repo_exists": bool(sofa_repo and os.path.exists(os.path.join(sofa_repo, "infer.py"))),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Probe whether shared MFA/SOFA runtime paths still resolve after moving the built EXE."
    )
    parser.add_argument("--language", default="korean", choices=["korean", "japanese"])
    parser.add_argument("--fake-exe-path", default="")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    fake_exe_path = args.fake_exe_path
    temp_dir = None
    if not fake_exe_path:
        temp_dir = tempfile.mkdtemp(prefix="utoa_portable_probe_")
        fake_exe_path = os.path.join(temp_dir, "MovedRelease", "UTAU_Auto_OTO.exe")
        os.makedirs(os.path.dirname(fake_exe_path), exist_ok=True)
        with open(fake_exe_path, "wb") as f:
            f.write(b"")

    report = probe_runtime_portability(fake_exe_path, language=args.language)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)

    if args.json_out:
        out_path = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
