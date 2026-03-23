import json
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import build


def test_copy_mfa_model_bundle_copies_only_model_roots(monkeypatch, tmp_path):
    runtime_root = tmp_path / "runtime"
    env_dir = runtime_root / ".env"
    env_dir.mkdir(parents=True)
    model_root = runtime_root / ".mfa_root_ascii"
    acoustic_dir = model_root / "acoustic"
    acoustic_dir.mkdir(parents=True)
    (acoustic_dir / "korean_mfa.zip").write_text("ko", encoding="utf-8")
    (acoustic_dir / "japanese_mfa.zip").write_text("ja", encoding="utf-8")

    model_status = {
        "korean_mfa": str(acoustic_dir / "korean_mfa.zip"),
        "japanese_mfa": str(acoustic_dir / "japanese_mfa.zip"),
    }

    monkeypatch.setattr(
        build,
        "_resolve_validated_mfa_runtime_bundle_source",
        lambda runtime_root_arg, require_models=False: (
            str(runtime_root),
            str(env_dir),
            str(env_dir / "Scripts" / "mfa.bat"),
            model_status,
            [str(model_root)],
        ),
    )

    release_dir = tmp_path / "release"
    bundle_dir = build._copy_mfa_model_bundle(str(release_dir), str(runtime_root), require_models=True)

    assert bundle_dir == str(release_dir / "mfa_runtime_bundle")
    assert (release_dir / "mfa_runtime_bundle" / ".mfa_root_ascii" / "acoustic" / "korean_mfa.zip").is_file()
    assert (release_dir / "mfa_runtime_bundle" / ".mfa_root_ascii" / "acoustic" / "japanese_mfa.zip").is_file()
    assert not (release_dir / "mfa_runtime_bundle" / ".env").exists()

    manifest = json.loads((release_dir / "mfa_runtime_bundle" / "bundle_manifest.json").read_text(encoding="utf-8"))
    assert manifest["bundle_kind"] == "models_only"
    assert manifest["copied_model_roots"] == [".mfa_root_ascii"]
    assert manifest["require_models"] is True
