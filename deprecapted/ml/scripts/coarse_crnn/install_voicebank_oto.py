"""Install a generated comparison OTO into each voicebank's `oto.ini`, with backup.

After `build_ab_comparison_oto.py` writes per-version and combined OTOs, the
listener still has to copy one of them into each voicebank's `oto.ini` slot
before rendering. This script automates that step safely:

- Reads the run manifest to discover voicebank wav directories.
- Auto-discovers the comparison root from the manifest's `run_id`
  (default `ml/eval/ab_listening/runs/<run_id>/comparison`).
- Backs up each voicebank's existing `oto.ini` to a timestamped file under a
  `.autooto_eval_backups/` directory inside the voicebank, so the original
  hand-tuned OTO is never lost.
- Copies `<comparison-root>/<voicebank_id>/<variant>.oto.ini` into the
  voicebank as `oto.ini`.

`--variant` selects which generated OTO to install:

- `combined` (default): all versions in one file, aliases suffixed with the
  version id. Recommended for workflow B (same USTx, swap aliases for A/B).
- `manual` / `autooto_baseline` / `autooto_F1` / any other version id: that
  single trimmed file. Use for workflow A (one version at a time).

`--restore` puts the latest backup back, in case you want to return the
voicebank to its original OTO after listening.

Examples
--------

Install combined OTO into both voicebanks (dry-run first)::

    python -m ml.scripts.coarse_crnn.install_voicebank_oto `
        --manifest ml/eval/ab_listening/manifests/2026-05-15_baseline.json `
        --variant combined --dry-run

Then for real::

    python -m ml.scripts.coarse_crnn.install_voicebank_oto `
        --manifest ml/eval/ab_listening/manifests/2026-05-15_baseline.json `
        --variant combined

Restore the most recent backup::

    python -m ml.scripts.coarse_crnn.install_voicebank_oto `
        --manifest ml/eval/ab_listening/manifests/2026-05-15_baseline.json `
        --restore
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import shutil
import sys
from typing import Any


BACKUP_DIR_NAME = ".autooto_eval_backups"
BACKUP_PREFIX = "oto.ini.autooto-eval.bak."


def _read_manifest(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        raise SystemExit(f"manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise SystemExit("manifest root must be an object")
    if "run_id" not in data or "voicebanks" not in data:
        raise SystemExit("manifest missing run_id or voicebanks")
    return data


def _resolve_comparison_root(*, manifest: dict[str, Any], explicit_root: str) -> str:
    if explicit_root:
        return os.path.abspath(explicit_root)
    run_id = str(manifest["run_id"])
    return os.path.abspath(
        os.path.join("ml", "eval", "ab_listening", "runs", run_id, "comparison")
    )


def _backup_path_for(*, wav_dir: str, timestamp: str) -> str:
    backup_dir = os.path.join(wav_dir, BACKUP_DIR_NAME)
    return os.path.join(backup_dir, f"{BACKUP_PREFIX}{timestamp}")


def _backup_existing_oto(*, wav_dir: str, timestamp: str) -> str | None:
    """Move the current oto.ini to a backup file. Returns the backup path, or None if no oto.ini existed."""
    current = os.path.join(wav_dir, "oto.ini")
    if not os.path.isfile(current):
        return None
    backup_dir = os.path.join(wav_dir, BACKUP_DIR_NAME)
    os.makedirs(backup_dir, exist_ok=True)
    target = _backup_path_for(wav_dir=wav_dir, timestamp=timestamp)
    shutil.copy2(current, target)
    return target


def _find_latest_backup(*, wav_dir: str) -> str | None:
    backup_dir = os.path.join(wav_dir, BACKUP_DIR_NAME)
    if not os.path.isdir(backup_dir):
        return None
    matches = sorted(glob.glob(os.path.join(backup_dir, f"{BACKUP_PREFIX}*")))
    return matches[-1] if matches else None


def _resolve_source_oto(*, comparison_root: str, vb_id: str, variant: str) -> str:
    return os.path.join(comparison_root, vb_id, f"{variant}.oto.ini")


def install_for_voicebank(
    *,
    vb_id: str,
    wav_dir: str,
    source_oto: str,
    timestamp: str,
    skip_backup: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not os.path.isdir(wav_dir):
        return {"voicebank_id": vb_id, "status": "wav_dir_missing", "wav_dir": wav_dir}
    if not os.path.isfile(source_oto):
        return {
            "voicebank_id": vb_id,
            "status": "source_oto_missing",
            "wav_dir": wav_dir,
            "source_oto": source_oto,
        }

    target = os.path.join(wav_dir, "oto.ini")
    result: dict[str, Any] = {
        "voicebank_id": vb_id,
        "wav_dir": wav_dir,
        "source_oto": source_oto,
        "target": target,
        "dry_run": bool(dry_run),
    }

    if dry_run:
        result["status"] = "dry_run"
        result["would_backup"] = (
            None if skip_backup or not os.path.isfile(target)
            else _backup_path_for(wav_dir=wav_dir, timestamp=timestamp)
        )
        return result

    backup_path: str | None = None
    if not skip_backup:
        backup_path = _backup_existing_oto(wav_dir=wav_dir, timestamp=timestamp)
    result["backup_path"] = backup_path
    shutil.copyfile(source_oto, target)
    result["status"] = "installed"
    return result


def restore_for_voicebank(*, vb_id: str, wav_dir: str, dry_run: bool = False) -> dict[str, Any]:
    if not os.path.isdir(wav_dir):
        return {"voicebank_id": vb_id, "status": "wav_dir_missing", "wav_dir": wav_dir}
    backup = _find_latest_backup(wav_dir=wav_dir)
    target = os.path.join(wav_dir, "oto.ini")
    result: dict[str, Any] = {
        "voicebank_id": vb_id,
        "wav_dir": wav_dir,
        "target": target,
        "dry_run": bool(dry_run),
    }
    if backup is None:
        result["status"] = "no_backup_found"
        return result
    result["backup_path"] = backup
    if dry_run:
        result["status"] = "dry_run"
        return result
    shutil.copyfile(backup, target)
    result["status"] = "restored"
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Install or restore voicebank oto.ini using generated comparison OTOs.")
    ap.add_argument("--manifest", required=True, help="Run manifest JSON used by the build/sample/comparison scripts.")
    ap.add_argument(
        "--variant",
        default="combined",
        help="Which comparison file to install: 'combined' (default) or any per-version id (e.g. 'manual', 'autooto_F1').",
    )
    ap.add_argument(
        "--comparison-root",
        default="",
        help="Override the comparison directory (default: ml/eval/ab_listening/runs/<run_id>/comparison).",
    )
    ap.add_argument(
        "--voicebanks",
        default="",
        help="Optional comma-separated voicebank_id filter. Default: all voicebanks in manifest.",
    )
    ap.add_argument("--restore", action="store_true", help="Restore each voicebank's latest backup instead of installing.")
    ap.add_argument("--no-backup", action="store_true", help="Skip backup before overwrite. Use only if you know what you are doing.")
    ap.add_argument("--dry-run", action="store_true", help="Show what would happen without modifying files.")
    args = ap.parse_args()

    manifest_path = os.path.abspath(args.manifest)
    manifest = _read_manifest(manifest_path)
    voicebanks: list[dict[str, Any]] = list(manifest.get("voicebanks") or [])
    if not voicebanks:
        raise SystemExit("manifest has no voicebanks")

    filter_ids = {t.strip() for t in str(args.voicebanks or "").split(",") if t.strip()}
    if filter_ids:
        voicebanks = [vb for vb in voicebanks if str(vb.get("id", "")) in filter_ids]
        if not voicebanks:
            raise SystemExit(f"--voicebanks filter matched nothing in manifest: {sorted(filter_ids)}")

    comparison_root = _resolve_comparison_root(manifest=manifest, explicit_root=args.comparison_root)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    summaries: list[dict[str, Any]] = []
    any_error = False
    for vb in voicebanks:
        vb_id = str(vb.get("id", "") or "")
        wav_dir = os.path.abspath(str(vb.get("wav_dir", "") or ""))
        if args.restore:
            result = restore_for_voicebank(vb_id=vb_id, wav_dir=wav_dir, dry_run=bool(args.dry_run))
        else:
            source_oto = _resolve_source_oto(
                comparison_root=comparison_root, vb_id=vb_id, variant=str(args.variant)
            )
            result = install_for_voicebank(
                vb_id=vb_id,
                wav_dir=wav_dir,
                source_oto=source_oto,
                timestamp=timestamp,
                skip_backup=bool(args.no_backup),
                dry_run=bool(args.dry_run),
            )
        summaries.append(result)
        status = result.get("status", "?")
        if status in {"installed", "restored", "dry_run"}:
            extra = ""
            if status == "installed" and result.get("backup_path"):
                extra = f" backup={result['backup_path']}"
            elif status == "restored" and result.get("backup_path"):
                extra = f" from={result['backup_path']}"
            elif status == "dry_run" and result.get("would_backup"):
                extra = f" would_backup={result['would_backup']}"
            print(f"[install] {vb_id}: {status} target={result.get('target')}{extra}")
        else:
            any_error = True
            print(f"[install] {vb_id}: ERROR status={status} detail={result}")

    print(json.dumps({"timestamp": timestamp, "comparison_root": comparison_root, "results": summaries}, ensure_ascii=False, indent=2))
    return 1 if any_error else 0


if __name__ == "__main__":
    sys.exit(main())
