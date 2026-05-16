"""Build the OTO outputs + labeling sheet for an A/B listening run.

Reads a manifest (see ml/eval/ab_listening/manifest.example.json) and:
- For each voicebank × autooto-kind version, runs generate_oto_with_boundary_decoder
  with the version's env overrides and writes the OTO to runs/<run_id>/<version_id>/<voicebank_id>/oto.ini.
- For external/moresampler-kind versions, copies the user-prepared OTO from
  manual_oto_dir/<voicebank_id>/oto.ini into the same layout.
- Emits a case manifest and a blank labeling CSV at labels/<run_id>.csv.

Manual steps (out of scope here):
- Authoring USTx files per voicebank.
- Running moresampler / external tools to produce their OTO outputs.
- Rendering each (voicebank × version) pair in OpenUtau using the produced OTO.
- Listening to renders and filling defect columns in the labeling CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from typing import Any

from core.coarse_crnn.boundary_generator import generate_oto_with_boundary_decoder
from core.oto_file_utils import parse_oto_line, read_text_with_fallback


DEFECT_COLUMNS: tuple[str, ...] = (
    "click",
    "consonant_clip",
    "vc_misalign",
    "overlap_unnatural",
    "cutoff_overreach",
)


def _read_manifest(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        raise SystemExit(f"manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise SystemExit("manifest root must be an object")
    for key in ("run_id", "voicebanks", "versions"):
        if key not in data:
            raise SystemExit(f"manifest missing required key: {key}")
    return data


def _safe_case_token(alias: str) -> str:
    raw = str(alias or "").strip()
    if not raw:
        return "blank"
    return re.sub(r"[^0-9A-Za-z가-힣ぁ-んァ-ヶー]+", "_", raw)[:32] or "blank"


def _enumerate_cases(*, voicebank: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse the voicebank's source_oto and emit one case per alias line."""
    source = str(voicebank.get("source_oto", "") or "")
    if not os.path.isfile(source):
        raise SystemExit(f"voicebank {voicebank.get('id')}: source_oto not found at {source}")
    cases: list[dict[str, Any]] = []
    text = read_text_with_fallback(source)
    for line_idx, raw in enumerate(text.splitlines()):
        parsed = parse_oto_line(raw)
        if not parsed:
            continue
        wav_name = str(parsed.get("wav", "") or "")
        alias = str(parsed.get("alias", "") or "")
        if not wav_name:
            continue
        cases.append(
            {
                "case_id": f"{voicebank['id']}__{line_idx:04d}__{_safe_case_token(alias)}",
                "voicebank_id": str(voicebank["id"]),
                "language": str(voicebank.get("language", "") or ""),
                "wav_name": wav_name,
                "alias": alias,
                "line_index": int(line_idx),
            }
        )
    return cases


def _resolve_run_dir(out_root: str, run_id: str) -> str:
    return os.path.abspath(os.path.join(out_root, run_id))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _apply_env_overrides(env: dict[str, str] | None) -> dict[str, str | None]:
    """Apply env overrides and return a dict for restoring the previous state."""
    restore: dict[str, str | None] = {}
    if not env:
        return restore
    for key, value in env.items():
        restore[str(key)] = os.environ.get(str(key))
        os.environ[str(key)] = str(value)
    return restore


def _restore_env(restore: dict[str, str | None]) -> None:
    for key, value in restore.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _generate_autooto_oto(
    *,
    voicebank: dict[str, Any],
    version: dict[str, Any],
    out_oto: str,
    log: list[str],
) -> tuple[int, int, list[str]]:
    model_path = str(version.get("model_path", "") or "")
    if not os.path.isfile(model_path):
        return 0, 0, [f"version {version['id']}: model_path not found at {model_path}"]
    special = set(voicebank.get("special_aliases", []) or [])
    restore = _apply_env_overrides(version.get("env"))
    try:
        processed, total, errors = generate_oto_with_boundary_decoder(
            wav_dir=str(voicebank["wav_dir"]),
            out_path=out_oto,
            source_oto_path=str(voicebank["source_oto"]),
            language=str(voicebank.get("language", "korean") or "korean"),
            format_type=str(voicebank.get("format_type", "") or ""),
            model_path=model_path,
            device=str(version.get("device", "auto") or "auto"),
            alias_suffix=str(version.get("alias_suffix", "") or ""),
            callback=lambda text: log.append(text),
            special_aliases=special,
        )
    finally:
        _restore_env(restore)
    return processed, total, errors


def _copy_external_oto(
    *,
    voicebank: dict[str, Any],
    version: dict[str, Any],
    out_oto: str,
) -> tuple[bool, str]:
    manual_root = str(version.get("manual_oto_dir", "") or "")
    if not manual_root:
        return False, f"version {version['id']}: manual_oto_dir not set"
    candidate = os.path.join(manual_root, str(voicebank["id"]), "oto.ini")
    if not os.path.isfile(candidate):
        return False, f"version {version['id']}: missing OTO at {candidate}"
    _ensure_dir(os.path.dirname(out_oto))
    shutil.copyfile(candidate, out_oto)
    return True, candidate


def _write_cases_manifest(*, cases: list[dict[str, Any]], path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"cases": cases}, handle, ensure_ascii=False, indent=2)


def _write_labeling_csv(*, cases: list[dict[str, Any]], versions: list[dict[str, Any]], path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    headers = [
        "case_id",
        "voicebank_id",
        "language",
        "wav_name",
        "alias",
        "version_id",
        *DEFECT_COLUMNS,
        "notes",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for case in cases:
            for version in versions:
                row = [
                    case["case_id"],
                    case["voicebank_id"],
                    case["language"],
                    case["wav_name"],
                    case["alias"],
                    version["id"],
                    *("" for _ in DEFECT_COLUMNS),
                    "",
                ]
                writer.writerow(row)


def _write_run_summary(*, summary: dict[str, Any], path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build OTO outputs and labeling sheet for an A/B listening run.")
    ap.add_argument("--manifest", required=True, help="Path to the run manifest JSON.")
    ap.add_argument(
        "--out-root",
        default=os.path.join("ml", "eval", "ab_listening", "runs"),
        help="Root directory for run output (default: ml/eval/ab_listening/runs).",
    )
    ap.add_argument(
        "--labels-root",
        default=os.path.join("ml", "eval", "ab_listening", "labels"),
        help="Root directory for labeling CSV (default: ml/eval/ab_listening/labels).",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="If set, skip generation when the target OTO already exists.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the manifest and print the plan without running generation.",
    )
    args = ap.parse_args()

    manifest_path = os.path.abspath(args.manifest)
    manifest = _read_manifest(manifest_path)
    run_id = str(manifest["run_id"]).strip()
    if not run_id:
        raise SystemExit("manifest run_id is empty")
    voicebanks: list[dict[str, Any]] = list(manifest.get("voicebanks") or [])
    versions: list[dict[str, Any]] = list(manifest.get("versions") or [])
    if not voicebanks:
        raise SystemExit("manifest has no voicebanks")
    if not versions:
        raise SystemExit("manifest has no versions")

    run_dir = _resolve_run_dir(args.out_root, run_id)
    labels_dir = os.path.abspath(args.labels_root)
    labels_csv = os.path.join(labels_dir, f"{run_id}.csv")
    cases_path = os.path.join(run_dir, "cases.json")
    summary_path = os.path.join(run_dir, "run_summary.json")

    plan_lines: list[str] = []
    plan_lines.append(f"run_id={run_id} voicebanks={len(voicebanks)} versions={len(versions)}")
    for vb in voicebanks:
        plan_lines.append(f"  vb {vb['id']} lang={vb.get('language','?')} wav_dir={vb.get('wav_dir')}")
    for v in versions:
        plan_lines.append(f"  ver {v['id']} kind={v.get('kind','?')}")
    print("\n".join(plan_lines))

    all_cases: list[dict[str, Any]] = []
    for vb in voicebanks:
        cases = _enumerate_cases(voicebank=vb)
        if cases:
            all_cases.extend(cases)
        print(f"[cases] {vb['id']} cases={len(cases)}")

    if args.dry_run:
        print("[dry-run] no files written")
        return 0

    _write_cases_manifest(cases=all_cases, path=cases_path)
    _write_labeling_csv(cases=all_cases, versions=versions, path=labels_csv)
    print(f"[cases] wrote {cases_path}")
    print(f"[labels] wrote {labels_csv} rows={len(all_cases) * len(versions)}")

    version_results: list[dict[str, Any]] = []
    overall_errors: list[str] = []

    for version in versions:
        ver_id = str(version["id"])
        kind = str(version.get("kind", "")).strip().lower()
        ver_dir = os.path.join(run_dir, ver_id)
        _ensure_dir(ver_dir)
        per_vb: list[dict[str, Any]] = []
        for vb in voicebanks:
            vb_id = str(vb["id"])
            out_oto = os.path.join(ver_dir, vb_id, "oto.ini")
            if args.skip_existing and os.path.isfile(out_oto):
                per_vb.append({"voicebank_id": vb_id, "status": "skipped_existing", "out_oto": out_oto})
                print(f"[skip] {ver_id}/{vb_id} (exists)")
                continue
            _ensure_dir(os.path.dirname(out_oto))
            t0 = time.time()
            if kind == "autooto":
                log_lines: list[str] = []
                processed, total, errors = _generate_autooto_oto(
                    voicebank=vb, version=version, out_oto=out_oto, log=log_lines
                )
                duration = time.time() - t0
                status = "ok" if (processed > 0 and not errors) else "errors"
                per_vb.append(
                    {
                        "voicebank_id": vb_id,
                        "status": status,
                        "out_oto": out_oto,
                        "processed": int(processed),
                        "total": int(total),
                        "duration_s": round(duration, 2),
                        "errors": errors[:10],
                        "log_tail": log_lines[-12:],
                    }
                )
                if errors:
                    overall_errors.extend(f"{ver_id}/{vb_id}: {e}" for e in errors)
                print(
                    f"[gen] {ver_id}/{vb_id} processed={processed}/{total} "
                    f"errors={len(errors)} dur={duration:.1f}s"
                )
            elif kind in {"external", "moresampler"}:
                ok, info = _copy_external_oto(voicebank=vb, version=version, out_oto=out_oto)
                duration = time.time() - t0
                if not ok:
                    overall_errors.append(f"{ver_id}/{vb_id}: {info}")
                    per_vb.append(
                        {"voicebank_id": vb_id, "status": "missing_manual_oto", "detail": info}
                    )
                    print(f"[copy] {ver_id}/{vb_id} MISSING: {info}")
                else:
                    per_vb.append(
                        {
                            "voicebank_id": vb_id,
                            "status": "copied",
                            "out_oto": out_oto,
                            "source": info,
                            "duration_s": round(duration, 2),
                        }
                    )
                    print(f"[copy] {ver_id}/{vb_id} <- {info}")
            else:
                overall_errors.append(f"version {ver_id}: unknown kind '{kind}'")
                per_vb.append({"voicebank_id": vb_id, "status": "unknown_kind"})
                print(f"[skip] {ver_id}/{vb_id} unknown kind={kind}")
        version_results.append({"version_id": ver_id, "kind": kind, "results": per_vb})

    summary = {
        "run_id": run_id,
        "manifest": manifest_path,
        "out_root": os.path.abspath(args.out_root),
        "labels_csv": labels_csv,
        "case_count": len(all_cases),
        "version_count": len(versions),
        "voicebank_count": len(voicebanks),
        "versions": version_results,
        "overall_errors": overall_errors[:50],
    }
    _write_run_summary(summary=summary, path=summary_path)
    print(f"[summary] wrote {summary_path}")
    if overall_errors:
        print(f"[done] {len(overall_errors)} error(s) — see summary for detail")
        return 1
    print("[done] all OTOs generated/copied successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
