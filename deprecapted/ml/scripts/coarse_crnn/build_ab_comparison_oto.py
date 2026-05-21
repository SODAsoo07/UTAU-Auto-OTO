"""Generate listener-friendly comparison OTOs covering every source alias.

Builds two flavors of OTO per voicebank, both honoring this invariant:
**alias text comes verbatim from `source_oto.ini`**, plus an optional version
suffix on the combined file. Parameter values come from each version's
generated OTO under `runs/<run_id>/<version>/<voicebank>/oto.ini`. If a
version is missing a particular (wav, alias) pair, the row falls back to the
source OTO's parameters so the line still renders.

Outputs per voicebank
---------------------
- `comparison/<vb>/<version>.oto.ini`
  Full alias set with that version's parameters. Drop-in replacement for the
  voicebank's `oto.ini` (workflow A: install one version, render same USTx,
  swap to the next version, repeat).

- `comparison/<vb>/combined.oto.ini`
  All source aliases in two forms:
  1. Unsuffixed (source parameters) — covers any USTx note whose alias is the
     plain source form.
  2. Suffixed `<alias>__<version>` for every version — covers USTx notes the
     listener has rewritten to compare versions directly (workflow B).

Sample CSVs from `sample_ab_labels.py` are no longer used here; they only
control which cases the listener actually labels, not the OTO scope.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any

from core.oto_file_utils import parse_oto_line, read_text_with_fallback


def _read_manifest(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        raise SystemExit(f"manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise SystemExit("manifest root must be an object")
    for key in ("run_id", "voicebanks"):
        if key not in data:
            raise SystemExit(f"manifest missing required key: {key}")
    return data


def _resolve_run_dir(*, manifest: dict[str, Any], explicit_run_dir: str) -> str:
    if explicit_run_dir:
        return os.path.abspath(explicit_run_dir)
    run_id = str(manifest["run_id"])
    return os.path.abspath(os.path.join("ml", "eval", "ab_listening", "runs", run_id))


def _discover_versions(run_dir: str, vb_id: str) -> list[str]:
    if not os.path.isdir(run_dir):
        return []
    out: list[str] = []
    for entry in sorted(os.listdir(run_dir)):
        candidate = os.path.join(run_dir, entry, vb_id, "oto.ini")
        if os.path.isfile(candidate):
            out.append(entry)
    return out


def _read_oto_rows(path: str) -> list[dict[str, Any]]:
    """Return parsed OTO rows in their original line order."""
    if not os.path.isfile(path):
        return []
    text = read_text_with_fallback(path)
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        parsed = parse_oto_line(line)
        if not parsed:
            continue
        out.append(parsed)
    return out


def _build_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Index rows by (wav, alias). Multiple matches preserve order for occurrence indexing."""
    table: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["wav"]).strip(), str(row["alias"]).strip())
        table[key].append(row)
    return table


def _lookup_occurrence(
    table: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    wav: str,
    alias: str,
    occurrence: int,
) -> dict[str, Any] | None:
    bucket = table.get((str(wav).strip(), str(alias).strip()))
    if not bucket:
        return None
    if occurrence < 0 or occurrence >= len(bucket):
        return None
    return bucket[occurrence]


def _format_oto_line(*, wav: str, alias: str, params: dict[str, Any]) -> str:
    return (
        f"{wav}={alias},"
        f"{float(params['offset']):.3f},"
        f"{float(params['cons']):.3f},"
        f"{float(params['cutoff']):.3f},"
        f"{float(params['pre']):.3f},"
        f"{float(params['ovl']):.3f}"
    )


def _write_oto(path: str, lines: list[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    text = "\n".join(lines)
    if lines and not text.endswith("\n"):
        text += "\n"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def build_per_version_oto(
    *,
    source_rows: list[dict[str, Any]],
    version_rows: list[dict[str, Any]],
    out_path: str,
) -> dict[str, Any]:
    """Emit one OTO containing every source alias with this version's params (or source fallback)."""
    version_lookup = _build_lookup(version_rows)
    seen: dict[tuple[str, str], int] = defaultdict(int)
    lines: list[str] = []
    missing = 0
    for source in source_rows:
        wav = str(source["wav"]).strip()
        alias = str(source["alias"]).strip()
        occ = seen[(wav, alias)]
        seen[(wav, alias)] += 1
        version_row = _lookup_occurrence(version_lookup, wav=wav, alias=alias, occurrence=occ)
        if version_row is None:
            version_row = source
            missing += 1
        lines.append(_format_oto_line(wav=wav, alias=alias, params=version_row))
    _write_oto(out_path, lines)
    return {
        "out_path": out_path,
        "row_count": len(lines),
        "missing_in_version": missing,
    }


def build_combined_oto(
    *,
    source_rows: list[dict[str, Any]],
    version_rows_by_id: dict[str, list[dict[str, Any]]],
    out_path: str,
    alias_separator: str = "__",
    include_unsuffixed: bool = True,
) -> dict[str, Any]:
    """Emit a combined OTO covering source aliases with every version, plus optional unsuffixed pass.

    Unsuffixed block (only when include_unsuffixed=True) lets a USTx render
    untouched: any plain source alias still resolves to source parameters.

    Suffixed blocks let the listener swap a note's alias to `<alias>__<version>`
    for direct A/B comparison without swapping OTO files.
    """
    lines: list[str] = []
    per_version_missing: dict[str, int] = {}

    if include_unsuffixed:
        lines.append("# --- source (unsuffixed) ---")
        for source in source_rows:
            lines.append(
                _format_oto_line(
                    wav=str(source["wav"]).strip(),
                    alias=str(source["alias"]).strip(),
                    params=source,
                )
            )

    for version_id, version_rows in version_rows_by_id.items():
        lookup = _build_lookup(version_rows)
        seen: dict[tuple[str, str], int] = defaultdict(int)
        missing = 0
        lines.append(f"# --- {version_id} ---")
        for source in source_rows:
            wav = str(source["wav"]).strip()
            alias = str(source["alias"]).strip()
            occ = seen[(wav, alias)]
            seen[(wav, alias)] += 1
            row = _lookup_occurrence(lookup, wav=wav, alias=alias, occurrence=occ)
            if row is None:
                row = source
                missing += 1
            suffixed = f"{alias}{alias_separator}{version_id}"
            lines.append(_format_oto_line(wav=wav, alias=suffixed, params=row))
        per_version_missing[version_id] = missing

    _write_oto(out_path, lines)
    return {
        "out_path": out_path,
        "row_count": len(lines),
        "include_unsuffixed": bool(include_unsuffixed),
        "missing_per_version": per_version_missing,
    }


def build_comparison_for_voicebank(
    *,
    vb_id: str,
    source_oto_path: str,
    run_dir: str,
    out_dir: str,
    versions: list[str] | None = None,
    alias_separator: str = "__",
    include_unsuffixed: bool = True,
) -> dict[str, Any]:
    source_rows = _read_oto_rows(source_oto_path)
    if not source_rows:
        return {
            "voicebank_id": vb_id,
            "status": "source_oto_empty_or_missing",
            "source_oto": source_oto_path,
        }
    discovered = _discover_versions(run_dir, vb_id) if versions is None else list(versions)
    if not discovered:
        return {
            "voicebank_id": vb_id,
            "status": "no_versions_found",
            "source_oto": source_oto_path,
            "versions": [],
        }

    version_rows_by_id: dict[str, list[dict[str, Any]]] = {}
    per_version_results: dict[str, dict[str, Any]] = {}
    for version_id in discovered:
        version_oto = os.path.join(run_dir, version_id, vb_id, "oto.ini")
        rows = _read_oto_rows(version_oto)
        version_rows_by_id[version_id] = rows
        per_version_path = os.path.join(out_dir, vb_id, f"{version_id}.oto.ini")
        per_version_results[version_id] = build_per_version_oto(
            source_rows=source_rows,
            version_rows=rows,
            out_path=per_version_path,
        )

    combined_path = os.path.join(out_dir, vb_id, "combined.oto.ini")
    combined_result = build_combined_oto(
        source_rows=source_rows,
        version_rows_by_id=version_rows_by_id,
        out_path=combined_path,
        alias_separator=alias_separator,
        include_unsuffixed=include_unsuffixed,
    )

    return {
        "voicebank_id": vb_id,
        "status": "ok",
        "source_oto": source_oto_path,
        "source_alias_count": len(source_rows),
        "versions": discovered,
        "per_version": per_version_results,
        "combined": combined_result,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate comparison OTOs (per-version full + combined) for every source alias.")
    ap.add_argument("--manifest", required=True, help="Run manifest JSON (provides voicebank source_oto paths and run_id).")
    ap.add_argument("--run-dir", default="", help="Override run output directory (default: ml/eval/ab_listening/runs/<run_id>).")
    ap.add_argument("--out-root", default="", help="Override output root (default: <run-dir>/comparison).")
    ap.add_argument("--alias-separator", default="__", help="Separator between source alias and version suffix (default: '__').")
    ap.add_argument(
        "--no-unsuffixed",
        action="store_true",
        help="Skip the unsuffixed source-alias block in the combined file (smaller but USTx must use suffixed aliases for every note).",
    )
    ap.add_argument(
        "--versions",
        default="",
        help="Optional comma-separated version_id filter (default: auto-discover).",
    )
    ap.add_argument(
        "--voicebanks",
        default="",
        help="Optional comma-separated voicebank_id filter (default: all in manifest).",
    )
    ap.add_argument("--summary-out", default="", help="Optional summary JSON path (default: <out-root>/comparison_summary.json).")
    args = ap.parse_args()

    manifest = _read_manifest(os.path.abspath(args.manifest))
    run_dir = _resolve_run_dir(manifest=manifest, explicit_run_dir=args.run_dir)
    out_root = os.path.abspath(args.out_root or os.path.join(run_dir, "comparison"))
    version_filter = [t.strip() for t in str(args.versions or "").split(",") if t.strip()]
    vb_filter = {t.strip() for t in str(args.voicebanks or "").split(",") if t.strip()}

    voicebanks = list(manifest.get("voicebanks") or [])
    if vb_filter:
        voicebanks = [vb for vb in voicebanks if str(vb.get("id", "")) in vb_filter]
        if not voicebanks:
            raise SystemExit(f"--voicebanks filter matched nothing: {sorted(vb_filter)}")

    summaries: list[dict[str, Any]] = []
    any_error = False
    for vb in voicebanks:
        vb_id = str(vb.get("id", "") or "")
        source_oto = str(vb.get("source_oto", "") or "")
        discovered = _discover_versions(run_dir, vb_id)
        if version_filter:
            versions = [v for v in version_filter if v in discovered]
        else:
            versions = discovered
        result = build_comparison_for_voicebank(
            vb_id=vb_id,
            source_oto_path=source_oto,
            run_dir=run_dir,
            out_dir=out_root,
            versions=versions,
            alias_separator=str(args.alias_separator),
            include_unsuffixed=(not bool(args.no_unsuffixed)),
        )
        summaries.append(result)
        if result["status"] != "ok":
            any_error = True
            print(f"[comparison] {vb_id}: ERROR status={result['status']}")
            continue
        miss_pairs = result["combined"].get("missing_per_version", {})
        miss_text = " ".join(f"{v}=miss{n}" for v, n in miss_pairs.items() if n > 0)
        miss_suffix = f" [{miss_text}]" if miss_text else ""
        print(
            f"[comparison] {vb_id}: source={result['source_alias_count']} aliases "
            f"versions={','.join(result['versions'])} "
            f"combined_rows={result['combined']['row_count']}{miss_suffix}"
        )

    summary_path = args.summary_out or os.path.join(out_root, "comparison_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "manifest": os.path.abspath(args.manifest),
                "run_dir": run_dir,
                "out_root": out_root,
                "alias_separator": str(args.alias_separator),
                "include_unsuffixed": not bool(args.no_unsuffixed),
                "voicebanks": summaries,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[comparison] wrote summary {summary_path}")
    return 1 if any_error else 0


if __name__ == "__main__":
    sys.exit(main())
