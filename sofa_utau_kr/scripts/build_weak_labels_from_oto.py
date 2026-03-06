#!/usr/bin/env python
"""
Build SOFA weak-label dataset from UTAU oto.ini.

Output layout:
  <out_root>/weak_label/<subset_name>/
    wavs/
    transcriptions.csv  # columns: name, ph_seq
"""

import argparse
import csv
import json
import os
import shutil
from typing import Dict, Iterable, List, Tuple


DEFAULT_SP_TOKENS = {"", "-", "_", "sp", "sil", "pau", "br", "r", "R", "SP"}


def parse_oto_line(line: str) -> Tuple[str, str]:
    """Return (wav_name, alias). Raises ValueError when format is invalid."""
    text = (line or "").strip()
    if not text or text.startswith("#") or text.startswith(";"):
        raise ValueError("empty_or_comment")
    if "=" not in text:
        raise ValueError("missing_equal")
    left, right = text.split("=", 1)
    wav_name = left.strip()
    parts = [p.strip() for p in right.split(",")]
    alias = parts[0] if parts else ""
    if not wav_name:
        raise ValueError("empty_wav_name")
    return wav_name, alias


def load_phoneme_map(path: str) -> Dict[str, List[str]]:
    """
    Supported map formats:
    - JSON object: {"alias": "ph1 ph2", ...} or {"alias": ["ph1", "ph2"]}
    - TXT/TSV: each line "alias<TAB>ph1 ph2"
    """
    mapping: Dict[str, List[str]] = {}
    if not path:
        return mapping
    if not os.path.exists(path):
        raise FileNotFoundError(f"phoneme map not found: {path}")

    lower = path.lower()
    if lower.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("phoneme map json must be an object")
        for key, value in data.items():
            k = str(key).strip()
            if not k:
                continue
            if isinstance(value, list):
                tokens = [str(v).strip() for v in value if str(v).strip()]
            else:
                tokens = [t for t in str(value).strip().split(" ") if t]
            if tokens:
                mapping[k] = tokens
        return mapping

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#") or text.startswith(";"):
                continue
            if "\t" not in text:
                continue
            key, value = text.split("\t", 1)
            k = key.strip()
            tokens = [t for t in value.strip().split(" ") if t]
            if k and tokens:
                mapping[k] = tokens
    return mapping


def enforce_sp_rules(tokens: Iterable[str], sp_token: str = "SP") -> List[str]:
    normalized: List[str] = []
    for token in tokens:
        t = str(token).strip()
        if not t:
            continue
        if t == sp_token:
            if normalized and normalized[-1] == sp_token:
                continue
            normalized.append(sp_token)
            continue
        normalized.append(t)

    if not normalized or normalized[0] != sp_token:
        normalized.insert(0, sp_token)
    if normalized[-1] != sp_token:
        normalized.append(sp_token)
    return normalized


def normalize_alias_to_ph_seq(
    alias: str,
    phoneme_map: Dict[str, List[str]] = None,
    sp_tokens: Iterable[str] = None,
    strip_suffixes: Iterable[str] = None,
) -> str:
    phoneme_map = phoneme_map or {}
    sp_set = {str(x).strip() for x in (sp_tokens or DEFAULT_SP_TOKENS)}
    strip_suffixes = [str(s).strip() for s in (strip_suffixes or []) if str(s).strip()]

    text = (alias or "").replace("\u3000", " ").strip()
    for suffix in strip_suffixes:
        if suffix and text.endswith(suffix):
            text = text[: -len(suffix)].rstrip()
            break

    raw_tokens = [tok for tok in text.split(" ") if tok]
    if not raw_tokens:
        return "SP"

    expanded: List[str] = []
    for token in raw_tokens:
        if token in sp_set:
            expanded.append("SP")
            continue
        mapped = phoneme_map.get(token)
        if mapped:
            expanded.extend(mapped)
            continue
        expanded.append(token)

    final_tokens = []
    for token in expanded:
        t = str(token).strip()
        if not t:
            continue
        if t in sp_set:
            final_tokens.append("SP")
        else:
            final_tokens.append(t)
    final_tokens = enforce_sp_rules(final_tokens, sp_token="SP")
    return " ".join(final_tokens)


def resolve_wav_path(wav_dir: str, wav_name: str) -> str:
    candidate = wav_name.strip()
    if os.path.isabs(candidate) and os.path.exists(candidate):
        return candidate
    joined = os.path.join(wav_dir, candidate)
    if os.path.exists(joined):
        return joined
    # fallback: use basename only
    base_joined = os.path.join(wav_dir, os.path.basename(candidate))
    if os.path.exists(base_joined):
        return base_joined
    return ""


def write_rows_csv_json(rows: List[dict], csv_path: str, json_path: str) -> None:
    rows = list(rows or [])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(rows, jf, indent=2, ensure_ascii=False)

    keys = sorted({k for row in rows for k in row.keys()})
    with open(csv_path, "w", encoding="utf-8", newline="") as cf:
        if not keys:
            cf.write("")
            return
        writer = csv.DictWriter(cf, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_weak_labels(
    oto_path: str,
    wav_dir: str,
    out_root: str,
    subset_name: str,
    phoneme_map: Dict[str, List[str]],
    sp_tokens: Iterable[str],
    strip_suffixes: Iterable[str],
    copy_wavs: bool,
) -> dict:
    subset_dir = os.path.join(out_root, "weak_label", subset_name)
    wav_out_dir = os.path.join(subset_dir, "wavs")
    os.makedirs(subset_dir, exist_ok=True)
    if copy_wavs:
        os.makedirs(wav_out_dir, exist_ok=True)

    trans_rows = []
    skip_rows = []
    seen_names = set()

    with open(oto_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            try:
                wav_name, alias = parse_oto_line(line)
            except ValueError as e:
                reason = str(e)
                if reason == "empty_or_comment":
                    continue
                skip_rows.append(
                    {
                        "line_no": idx,
                        "raw_line": line.rstrip("\n"),
                        "reason": reason,
                    }
                )
                continue

            wav_path = resolve_wav_path(wav_dir, wav_name)
            if not wav_path:
                skip_rows.append(
                    {
                        "line_no": idx,
                        "wav_name": wav_name,
                        "alias": alias,
                        "reason": "missing_wav",
                    }
                )
                continue

            base = os.path.splitext(os.path.basename(wav_path))[0]
            if base in seen_names:
                skip_rows.append(
                    {
                        "line_no": idx,
                        "wav_name": wav_name,
                        "alias": alias,
                        "reason": "duplicate_wav_entry",
                    }
                )
                continue

            ph_seq = normalize_alias_to_ph_seq(
                alias=alias,
                phoneme_map=phoneme_map,
                sp_tokens=sp_tokens,
                strip_suffixes=strip_suffixes,
            )
            if not ph_seq.strip():
                skip_rows.append(
                    {
                        "line_no": idx,
                        "wav_name": wav_name,
                        "alias": alias,
                        "reason": "empty_ph_seq",
                    }
                )
                continue

            if copy_wavs:
                target_wav = os.path.join(wav_out_dir, os.path.basename(wav_path))
                shutil.copy2(wav_path, target_wav)

            trans_rows.append({"name": base, "ph_seq": ph_seq})
            seen_names.add(base)

    transcriptions_csv = os.path.join(subset_dir, "transcriptions.csv")
    with open(transcriptions_csv, "w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=["name", "ph_seq"])
        writer.writeheader()
        for row in trans_rows:
            writer.writerow(row)

    reports_dir = os.path.join(out_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    skip_csv = os.path.join(reports_dir, f"weak_label_{subset_name}_skip_report.csv")
    skip_json = os.path.join(reports_dir, f"weak_label_{subset_name}_skip_report.json")
    write_rows_csv_json(skip_rows, skip_csv, skip_json)

    summary = {
        "oto_path": oto_path,
        "wav_dir": wav_dir,
        "out_subset_dir": subset_dir,
        "transcriptions_csv": transcriptions_csv,
        "kept_rows": len(trans_rows),
        "skipped_rows": len(skip_rows),
        "copied_wavs": len(trans_rows) if copy_wavs else 0,
        "skip_report_csv": skip_csv,
        "skip_report_json": skip_json,
    }
    summary_path = os.path.join(reports_dir, f"weak_label_{subset_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2, ensure_ascii=False)
    summary["summary_json"] = summary_path
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oto", required=True, help="Path to manual oto.ini")
    parser.add_argument("--wav-dir", required=True, help="Folder containing wav files")
    parser.add_argument("--out-root", required=True, help="Output root folder (e.g. sofa_utau_kr/data)")
    parser.add_argument("--subset-name", default="utau_kr", help="Subset name under weak_label/")
    parser.add_argument("--phoneme-map", default="", help="Optional phoneme map path (json or tsv)")
    parser.add_argument(
        "--sp-tokens",
        default="SP,sp,sil,pau,br,R,-,_",
        help="Comma-separated silence tokens",
    )
    parser.add_argument(
        "--strip-suffixes",
        default="",
        help="Comma-separated suffixes to strip from alias (e.g. _C4,_D4)",
    )
    parser.add_argument(
        "--no-copy-wavs",
        action="store_true",
        help="Do not copy wav files into weak_label/<subset>/wavs",
    )
    args = parser.parse_args()

    phoneme_map = load_phoneme_map(args.phoneme_map) if args.phoneme_map else {}
    sp_tokens = [t.strip() for t in args.sp_tokens.split(",") if t.strip()]
    strip_suffixes = [t.strip() for t in args.strip_suffixes.split(",") if t.strip()]
    summary = build_weak_labels(
        oto_path=args.oto,
        wav_dir=args.wav_dir,
        out_root=args.out_root,
        subset_name=args.subset_name,
        phoneme_map=phoneme_map,
        sp_tokens=sp_tokens,
        strip_suffixes=strip_suffixes,
        copy_wavs=not args.no_copy_wavs,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

