from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
from typing import Dict, Iterable, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_OTO_ENCODINGS = ("utf-8-sig", "cp932", "cp949", "euc-kr", "utf-8", "latin-1")
_DEFAULT_SPECIAL_WAV_STEMS = {
    "_breath",
    "breath",
    "breaths",
    "inhale",
    "_inhale",
    "exhale",
    "_exhale",
    "sil",
    "_sil",
    "silence",
    "_silence",
    "pau",
    "_pau",
    "pause",
    "_pause",
    "rest",
    "_rest",
    "noise",
    "_noise",
    "noisefile",
    "sample",
    "_sample",
}


def _safe_name(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "case"
    out = []
    for ch in raw:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return re.sub(r"_+", "_", "".join(out)).strip("_") or "case"


def _read_oto_lines(path: str) -> List[str]:
    for enc in _OTO_ENCODINGS:
        try:
            with open(path, "r", encoding=enc) as handle:
                return [line.strip() for line in handle if line.strip() and "=" in line]
        except Exception:
            continue
    return []


def _wav_stem(path_or_name: str) -> str:
    return os.path.splitext(os.path.basename(str(path_or_name or "").strip()))[0].strip().lower()


def _is_special_wav_name(path_or_name: str, special_stems: set[str]) -> bool:
    stem = _wav_stem(path_or_name)
    return bool(stem and stem in special_stems)


def _quote_yaml(value: object) -> str:
    text = str(value if value is not None else "")
    return json.dumps(text, ensure_ascii=False)


def _iter_oto_paths(dataset_root: str) -> Iterable[str]:
    root = os.path.abspath(dataset_root)
    for base, _dirs, files in os.walk(root):
        if "oto.ini" in {str(name) for name in files}:
            yield os.path.join(base, "oto.ini")


def _has_wav(root: str) -> bool:
    for _base, _dirs, files in os.walk(root):
        if any(str(name).lower().endswith(".wav") for name in files):
            return True
    return False


def _iter_wavs(root: str) -> Iterable[str]:
    for base, _dirs, files in os.walk(root):
        for name in files:
            if str(name).lower().endswith(".wav"):
                yield os.path.join(base, str(name))


def _copy_or_link_file(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _prepare_eval_voicebank(source_dir: str, out_dir: str, special_stems: set[str]) -> Dict[str, object]:
    copied = 0
    skipped: List[str] = []
    os.makedirs(out_dir, exist_ok=True)
    for wav_path in _iter_wavs(source_dir):
        rel = os.path.relpath(wav_path, source_dir)
        if _is_special_wav_name(rel, special_stems):
            skipped.append(rel.replace("\\", "/"))
            continue
        _copy_or_link_file(wav_path, os.path.join(out_dir, rel))
        copied += 1
    return {"voicebank_dir": os.path.abspath(out_dir), "copied_wavs": copied, "skipped_wavs": skipped}


def _infer_lang_format(dataset_root: str, oto_path: str) -> tuple[str, str]:
    rel = os.path.relpath(os.path.abspath(oto_path), os.path.abspath(dataset_root))
    parts = rel.split(os.sep)
    language = parts[0].lower() if len(parts) >= 1 else ""
    fmt = parts[1].lower() if len(parts) >= 2 else ""
    return language, fmt


def _matches_filter(value: str, allowed: List[str]) -> bool:
    if not allowed:
        return True
    return str(value or "").strip().lower() in {x.strip().lower() for x in allowed if x.strip()}


def _write_base_oto(
    source_oto: str,
    out_path: str,
    *,
    zero_timing: bool,
    special_stems: set[str],
) -> int:
    rows = _read_oto_lines(source_oto)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        for line in rows:
            if "=" not in line:
                continue
            left, right = line.split("=", 1)
            if _is_special_wav_name(left, special_stems):
                continue
            alias = right.split(",", 1)[0].strip()
            if not alias:
                continue
            if zero_timing:
                handle.write(f"{left.strip()}={alias},0,0,0,0,0\n")
            else:
                handle.write(line.rstrip() + "\n")
            written += 1
    return written


def _write_batch_yaml(path: str, cases: List[Dict[str, object]], *, aligner: str, fallback_aligner: str) -> None:
    lines = [
        "defaults:",
        "  enabled: true",
        "  align_if_missing: true",
        f"  primary_aligner: {_quote_yaml(aligner)}",
        f"  fallback_aligner: {_quote_yaml(fallback_aligner)}",
        "  replace_oto_ini: false",
        "  generate_openutau: false",
        "  ml_policy: \"off\"",
        "cases:",
    ]
    for case in cases:
        lines.extend(
            [
                f"  - name: {_quote_yaml(case['name'])}",
                f"    language: {_quote_yaml(case['language'])}",
                f"    auto_format: {_quote_yaml(case['format'])}",
                f"    voicebank_dir: {_quote_yaml(case['voicebank_dir'])}",
                f"    textgrid_dir: {_quote_yaml(case['textgrid_dir'])}",
                f"    out_dir: {_quote_yaml(case['out_dir'])}",
                f"    output_name: {_quote_yaml(case['output_name'])}",
                f"    base_oto: {_quote_yaml(case['base_oto'])}",
                "    no_base_oto: false",
                "    enabled: true",
            ]
        )
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build batch-generation cases from dataset_staged manual oto.ini files."
    )
    parser.add_argument("--dataset-root", default=os.path.join(ROOT_DIR, "dataset_staged"))
    parser.add_argument("--out-dir", default=os.path.join(ROOT_DIR, "eval_runs", "dataset_oto_eval"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--language", action="append", default=[], help="Filter language, repeatable.")
    parser.add_argument("--format", action="append", default=[], help="Filter format, repeatable.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--aligner", default="sequence")
    parser.add_argument("--fallback-aligner", default="")
    parser.add_argument(
        "--include-special-files",
        action="store_true",
        help="Do not filter non-phonated/special WAV files such as _breath.wav during evaluation.",
    )
    parser.add_argument(
        "--skip-wav-name",
        action="append",
        default=[],
        help="Additional exact WAV stem/name to skip during evaluation. Repeatable.",
    )
    parser.add_argument(
        "--base-mode",
        choices=("zero", "manual"),
        default="zero",
        help="zero avoids leaking manual timing into generated OTO; manual uses source oto.ini directly.",
    )
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    if not os.path.isdir(dataset_root):
        raise SystemExit(f"dataset root not found: {dataset_root}")

    run_name = _safe_name(args.run_name or f"{args.aligner}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_root = os.path.abspath(os.path.join(args.out_dir, run_name))
    base_root = os.path.join(out_root, "base_oto")
    eval_voicebank_root = os.path.join(out_root, "eval_voicebanks")
    generated_root = os.path.join(out_root, "generated")
    textgrid_root = os.path.join(out_root, "textgrids")
    os.makedirs(out_root, exist_ok=True)
    special_stems = set()
    if not args.include_special_files:
        special_stems.update(_DEFAULT_SPECIAL_WAV_STEMS)
    special_stems.update(_wav_stem(name) for name in args.skip_wav_name if _wav_stem(name))

    cases: List[Dict[str, object]] = []
    seen_names: Dict[str, int] = {}
    for oto_path in _iter_oto_paths(dataset_root):
        language, fmt = _infer_lang_format(dataset_root, oto_path)
        if not _matches_filter(language, args.language):
            continue
        if not _matches_filter(fmt, args.format):
            continue
        voicebank_dir = os.path.dirname(oto_path)
        if not _has_wav(voicebank_dir):
            continue
        rel_dir = os.path.relpath(voicebank_dir, dataset_root)
        base_name = _safe_name(rel_dir)
        count = seen_names.get(base_name, 0)
        seen_names[base_name] = count + 1
        case_id = base_name if count == 0 else f"{base_name}_{count:02d}"
        eval_voicebank = _prepare_eval_voicebank(
            voicebank_dir,
            os.path.join(eval_voicebank_root, case_id),
            special_stems,
        )
        eval_voicebank_dir = str(eval_voicebank["voicebank_dir"])
        if int(eval_voicebank.get("copied_wavs", 0) or 0) <= 0:
            continue
        if args.base_mode == "zero":
            base_oto = os.path.join(base_root, case_id + ".ini")
            row_count = _write_base_oto(oto_path, base_oto, zero_timing=True, special_stems=special_stems)
        else:
            base_oto = os.path.join(base_root, case_id + ".ini")
            row_count = _write_base_oto(oto_path, base_oto, zero_timing=False, special_stems=special_stems)
        if row_count <= 0:
            continue
        case_out_dir = os.path.join(generated_root, case_id)
        case = {
            "id": case_id,
            "name": case_id,
            "language": language,
            "format": fmt,
            "source_voicebank_dir": os.path.abspath(voicebank_dir),
            "voicebank_dir": os.path.abspath(eval_voicebank_dir),
            "manual_oto": os.path.abspath(oto_path),
            "base_oto": os.path.abspath(base_oto),
            "textgrid_dir": os.path.join(textgrid_root, case_id),
            "out_dir": case_out_dir,
            "output_name": f"oto.{args.aligner}.ini",
            "candidate_oto": os.path.join(case_out_dir, f"oto.{args.aligner}.ini"),
            "manual_rows": row_count,
            "skipped_wavs": eval_voicebank.get("skipped_wavs", []),
        }
        cases.append(case)
        if args.limit and len(cases) >= int(args.limit):
            break

    if not cases:
        raise SystemExit("no dataset cases matched the requested filters")

    yaml_path = os.path.join(out_root, f"batch_{_safe_name(args.aligner)}.yaml")
    manifest_path = os.path.join(out_root, "manifest.json")
    _write_batch_yaml(yaml_path, cases, aligner=args.aligner, fallback_aligner=args.fallback_aligner)
    manifest = {
        "dataset_root": dataset_root,
        "run_name": run_name,
        "aligner": args.aligner,
        "fallback_aligner": args.fallback_aligner,
        "base_mode": args.base_mode,
        "batch_config": yaml_path,
        "case_count": len(cases),
        "cases": cases,
    }
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"[DONE] cases={len(cases)}")
    print(f"[DONE] batch_config={yaml_path}")
    print(f"[DONE] manifest={manifest_path}")
    print(f"[NEXT] python scripts/run_oto_generation_batch.py --config {yaml_path}")
    print(f"[NEXT] python scripts/score_oto_eval.py --manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
