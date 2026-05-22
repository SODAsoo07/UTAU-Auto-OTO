from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.generation.common.oto_file_utils import parse_oto_line, read_text_with_fallback
from core.model_context.filename import parse_filename_context
from core.model_context.manual_oto_anchor import (
    classify_manual_oto_alias_role,
    manual_oto_params_to_anchor_record,
)
from core.model_context.oto_params import wav_duration_ms


SCHEMA_VERSION = "manual_oto_anchor_manifest_v1"
DEFAULT_DATASET_ROOT = (
    r"C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\dataset_staged"
)
KNOWN_LANGUAGES = frozenset({"japanese", "korean", "english", "ja", "jp", "ko", "kr", "en"})
KNOWN_FORMATS = frozenset({"cv", "cvvc", "vcv", "cvc", "cmpx", "vccv", "cv-vc"})


def _norm_format(value: object) -> str:
    text = str(value or "").strip().lower()
    return {"cv": "cv", "cvvc": "cvvc", "vcv": "vcv", "cvc": "cvc", "cmpx": "cmpx"}.get(text, text)


def _infer_layout(oto_path: Path, dataset_root: Path) -> tuple[str, str, str]:
    try:
        rel_parts = oto_path.parent.relative_to(dataset_root).parts
    except ValueError:
        rel_parts = oto_path.parent.parts

    root_name = dataset_root.name.lower()
    parent_name = dataset_root.parent.name.lower()
    language = ""
    format_type = ""
    voicebank_parts: tuple[str, ...] = ()

    if root_name in {"alignmented", "aligned", "alignment"}:
        language = parent_name if parent_name in KNOWN_LANGUAGES else ""
        if rel_parts:
            format_type = _norm_format(rel_parts[0])
            voicebank_parts = rel_parts[1:]
    elif root_name in KNOWN_FORMATS:
        format_type = _norm_format(root_name)
        language = parent_name if parent_name in KNOWN_LANGUAGES else ""
        voicebank_parts = rel_parts
    elif rel_parts and rel_parts[0].lower() in KNOWN_LANGUAGES:
        language = rel_parts[0].lower()
        if len(rel_parts) >= 3 and rel_parts[1].lower() in {"alignmented", "aligned", "alignment"}:
            format_type = _norm_format(rel_parts[2])
            voicebank_parts = rel_parts[3:]
        elif len(rel_parts) >= 2:
            format_type = _norm_format(rel_parts[1])
            voicebank_parts = rel_parts[2:]
    elif rel_parts and rel_parts[0].lower() in KNOWN_FORMATS:
        language = parent_name if parent_name in KNOWN_LANGUAGES else ""
        format_type = _norm_format(rel_parts[0])
        voicebank_parts = rel_parts[1:]
    elif len(rel_parts) >= 3 and rel_parts[1].lower() in {"alignmented", "aligned", "alignment"}:
        language = rel_parts[0].lower()
        format_type = _norm_format(rel_parts[2])
        voicebank_parts = rel_parts[3:]
    elif len(rel_parts) >= 2:
        language = rel_parts[0].lower()
        format_type = _norm_format(rel_parts[1])
        voicebank_parts = rel_parts[2:]
    else:
        voicebank_parts = rel_parts
    voicebank_id = "/".join(voicebank_parts) if voicebank_parts else oto_path.parent.name
    return language, format_type, voicebank_id


def _iter_oto_paths(dataset_root: Path) -> Iterable[Path]:
    yield from sorted(path for path in dataset_root.rglob("oto.ini") if path.is_file())


def _row_params(parsed: dict[str, object]) -> dict[str, float]:
    return {
        "offset": float(parsed.get("offset", 0.0) or 0.0),
        "consonant": float(parsed.get("cons", parsed.get("consonant", 0.0)) or 0.0),
        "cutoff": float(parsed.get("cutoff", 0.0) or 0.0),
        "preutterance": float(parsed.get("pre", parsed.get("preutterance", 0.0)) or 0.0),
        "overlap": float(parsed.get("ovl", parsed.get("overlap", 0.0)) or 0.0),
    }


def _manual_row_to_manifest(
    *,
    parsed: dict[str, object],
    oto_path: Path,
    wav_path: Path,
    duration_ms: float,
    language: str,
    format_type: str,
    voicebank_id: str,
    slot_index: int,
    slot_count: int,
    prev_alias: str,
    next_alias: str,
    line_index: int,
) -> dict[str, object]:
    wav_name = str(parsed.get("wav", "") or "").strip()
    alias = str(parsed.get("alias", "") or "").strip()
    role = classify_manual_oto_alias_role(alias, language=language, format_type=format_type)
    params = _row_params(parsed)
    anchor = manual_oto_params_to_anchor_record(params, duration_ms=duration_ms)
    filename = parse_filename_context(wav_name, language=language, format_type=format_type)
    return {
        "schema_version": SCHEMA_VERSION,
        "label_source": "manual_oto_anchor",
        "label_quality": anchor.label_quality,
        "usable_as_target": anchor.usable_as_target,
        "language": language,
        "format_type": format_type,
        "voicebank_id": voicebank_id,
        "source_oto_path": str(oto_path),
        "oto_line_index": int(line_index),
        "wav_path": str(wav_path),
        "wav_name": wav_name,
        "duration_ms": float(duration_ms),
        "alias": alias,
        "alias_norm": alias.strip(),
        "alias_role": role,
        "alias_type": role,
        "prev_alias": prev_alias,
        "next_alias": next_alias,
        "slot_index": int(slot_index),
        "slot_count": int(slot_count),
        "slot_pos_norm": float(slot_index / max(1, slot_count - 1)) if slot_count > 1 else 0.0,
        "filename_tokens": list(filename.tokens),
        "filename_canonical_tokens": list(filename.canonical_tokens),
        "manual_oto": anchor.raw_params,
        "anchors_abs_ms": anchor.anchors_abs_ms,
        "anchors_repaired_abs_ms": anchor.repaired_anchors_abs_ms,
        "warnings": list(anchor.warnings),
    }


def build_manual_oto_anchor_manifest(
    *,
    dataset_root: Path,
    out_path: Path,
    max_rows: int | None = None,
    max_otos: int | None = None,
) -> dict[str, object]:
    dataset_root = dataset_root.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict[str, object]] = []
    stats: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    format_role_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    by_bank: dict[str, Counter[str]] = defaultdict(Counter)

    oto_paths = list(_iter_oto_paths(dataset_root))
    if max_otos is not None:
        oto_paths = oto_paths[: max(0, int(max_otos))]
    for oto_path in oto_paths:
        stats["oto_files_seen"] += 1
        language, format_type, voicebank_id = _infer_layout(oto_path, dataset_root)
        parsed_rows: list[tuple[int, dict[str, object]]] = []
        text = read_text_with_fallback(str(oto_path))
        for line_index, raw in enumerate(text.splitlines()):
            parsed = parse_oto_line(raw)
            if not parsed:
                stats["unparseable_or_empty_lines"] += 1
                continue
            wav_name = str(parsed.get("wav", "") or "").strip()
            alias = str(parsed.get("alias", "") or "").strip()
            if not wav_name or not alias:
                stats["blank_wav_or_alias"] += 1
                continue
            parsed_rows.append((line_index, parsed))

        by_wav: dict[str, list[tuple[int, dict[str, object]]]] = defaultdict(list)
        for line_index, parsed in parsed_rows:
            by_wav[str(parsed.get("wav", "") or "")].append((line_index, parsed))

        for wav_name, wav_rows in by_wav.items():
            wav_path = oto_path.parent / wav_name
            if not wav_path.is_file():
                stats["missing_wav_rows"] += len(wav_rows)
                by_bank[voicebank_id]["missing_wav_rows"] += len(wav_rows)
                continue
            duration = wav_duration_ms(str(wav_path), missing_value=0.0)
            slot_count = len(wav_rows)
            for idx, (line_index, parsed) in enumerate(wav_rows):
                prev_alias = str(wav_rows[idx - 1][1].get("alias", "") or "") if idx > 0 else ""
                next_alias = str(wav_rows[idx + 1][1].get("alias", "") or "") if idx + 1 < slot_count else ""
                row = _manual_row_to_manifest(
                    parsed=parsed,
                    oto_path=oto_path,
                    wav_path=wav_path,
                    duration_ms=duration,
                    language=language,
                    format_type=format_type,
                    voicebank_id=voicebank_id,
                    slot_index=idx,
                    slot_count=slot_count,
                    prev_alias=prev_alias,
                    next_alias=next_alias,
                    line_index=line_index,
                )
                rows_out.append(row)
                stats["rows_written"] += 1
                role = str(row["alias_role"])
                role_counts[role] += 1
                format_role_counts[f"{language}|{format_type}|{role}"] += 1
                quality_counts[str(row["label_quality"])] += 1
                by_bank[voicebank_id]["rows"] += 1
                by_bank[voicebank_id][f"quality:{row['label_quality']}"] += 1
                for flag in row.get("warnings", []):
                    warning_counts[str(flag)] += 1
                if max_rows is not None and len(rows_out) >= max_rows:
                    break
            if max_rows is not None and len(rows_out) >= max_rows:
                break
        if max_rows is not None and len(rows_out) >= max_rows:
            break

    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows_out:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "out_path": str(out_path),
        "stats": dict(stats),
        "role_counts": dict(role_counts),
        "format_role_counts": dict(format_role_counts),
        "quality_counts": dict(quality_counts),
        "warning_counts": dict(warning_counts),
        "top_banks": {
            key: dict(value)
            for key, value in sorted(by_bank.items(), key=lambda item: item[1].get("rows", 0), reverse=True)[:50]
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a manifest that treats manual oto.ini timings as OTO anchor supervision."
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--out",
        default=r"ml_workspace\manual_oto_anchor\manual_oto_anchor_manifest.jsonl",
    )
    parser.add_argument(
        "--summary-out",
        default=r"ml_workspace\manual_oto_anchor\manual_oto_anchor_summary.json",
    )
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-otos", type=int)
    args = parser.parse_args()

    summary = build_manual_oto_anchor_manifest(
        dataset_root=Path(args.dataset_root),
        out_path=Path(args.out),
        max_rows=args.max_rows,
        max_otos=args.max_otos,
    )
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
