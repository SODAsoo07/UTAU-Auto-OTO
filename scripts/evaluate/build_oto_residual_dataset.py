"""Build a trainable OTO residual dataset from gold-vs-generated OTO pairs.

This script is intentionally evaluation/training tooling. Runtime HSMM
generation must not depend on it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.oto_ml_features import TARGET_NAMES, write_dataset_csv
from core.oto_normalization import canonicalize_alias_for_matching, normalize_wav_key
from scripts.dev.diff_oto import build_report
from scripts.evaluate.stratified_oto_residuals import row_role_tags

PARAM_TO_TARGET = {
    "offset": "delta_offset",
    "consonant": "delta_cons",
    "cutoff_end_abs": "delta_cutoff",
    "preutterance": "delta_pre",
    "overlap": "delta_ovl",
}
TARGET_TO_PARAM = {target: param for param, target in PARAM_TO_TARGET.items()}

_ZERO_TIMING_EPS_MS = 1e-3
_KOREAN_LARGE_RESIDUAL_DOWNWEIGHT_MS = 500.0
_KOREAN_CATASTROPHIC_RESIDUAL_SKIP_MS = 1200.0


def _read_json(path: str) -> object:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _write_plain_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _stable_id(*parts: object, length: int = 16) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:length]


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _case_list_from_payload(payload: object) -> list[dict]:
    if isinstance(payload, dict):
        if isinstance(payload.get("cases"), list):
            return [dict(item) for item in payload["cases"] if isinstance(item, dict)]
        return [dict(payload)]
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    raise ValueError("cases JSON must be a dict, a list, or {'cases': [...]}")


def load_cases(*, cases_json: str = "", single_case: dict | None = None) -> list[dict]:
    cases: list[dict] = []
    if cases_json:
        cases.extend(_case_list_from_payload(_read_json(cases_json)))
    if single_case:
        gold = str(single_case.get("gold") or "").strip()
        pred = str(single_case.get("pred") or "").strip()
        if gold or pred:
            cases.append(dict(single_case))
    normalized: list[dict] = []
    for idx, case in enumerate(cases):
        gold = os.path.abspath(str(case.get("gold") or case.get("manual") or ""))
        pred = os.path.abspath(str(case.get("pred") or case.get("generated") or ""))
        if not gold or not os.path.isfile(gold):
            raise FileNotFoundError(f"gold oto.ini not found for case {idx}: {gold}")
        if not pred or not os.path.isfile(pred):
            raise FileNotFoundError(f"pred oto.ini not found for case {idx}: {pred}")
        wav_dir_raw = str(case.get("wav_dir") or case.get("wav-dir") or "").strip()
        wav_dir = os.path.abspath(wav_dir_raw) if wav_dir_raw else os.path.dirname(gold)
        case_id = str(case.get("case_id") or case.get("id") or Path(gold).parent.name or f"case_{idx:03d}")
        language = str(case.get("language") or case.get("lang") or "").strip().lower()
        format_type = str(case.get("format_type") or case.get("format") or "").strip().lower()
        normalized.append(
            {
                "case_id": case_id,
                "language": language,
                "format_type": format_type,
                "gold": gold,
                "pred": pred,
                "wav_dir": wav_dir,
            }
        )
    if not normalized:
        raise ValueError("no cases were provided")
    return normalized


def _cutoff_abs(row: dict, side: str) -> tuple[float, str]:
    info = row.get("cutoff_end") or {}
    key = f"{side}_cutoff_end_abs"
    mode_key = f"{side}_cutoff_end_mode"
    if key in info:
        return _float(info.get(key)), str(info.get(mode_key) or "")
    source = row.get(side) or {}
    offset = _float(source.get("offset"))
    cutoff = _float(source.get("cutoff"))
    return offset + abs(cutoff), "offset_relative_fallback"


def _family_counts(rows: list[dict]) -> dict[str, Counter]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        wav = str(row.get("wav") or "")
        family = str(row.get("alias_family") or "")
        counts[wav]["rows"] += 1
        counts[wav][family] += 1
    return counts


def _ordered_rows_for_case(per_row: list[dict]) -> list[dict]:
    indexed = [(idx, row) for idx, row in enumerate(per_row)]
    return [
        row
        for _idx, row in sorted(
            indexed,
            key=lambda item: (
                str(item[1].get("wav") or ""),
                _float((item[1].get("pred") or {}).get("offset")),
                str(item[1].get("alias") or ""),
                int(item[1].get("occurrence") or 0),
                item[0],
            ),
        )
    ]


def _neighbor_family(rows: list[dict], index: int, direction: int) -> str:
    pos = index + direction
    if pos < 0 or pos >= len(rows):
        return ""
    return str(rows[pos].get("alias_family") or "")


def _sample_weight(tags: dict[str, str]) -> float:
    family = str(tags.get("alias_family") or "")
    right_class = str(tags.get("vc_right_class") or "")
    weight = 1.0
    if family in {"vc", "vv", "vcv"}:
        weight *= 1.15
    if right_class in {"nasal", "liquid", "glide", "yoon"}:
        weight *= 1.15
    return round(weight, 4)


def _near_zero(value: object, *, eps: float = _ZERO_TIMING_EPS_MS) -> bool:
    return abs(_float(value)) <= float(eps)


def _manual_timing_has_zero_anchor_profile(gold: dict) -> bool:
    return (
        _near_zero(gold.get("offset"))
        and _near_zero(gold.get("consonant"))
        and _near_zero(gold.get("preutterance"))
        and _near_zero(gold.get("overlap"))
    )


def _row_training_quality(
    *,
    language: str,
    family: str,
    gold: dict,
    deltas: dict[str, float],
    row_index: int,
) -> dict[str, object]:
    """Return conservative default training quality for gold-vs-generated rows.

    This does not delete rows from the diagnostic dataset. It only marks rows
    that should not become LightGBM supervision by default, because a generated
    or partially blank source OTO can otherwise teach one-step shifts as truth.
    """
    lang = str(language or "").strip().lower()
    family_norm = str(family or "").strip().lower()
    abs_offset = abs(float(deltas.get("delta_offset", 0.0)))
    keep = 1
    reason = ""
    quality = 100.0

    if family_norm != "policy_breath" and _manual_timing_has_zero_anchor_profile(gold):
        keep = 0
        reason = "manual_zero_anchor_profile"
        quality = 0.0
    elif (
        lang in {"korean", "ko", "kr"}
        and family_norm in {"cv", "cv_head", "v", "vv", "vc", "vcv"}
        and row_index > 0
        and _near_zero(gold.get("offset"))
    ):
        keep = 0
        reason = "manual_zero_offset_noninitial_sequence_row"
        quality = 0.0
    elif lang in {"korean", "ko", "kr"} and abs_offset >= _KOREAN_CATASTROPHIC_RESIDUAL_SKIP_MS:
        keep = 0
        reason = "korean_catastrophic_offset_residual"
        quality = 0.0
    elif lang in {"korean", "ko", "kr"} and abs_offset >= _KOREAN_LARGE_RESIDUAL_DOWNWEIGHT_MS:
        quality = 45.0

    return {
        "train_keep_default": int(keep),
        "train_skip_reason": reason,
        "train_quality_score": float(quality),
        "skipped_reason": reason,
    }


def build_residual_rows(
    cases: list[dict],
    *,
    cutoff_end_mode: str = "compatible",
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    case_summaries: list[dict] = []
    for case in cases:
        report, per_row = build_report(
            gold_path=str(case["gold"]),
            pred_path=str(case["pred"]),
            wav_dir=str(case["wav_dir"]),
            cutoff_end_mode=cutoff_end_mode,
        )
        ordered = _ordered_rows_for_case(per_row)
        by_wav: dict[str, list[dict]] = defaultdict(list)
        for row in ordered:
            by_wav[str(row.get("wav") or "")].append(row)
        wav_counts = _family_counts(ordered)
        source_oto_id = _stable_id(case["gold"], case["pred"])
        case_summaries.append(
            {
                "case_id": case["case_id"],
                "language": case["language"],
                "format_type": case["format_type"],
                "gold": case["gold"],
                "pred": case["pred"],
                "wav_dir": case["wav_dir"],
                "rows_matched": int(report.get("rows_matched", 0)),
                "only_in_gold": int(report.get("only_in_gold", 0)),
                "only_in_pred": int(report.get("only_in_pred", 0)),
            }
        )
        line_index = 0
        for wav, wav_rows in sorted(by_wav.items()):
            file_rows = len(wav_rows)
            for row_index, row in enumerate(wav_rows):
                alias = str(row.get("alias") or "")
                family = str(row.get("alias_family") or "")
                tags = row_role_tags(row)
                gold = dict(row.get("gold") or {})
                pred = dict(row.get("pred") or {})
                gold_cutoff_abs, gold_cutoff_mode = _cutoff_abs(row, "gold")
                pred_cutoff_abs, pred_cutoff_mode = _cutoff_abs(row, "pred")
                delta_cutoff = gold_cutoff_abs - pred_cutoff_abs
                deltas = {
                    "delta_offset": _float(gold.get("offset")) - _float(pred.get("offset")),
                    "delta_cons": _float(gold.get("consonant")) - _float(pred.get("consonant")),
                    "delta_cutoff": delta_cutoff,
                    "delta_pre": _float(gold.get("preutterance")) - _float(pred.get("preutterance")),
                    "delta_ovl": _float(gold.get("overlap")) - _float(pred.get("overlap")),
                }
                quality = _row_training_quality(
                    language=str(case["language"]),
                    family=family,
                    gold=gold,
                    deltas=deltas,
                    row_index=row_index,
                )
                sample_weight = round(
                    _sample_weight(tags) * max(0.0, _float(quality.get("train_quality_score"), 100.0)) / 100.0,
                    4,
                )
                row_uid = _stable_id(
                    case["case_id"],
                    wav,
                    alias,
                    int(row.get("occurrence") or 0),
                    line_index,
                    source_oto_id,
                    length=24,
                )
                counts = wav_counts[wav]
                target_abs = {f"abs_{name}": abs(value) for name, value in deltas.items()}
                payload = {
                    "row_uid": row_uid,
                    "case_id": case["case_id"],
                    "voicebank_id": case["case_id"],
                    "language": case["language"],
                    "format_type": case["format_type"],
                    "wav": wav,
                    "alias": alias,
                    "wav_norm": normalize_wav_key(wav),
                    "alias_norm": canonicalize_alias_for_matching(str(case["language"]), alias),
                    "occurrence_index": int(row.get("occurrence") or 0),
                    "line_index": line_index,
                    "source_oto_id": source_oto_id,
                    "source_row_id": row_uid,
                    "group_id": f"{case['case_id']}::{normalize_wav_key(wav)}",
                    "case_wav": f"{case['case_id']}::{wav}",
                    "alias_type": family,
                    "alias_family": family,
                    "alias_group": str(tags.get("alias_family_x_vc_right_class") or family),
                    "vc_left_token": str(tags.get("vc_left_token") or ""),
                    "vc_right_token": str(tags.get("vc_right_token") or ""),
                    "vc_right_class": str(tags.get("vc_right_class") or ""),
                    "vc_pair": str(tags.get("vc_pair") or ""),
                    "row_index_in_wav": row_index,
                    "row_ratio_in_wav": (row_index / max(1, file_rows - 1)) if file_rows > 1 else 0.0,
                    "file_row_count": file_rows,
                    "file_cv_count": int(counts.get("cv", 0)) + int(counts.get("cv_head", 0)),
                    "file_vc_count": int(counts.get("vc", 0)) + int(counts.get("terminal_v_dash", 0)),
                    "file_vv_count": int(counts.get("vv", 0)),
                    "file_vcv_count": int(counts.get("vcv", 0)),
                    "file_br_count": int(counts.get("br", 0)),
                    "file_mono_count": int(counts.get("v", 0)),
                    "file_cv_ratio": (int(counts.get("cv", 0)) + int(counts.get("cv_head", 0))) / max(1, file_rows),
                    "file_vc_ratio": (int(counts.get("vc", 0)) + int(counts.get("terminal_v_dash", 0))) / max(1, file_rows),
                    "file_vc_cv_ratio": (int(counts.get("vc", 0)) + int(counts.get("terminal_v_dash", 0))) / max(
                        1,
                        int(counts.get("cv", 0)) + int(counts.get("cv_head", 0)),
                    ),
                    "file_cv_vc_balance": (
                        (int(counts.get("cv", 0)) + int(counts.get("cv_head", 0)))
                        - (int(counts.get("vc", 0)) + int(counts.get("terminal_v_dash", 0)))
                    )
                    / max(1, file_rows),
                    "is_head_row": 1.0 if family == "cv_head" or alias.strip().startswith("-") else 0.0,
                    "is_tail_row": 1.0 if family == "terminal_v_dash" or alias.strip().endswith("-") else 0.0,
                    "wav_duration_ms": _float(row.get("duration_ms")),
                    "base_offset": _float(pred.get("offset")),
                    "base_cons": _float(pred.get("consonant")),
                    "base_cutoff_abs": pred_cutoff_abs,
                    "base_cutoff_raw": _float(pred.get("cutoff")),
                    "base_cutoff_mode": pred_cutoff_mode,
                    "base_pre": _float(pred.get("preutterance")),
                    "base_ovl": _float(pred.get("overlap")),
                    "base_cons_gap": _float(pred.get("consonant")) - _float(pred.get("preutterance")),
                    "base_cut_gap": pred_cutoff_abs - _float(pred.get("offset")),
                    "base_ovl_ratio": _float(pred.get("overlap")) / max(1.0, _float(pred.get("preutterance"))),
                    "expected_anchor_ms": _float(pred.get("offset")) + _float(pred.get("preutterance")),
                    "base_offset_to_expected_ms": -_float(pred.get("preutterance")),
                    "base_pre_to_expected_ms": 0.0,
                    "base_cutoff_to_next_anchor_ms": 0.0,
                    "onset_class": str(tags.get("vc_right_class") or ""),
                    "voicing_class": "",
                    "coda_type": str(tags.get("vc_right_token") or ""),
                    "vowel_class": str(tags.get("vc_left_token") or ""),
                    "mora_position": family,
                    "bridge_type": str(tags.get("vc_pair") or ""),
                    "is_nasal_or_sonorant": 1.0
                    if str(tags.get("vc_right_class") or "") in {"nasal", "liquid", "glide"}
                    else 0.0,
                    "prev_alias_type": _neighbor_family(wav_rows, row_index, -1),
                    "next_alias_type": _neighbor_family(wav_rows, row_index, 1),
                    "manual_offset": _float(gold.get("offset")),
                    "manual_cons": _float(gold.get("consonant")),
                    "manual_cutoff": _float(gold.get("cutoff")),
                    "manual_cutoff_abs": gold_cutoff_abs,
                    "manual_cutoff_mode": gold_cutoff_mode,
                    "manual_pre": _float(gold.get("preutterance")),
                    "manual_ovl": _float(gold.get("overlap")),
                    "label_source": "gold_oto_residual",
                    "sample_weight": sample_weight,
                    "blank_risk_score": 0.0,
                    "blank_risk_flag": 0,
                    "train_keep_default": int(quality["train_keep_default"]),
                    "train_skip_reason": str(quality["train_skip_reason"]),
                    "train_quality_score": float(quality["train_quality_score"]),
                    "skipped_reason": str(quality["skipped_reason"]),
                    "aux_vowel_start_rel": 0.0,
                    "aux_vowel_end_rel": 0.0,
                    "aux_next_onset_rel": 0.0,
                    "delta_cutoff_end_abs": delta_cutoff,
                }
                payload.update(deltas)
                payload.update(target_abs)
                rows.append(payload)
                line_index += 1
    summary = {
        "cases": case_summaries,
        "rows_total": len(rows),
        "cutoff_end_mode": cutoff_end_mode,
        "target_names": list(TARGET_NAMES),
        "by_alias_family": dict(sorted(Counter(row["alias_family"] for row in rows).items())),
        "by_vc_right_class": dict(sorted(Counter(row["vc_right_class"] for row in rows if row["vc_right_class"]).items())),
        "train_keep_default_rows": sum(int(row.get("train_keep_default", 0) or 0) for row in rows),
        "train_skip_default_rows": sum(1 for row in rows if int(row.get("train_keep_default", 0) or 0) <= 0),
        "by_train_skip_reason": dict(
            sorted(Counter(str(row.get("train_skip_reason") or "") for row in rows if row.get("train_skip_reason")).items())
        ),
        "target_mae_ms": {
            target: round(sum(abs(_float(row.get(target))) for row in rows) / max(1, len(rows)), 3)
            for target in TARGET_NAMES
        },
    }
    return rows, summary


def split_rows(
    rows: list[dict],
    *,
    heldout_ratio: float = 0.2,
    seed: int = 42,
    group_column: str = "group_id",
) -> tuple[list[dict], list[dict], dict]:
    if not rows:
        return [], [], {"group_column": group_column, "train_groups": 0, "heldout_groups": 0}
    groups = sorted({str(row.get(group_column) or row.get("group_id") or row.get("row_uid")) for row in rows})
    shuffled = list(groups)
    random.Random(int(seed)).shuffle(shuffled)
    if len(shuffled) <= 1:
        heldout_groups: set[str] = set()
    else:
        heldout_n = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * float(heldout_ratio)))))
        heldout_groups = set(shuffled[:heldout_n])
    train: list[dict] = []
    heldout: list[dict] = []
    for row in rows:
        group = str(row.get(group_column) or row.get("group_id") or row.get("row_uid"))
        target = heldout if group in heldout_groups else train
        out = dict(row)
        out["split"] = "heldout" if target is heldout else "train"
        target.append(out)
    split_summary = {
        "group_column": group_column,
        "heldout_ratio": float(heldout_ratio),
        "seed": int(seed),
        "groups_total": len(groups),
        "train_groups": len(set(groups) - heldout_groups),
        "heldout_groups": len(heldout_groups),
        "train_rows": len(train),
        "heldout_rows": len(heldout),
        "train_by_alias_family": dict(sorted(Counter(row["alias_family"] for row in train).items())),
        "heldout_by_alias_family": dict(sorted(Counter(row["alias_family"] for row in heldout).items())),
    }
    return train, heldout, split_summary


def write_outputs(out_dir: str, rows: list[dict], train: list[dict], heldout: list[dict], summary: dict) -> dict:
    out_dir_abs = os.path.abspath(out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)
    paths = {
        "dataset_jsonl": os.path.join(out_dir_abs, "residual_dataset.jsonl"),
        "train_jsonl": os.path.join(out_dir_abs, "train.jsonl"),
        "heldout_jsonl": os.path.join(out_dir_abs, "heldout.jsonl"),
        "dataset_csv": os.path.join(out_dir_abs, "residual_dataset.csv"),
        "train_csv": os.path.join(out_dir_abs, "train.csv"),
        "heldout_csv": os.path.join(out_dir_abs, "heldout.csv"),
        "debug_csv": os.path.join(out_dir_abs, "residual_dataset.debug.csv"),
        "summary": os.path.join(out_dir_abs, "summary.json"),
    }
    _write_jsonl(paths["dataset_jsonl"], rows)
    _write_jsonl(paths["train_jsonl"], train)
    _write_jsonl(paths["heldout_jsonl"], heldout)
    write_dataset_csv(paths["dataset_csv"], rows, append=False)
    write_dataset_csv(paths["train_csv"], train, append=False)
    write_dataset_csv(paths["heldout_csv"], heldout, append=False)
    _write_plain_csv(paths["debug_csv"], rows)
    summary_out = dict(summary)
    summary_out["outputs"] = paths
    _write_json(paths["summary"], summary_out)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-json", default="", help="JSON list/dict with case_id, language, format_type, gold, pred, wav_dir")
    parser.add_argument("--case-id", default="", help="Single-case id when not using --cases-json")
    parser.add_argument("--language", default="", help="Single-case language")
    parser.add_argument("--format-type", default="", help="Single-case format type")
    parser.add_argument("--gold", default="", help="Single-case manual/gold oto.ini")
    parser.add_argument("--pred", default="", help="Single-case generated oto.ini")
    parser.add_argument("--wav-dir", default="", help="Single-case WAV directory")
    parser.add_argument(
        "--cutoff-end-mode",
        choices=("raw", "relative", "utau", "compatible"),
        default="compatible",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--heldout-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-column", default="group_id")
    args = parser.parse_args()

    cases = load_cases(
        cases_json=args.cases_json,
        single_case={
            "case_id": args.case_id,
            "language": args.language,
            "format_type": args.format_type,
            "gold": args.gold,
            "pred": args.pred,
            "wav_dir": args.wav_dir,
        },
    )
    rows, summary = build_residual_rows(cases, cutoff_end_mode=args.cutoff_end_mode)
    train, heldout, split_summary = split_rows(
        rows,
        heldout_ratio=float(args.heldout_ratio),
        seed=int(args.seed),
        group_column=str(args.group_column or "group_id"),
    )
    summary["split"] = split_summary
    paths = write_outputs(args.out_dir, rows, train, heldout, summary)
    print(
        json.dumps(
            {
                "rows_total": len(rows),
                "train_rows": len(train),
                "heldout_rows": len(heldout),
                "out_dir": os.path.abspath(args.out_dir),
                "summary": paths["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
