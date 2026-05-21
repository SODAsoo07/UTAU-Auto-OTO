from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.ja_lab_generator import parse_ja_filename
from core.ja_oto_mapping import classify_ja_alias
from core.model_context.oto_params import safe_cutoff_abs_ms, wav_duration_ms
from core.oto_file_utils import parse_oto_line, read_text_with_fallback

PAUSE_LABELS = {
    "",
    "sp",
    "spn",
    "sil",
    "pau",
    "ap",
    "br",
    "bre",
    "breath",
    "r",
}
SMALL_JA = set("ぁぃぅぇぉゃゅょゎァィゥェォャュョヮっッ")


@dataclass(frozen=True)
class OtoRow:
    wav: str
    alias: str
    offset_ms: float
    cons_ms: float
    cutoff: float
    pre_ms: float
    ovl_ms: float
    alias_type: str


@dataclass(frozen=True)
class LabRow:
    start: int
    end: int
    label: str


def _iter_wavs(root: str) -> List[str]:
    out: List[str] = []
    for name in sorted(os.listdir(root)):
        if name.lower().endswith(".wav"):
            out.append(os.path.join(root, name))
    return out


def _discover_oto_targets(dataset_root: str) -> List[Tuple[str, str]]:
    root = os.path.abspath(str(dataset_root or "").strip())
    if not os.path.isdir(root):
        return []
    out: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for dp, _dns, fns in os.walk(root):
        if "oto.ini" not in fns:
            continue
        voicebank = os.path.abspath(dp)
        oto = os.path.join(voicebank, "oto.ini")
        key = (voicebank + "||" + oto).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append((voicebank, oto))
    out.sort(key=lambda x: x[0].lower())
    return out


def _sanitize_rel_key(path: str) -> str:
    text = str(path or "").strip().replace("\\", "/")
    text = re.sub(r"[^0-9A-Za-z_\-./]+", "_", text)
    text = text.strip("./")
    if not text:
        return "root"
    return text.replace("/", "__")


def _classify_alias_type(language: str, alias: str) -> str:
    lang = str(language or "").strip().lower()
    if lang == "japanese":
        try:
            return str(classify_ja_alias(alias)).strip().lower()
        except Exception:
            return ""
    text = str(alias or "").strip().lower()
    if not text:
        return ""
    if text.startswith("- "):
        return "cv_head"
    if " " in text:
        return "vc"
    return "cv"


def _read_oto_rows(oto_path: str, language: str) -> List[OtoRow]:
    text = read_text_with_fallback(oto_path)
    out: List[OtoRow] = []
    for raw in text.splitlines():
        row = parse_oto_line(raw)
        if not row:
            continue
        alias = str(row.get("alias", "") or "").strip()
        alias_type = _classify_alias_type(language, alias)
        out.append(
            OtoRow(
                wav=str(row.get("wav", "") or "").strip(),
                alias=alias,
                offset_ms=float(row.get("offset", 0.0) or 0.0),
                cons_ms=float(row.get("cons", 0.0) or 0.0),
                cutoff=float(row.get("cutoff", 0.0) or 0.0),
                pre_ms=float(row.get("pre", 0.0) or 0.0),
                ovl_ms=float(row.get("ovl", 0.0) or 0.0),
                alias_type=alias_type,
            )
        )
    return out


def _group_rows_by_wav(rows: Sequence[OtoRow]) -> Dict[str, List[OtoRow]]:
    out: Dict[str, List[OtoRow]] = defaultdict(list)
    for row in rows:
        key = str(row.wav or "").strip().lower()
        if key:
            out[key].append(row)
    for key, items in out.items():
        out[key] = sorted(items, key=lambda x: float(x.offset_ms))
    return out


def _normalize_label(label: str) -> str:
    return str(label or "").strip().lower()


def _is_pause(label: str) -> bool:
    return _normalize_label(label) in PAUSE_LABELS


def _infer_expected_word_count(base: str, language: str) -> int:
    stem = str(base or "").strip().lstrip("_")
    if not stem:
        return 0
    lang = str(language or "").strip().lower()
    if lang == "japanese":
        try:
            parsed = parse_ja_filename(stem) or []
        except Exception:
            parsed = []
        if parsed and int(len(parsed)) >= 2:
            return int(len(parsed))
        mora = 0
        for ch in stem:
            if ch in {"_", "-", " ", "|", "/", "."}:
                continue
            code = ord(ch)
            is_kana = (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF)
            if not is_kana:
                continue
            if ch in SMALL_JA:
                if mora <= 0:
                    mora = 1
                continue
            mora += 1
        if mora > 0:
            return int(mora)
    parts = [p for p in re.split(r"[_\s\-|/]+", stem) if p.strip()]
    return int(len(parts))


def _write_lab(path: str, rows: Sequence[LabRow]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{int(row.start)} {int(row.end)} {row.label}\n")


def _quality_report(rows: Sequence[LabRow], wav_end_100ns: int, expected_words: int) -> Dict[str, object]:
    non_pause = [r for r in rows if not _is_pause(r.label)]
    speech_100ns = sum(max(0, int(r.end) - int(r.start)) for r in non_pause)
    speech_ratio = float(speech_100ns) / float(max(1, wav_end_100ns))
    non_pause_count = int(len(non_pause))

    errors: List[str] = []
    if non_pause_count < 1:
        errors.append("too_few_non_pause_labels")
    if speech_ratio < 0.03:
        errors.append("speech_ratio_too_low")
    if speech_ratio > 0.999:
        errors.append("speech_ratio_too_high")

    first_non_pause = next((r for r in rows if not _is_pause(r.label)), None)
    last_non_pause = next((r for r in reversed(rows) if not _is_pause(r.label)), None)
    first_ratio = (
        float(first_non_pause.start) / float(max(1, wav_end_100ns))
        if first_non_pause is not None
        else 0.0
    )
    tail_pause_ratio = (
        float(wav_end_100ns - int(last_non_pause.end)) / float(max(1, wav_end_100ns))
        if last_non_pause is not None
        else 1.0
    )
    if first_ratio > 0.80:
        errors.append("late_first_non_pause")
    if tail_pause_ratio > 0.95:
        errors.append("long_tail_pause")

    return {
        "accepted": bool(not errors),
        "errors": errors,
        "non_pause_count": non_pause_count,
        "speech_ratio": float(speech_ratio),
        "first_non_pause_ratio": float(first_ratio),
        "tail_pause_ratio": float(tail_pause_ratio),
    }


def _write_json(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: str, rows: Iterable[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build sequence-align training manifests from OTO CV aliases.")
    parser.add_argument("--voicebank", default="", help="Voicebank root containing wav files and oto.ini")
    parser.add_argument("--dataset-root", default="", help="Root folder to recursively discover voicebank oto.ini files")
    parser.add_argument("--oto-path", default="", help="Optional explicit oto.ini path")
    parser.add_argument("--language", default="japanese", choices=["japanese", "korean"], help="Language hint")
    parser.add_argument("--out-root", default="", help="Output root folder")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    args = parser.parse_args()

    voicebank_raw = str(args.voicebank or "").strip()
    dataset_root_raw = str(args.dataset_root or "").strip()
    if bool(voicebank_raw) == bool(dataset_root_raw):
        raise SystemExit("use exactly one of --voicebank or --dataset-root")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if str(args.out_root or "").strip():
        out_root = os.path.abspath(str(args.out_root))
    else:
        auto_tag = os.path.basename(os.path.abspath(voicebank_raw)) if voicebank_raw else os.path.basename(os.path.abspath(dataset_root_raw))
        out_root = os.path.abspath(
            os.path.join(
                os.getcwd(),
                "dataset_workspace",
                "sequence_oto_cv_preprocessed",
                f"{auto_tag}_{stamp}",
            )
        )

    lab_dir = os.path.join(out_root, "labs")
    os.makedirs(lab_dir, exist_ok=True)

    targets: List[Tuple[str, str]] = []
    if voicebank_raw:
        voicebank = os.path.abspath(voicebank_raw)
        if not os.path.isdir(voicebank):
            raise SystemExit(f"voicebank not found: {voicebank}")
        oto_path = os.path.abspath(str(args.oto_path or "").strip()) if str(args.oto_path or "").strip() else os.path.join(voicebank, "oto.ini")
        if not os.path.isfile(oto_path):
            raise SystemExit(f"oto.ini not found: {oto_path}")
        targets.append((voicebank, oto_path))
        root_for_rel = os.path.dirname(voicebank)
    else:
        dataset_root = os.path.abspath(dataset_root_raw)
        if not os.path.isdir(dataset_root):
            raise SystemExit(f"dataset-root not found: {dataset_root}")
        targets = _discover_oto_targets(dataset_root)
        if not targets:
            raise SystemExit(f"no oto.ini discovered in: {dataset_root}")
        root_for_rel = dataset_root

    all_rows: List[Dict[str, object]] = []
    accepted_rows: List[Dict[str, object]] = []
    speech_ratios: List[float] = []
    cv_alias_counter = Counter()
    voicebank_stats: List[Dict[str, object]] = []

    for voicebank, oto_path in targets:
        oto_rows = _read_oto_rows(oto_path, args.language)
        rows_by_wav = _group_rows_by_wav(oto_rows)
        wav_paths = _iter_wavs(voicebank)
        rel_key = _sanitize_rel_key(os.path.relpath(voicebank, root_for_rel))
        vb_lab_dir = os.path.join(lab_dir, rel_key)
        os.makedirs(vb_lab_dir, exist_ok=True)

        vb_total = 0
        vb_accepted = 0
        for wav_path in wav_paths:
            vb_total += 1
            wav_name = os.path.basename(wav_path)
            wav_key = wav_name.strip().lower()
            source_base = os.path.splitext(wav_name)[0]
            base = f"{rel_key}__{source_base}"
            wav_duration_ms_value = wav_duration_ms(wav_path)
            wav_end_100ns = int(round(wav_duration_ms_value * 10_000.0))
            if wav_end_100ns <= 0:
                continue

            wav_oto_rows = rows_by_wav.get(wav_key, [])
            cv_rows = [r for r in wav_oto_rows if r.alias_type in {"cv", "cv_head"}]
            for r in cv_rows:
                cv_alias_counter[str(r.alias_type)] += 1

            quality_errors: List[str] = []
            if not cv_rows:
                quality_errors.append("no_cv_alias_rows")

            if cv_rows:
                start_ms = min(max(0.0, float(r.offset_ms)) for r in cv_rows)
                end_ms = max(
                    safe_cutoff_abs_ms(
                        float(r.offset_ms),
                        float(r.cutoff),
                        wav_duration_ms_value,
                        convention="legacy_candidate",
                    )
                    for r in cv_rows
                )
                start_ms = max(0.0, min(start_ms, wav_duration_ms_value))
                end_ms = max(start_ms + 1.0, min(end_ms, wav_duration_ms_value))
            else:
                start_ms = 0.0
                end_ms = wav_duration_ms_value

            start_100ns = int(round(start_ms * 10_000.0))
            end_100ns = int(round(end_ms * 10_000.0))
            start_100ns = max(0, min(start_100ns, wav_end_100ns))
            end_100ns = max(start_100ns + 1, min(end_100ns, wav_end_100ns))

            lab_rows: List[LabRow] = []
            if start_100ns > 0:
                lab_rows.append(LabRow(start=0, end=start_100ns, label="SP"))
            lab_rows.append(LabRow(start=start_100ns, end=end_100ns, label="CVSPAN"))
            if end_100ns < wav_end_100ns:
                lab_rows.append(LabRow(start=end_100ns, end=wav_end_100ns, label="SP"))

            expected_words = _infer_expected_word_count(source_base, args.language)
            quality = _quality_report(lab_rows, wav_end_100ns, expected_words)
            merged_errors = list(quality.get("errors", []) or []) + quality_errors
            accepted = bool(not merged_errors)

            clean_lab_path = os.path.join(vb_lab_dir, source_base + ".lab")
            _write_lab(clean_lab_path, lab_rows)

            item = {
                "base": base,
                "source_base": source_base,
                "source_voicebank": voicebank,
                "wav_path": wav_path,
                "raw_lab_path": oto_path,
                "clean_lab_path": clean_lab_path,
                "wav_end_100ns": int(wav_end_100ns),
                "scale_factor": 1.0,
                "scale_anchor_end": int(wav_end_100ns),
                "tail_pause_repaired": False,
                "expected_words": int(expected_words),
                "accepted": bool(accepted),
                "quality_errors": merged_errors,
                "non_pause_count": int(quality.get("non_pause_count", 1) or 1),
                "speech_ratio": float(quality.get("speech_ratio", 0.0) or 0.0),
                "first_non_pause_ratio": float(quality.get("first_non_pause_ratio", 0.0) or 0.0),
                "tail_pause_ratio": float(quality.get("tail_pause_ratio", 0.0) or 0.0),
                "cv_alias_count": int(len(cv_rows)),
                "cv_alias_types": dict(Counter([r.alias_type for r in cv_rows])),
            }
            all_rows.append(item)
            speech_ratios.append(float(item["speech_ratio"]))
            if accepted:
                accepted_rows.append(item)
                vb_accepted += 1
        voicebank_stats.append(
            {
                "voicebank": voicebank,
                "oto_path": oto_path,
                "wav_pairs": int(vb_total),
                "accepted_pairs": int(vb_accepted),
            }
        )

    accepted_rows = sorted(accepted_rows, key=lambda x: str(x.get("base", "")))
    val_ratio = max(0.05, min(0.5, float(args.val_ratio)))
    val_count = max(1, int(round(len(accepted_rows) * val_ratio))) if accepted_rows else 0
    val_rows = accepted_rows[-val_count:] if val_count > 0 else []
    train_rows = accepted_rows[:-val_count] if val_count > 0 else accepted_rows

    _write_jsonl(os.path.join(out_root, "manifest_all.jsonl"), all_rows)
    _write_jsonl(os.path.join(out_root, "manifest_accepted.jsonl"), accepted_rows)
    _write_jsonl(os.path.join(out_root, "manifest_train.jsonl"), train_rows)
    _write_jsonl(os.path.join(out_root, "manifest_val.jsonl"), val_rows)

    summary = {
        "mode": "voicebank" if voicebank_raw else "dataset_root",
        "voicebank": os.path.abspath(voicebank_raw) if voicebank_raw else "",
        "dataset_root": os.path.abspath(dataset_root_raw) if dataset_root_raw else "",
        "language": args.language,
        "out_root": out_root,
        "wav_pairs": int(len(all_rows)),
        "accepted_pairs": int(len(accepted_rows)),
        "rejected_pairs": int(len(all_rows) - len(accepted_rows)),
        "train_pairs": int(len(train_rows)),
        "val_pairs": int(len(val_rows)),
        "voicebank_count": int(len(targets)),
        "voicebank_stats_top": sorted(
            voicebank_stats,
            key=lambda x: int(x.get("accepted_pairs", 0)),
            reverse=True,
        )[:40],
        "cv_alias_type_counts": dict(cv_alias_counter),
        "speech_ratio_mean": float(statistics.mean(speech_ratios)) if speech_ratios else 0.0,
        "rejected_examples": [
            {
                "base": str(x.get("base", "")),
                "errors": list(x.get("quality_errors", []) or []),
                "cv_alias_count": int(x.get("cv_alias_count", 0) or 0),
            }
            for x in all_rows
            if not bool(x.get("accepted", False))
        ][:30],
    }
    _write_json(os.path.join(out_root, "summary.json"), summary)

    print(f"[PREP-OTO] wav_pairs={summary['wav_pairs']}")
    print(f"[PREP-OTO] accepted_pairs={summary['accepted_pairs']}")
    print(f"[PREP-OTO] train_pairs={summary['train_pairs']} val_pairs={summary['val_pairs']}")
    print(f"[PREP-OTO] out_root={out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
