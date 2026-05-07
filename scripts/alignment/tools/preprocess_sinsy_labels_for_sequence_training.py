from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import wave
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Set, Tuple

try:
    from core.ja_lab_generator import parse_ja_filename
except Exception:  # pragma: no cover
    parse_ja_filename = None

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
AP_LABELS = {"ap", "br", "bre", "breath"}
DEFAULT_PAUSE_LIKE_PHONE_LABELS = {"vf", "axh", "exh"}
_SMALL_JA = set("ゃゅょぁぃぅぇぉャュョァィゥェォゎヮっッゕゖ")


@dataclass
class LabRow:
    start: int
    end: int
    label: str


def _read_wav_duration_100ns(path: str) -> int:
    with wave.open(path, "rb") as wf:
        fr = int(wf.getframerate() or 1)
        n = int(wf.getnframes() or 0)
    sec = float(n) / float(max(fr, 1))
    return int(round(sec * 10_000_000.0))


def _read_lab_rows(path: str) -> List[LabRow]:
    rows: List[LabRow] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            line = str(raw or "").strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                start = int(float(parts[0]))
                end = int(float(parts[1]))
            except Exception:
                continue
            label = " ".join(parts[2:]).strip()
            rows.append(LabRow(start=start, end=end, label=label))
    return rows


def _normalize_label(label: str) -> str:
    return str(label or "").strip().lower()


def _normalize_training_label(
    label: str,
    *,
    label_scheme: str,
    pause_like_phone_labels: Set[str],
) -> str:
    raw = str(label or "").strip()
    if not raw:
        return ""
    low = raw.lower()
    scheme = str(label_scheme or "none").strip().lower()
    if scheme == "none":
        return raw
    if low in PAUSE_LABELS:
        return "AP" if low in AP_LABELS else "SP"
    if low in pause_like_phone_labels:
        return "AP"
    return raw


def _apply_label_normalization(
    rows: Sequence[LabRow],
    *,
    label_scheme: str,
    pause_like_phone_labels: Set[str],
) -> Tuple[List[LabRow], int, Dict[str, int]]:
    out: List[LabRow] = []
    remapped = 0
    remap_pairs: Dict[str, int] = {}
    for row in rows:
        new_label = _normalize_training_label(
            row.label,
            label_scheme=label_scheme,
            pause_like_phone_labels=pause_like_phone_labels,
        )
        out.append(LabRow(start=int(row.start), end=int(row.end), label=str(new_label)))
        if str(new_label) != str(row.label):
            remapped += 1
            key = f"{row.label} -> {new_label}"
            remap_pairs[key] = int(remap_pairs.get(key, 0) + 1)
    return out, int(remapped), remap_pairs


def _is_pause(label: str) -> bool:
    return _normalize_label(label) in PAUSE_LABELS


def _infer_expected_word_count(base: str, language: str) -> int:
    stem = str(base or "").strip().lstrip("_")
    if not stem:
        return 0
    lang = str(language or "").strip().lower()
    if lang == "japanese" and callable(parse_ja_filename):
        try:
            parsed = parse_ja_filename(stem) or []
        except Exception:
            parsed = []
        if parsed:
            if int(len(parsed)) >= 2:
                return int(len(parsed))
    if lang == "japanese":
        mora = 0
        for ch in stem:
            if ch in {"_", "-", " ", "|", "/", "."}:
                continue
            code = ord(ch)
            is_kana = (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF)
            if not is_kana:
                continue
            if ch in _SMALL_JA:
                if mora <= 0:
                    mora = 1
                continue
            mora += 1
        if mora > 0:
            return int(mora)
    parts = [p for p in re.split(r"[_\s\-|/]+", stem) if p.strip()]
    return int(len(parts))


def _scale_rows_to_wav(
    rows: Sequence[LabRow],
    wav_end_100ns: int,
) -> Tuple[List[LabRow], float, bool, int]:
    if not rows:
        return [], 1.0, False, 0
    if wav_end_100ns <= 0:
        return [], 1.0, False, 0

    sorted_rows = sorted(
        [LabRow(int(r.start), int(r.end), str(r.label)) for r in rows if int(r.end) > int(r.start)],
        key=lambda x: (x.start, x.end),
    )
    if not sorted_rows:
        return [], 1.0, False, 0

    raw_end = max(int(r.end) for r in sorted_rows)
    if raw_end <= 0:
        return [], 1.0, False, 0

    last_non_pause_end = 0
    for row in reversed(sorted_rows):
        if not _is_pause(row.label):
            last_non_pause_end = int(row.end)
            break

    anchor_end = int(raw_end)
    tail_repaired = False
    if last_non_pause_end > 0 and raw_end > int(wav_end_100ns * 1.2):
        trailing_share = float(max(0, raw_end - last_non_pause_end)) / float(max(1, raw_end))
        if trailing_share >= 0.30:
            anchor_end = int(last_non_pause_end)
            tail_repaired = True

    scale = float(wav_end_100ns) / float(max(1, anchor_end))

    out: List[LabRow] = []
    for row in sorted_rows:
        start = int(round(float(row.start) * scale))
        end = int(round(float(row.end) * scale))
        start = max(0, min(start, wav_end_100ns))
        end = max(0, min(end, wav_end_100ns))
        if end <= start:
            continue
        if out and start < out[-1].end:
            start = int(out[-1].end)
        if end <= start:
            continue
        out.append(LabRow(start=start, end=end, label=str(row.label)))

    if not out:
        return [LabRow(start=0, end=int(wav_end_100ns), label="SP")], float(scale), tail_repaired, int(anchor_end)

    if out[0].start > 0:
        out.insert(0, LabRow(start=0, end=int(out[0].start), label="SP"))
    if out[-1].end < wav_end_100ns:
        out.append(LabRow(start=int(out[-1].end), end=int(wav_end_100ns), label="SP"))
    out[-1].end = int(wav_end_100ns)
    return out, float(scale), tail_repaired, int(anchor_end)


def _compress_adjacent(rows: Sequence[LabRow]) -> List[LabRow]:
    out: List[LabRow] = []
    for row in rows:
        if not out:
            out.append(LabRow(row.start, row.end, row.label))
            continue
        prev = out[-1]
        if _normalize_label(prev.label) == _normalize_label(row.label) and prev.end == row.start:
            prev.end = row.end
        else:
            out.append(LabRow(row.start, row.end, row.label))
    return out


def _write_lab(path: str, rows: Sequence[LabRow]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{int(row.start)} {int(row.end)} {row.label}\n")


def _quality_report(rows: Sequence[LabRow], wav_end_100ns: int, expected_words: int, scale_factor: float) -> Dict[str, object]:
    non_pause = [r for r in rows if not _is_pause(r.label)]
    speech_100ns = sum(max(0, int(r.end) - int(r.start)) for r in non_pause)
    speech_ratio = float(speech_100ns) / float(max(1, wav_end_100ns))
    non_pause_count = int(len(non_pause))

    errors: List[str] = []
    if non_pause_count < 3:
        errors.append("too_few_non_pause_labels")
    if speech_ratio < 0.08:
        errors.append("speech_ratio_too_low")
    if speech_ratio > 0.995:
        errors.append("speech_ratio_too_high")
    if expected_words > 0:
        if non_pause_count < max(2, int(round(expected_words * 0.35))):
            errors.append("non_pause_under_expected")
        if non_pause_count > int(expected_words * 6.0 + 6):
            errors.append("non_pause_over_expected")
    if scale_factor < 0.12 or scale_factor > 1.2:
        errors.append("scale_factor_outlier")

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
    if first_ratio > 0.45:
        errors.append("late_first_non_pause")
    if tail_pause_ratio > 0.70:
        errors.append("long_tail_pause")

    return {
        "accepted": bool(not errors),
        "errors": errors,
        "non_pause_count": non_pause_count,
        "speech_ratio": float(speech_ratio),
        "first_non_pause_ratio": float(first_ratio),
        "tail_pause_ratio": float(tail_pause_ratio),
    }


def _iter_bases(wav_folder: str, lab_folder: str) -> List[str]:
    wav_names = os.listdir(wav_folder) if os.path.isdir(wav_folder) else []
    lab_names = os.listdir(lab_folder) if os.path.isdir(lab_folder) else []
    wav_bases = {os.path.splitext(n)[0] for n in wav_names if n.lower().endswith(".wav")}
    lab_bases = {os.path.splitext(n)[0] for n in lab_names if n.lower().endswith(".lab")}
    return sorted(wav_bases & lab_bases)


def _write_json(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess Sinsy lab files for sequence aligner tuning.")
    parser.add_argument("--source", required=True, help="Folder with paired wav/lab files")
    parser.add_argument("--wav-dir", default="", help="Optional explicit wav folder (when split layout is used)")
    parser.add_argument("--lab-dir", default="", help="Optional explicit lab folder (when split layout is used)")
    parser.add_argument("--language", default="japanese", choices=["japanese", "korean"], help="Language hint")
    parser.add_argument("--out-root", default="", help="Output root folder")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument(
        "--label-scheme",
        default="none",
        choices=["none", "ja_phone_v1"],
        help="Optional label normalization scheme before quality filtering",
    )
    parser.add_argument(
        "--pause-like-phone-labels",
        default="vf,axh,exh",
        help="CSV labels mapped to AP when --label-scheme=ja_phone_v1",
    )
    args = parser.parse_args()

    source = os.path.abspath(str(args.source or "").strip())
    if not os.path.isdir(source):
        raise SystemExit(f"source not found: {source}")
    explicit_wav_dir = os.path.abspath(str(args.wav_dir or "").strip()) if str(args.wav_dir or "").strip() else ""
    explicit_lab_dir = os.path.abspath(str(args.lab_dir or "").strip()) if str(args.lab_dir or "").strip() else ""
    wav_dir = explicit_wav_dir if explicit_wav_dir else source
    lab_dir = explicit_lab_dir if explicit_lab_dir else source
    if not os.path.isdir(wav_dir):
        raise SystemExit(f"wav-dir not found: {wav_dir}")
    if not os.path.isdir(lab_dir):
        raise SystemExit(f"lab-dir not found: {lab_dir}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if str(args.out_root or "").strip():
        out_root = os.path.abspath(str(args.out_root))
    else:
        out_root = os.path.abspath(
            os.path.join(
                os.getcwd(),
                "dataset_workspace",
                "sequence_sinsy_preprocessed",
                f"{os.path.basename(source)}_{stamp}",
            )
        )

    clean_lab_dir = os.path.join(out_root, "labs")
    os.makedirs(clean_lab_dir, exist_ok=True)

    accepted_rows: List[Dict[str, object]] = []
    all_rows: List[Dict[str, object]] = []
    scales: List[float] = []
    speech_ratios: List[float] = []
    tail_repaired_count = 0
    remapped_total = 0
    remap_hist: Dict[str, int] = {}
    pause_like_phone_labels = {
        token.strip().lower()
        for token in str(args.pause_like_phone_labels or "").split(",")
        if token.strip()
    }
    if not pause_like_phone_labels:
        pause_like_phone_labels = set(DEFAULT_PAUSE_LIKE_PHONE_LABELS)

    for base in _iter_bases(wav_dir, lab_dir):
        wav_path = os.path.join(wav_dir, base + ".wav")
        lab_path = os.path.join(lab_dir, base + ".lab")
        wav_end = _read_wav_duration_100ns(wav_path)
        raw_rows = _read_lab_rows(lab_path)
        norm_rows, remapped_count, remap_pairs = _apply_label_normalization(
            raw_rows,
            label_scheme=str(args.label_scheme),
            pause_like_phone_labels=pause_like_phone_labels,
        )
        remapped_total += int(remapped_count)
        for key, value in remap_pairs.items():
            remap_hist[key] = int(remap_hist.get(key, 0) + int(value))
        scaled_rows, scale_factor, tail_repaired, scale_anchor_end = _scale_rows_to_wav(norm_rows, wav_end)
        clean_rows = _compress_adjacent(scaled_rows)
        expected_words = _infer_expected_word_count(base, args.language)
        quality = _quality_report(clean_rows, wav_end, expected_words, scale_factor)
        if bool(tail_repaired):
            tail_repaired_count += 1

        clean_lab_path = os.path.join(clean_lab_dir, base + ".lab")
        _write_lab(clean_lab_path, clean_rows)

        item = {
            "base": base,
            "wav_path": wav_path,
            "raw_lab_path": lab_path,
            "clean_lab_path": clean_lab_path,
            "wav_end_100ns": int(wav_end),
            "label_scheme": str(args.label_scheme),
            "label_remapped_count": int(remapped_count),
            "scale_factor": float(scale_factor),
            "scale_anchor_end": int(scale_anchor_end),
            "tail_pause_repaired": bool(tail_repaired),
            "expected_words": int(expected_words),
            "accepted": bool(quality.get("accepted", False)),
            "quality_errors": list(quality.get("errors", []) or []),
            "non_pause_count": int(quality.get("non_pause_count", 0) or 0),
            "speech_ratio": float(quality.get("speech_ratio", 0.0) or 0.0),
            "first_non_pause_ratio": float(quality.get("first_non_pause_ratio", 0.0) or 0.0),
            "tail_pause_ratio": float(quality.get("tail_pause_ratio", 0.0) or 0.0),
        }
        all_rows.append(item)
        scales.append(float(scale_factor))
        speech_ratios.append(float(item["speech_ratio"]))
        if item["accepted"]:
            accepted_rows.append(item)

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
        "source": source,
        "wav_dir": wav_dir,
        "lab_dir": lab_dir,
        "language": args.language,
        "out_root": out_root,
        "label_scheme": str(args.label_scheme),
        "pause_like_phone_labels": sorted(list(pause_like_phone_labels)),
        "total_pairs": int(len(all_rows)),
        "accepted_pairs": int(len(accepted_rows)),
        "rejected_pairs": int(len(all_rows) - len(accepted_rows)),
        "train_pairs": int(len(train_rows)),
        "val_pairs": int(len(val_rows)),
        "tail_pause_repaired_pairs": int(tail_repaired_count),
        "label_remapped_rows": int(remapped_total),
        "top_label_remaps": [
            {"pair": str(pair), "count": int(count)}
            for pair, count in sorted(remap_hist.items(), key=lambda item: int(item[1]), reverse=True)[:20]
        ],
        "scale_factor_mean": float(statistics.mean(scales)) if scales else 0.0,
        "speech_ratio_mean": float(statistics.mean(speech_ratios)) if speech_ratios else 0.0,
        "rejected_examples": [
            {
                "base": str(x.get("base", "")),
                "errors": list(x.get("quality_errors", []) or []),
            }
            for x in all_rows
            if not bool(x.get("accepted", False))
        ][:30],
    }
    _write_json(os.path.join(out_root, "summary.json"), summary)

    print(f"[PREP] total_pairs={summary['total_pairs']}")
    print(f"[PREP] accepted_pairs={summary['accepted_pairs']}")
    print(f"[PREP] train_pairs={summary['train_pairs']} val_pairs={summary['val_pairs']}")
    print(f"[PREP] label_scheme={summary['label_scheme']} remapped_rows={summary['label_remapped_rows']}")
    print(f"[PREP] out_root={out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
