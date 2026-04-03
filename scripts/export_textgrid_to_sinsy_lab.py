from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

try:
    import textgrid  # type: ignore
except Exception:  # pragma: no cover
    textgrid = None


_PAUSE_MARKS = {"", "sp", "spn", "sil", "pau", "ap", "br", "bre", "breath", "r"}


@dataclass
class LabRow:
    start: int
    end: int
    label: str


def _safe_mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _list_textgrids(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(".textgrid")
    )


def _norm_label(mark: str) -> str:
    raw = str(mark or "").strip()
    if raw.lower() in _PAUSE_MARKS:
        return "SP"
    return raw


def _load_tier_intervals(path: str, tier_name: str) -> Tuple[float, List[Tuple[float, float, str]]]:
    if textgrid is None:
        raise RuntimeError("python-textgrid is not available")
    tg = textgrid.TextGrid.fromFile(path)
    max_t = float(getattr(tg, "maxTime", 0.0) or 0.0)
    target = None
    for tier in list(getattr(tg, "tiers", [])):
        if str(getattr(tier, "name", "") or "").strip().lower() == str(tier_name).strip().lower():
            target = tier
            break
    if target is None:
        raise RuntimeError(f"tier not found: {tier_name}")

    items: List[Tuple[float, float, str]] = []
    for it in list(getattr(target, "intervals", []) or []):
        st = float(getattr(it, "minTime", 0.0) or 0.0)
        ed = float(getattr(it, "maxTime", 0.0) or 0.0)
        if ed <= st:
            continue
        label = _norm_label(str(getattr(it, "mark", "") or ""))
        items.append((st, ed, label))
    return max_t, items


def _to_100ns(sec: float) -> int:
    return int(round(float(sec) * 10_000_000.0))


def _sanitize_rows(max_t: float, rows: Sequence[Tuple[float, float, str]]) -> List[LabRow]:
    max_100ns = max(0, _to_100ns(max_t))
    out: List[LabRow] = []

    for st, ed, lb in sorted(rows, key=lambda x: (float(x[0]), float(x[1]))):
        s = max(0, min(max_100ns, _to_100ns(st)))
        e = max(0, min(max_100ns, _to_100ns(ed)))
        if e <= s:
            continue
        label = str(lb or "").strip() or "SP"
        if out and s < out[-1].end:
            s = out[-1].end
        if e <= s:
            continue
        out.append(LabRow(start=s, end=e, label=label))

    if not out and max_100ns > 0:
        return [LabRow(start=0, end=max_100ns, label="SP")]

    if out and out[0].start > 0:
        out.insert(0, LabRow(start=0, end=out[0].start, label="SP"))
    if out and out[-1].end < max_100ns:
        out.append(LabRow(start=out[-1].end, end=max_100ns, label="SP"))

    compact: List[LabRow] = []
    for row in out:
        if not compact:
            compact.append(LabRow(row.start, row.end, row.label))
            continue
        prev = compact[-1]
        if prev.label == row.label and prev.end == row.start:
            prev.end = row.end
        else:
            compact.append(LabRow(row.start, row.end, row.label))

    if compact:
        compact[-1].end = max_100ns
    return compact


def _write_lab(path: str, rows: Sequence[LabRow]) -> None:
    _safe_mkdir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{int(row.start)} {int(row.end)} {row.label}\n")


def _copy_wav_if_exists(base: str, wav_dir: str, out_dir: str) -> str:
    if not wav_dir:
        return ""
    src = os.path.join(wav_dir, base + ".wav")
    if not os.path.isfile(src):
        return ""
    dst = os.path.join(out_dir, base + ".wav")
    shutil.copy2(src, dst)
    return dst


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export TextGrid tier intervals into Sinsy .lab format (100ns)."
    )
    parser.add_argument("--textgrid-dir", required=True, help="Input TextGrid folder")
    parser.add_argument("--out-dir", required=True, help="Output .lab folder")
    parser.add_argument(
        "--tier",
        default="words",
        choices=["words", "phones"],
        help="Tier to export",
    )
    parser.add_argument(
        "--wav-dir",
        default="",
        help="Optional WAV source folder (copy same basename WAV into out-dir)",
    )
    parser.add_argument("--clean", action="store_true", help="Remove old .lab/.wav/.json in out-dir")
    args = parser.parse_args()

    tg_dir = os.path.abspath(str(args.textgrid_dir))
    out_dir = _safe_mkdir(os.path.abspath(str(args.out_dir)))
    wav_dir = os.path.abspath(str(args.wav_dir)) if str(args.wav_dir or "").strip() else ""
    tier_name = str(args.tier or "words").strip().lower()

    if not os.path.isdir(tg_dir):
        raise SystemExit(f"textgrid-dir not found: {tg_dir}")
    if textgrid is None:
        raise SystemExit("python-textgrid is required but not available")

    if args.clean:
        for name in os.listdir(out_dir):
            low = name.lower()
            if low.endswith(".lab") or low.endswith(".wav") or low in {"manifest.json"}:
                try:
                    os.remove(os.path.join(out_dir, name))
                except Exception:
                    pass

    tg_files = _list_textgrids(tg_dir)
    results: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []

    for tg_path in tg_files:
        base = os.path.splitext(os.path.basename(tg_path))[0]
        try:
            max_t, rows = _load_tier_intervals(tg_path, tier_name)
            clean_rows = _sanitize_rows(max_t, rows)
            out_lab = os.path.join(out_dir, base + ".lab")
            _write_lab(out_lab, clean_rows)
            out_wav = _copy_wav_if_exists(base, wav_dir, out_dir)
            results.append(
                {
                    "base": base,
                    "textgrid": tg_path,
                    "lab": out_lab,
                    "wav": out_wav,
                    "rows": int(len(clean_rows)),
                }
            )
        except Exception as exc:
            errors.append({"base": base, "textgrid": tg_path, "error": str(exc)})

    payload = {
        "textgrid_dir": tg_dir,
        "out_dir": out_dir,
        "tier": tier_name,
        "wav_dir": wav_dir,
        "converted": int(len(results)),
        "errors": int(len(errors)),
        "items": results,
        "error_items": errors,
    }
    manifest = os.path.join(out_dir, "manifest.json")
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[DONE] tier={tier_name} converted={len(results)} errors={len(errors)}")
    print(f"[DONE] out_dir={out_dir}")
    print(f"[DONE] manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
