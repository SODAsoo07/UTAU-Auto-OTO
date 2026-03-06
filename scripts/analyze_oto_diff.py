import argparse
import json
import os
import re
import sys
from collections import defaultdict


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.ja_oto_mapping import classify_ja_alias
from core.oto_generator import classify_alias


def _read_text(path):
    encodings = ("utf-8-sig", "utf-8", "cp932", "cp949")
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read().splitlines(), enc
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"failed to read '{path}': {last_err}")


def _norm(s):
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def _classify_alias(alias, language):
    lang = (language or "auto").strip().lower()
    a = str(alias or "").strip()
    if lang == "japanese":
        return classify_ja_alias(a, {})
    if lang == "korean":
        return classify_alias(a)
    if re.search(r"[\u3040-\u30ff]", a):
        return classify_ja_alias(a, {})
    return classify_alias(a)


def _parse_oto(path, language):
    lines, enc = _read_text(path)
    rows = {}
    occ = defaultdict(int)
    malformed = 0
    for idx, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            malformed += 1
            continue
        wav, rhs = line.split("=", 1)
        parts = [p.strip() for p in rhs.split(",")]
        if len(parts) < 6:
            malformed += 1
            continue
        alias = parts[0]
        try:
            offset = float(parts[1])
            cons = float(parts[2])
            cutoff = float(parts[3])
            pre = float(parts[4])
            ovl = float(parts[5])
        except Exception:
            malformed += 1
            continue

        key_base = (_norm(wav), _norm(alias))
        key_idx = occ[key_base]
        occ[key_base] += 1
        key = (key_base[0], key_base[1], key_idx)
        rows[key] = {
            "wav": wav.strip(),
            "alias": alias,
            "offset": offset,
            "cons": cons,
            "cutoff": cutoff,
            "pre": pre,
            "ovl": ovl,
            "pre_abs": offset + pre,
            "cons_abs": offset + cons,
            "cutoff_abs": offset + (abs(cutoff) if cutoff < 0 else cutoff),
            "alias_type": _classify_alias(alias, language),
            "line_no": idx,
        }
    return rows, {"encoding": enc, "malformed_lines": malformed, "line_count": len(lines)}


def _mean(values):
    return sum(values) / float(len(values)) if values else 0.0


def analyze(base_oto, auto_oto, language="auto", topn=30):
    base_rows, base_meta = _parse_oto(base_oto, language)
    auto_rows, auto_meta = _parse_oto(auto_oto, language)

    base_keys = set(base_rows.keys())
    auto_keys = set(auto_rows.keys())
    shared = sorted(base_keys & auto_keys)

    fields = ("offset", "cons", "cutoff", "pre", "ovl", "pre_abs", "cons_abs", "cutoff_abs")
    field_abs = {f: [] for f in fields}
    by_type = defaultdict(lambda: {f: [] for f in fields})
    suspicious = []

    for k in shared:
        b = base_rows[k]
        a = auto_rows[k]
        alias_type = a.get("alias_type") or b.get("alias_type") or "unknown"
        diffs = {}
        for f in fields:
            d = abs(float(a[f]) - float(b[f]))
            field_abs[f].append(d)
            by_type[alias_type][f].append(d)
            diffs[f] = d
        jump_score = max(diffs["pre_abs"], diffs["offset"], diffs["cons_abs"])
        if jump_score >= 120.0:
            suspicious.append({
                "wav": a["wav"],
                "alias": a["alias"],
                "alias_type": alias_type,
                "jump_score": jump_score,
                "diff_offset": diffs["offset"],
                "diff_pre_abs": diffs["pre_abs"],
                "diff_cons_abs": diffs["cons_abs"],
                "diff_cutoff_abs": diffs["cutoff_abs"],
            })

    suspicious.sort(key=lambda x: x["jump_score"], reverse=True)

    by_type_summary = {}
    for alias_type, vals in sorted(by_type.items()):
        by_type_summary[alias_type] = {f"mae_{f}": _mean(v) for f, v in vals.items()}
        by_type_summary[alias_type]["rows"] = len(vals["offset"])

    report = {
        "base_oto": os.path.abspath(base_oto),
        "auto_oto": os.path.abspath(auto_oto),
        "language": language,
        "counts": {
            "base_rows": len(base_rows),
            "auto_rows": len(auto_rows),
            "shared_rows": len(shared),
            "base_only": len(base_keys - auto_keys),
            "auto_only": len(auto_keys - base_keys),
            "suspicious_rows": len(suspicious),
        },
        "parse_meta": {
            "base": base_meta,
            "auto": auto_meta,
        },
        "mae": {f"mae_{f}": _mean(vals) for f, vals in field_abs.items()},
        "alias_type_mae": by_type_summary,
        "suspicious_top": suspicious[: max(1, int(topn))],
    }
    return report


def main():
    ap = argparse.ArgumentParser(description="Compare two oto.ini files and report per-alias timing deltas.")
    ap.add_argument("--base-oto", required=True, help="Reference oto.ini path")
    ap.add_argument("--auto-oto", required=True, help="Generated oto.ini path")
    ap.add_argument("--report", required=True, help="Output report path (.json)")
    ap.add_argument("--language", default="auto", choices=["auto", "korean", "japanese"])
    ap.add_argument("--topn", type=int, default=30, help="Number of suspicious aliases in report")
    args = ap.parse_args()

    report = analyze(args.base_oto, args.auto_oto, language=args.language, topn=args.topn)
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OTODiff] shared_rows={report['counts']['shared_rows']} suspicious={report['counts']['suspicious_rows']}")
    print(f"[OTODiff] mae_pre_abs={report['mae']['mae_pre_abs']:.3f} mae_offset={report['mae']['mae_offset']:.3f}")
    print(f"[OTODiff] report={os.path.abspath(args.report)}")


if __name__ == "__main__":
    main()

