from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PARAM_KEYS = ("offset", "consonant", "cutoff_abs", "preutterance", "overlap")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two OTO evaluation JSON files.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--min-context-files", type=int, default=20)
    args = parser.parse_args()

    base = _read_json(args.base)
    candidate = _read_json(args.candidate)
    result = {
        "base": str(Path(args.base).resolve()),
        "candidate": str(Path(args.candidate).resolve()),
        "overall": _compare_overall(base, candidate),
        "by_format": _compare_group(base.get("by_format") or {}, candidate.get("by_format") or {}),
        "by_alias_context": _compare_group(
            base.get("by_alias_context") or {},
            candidate.get("by_alias_context") or {},
            min_files=int(args.min_context_files),
        ),
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_compact(result), ensure_ascii=False, indent=2))
    return 0


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"evaluation JSON must be an object: {path}")
    return payload


def _compare_overall(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_params = base.get("param_mae_ms") or {}
    cand_params = candidate.get("param_mae_ms") or {}
    out = {
        "evaluated_items": {
            "base": int(base.get("evaluated_items", 0) or 0),
            "candidate": int(candidate.get("evaluated_items", 0) or 0),
        },
        "preutterance_acc_50ms": _metric_delta(base.get("preutterance_acc_50ms"), candidate.get("preutterance_acc_50ms")),
        "preutterance_acc_20ms": _metric_delta(base.get("preutterance_acc_20ms"), candidate.get("preutterance_acc_20ms")),
        "position_bad_rate_250ms": _metric_delta(base.get("position_bad_rate_250ms"), candidate.get("position_bad_rate_250ms")),
        "param_mae_ms": {},
    }
    for key in PARAM_KEYS:
        out["param_mae_ms"][key] = _metric_delta(base_params.get(key), cand_params.get(key))
    return out


def _compare_group(base: dict[str, Any], candidate: dict[str, Any], *, min_files: int = 1) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(set(base) | set(candidate)):
        b = base.get(key) or {}
        c = candidate.get(key) or {}
        files = max(int(b.get("files", 0) or 0), int(c.get("files", 0) or 0))
        if files < int(min_files):
            continue
        out[key] = {
            "files": {"base": int(b.get("files", 0) or 0), "candidate": int(c.get("files", 0) or 0)},
            "preutterance_mae_ms": _metric_delta(b.get("preutterance_mae_ms"), c.get("preutterance_mae_ms")),
            "preutterance_acc_50ms": _metric_delta(b.get("preutterance_acc_50ms"), c.get("preutterance_acc_50ms")),
            "offset_mae_ms": _metric_delta(b.get("offset_mae_ms"), c.get("offset_mae_ms")),
            "cutoff_abs_mae_ms": _metric_delta(b.get("cutoff_abs_mae_ms"), c.get("cutoff_abs_mae_ms")),
            "position_bad_rate_250ms": _metric_delta(b.get("position_bad_rate_250ms"), c.get("position_bad_rate_250ms")),
        }
        if "consonant_mae_ms" in b or "consonant_mae_ms" in c:
            out[key]["consonant_mae_ms"] = _metric_delta(b.get("consonant_mae_ms"), c.get("consonant_mae_ms"))
        if "overlap_mae_ms" in b or "overlap_mae_ms" in c:
            out[key]["overlap_mae_ms"] = _metric_delta(b.get("overlap_mae_ms"), c.get("overlap_mae_ms"))
    return out


def _metric_delta(base: object, candidate: object) -> dict[str, float | None]:
    b = _float_or_none(base)
    c = _float_or_none(candidate)
    return {"base": b, "candidate": c, "delta": (c - b) if b is not None and c is not None else None}


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    contexts = result.get("by_alias_context") or {}
    ranked = sorted(
        contexts.items(),
        key=lambda item: abs(float(((item[1].get("preutterance_mae_ms") or {}).get("delta")) or 0.0)),
        reverse=True,
    )
    compact = dict(result)
    compact["by_alias_context"] = dict(ranked[:20])
    return compact


if __name__ == "__main__":
    raise SystemExit(main())
