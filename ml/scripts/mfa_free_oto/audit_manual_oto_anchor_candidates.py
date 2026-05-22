from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.mfa_free_oto.manual_oto_candidates import ALL_ANCHORS, extract_manual_oto_candidate_tracks


DEFAULT_MANIFEST = r"ml_workspace\manual_oto_anchor\splits\manual_oto_anchor_filtered.jsonl"
DEFAULT_OUT = r"ml_workspace\manual_oto_anchor\candidate_audit\candidate_recall_summary.json"


ANCHORS = ALL_ANCHORS


def _load_rows(path: Path, *, max_rows: int | None = None, max_wavs: int | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_wavs: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            wav = str(row.get("wav_path", "") or "")
            if max_wavs is not None and wav not in seen_wavs and len(seen_wavs) >= max_wavs:
                continue
            seen_wavs.add(wav)
            rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def _candidate_times_for_wav(
    wav_path: str,
    *,
    encoder: str,
    min_score: float,
    top_k_fallback: int,
) -> dict[str, list[float]]:
    tracks = extract_manual_oto_candidate_tracks(
        wav_path,
        encoder=encoder,
        min_score=float(min_score),
        top_k_fallback=int(top_k_fallback),
    )
    return {anchor: tracks.candidate_times_ms(anchor) for anchor in ANCHORS}


def _nearest_error(target: float, candidates: list[float]) -> float | None:
    if not candidates:
        return None
    return min(abs(float(candidate) - float(target)) for candidate in candidates)


def audit_candidate_recall(
    *,
    manifest_path: Path,
    out_path: Path,
    encoder: str = "acoustic",
    max_rows: int | None = None,
    max_wavs: int | None = None,
    min_score: float = 0.28,
    top_k_fallback: int = 24,
) -> dict[str, object]:
    rows = _load_rows(manifest_path, max_rows=max_rows, max_wavs=max_wavs)
    by_wav: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_wav[str(row.get("wav_path", "") or "")].append(row)

    candidate_cache: dict[str, dict[str, list[float]]] = {}
    errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in ANCHORS}
    misses_by_anchor: Counter[str] = Counter()
    total_by_anchor: Counter[str] = Counter()
    by_role_anchor_total: Counter[str] = Counter()
    by_role_anchor_hit30: Counter[str] = Counter()
    by_role_anchor_hit60: Counter[str] = Counter()
    failures: list[dict[str, object]] = []
    wav_failures: list[dict[str, object]] = []

    for wav_path, wav_rows in by_wav.items():
        try:
            candidate_cache[wav_path] = _candidate_times_for_wav(
                wav_path,
                encoder=encoder,
                min_score=float(min_score),
                top_k_fallback=int(top_k_fallback),
            )
        except Exception as exc:
            wav_failures.append({"wav_path": wav_path, "error": str(exc)})
            continue
        candidates = candidate_cache[wav_path]
        for row in wav_rows:
            role = str(row.get("alias_role", "") or "")
            anchors = dict(row.get("anchors_abs_ms") or {})
            for anchor in ANCHORS:
                if anchor not in anchors:
                    continue
                target = float(anchors[anchor])
                total_by_anchor[anchor] += 1
                key = f"{role}|{anchor}"
                by_role_anchor_total[key] += 1
                err = _nearest_error(target, candidates.get(anchor, []))
                if err is None:
                    misses_by_anchor[anchor] += 1
                    failures.append(
                        {
                            "wav_path": wav_path,
                            "alias": row.get("alias", ""),
                            "alias_role": role,
                            "anchor": anchor,
                            "target_ms": target,
                            "reason": "no_candidates",
                        }
                    )
                    continue
                errors_by_anchor[anchor].append(float(err))
                if err <= 30.0:
                    by_role_anchor_hit30[key] += 1
                if err <= 60.0:
                    by_role_anchor_hit60[key] += 1
                if err > 60.0 and len(failures) < 200:
                    failures.append(
                        {
                            "wav_path": wav_path,
                            "alias": row.get("alias", ""),
                            "alias_role": role,
                            "anchor": anchor,
                            "target_ms": target,
                            "nearest_error_ms": float(err),
                        }
                    )

    def anchor_summary(anchor: str) -> dict[str, object]:
        values = errors_by_anchor[anchor]
        total = int(total_by_anchor[anchor])
        missed = int(misses_by_anchor[anchor])
        hit30 = sum(1 for value in values if value <= 30.0)
        hit60 = sum(1 for value in values if value <= 60.0)
        return {
            "total": total,
            "candidate_missing": missed,
            "hit30": hit30,
            "hit60": hit60,
            "recall30": float(hit30 / total) if total else 0.0,
            "recall60": float(hit60 / total) if total else 0.0,
            "median_error_ms": float(median(values)) if values else None,
            "p90_error_ms": float(np.percentile(np.asarray(values, dtype=np.float32), 90.0)) if values else None,
        }

    by_role_anchor: dict[str, dict[str, object]] = {}
    for key, total in sorted(by_role_anchor_total.items()):
        by_role_anchor[key] = {
            "total": int(total),
            "recall30": float(by_role_anchor_hit30[key] / total) if total else 0.0,
            "recall60": float(by_role_anchor_hit60[key] / total) if total else 0.0,
        }

    summary = {
        "manifest_path": str(manifest_path),
        "out_path": str(out_path),
        "encoder": encoder,
        "rows": len(rows),
        "wavs": len(by_wav),
        "wav_failures": wav_failures[:100],
        "anchor_summary": {anchor: anchor_summary(anchor) for anchor in ANCHORS},
        "by_role_anchor": by_role_anchor,
        "worst_or_missing_examples": failures[:200],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit audio candidate recall around manual OTO anchors.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--encoder", default="acoustic")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-wavs", type=int)
    parser.add_argument("--min-score", type=float, default=0.28)
    parser.add_argument("--top-k-fallback", type=int, default=24)
    args = parser.parse_args()
    summary = audit_candidate_recall(
        manifest_path=Path(args.manifest),
        out_path=Path(args.out),
        encoder=str(args.encoder or "acoustic"),
        max_rows=args.max_rows,
        max_wavs=args.max_wavs,
        min_score=float(args.min_score),
        top_k_fallback=int(args.top_k_fallback),
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
