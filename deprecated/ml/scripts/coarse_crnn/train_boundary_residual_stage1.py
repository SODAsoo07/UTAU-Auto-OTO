from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.coarse_crnn.alias_role import normalize_role
from core.coarse_crnn.boundary_residual import DEFAULT_DELTA_CLIPS, RESIDUAL_TARGET_NAMES
from core.runtime.runtime_encoding import bootstrap_utf8_runtime

try:
    import pandas as pd
except Exception as _pd_exc:  # pragma: no cover
    pd = None
    PD_IMPORT_ERROR = _pd_exc
else:
    PD_IMPORT_ERROR = None


def _require_stack() -> None:
    if pd is None:
        raise RuntimeError(f"pandas is required: {PD_IMPORT_ERROR}")


def main() -> int:
    bootstrap_utf8_runtime()
    _require_stack()
    ap = argparse.ArgumentParser(description="Train stage1 role-bias residual corrector for boundary decoder.")
    ap.add_argument("--dataset-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--role-column", default="role")
    ap.add_argument("--min-role-rows", type=int, default=40)
    ap.add_argument("--shrink", type=float, default=80.0, help="Shrinkage strength toward global median.")
    args = ap.parse_args()

    csv_path = os.path.abspath(str(args.dataset_csv))
    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    frame = pd.read_csv(csv_path, encoding="utf-8")
    role_col = str(args.role_column or "role").strip()
    if role_col not in frame.columns:
        raise SystemExit(f"missing role column: {role_col}")
    roles = frame[role_col].fillna("other").astype(str).map(normalize_role)
    min_rows = max(1, int(args.min_role_rows))
    shrink = max(0.0, float(args.shrink))

    targets: dict[str, dict[str, object]] = {}
    summary: dict[str, object] = {
        "rows": int(len(frame)),
        "dataset_csv": csv_path,
        "min_role_rows": int(min_rows),
        "shrink": float(shrink),
    }
    for target in RESIDUAL_TARGET_NAMES:
        if target not in frame.columns:
            continue
        raw = pd.to_numeric(frame[target], errors="coerce").fillna(0.0)
        lo, hi = DEFAULT_DELTA_CLIPS.get(target, (-500.0, 500.0))
        clipped = raw.clip(lower=float(lo), upper=float(hi))
        global_median = float(clipped.median())
        by_role: dict[str, float] = {}
        role_counts: dict[str, int] = {}
        for role in sorted(set(roles.tolist())):
            mask = roles == role
            count = int(mask.sum())
            if count < min_rows:
                continue
            role_med = float(clipped[mask].median())
            role_val = (role_med * float(count) + global_median * float(shrink)) / max(1.0, float(count) + float(shrink))
            by_role[str(role)] = float(role_val)
            role_counts[str(role)] = int(count)
        targets[target] = {
            "default": float(global_median),
            "by_role": by_role,
            "role_counts": role_counts,
            "clip": [float(lo), float(hi)],
        }

    payload = {
        "backend": "boundary_residual_stage1_role_bias_v1",
        "version": "v1",
        "targets": targets,
        "summary": summary,
    }
    out_path = os.path.join(out_dir, "stage1_role_bias.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"out": out_path, "targets": list(targets.keys()), "rows": int(len(frame))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
