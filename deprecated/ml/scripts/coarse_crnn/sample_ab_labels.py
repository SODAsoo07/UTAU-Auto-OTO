"""Subsample an A/B listening label CSV to a manageable listening set.

Full label CSVs can run to thousands of rows. Listeners can realistically
label 30~60 cases per voicebank with high attention. This script picks a
subset of `case_id`s (preserving all version rows for each selected case so
A/B comparison is possible) and writes a filtered CSV plus a sampling
summary JSON.

Sampling modes
- uniform (default): random sample of case_ids per voicebank.
- role-stratified (`--stratify-by-role`): ensure proportional coverage of
  the role categories that matter most for the roadmap. Defaults emphasise
  Korean coda-heavy transitions (vc, v-cv) since those are where F-1 should
  show its largest improvement.

Examples
--------

Uniform sampling, 40 cases per voicebank::

    python -m ml.scripts.coarse_crnn.sample_ab_labels \\
        --labels ml/eval/ab_listening/labels/2026-05-15_baseline.csv \\
        --out ml/eval/ab_listening/labels/2026-05-15_baseline.sample.csv \\
        --per-voicebank 40 --seed 0

Role-stratified, default quotas::

    python -m ml.scripts.coarse_crnn.sample_ab_labels \\
        --labels ml/eval/ab_listening/labels/2026-05-15_baseline.csv \\
        --out ml/eval/ab_listening/labels/2026-05-15_baseline.sample.csv \\
        --per-voicebank 50 --stratify-by-role --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from typing import Any

from core.coarse_crnn.alias_role import classify_alias_role, normalize_role
from core.coarse_crnn.boundary_targets import _infer_alias_type as _boundary_alias_type
from core.coarse_crnn.boundary_targets import _infer_transition_type as _boundary_transition_type
from core.coarse_crnn.lang import normalize_language


# Per-role quota share for `--stratify-by-role` when the user does not
# override with `--role-quota`. Sums do not need to be exactly 1.0;
# the script normalizes them per voicebank.
DEFAULT_ROLE_QUOTA: dict[str, float] = {
    "vc": 0.30,        # Korean coda transition — top priority for F-1
    "v-cv": 0.18,      # CVVC transition
    "vv": 0.10,        # vowel transition
    "cv": 0.18,        # body
    "-cv": 0.10,       # head
    "v": 0.06,
    "v-": 0.04,
    "cv-": 0.02,
    "br": 0.01,
    "endbr": 0.01,
    "other": 0.00,
}


def _role_for_row(row: dict[str, Any]) -> str:
    """Run the same alias→role classifier the OTO pipeline uses.

    Reusing `boundary_targets._infer_alias_type` and `_infer_transition_type`
    keeps the sampler's role buckets aligned with the rest of the pipeline,
    so the stratified quotas actually match production roles.
    """
    alias = str(row.get("alias", "") or "")
    lang = normalize_language(str(row.get("language", "") or "")) or "korean"
    alias_type = _boundary_alias_type(alias, language=lang)
    transition_type = _boundary_transition_type(alias, language=lang)
    role = classify_alias_role(
        lang,
        alias,
        alias_type=alias_type,
        transition_type=transition_type,
        is_special=False,
    )
    return normalize_role(role)


def _read_rows(path: str) -> tuple[list[str], list[dict[str, Any]]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return fieldnames, rows


def _group_cases(rows: list[dict[str, Any]]) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, str]],
]:
    """Return (rows_by_case_id, case_meta_by_case_id).

    case_meta carries voicebank_id / alias / language for sampling decisions.
    """
    rows_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    meta: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = str(row.get("case_id", "") or "")
        if not case_id:
            continue
        rows_by_case[case_id].append(row)
        if case_id not in meta:
            meta[case_id] = {
                "voicebank_id": str(row.get("voicebank_id", "") or ""),
                "alias": str(row.get("alias", "") or ""),
                "language": str(row.get("language", "") or ""),
            }
    return rows_by_case, meta


def _normalize_quota(quota: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in quota.values())
    if total <= 0.0:
        return {}
    return {role: max(0.0, float(v)) / total for role, v in quota.items()}


def _sample_per_voicebank_uniform(
    *,
    case_ids: list[str],
    target_count: int,
    rng: random.Random,
) -> list[str]:
    if len(case_ids) <= target_count:
        return list(case_ids)
    return rng.sample(case_ids, target_count)


def _sample_per_voicebank_stratified(
    *,
    case_ids: list[str],
    case_roles: dict[str, str],
    target_count: int,
    quota: dict[str, float],
    rng: random.Random,
) -> tuple[list[str], dict[str, int]]:
    """Sample case_ids honoring role-quota shares with backfill if shortages."""
    if len(case_ids) <= target_count:
        return list(case_ids), {}

    quota_norm = _normalize_quota(quota)
    # Per-role available case list
    by_role: dict[str, list[str]] = defaultdict(list)
    for cid in case_ids:
        by_role[case_roles.get(cid, "other")].append(cid)
    for role in by_role:
        rng.shuffle(by_role[role])

    # Initial target count per role (rounded), then iteratively backfill from
    # whichever role still has capacity if a role is short of its target.
    role_targets: dict[str, int] = {}
    for role, share in quota_norm.items():
        role_targets[role] = int(round(share * float(target_count)))
    # Distribute rounding remainder
    remainder = target_count - sum(role_targets.values())
    roles_sorted_by_share = sorted(quota_norm.keys(), key=lambda r: -quota_norm[r])
    i = 0
    while remainder > 0 and roles_sorted_by_share:
        role_targets[roles_sorted_by_share[i % len(roles_sorted_by_share)]] += 1
        remainder -= 1
        i += 1

    # Track each picked case's role so backfill is reflected in the summary.
    case_role: dict[str, str] = {cid: role for role in by_role for cid in by_role[role]}

    picked: list[str] = []
    leftover_cases: list[str] = []
    for role, want in role_targets.items():
        avail = by_role.get(role, [])
        take = min(want, len(avail))
        picked.extend(avail[:take])
        leftover_cases.extend(avail[take:])
    deficit = target_count - len(picked)
    if deficit > 0 and leftover_cases:
        rng.shuffle(leftover_cases)
        picked.extend(leftover_cases[:deficit])

    actual_per_role: dict[str, int] = {role: 0 for role in role_targets}
    for cid in picked:
        role = case_role.get(cid, "other")
        actual_per_role[role] = actual_per_role.get(role, 0) + 1
    return picked, actual_per_role


def sample_label_rows(
    *,
    rows: list[dict[str, Any]],
    per_voicebank: int,
    stratify_by_role: bool,
    role_quota: dict[str, float],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sample case_ids per voicebank; return (filtered_rows, summary)."""
    rng = random.Random(int(seed))
    rows_by_case, meta = _group_cases(rows)
    by_voicebank: dict[str, list[str]] = defaultdict(list)
    for case_id, m in meta.items():
        by_voicebank[m["voicebank_id"]].append(case_id)

    role_cache: dict[str, str] = {}
    if stratify_by_role:
        for case_id, m in meta.items():
            # Pull role from the first row of the case (any version row has same alias).
            row0 = rows_by_case[case_id][0]
            role_cache[case_id] = _role_for_row(row0)

    sampled_cases: list[str] = []
    summary_per_vb: dict[str, dict[str, Any]] = {}
    for vb_id, ids in sorted(by_voicebank.items()):
        rng.shuffle(ids)
        if stratify_by_role:
            picked, per_role_counts = _sample_per_voicebank_stratified(
                case_ids=ids,
                case_roles=role_cache,
                target_count=per_voicebank,
                quota=role_quota,
                rng=rng,
            )
        else:
            picked = _sample_per_voicebank_uniform(
                case_ids=ids, target_count=per_voicebank, rng=rng
            )
            per_role_counts = {}
        sampled_cases.extend(picked)
        summary_per_vb[vb_id] = {
            "total_cases": len(ids),
            "sampled_cases": len(picked),
            "per_role_counts": per_role_counts,
        }

    sampled_set = set(sampled_cases)
    filtered: list[dict[str, Any]] = []
    for case_id in sorted(sampled_cases):
        # Keep all version rows for this case; preserve their stable order
        # (the build script writes them in version-definition order).
        filtered.extend(rows_by_case[case_id])

    summary = {
        "per_voicebank": summary_per_vb,
        "sampled_case_count": len(sampled_set),
        "sampled_row_count": len(filtered),
        "stratified": bool(stratify_by_role),
        "seed": int(seed),
        "role_quota": role_quota if stratify_by_role else {},
    }
    return filtered, summary


def _parse_role_quota_arg(raw: str) -> dict[str, float]:
    if not raw:
        return dict(DEFAULT_ROLE_QUOTA)
    if os.path.isfile(raw):
        with open(raw, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise SystemExit(f"role-quota JSON root must be an object: {raw}")
        return {str(k): float(v) for k, v in data.items()}
    # inline "role=value,role=value"
    out: dict[str, float] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise SystemExit(f"invalid role-quota token (expected role=value): {token}")
        role, value = token.split("=", 1)
        out[role.strip()] = float(value.strip())
    return out


def _write_rows(*, rows: list[dict[str, Any]], fieldnames: list[str], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    ap = argparse.ArgumentParser(description="Sample an A/B listening label CSV down to a manageable size.")
    ap.add_argument("--labels", required=True, help="Path to the full label CSV (built by build_ab_listening_set).")
    ap.add_argument("--out", required=True, help="Path to write the sampled CSV.")
    ap.add_argument("--per-voicebank", type=int, default=40, help="Target number of cases per voicebank (default: 40).")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling (default: 0).")
    ap.add_argument(
        "--stratify-by-role",
        action="store_true",
        help="Enable role-stratified sampling using DEFAULT_ROLE_QUOTA or --role-quota.",
    )
    ap.add_argument(
        "--role-quota",
        default="",
        help=(
            "Optional role quota override. Either a path to a JSON file mapping role -> weight, "
            "or an inline string like 'vc=0.35,v-cv=0.20,cv=0.20'."
        ),
    )
    ap.add_argument("--summary-out", default="", help="Optional path to write the sampling summary JSON.")
    args = ap.parse_args()

    fieldnames, rows = _read_rows(args.labels)
    if not rows:
        print(f"[sample] no rows in {args.labels}")
        return 1
    quota = _parse_role_quota_arg(args.role_quota) if args.stratify_by_role else {}

    filtered, summary = sample_label_rows(
        rows=rows,
        per_voicebank=int(args.per_voicebank),
        stratify_by_role=bool(args.stratify_by_role),
        role_quota=quota,
        seed=int(args.seed),
    )

    _write_rows(rows=filtered, fieldnames=fieldnames, path=args.out)
    print(
        f"[sample] wrote {args.out} cases={summary['sampled_case_count']} "
        f"rows={summary['sampled_row_count']} stratified={summary['stratified']}"
    )
    for vb_id, info in summary["per_voicebank"].items():
        line = f"  {vb_id}: sampled={info['sampled_cases']}/{info['total_cases']}"
        per_role = info.get("per_role_counts") or {}
        if per_role:
            top = sorted(per_role.items(), key=lambda kv: -kv[1])[:6]
            top_str = " ".join(f"{role}={n}" for role, n in top if n > 0)
            line += f" roles=({top_str})"
        print(line)

    summary_path = args.summary_out
    if not summary_path:
        base, ext = os.path.splitext(args.out)
        summary_path = base + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"[sample] wrote summary {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
