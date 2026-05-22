from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.mfa_free_oto.acoustic_nucleus import AcousticNucleusConfig, relabel_vowel_nuclei_manifest
from core.mfa_free_oto.htk_lab import build_gold_manifest_from_htk_lab_dirs
from core.mfa_free_oto.metrics import boundary_error_metrics
from core.mfa_free_oto.review_overlay import write_review_html
from core.mfa_free_oto.runtime_inference import predict_wav


@dataclass(frozen=True)
class RowEval:
    row_id: str
    wav_name: str
    wav_path: str
    cv: dict
    nucleus: dict
    status: str
    catastrophic: bool
    one_step_shift: bool
    slot_warnings: tuple[str, ...]
    tags: tuple[str, ...]
    fallback_reason: str
    decoded_events: list[dict]
    label_events: list[dict]
    overlay_relpath: str | None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate No-MFA runtime accuracy against manual HTK .lab labels only (TextGrid excluded)."
    )
    parser.add_argument(
        "--lab-root",
        default=r"C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\dataset_staged\japanese\alignmented\CV\ryosuke_rentan\C4",
        help="Root directory containing paired .lab/.wav files.",
    )
    parser.add_argument(
        "--out-dir",
        default=r"C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\ml\eval\no_mfa_runtime_c4_lab",
        help="Output directory for JSON/HTML/overlay artifacts.",
    )
    parser.add_argument("--language", default="japanese")
    parser.add_argument("--format-type", default="CV")
    parser.add_argument("--encoder", default="acoustic_world_v1")
    parser.add_argument("--checkpoint", default="", help="Optional checkpoint. Empty means rule-based runtime path.")
    parser.add_argument("--baseline-encoder", default="acoustic_world_v1")
    parser.add_argument("--baseline-checkpoint", default="", help="Baseline checkpoint. Empty means rule-based baseline.")
    parser.add_argument("--device", default="", help="Optional torch device, e.g. cpu/cuda.")
    parser.add_argument("--boundary-tolerance-ms", type=float, default=40.0)
    parser.add_argument("--nucleus-tolerance-ms", type=float, default=50.0)
    parser.add_argument("--gate-max-mis-mapping-rate", type=float, default=0.70)
    parser.add_argument("--gate-max-cv-p90-ms", type=float, default=24.0)
    parser.add_argument("--gate-max-nucleus-p90-ms", type=float, default=40.0)
    parser.add_argument("--gate-max-hard-fail-rate", type=float, default=0.35)
    parser.add_argument("--max-overlays", type=int, default=80, help="Render overlays for worst N rows.")
    parser.add_argument(
        "--nucleus-reference",
        choices=("lab-midpoint", "acoustic"),
        default="lab-midpoint",
        help="Reference vowel_nucleus policy. acoustic recomputes nuclei from acoustic evidence and keeps lab midpoint diagnostics.",
    )
    parser.add_argument("--nucleus-encoder", default="", help="Encoder for acoustic nucleus reference. Defaults to --encoder.")
    parser.add_argument("--disable-slot-viterbi", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manual_gold_manifest.jsonl"

    rows, build_summary = build_gold_manifest_from_htk_lab_dirs(
        [args.lab_root],
        out_path=manifest_path,
        language=args.language,
        format_type=args.format_type,
        dataset_id="C4",
        relative_to=None,
    )
    if args.nucleus_reference == "acoustic":
        acoustic_manifest_path = out_dir / "manual_gold_manifest.acoustic_nucleus.jsonl"
        rows, nucleus_summary = relabel_vowel_nuclei_manifest(
            manifest_path,
            acoustic_manifest_path,
            encoder=(args.nucleus_encoder or args.encoder),
            device=(args.device or None),
            config=AcousticNucleusConfig(),
        )
        manifest_path = acoustic_manifest_path
    else:
        nucleus_summary = None

    if not rows:
        raise SystemExit(f"No .lab/.wav rows found: {args.lab_root}")

    per_row: list[RowEval] = []
    cv_ref_all: list[float] = []
    cv_pred_all: list[float] = []
    vn_ref_all: list[float] = []
    vn_pred_all: list[float] = []
    hard_fail = 0
    partial_fail = 0
    soft_warn = 0
    catastrophic = 0
    one_step_shift = 0
    rule_based = 0
    runtime_meta_snapshot: dict[str, object] = {}
    runtime_payload_by_row_id: dict[str, dict] = {}

    for row in rows:
        prediction = predict_wav(
            row["wav_path"],
            checkpoint_path=args.checkpoint or "",
            expected_phones=_expected_phones(row),
            encoder=args.encoder,
            device=(args.device or None),
            use_slot_viterbi=not args.disable_slot_viterbi,
        )
        fallback_reason = str(prediction.posterior.metadata.get("rule_fallback_reason") or "")
        if not runtime_meta_snapshot:
            runtime_meta_snapshot = dict(prediction.posterior.metadata or {})
        if bool(prediction.posterior.metadata.get("rule_based")):
            rule_based += 1
        decoded_events = [
            {
                "label": str(event.label),
                "selected_time_ms": float(event.selected_time_ms),
                "score": float(event.score),
                "expected_phone": str(event.expected_phone or ""),
                "frame_index": int(event.frame_index),
            }
            for event in prediction.decoded_events
        ]
        row_key = str(row.get("row_id") or row.get("wav_name") or "")
        runtime_payload_by_row_id[row_key] = {
            "row": row,
            "posterior": prediction.posterior,
            "decoded_events": decoded_events,
        }
        label_events = list(row.get("events") or [])
        ref_cv = _event_times(label_events, "cv_boundary")
        pred_cv = _pred_event_times(decoded_events, "cv_boundary")
        ref_vn = _event_times(label_events, "vowel_nucleus")
        pred_vn = _pred_event_times(decoded_events, "vowel_nucleus")

        cv_metric = boundary_error_metrics(ref_cv, pred_cv, tolerance_ms=float(args.boundary_tolerance_ms))
        vn_metric = boundary_error_metrics(ref_vn, pred_vn, tolerance_ms=float(args.nucleus_tolerance_ms))
        row_one_step_shift = _detect_one_step_shift(ref_cv, pred_cv)
        row_status = _row_status(cv_metric=cv_metric, vn_metric=vn_metric)
        row_catastrophic = row_status in {"hard_fail", "partial_fail"}

        if row_one_step_shift:
            one_step_shift += 1
        if row_status == "hard_fail":
            hard_fail += 1
        elif row_status == "partial_fail":
            partial_fail += 1
        elif row_status == "soft_warn":
            soft_warn += 1
        if row_catastrophic:
            catastrophic += 1

        cv_ref_all.extend(ref_cv)
        cv_pred_all.extend(pred_cv)
        vn_ref_all.extend(ref_vn)
        vn_pred_all.extend(pred_vn)

        per_row.append(
            RowEval(
                row_id=row_key,
                wav_name=str(row.get("wav_name") or ""),
                wav_path=str(row.get("wav_path") or ""),
                cv=cv_metric,
                nucleus=vn_metric,
                status=row_status,
                catastrophic=row_catastrophic,
                one_step_shift=row_one_step_shift,
                slot_warnings=tuple(prediction.slot_result.warnings if prediction.slot_result is not None else ()),
                tags=tuple(_row_tags(cv_metric=cv_metric, vn_metric=vn_metric, slot_warnings=tuple(prediction.slot_result.warnings if prediction.slot_result is not None else ()))),
                fallback_reason=fallback_reason,
                decoded_events=decoded_events,
                label_events=label_events,
                overlay_relpath=None,
            )
        )

    cv_global = boundary_error_metrics(cv_ref_all, cv_pred_all, tolerance_ms=float(args.boundary_tolerance_ms))
    vn_global = boundary_error_metrics(vn_ref_all, vn_pred_all, tolerance_ms=float(args.nucleus_tolerance_ms))
    rows_total = len(per_row)
    mismatch_rate = float(hard_fail + partial_fail) / float(max(1, rows_total))
    hard_fail_rate = float(hard_fail) / float(max(1, rows_total))
    partial_fail_rate = float(partial_fail) / float(max(1, rows_total))
    soft_warn_rate = float(soft_warn) / float(max(1, rows_total))
    one_step_shift_rate = float(one_step_shift) / float(max(1, rows_total))
    nucleus_miss_rate = _miss_rate(vn_global)
    baseline = _evaluate_runtime_summary(
        rows,
        encoder=args.baseline_encoder,
        checkpoint=args.baseline_checkpoint,
        device=(args.device or None),
        boundary_tolerance_ms=float(args.boundary_tolerance_ms),
        nucleus_tolerance_ms=float(args.nucleus_tolerance_ms),
        use_slot_viterbi=not args.disable_slot_viterbi,
    )

    rank = sorted(
        enumerate(per_row),
        key=lambda item: _severity(item[1]),
        reverse=True,
    )
    overlay_count = max(0, min(int(args.max_overlays), len(rank)))
    updated_rows: list[RowEval] = list(per_row)
    for idx, (row_index, row_eval) in enumerate(rank[:overlay_count], start=1):
        overlay_name = f"{idx:03d}_{_safe_name(row_eval.row_id)}.html"
        overlay_path = overlay_dir / overlay_name
        row_payload = next((row for row in rows if str(row.get("row_id") or row.get("wav_name")) == row_eval.row_id), None)
        if row_payload is None:
            continue
        pred_payload = runtime_payload_by_row_id.get(row_eval.row_id)
        if pred_payload is None:
            continue
        write_review_html(
            overlay_path,
            row_payload,
            posterior=pred_payload["posterior"],
            decoded_events=pred_payload["decoded_events"],
            width=1360,
        )
        updated_rows[row_index] = RowEval(
            row_id=row_eval.row_id,
            wav_name=row_eval.wav_name,
            wav_path=row_eval.wav_path,
            cv=row_eval.cv,
            nucleus=row_eval.nucleus,
            status=row_eval.status,
            catastrophic=row_eval.catastrophic,
            one_step_shift=row_eval.one_step_shift,
            slot_warnings=row_eval.slot_warnings,
            tags=row_eval.tags,
            fallback_reason=row_eval.fallback_reason,
            decoded_events=row_eval.decoded_events,
            label_events=row_eval.label_events,
            overlay_relpath=f"overlays/{overlay_name}",
        )
    per_row = updated_rows

    result = {
        "dataset": {
            "lab_root": str(Path(args.lab_root).resolve()),
            "manifest_path": str(manifest_path),
            "rows": rows_total,
            "build_summary": {
                "rows": int(build_summary.rows),
                "labelled_rows": int(build_summary.labelled_rows),
                "skipped_without_wav": int(build_summary.skipped_without_wav),
                "dropped_segments": int(build_summary.dropped_segments),
            },
            "nucleus_reference": args.nucleus_reference,
            "nucleus_summary": nucleus_summary.__dict__ if nucleus_summary is not None else None,
        },
        "runtime": {
            "encoder": args.encoder,
            "checkpoint": args.checkpoint or "",
            "rule_based_rows": int(rule_based),
            "slot_viterbi_enabled": bool(not args.disable_slot_viterbi),
            "metadata_snapshot": runtime_meta_snapshot,
        },
        "tolerances_ms": {
            "cv_boundary": float(args.boundary_tolerance_ms),
            "vowel_nucleus": float(args.nucleus_tolerance_ms),
        },
        "metrics": {
            "mis_mapping_rate": mismatch_rate,
            "hard_fail_rate": hard_fail_rate,
            "partial_fail_rate": partial_fail_rate,
            "soft_warn_rate": soft_warn_rate,
            "one_step_shift_rate": one_step_shift_rate,
            "vc_pre_ge_80ms_rate": _vc_pre_ge_80ms_rate_from_rows(per_row),
            "nucleus_miss_rate": nucleus_miss_rate,
            "cv_match_rate": _match_rate(cv_global),
            "vowel_nucleus_match_rate": _match_rate(vn_global),
            "cv_miss_rate": _miss_rate(cv_global),
            "vowel_nucleus_miss_rate": _miss_rate(vn_global),
            "cv_boundary": cv_global,
            "vowel_nucleus": vn_global,
            "row_count": rows_total,
        },
        "gate": _gate_report(
            {
                "mis_mapping_rate": mismatch_rate,
                "cv_p90_error_ms": float(cv_global.get("p90_error_ms") or 0.0),
                "nucleus_p90_error_ms": float(vn_global.get("p90_error_ms") or 0.0),
                "hard_fail_rate": hard_fail_rate,
            },
            {
                "max_mis_mapping_rate": float(args.gate_max_mis_mapping_rate),
                "max_cv_p90_ms": float(args.gate_max_cv_p90_ms),
                "max_nucleus_p90_ms": float(args.gate_max_nucleus_p90_ms),
                "max_hard_fail_rate": float(args.gate_max_hard_fail_rate),
            },
        ),
        "baseline": baseline,
        "comparison": _comparison_from_baseline(
            metrics={
                "mis_mapping_rate": mismatch_rate,
                "one_step_shift_rate": one_step_shift_rate,
                "vc_pre_ge_80ms_rate": _vc_pre_ge_80ms_rate_from_rows(per_row),
                "nucleus_miss_rate": nucleus_miss_rate,
                "hard_fail_rate": hard_fail_rate,
            },
            baseline=(baseline.get("metrics") or {}),
        ),
        "status_report": _status_report(per_row),
        "rows": [_row_to_dict(item) for item in per_row],
        "note": "TextGrid is intentionally excluded. Only manual HTK .lab labels are used.",
    }

    metrics_json = out_dir / "metrics.json"
    metrics_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_html(out_dir / "summary.html", result)
    print(json.dumps({"metrics_json": str(metrics_json), "summary_html": str(out_dir / "summary.html")}, ensure_ascii=False))
    return 0


def _expected_phones(row: dict) -> list[str]:
    out: list[str] = []
    for item in row.get("expected_phones") or []:
        if isinstance(item, dict):
            phone = str(item.get("phone") or "").strip()
        else:
            phone = str(item or "").strip()
        if phone:
            out.append(phone)
    return out


def _event_times(events: Sequence[dict], label: str) -> list[float]:
    out: list[float] = []
    for event in events:
        if str(event.get("label") or "") != label:
            continue
        out.append(float(event.get("time_ms", 0.0)))
    return out


def _pred_event_times(events: Sequence[dict], label: str) -> list[float]:
    return [float(event["selected_time_ms"]) for event in events if str(event.get("label") or "") == label]


def _row_status(*, cv_metric: dict, vn_metric: dict) -> str:
    cv_ref = int(cv_metric.get("reference_count", 0))
    vn_ref = int(vn_metric.get("reference_count", 0))
    cv_hit = int(cv_metric.get("count", 0))
    vn_hit = int(vn_metric.get("count", 0))
    cv_hit_rate = float(cv_hit) / float(max(1, cv_ref))
    vn_hit_rate = float(vn_hit) / float(max(1, vn_ref))
    cv_hard_miss = cv_ref > 0 and cv_hit <= 0
    vn_hard_miss = vn_ref > 0 and vn_hit <= 0
    if cv_hard_miss or vn_hard_miss:
        return "hard_fail"
    severe_cv = cv_ref >= 2 and cv_hit_rate < 0.5
    severe_vn = vn_ref >= 2 and vn_hit_rate < 0.5
    if severe_cv or severe_vn:
        return "partial_fail"
    cv_warn = int(cv_metric.get("missed", 0)) > 0 or int(cv_metric.get("false_positive", 0)) > 0
    vn_warn = int(vn_metric.get("missed", 0)) > 0 or int(vn_metric.get("false_positive", 0)) > 0
    if cv_warn or vn_warn:
        return "soft_warn"
    return "ok"


def _miss_rate(metric: dict) -> float:
    ref = max(1, int(metric.get("reference_count", 0)))
    missed = int(metric.get("missed", 0))
    return float(missed) / float(ref)


def _match_rate(metric: dict) -> float:
    ref = max(1, int(metric.get("reference_count", 0)))
    matched = int(metric.get("count", 0))
    return float(matched) / float(ref)


def _detect_one_step_shift(reference_ms: Sequence[float], predicted_ms: Sequence[float]) -> bool:
    refs = [float(v) for v in reference_ms]
    preds = [float(v) for v in predicted_ms]
    if len(refs) < 3 or len(preds) < 3:
        return False
    offsets: list[int] = []
    for pred_idx, pred_t in enumerate(preds):
        nearest = min(range(len(refs)), key=lambda idx: abs(refs[idx] - pred_t))
        offsets.append(int(nearest) - int(pred_idx))
    pos = sum(1 for off in offsets if off == 1)
    neg = sum(1 for off in offsets if off == -1)
    dominant = max(pos, neg)
    return dominant >= max(3, int(len(offsets) * 0.6))


def _severity(row: RowEval) -> float:
    cv_miss = int(row.cv.get("missed", 0))
    cv_fp = int(row.cv.get("false_positive", 0))
    vn_miss = int(row.nucleus.get("missed", 0))
    vn_fp = int(row.nucleus.get("false_positive", 0))
    cv_p90 = float(row.cv.get("p90_error_ms") or 0.0)
    vn_p90 = float(row.nucleus.get("p90_error_ms") or 0.0)
    return (cv_miss * 4.0) + (cv_fp * 3.5) + (vn_miss * 3.0) + (vn_fp * 2.5) + (cv_p90 * 0.02) + (vn_p90 * 0.015)


def _safe_name(text: str) -> str:
    raw = str(text or "").strip().lower()
    if not raw:
        return "row"
    out = []
    for ch in raw:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "row"


def _row_to_dict(row: RowEval) -> dict:
    return {
        "row_id": row.row_id,
        "wav_name": row.wav_name,
        "wav_path": row.wav_path,
        "cv_boundary": row.cv,
        "vowel_nucleus": row.nucleus,
        "status": row.status,
        "catastrophic": row.catastrophic,
        "one_step_shift": row.one_step_shift,
        "slot_warnings": list(row.slot_warnings),
        "tags": list(row.tags),
        "fallback_reason": row.fallback_reason,
        "overlay": row.overlay_relpath,
    }


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f}"


def _write_summary_html(path: Path, result: dict) -> None:
    metrics = result.get("metrics") or {}
    comparison = result.get("comparison") or {}
    rows = result.get("rows") or []
    top_rows = sorted(
        rows,
        key=lambda row: (
            int((row.get("cv_boundary") or {}).get("missed", 0)) + int((row.get("vowel_nucleus") or {}).get("missed", 0)),
            float((row.get("cv_boundary") or {}).get("p90_error_ms") or 0.0),
        ),
        reverse=True,
    )[:80]
    table_rows: list[str] = []
    for row in top_rows:
        overlay = row.get("overlay") or ""
        link = f'<a href="{html.escape(overlay)}">open</a>' if overlay else "-"
        cv = row.get("cv_boundary") or {}
        vn = row.get("vowel_nucleus") or {}
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('row_id') or ''))}</td>"
            f"<td>{html.escape(str(row.get('status') or ''))}</td>"
            f"<td>{int(row.get('catastrophic') is True)}</td>"
            f"<td>{int(row.get('one_step_shift') is True)}</td>"
            f"<td>{int(cv.get('missed', 0))}/{int(cv.get('reference_count', 0))}</td>"
            f"<td>{_fmt(cv.get('median_error_ms'))}</td>"
            f"<td>{_fmt(cv.get('p90_error_ms'))}</td>"
            f"<td>{int(vn.get('missed', 0))}/{int(vn.get('reference_count', 0))}</td>"
            f"<td>{_fmt(vn.get('median_error_ms'))}</td>"
            f"<td>{_fmt(vn.get('p90_error_ms'))}</td>"
            f"<td>{link}</td>"
            "</tr>"
        )
    status_rows = result.get("status_report", {}).get("top_rows", [])
    status_list = "".join(
        f"<li><b>{html.escape(str(item.get('status') or ''))}</b> - {html.escape(str(item.get('row_id') or ''))} "
        f"(tags: {html.escape(','.join(item.get('tags') or []))})</li>"
        for item in status_rows
    )
    body = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>No-MFA Runtime C4 Eval</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; margin: 20px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ border: 1px solid #ddd; padding: 6px; text-align: left; }}
th {{ background: #f3f4f6; }}
.kpi {{ display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 10px; margin-bottom: 14px; }}
.box {{ border: 1px solid #ddd; padding: 10px; background: #fafafa; }}
.v {{ font-size: 20px; font-weight: 700; }}
code {{ background: #f3f4f6; padding: 1px 4px; }}
</style>
</head>
<body>
<h1>No-MFA Runtime Accuracy (HTK .lab only)</h1>
<p>TextGrid는 제외됨. 평가 라벨 소스는 수동 <code>.lab</code>만 사용.</p>
<div class="kpi">
  <div class="box"><div>rows</div><div class="v">{int(metrics.get("row_count", 0))}</div></div>
  <div class="box"><div>mis_mapping_rate</div><div class="v">{float(metrics.get("mis_mapping_rate", 0.0)):.3f}</div></div>
  <div class="box"><div>hard_fail_rate</div><div class="v">{float(metrics.get("hard_fail_rate", 0.0)):.3f}</div></div>
  <div class="box"><div>partial_fail_rate</div><div class="v">{float(metrics.get("partial_fail_rate", 0.0)):.3f}</div></div>
  <div class="box"><div>soft_warn_rate</div><div class="v">{float(metrics.get("soft_warn_rate", 0.0)):.3f}</div></div>
  <div class="box"><div>one_step_shift_rate</div><div class="v">{float(metrics.get("one_step_shift_rate", 0.0)):.3f}</div></div>
  <div class="box"><div>nucleus_miss_rate</div><div class="v">{float(metrics.get("nucleus_miss_rate", 0.0)):.3f}</div></div>
  <div class="box"><div>cv_match_rate</div><div class="v">{float(metrics.get("cv_match_rate", 0.0)):.3f}</div></div>
  <div class="box"><div>vowel_nucleus_match_rate</div><div class="v">{float(metrics.get("vowel_nucleus_match_rate", 0.0)):.3f}</div></div>
  <div class="box"><div>cv median err (ms)</div><div class="v">{_fmt((metrics.get("cv_boundary") or {}).get("median_error_ms"))}</div></div>
  <div class="box"><div>cv p90 err (ms)</div><div class="v">{_fmt((metrics.get("cv_boundary") or {}).get("p90_error_ms"))}</div></div>
  <div class="box"><div>nucleus median err (ms)</div><div class="v">{_fmt((metrics.get("vowel_nucleus") or {}).get("median_error_ms"))}</div></div>
  <div class="box"><div>nucleus p90 err (ms)</div><div class="v">{_fmt((metrics.get("vowel_nucleus") or {}).get("p90_error_ms"))}</div></div>
</div>
<h2>Gate</h2>
<p>pass: <b>{html.escape(str((result.get("gate") or {}).get("pass")))} </b></p>
<h2>Baseline Compare (rule-only vs world_v1-light)</h2>
<p>
  mis_mapping_rate delta: <b>{_fmt(comparison.get("delta_mis_mapping_rate"))}</b>,
  one_step_shift_rate delta: <b>{_fmt(comparison.get("delta_one_step_shift_rate"))}</b>,
  vc_pre_ge_80ms_rate delta: <b>{_fmt(comparison.get("delta_vc_pre_ge_80ms_rate"))}</b>,
  nucleus_miss_rate delta: <b>{_fmt(comparison.get("delta_nucleus_miss_rate"))}</b>
</p>
<h2>Worst Rows</h2>
<table>
  <thead>
    <tr>
      <th>row_id</th><th>status</th><th>catastrophic</th><th>one_step_shift</th>
      <th>cv missed/ref</th><th>cv median</th><th>cv p90</th>
      <th>nucleus missed/ref</th><th>nucleus median</th><th>nucleus p90</th><th>overlay</th>
    </tr>
  </thead>
  <tbody>
    {"".join(table_rows)}
  </tbody>
</table>
<h2>Status Top Rows</h2>
<ul>
{status_list}
</ul>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _row_tags(*, cv_metric: dict, vn_metric: dict, slot_warnings: Sequence[str]) -> list[str]:
    tags: list[str] = []
    if int(cv_metric.get("missed", 0)) > 0:
        tags.append("cv_missed")
    if int(vn_metric.get("missed", 0)) > 0:
        tags.append("nucleus_missed")
    if int(cv_metric.get("false_positive", 0)) > 0:
        tags.append("cv_false_positive")
    if int(vn_metric.get("false_positive", 0)) > 0:
        tags.append("nucleus_false_positive")
    cv_p90 = float(cv_metric.get("p90_error_ms") or 0.0)
    vn_p90 = float(vn_metric.get("p90_error_ms") or 0.0)
    if cv_p90 >= 35.0:
        tags.append("late_cv")
    if vn_p90 >= 45.0:
        tags.append("late_nucleus")
    for warning in slot_warnings:
        prefix = str(warning).split(":", 1)[0]
        if prefix == "tight_gap":
            tags.append("gap_violation")
        elif prefix == "weak_flux_slot":
            tags.append("weak_flux")
        elif prefix == "weak_voicing_slot":
            tags.append("weak_voicing")
        elif prefix == "low_score_slot":
            tags.append("low_posterior")
    uniq: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        uniq.append(tag)
    return uniq


def _status_report(rows: Sequence[RowEval]) -> dict:
    by_status: dict[str, list[RowEval]] = {"hard_fail": [], "partial_fail": [], "soft_warn": [], "ok": []}
    for row in rows:
        by_status.setdefault(row.status, []).append(row)
    status_counts = {key: len(items) for key, items in by_status.items()}
    top_rows: list[dict] = []
    for status in ("hard_fail", "partial_fail", "soft_warn"):
        ranked = sorted(
            by_status.get(status, []),
            key=_severity,
            reverse=True,
        )
        for row in ranked[:5]:
            top_rows.append(
                {
                    "status": status,
                    "row_id": row.row_id,
                    "tags": list(row.tags),
                    "cv_p90_ms": float(row.cv.get("p90_error_ms") or 0.0),
                    "vn_p90_ms": float(row.nucleus.get("p90_error_ms") or 0.0),
                    "overlay": row.overlay_relpath,
                }
            )
    return {"status_counts": status_counts, "top_rows": top_rows}


def _gate_report(values: dict[str, float], thresholds: dict[str, float]) -> dict:
    checks = {
        "mis_mapping_rate": float(values["mis_mapping_rate"]) <= float(thresholds["max_mis_mapping_rate"]),
        "cv_p90_error_ms": float(values["cv_p90_error_ms"]) <= float(thresholds["max_cv_p90_ms"]),
        "nucleus_p90_error_ms": float(values["nucleus_p90_error_ms"]) <= float(thresholds["max_nucleus_p90_ms"]),
        "hard_fail_rate": float(values["hard_fail_rate"]) <= float(thresholds["max_hard_fail_rate"]),
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "values": values,
        "thresholds": thresholds,
    }


def _evaluate_runtime_summary(
    rows: Sequence[dict],
    *,
    encoder: str,
    checkpoint: str,
    device: str | None,
    boundary_tolerance_ms: float,
    nucleus_tolerance_ms: float,
    use_slot_viterbi: bool,
) -> dict:
    cv_ref_all: list[float] = []
    cv_pred_all: list[float] = []
    vn_ref_all: list[float] = []
    vn_pred_all: list[float] = []
    rule_based = 0
    one_step_shift = 0
    hard_fail = 0
    partial_fail = 0
    per_row: list[RowEval] = []
    meta_snapshot: dict[str, object] = {}
    for row in rows:
        prediction = predict_wav(
            row["wav_path"],
            checkpoint_path=checkpoint or "",
            expected_phones=_expected_phones(row),
            encoder=encoder,
            device=device,
            use_slot_viterbi=use_slot_viterbi,
        )
        if not meta_snapshot:
            meta_snapshot = dict(prediction.posterior.metadata or {})
        if bool(prediction.posterior.metadata.get("rule_based")):
            rule_based += 1
        decoded_events = [
            {
                "label": str(event.label),
                "selected_time_ms": float(event.selected_time_ms),
            }
            for event in prediction.decoded_events
        ]
        label_events = list(row.get("events") or [])
        ref_cv = _event_times(label_events, "cv_boundary")
        pred_cv = _pred_event_times(decoded_events, "cv_boundary")
        ref_vn = _event_times(label_events, "vowel_nucleus")
        pred_vn = _pred_event_times(decoded_events, "vowel_nucleus")
        cv_metric = boundary_error_metrics(ref_cv, pred_cv, tolerance_ms=float(boundary_tolerance_ms))
        vn_metric = boundary_error_metrics(ref_vn, pred_vn, tolerance_ms=float(nucleus_tolerance_ms))
        status = _row_status(cv_metric=cv_metric, vn_metric=vn_metric)
        if status == "hard_fail":
            hard_fail += 1
        elif status == "partial_fail":
            partial_fail += 1
        if _detect_one_step_shift(ref_cv, pred_cv):
            one_step_shift += 1
        cv_ref_all.extend(ref_cv)
        cv_pred_all.extend(pred_cv)
        vn_ref_all.extend(ref_vn)
        vn_pred_all.extend(pred_vn)
        per_row.append(
            RowEval(
                row_id=str(row.get("row_id") or row.get("wav_name") or ""),
                wav_name=str(row.get("wav_name") or ""),
                wav_path=str(row.get("wav_path") or ""),
                cv=cv_metric,
                nucleus=vn_metric,
                status=status,
                catastrophic=status in {"hard_fail", "partial_fail"},
                one_step_shift=False,
                slot_warnings=(),
                tags=(),
                fallback_reason=str(prediction.posterior.metadata.get("rule_fallback_reason") or ""),
                decoded_events=[],
                label_events=[],
                overlay_relpath=None,
            )
        )
    rows_total = len(rows)
    cv_global = boundary_error_metrics(cv_ref_all, cv_pred_all, tolerance_ms=float(boundary_tolerance_ms))
    vn_global = boundary_error_metrics(vn_ref_all, vn_pred_all, tolerance_ms=float(nucleus_tolerance_ms))
    return {
        "runtime": {
            "encoder": encoder,
            "checkpoint": checkpoint or "",
            "rule_based_rows": int(rule_based),
            "metadata_snapshot": meta_snapshot,
        },
        "metrics": {
            "mis_mapping_rate": float(hard_fail + partial_fail) / float(max(1, rows_total)),
            "one_step_shift_rate": float(one_step_shift) / float(max(1, rows_total)),
            "nucleus_miss_rate": _miss_rate(vn_global),
            "vc_pre_ge_80ms_rate": _vc_pre_ge_80ms_rate_from_rows(per_row),
            "hard_fail_rate": float(hard_fail) / float(max(1, rows_total)),
        },
    }


def _comparison_from_baseline(*, metrics: dict[str, float], baseline: dict[str, float]) -> dict:
    return {
        "target": metrics,
        "baseline": baseline,
        "delta_mis_mapping_rate": float(metrics.get("mis_mapping_rate", 0.0)) - float(baseline.get("mis_mapping_rate", 0.0)),
        "delta_one_step_shift_rate": float(metrics.get("one_step_shift_rate", 0.0))
        - float(baseline.get("one_step_shift_rate", 0.0)),
        "delta_vc_pre_ge_80ms_rate": float(metrics.get("vc_pre_ge_80ms_rate", 0.0))
        - float(baseline.get("vc_pre_ge_80ms_rate", 0.0)),
        "delta_nucleus_miss_rate": float(metrics.get("nucleus_miss_rate", 0.0))
        - float(baseline.get("nucleus_miss_rate", 0.0)),
        "delta_hard_fail_rate": float(metrics.get("hard_fail_rate", 0.0)) - float(baseline.get("hard_fail_rate", 0.0)),
    }


def _vc_pre_ge_80ms_rate_from_rows(rows: Sequence[RowEval]) -> float:
    if not rows:
        return 0.0
    flagged = 0
    for row in rows:
        if any(tag in {"gap_violation"} for tag in row.tags):
            flagged += 1
    return float(flagged) / float(max(1, len(rows)))


if __name__ == "__main__":
    raise SystemExit(main())
