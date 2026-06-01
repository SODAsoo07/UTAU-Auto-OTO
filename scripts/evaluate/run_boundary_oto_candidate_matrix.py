"""Run boundary-gated OTO candidate evaluation over multiple gold cases.

The single-case evaluator (`apply_boundary_oto_candidates.py`) is useful for
one voicebank, but HSMM timing changes should not be promoted from a single
gold case. This runner executes a small config sweep over a cases JSON and
summarizes per-case/family deltas so a candidate can be judged by consistency.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class MatrixCase:
    case_id: str
    gold: str
    pred: str
    wav_dir: str
    language: str = "japanese"
    format_type: str = "cvvc"


@dataclass(frozen=True)
class MatrixConfig:
    name: str
    args: tuple[str, ...]


def load_cases(path: str) -> list[MatrixCase]:
    payload = _read_json(path)
    raw_cases = payload.get("cases") if isinstance(payload, Mapping) else None
    if not isinstance(raw_cases, list):
        raise ValueError(f"cases JSON must contain a list at 'cases': {path}")
    cases: list[MatrixCase] = []
    for idx, raw in enumerate(raw_cases):
        if not isinstance(raw, Mapping):
            raise ValueError(f"case {idx} is not an object")
        case_id = str(raw.get("case_id") or raw.get("name") or f"case_{idx}").strip()
        gold = str(raw.get("gold") or "").strip()
        pred = str(raw.get("pred") or "").strip()
        wav_dir = str(raw.get("wav_dir") or "").strip()
        if not case_id or not gold or not pred or not wav_dir:
            raise ValueError(f"case {idx} is missing case_id/gold/pred/wav_dir")
        cases.append(
            MatrixCase(
                case_id=case_id,
                gold=gold,
                pred=pred,
                wav_dir=wav_dir,
                language=str(raw.get("language") or "japanese"),
                format_type=str(raw.get("format_type") or "cvvc"),
            )
        )
    return cases


def validate_cases(cases: Sequence[MatrixCase], *, repo_root: Path) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for case in cases:
        missing = []
        for field in ("gold", "pred", "wav_dir"):
            value = getattr(case, field)
            if not _resolve_path(value, repo_root=repo_root).exists():
                missing.append(field)
        if missing:
            issues.append({"case_id": case.case_id, "missing": missing})
    return issues


def default_configs(preset: str) -> list[MatrixConfig]:
    name = str(preset or "safety_sweep").strip().lower().replace("-", "_")
    if name == "default_only":
        return [MatrixConfig("cv_vc_max60", ())]
    if name == "safety_sweep":
        return [
            MatrixConfig("cv_vc_max60", ()),
            MatrixConfig("cv_only_max60", ("--role-labels", "cv:vowel_onset")),
            MatrixConfig("vc_only_max60", ("--role-labels", "vc:vowel_end")),
            MatrixConfig("cv_vc_max80", ("--max-shift-ms", "80")),
            MatrixConfig("cv_vc_max60_margin05", ("--min-score-margin", "0.05")),
        ]
    raise ValueError(f"unknown preset: {preset}")


def load_configs(path: str, *, preset: str) -> list[MatrixConfig]:
    if not path:
        return default_configs(preset)
    payload = _read_json(path)
    raw_configs = payload.get("configs") if isinstance(payload, Mapping) else None
    if not isinstance(raw_configs, list):
        raise ValueError(f"configs JSON must contain a list at 'configs': {path}")
    out: list[MatrixConfig] = []
    for idx, raw in enumerate(raw_configs):
        if not isinstance(raw, Mapping):
            raise ValueError(f"config {idx} is not an object")
        name = str(raw.get("name") or raw.get("config") or f"config_{idx}").strip()
        raw_args = raw.get("args") or []
        if not isinstance(raw_args, list) or not all(isinstance(item, str) for item in raw_args):
            raise ValueError(f"config {name!r} args must be a list of strings")
        if not name:
            raise ValueError(f"config {idx} name is empty")
        out.append(MatrixConfig(name=name, args=tuple(raw_args)))
    return out


def build_apply_command(
    *,
    python_exe: str,
    case: MatrixCase,
    config: MatrixConfig,
    model: str,
    out_dir: Path,
    device: str,
    cutoff_end_mode: str,
    disable_acoustic_heuristics: bool,
    heuristic_weight: float,
) -> tuple[list[str], Path, Path]:
    case_dir = out_dir / _safe_path_name(config.name) / _safe_path_name(case.case_id)
    report_path = case_dir / "report.json"
    adjusted_path = case_dir / "adjusted.ini"
    cmd = [
        python_exe,
        "scripts/evaluate/apply_boundary_oto_candidates.py",
        "--pred",
        case.pred,
        "--gold",
        case.gold,
        "--wav-dir",
        case.wav_dir,
        "--model",
        model,
        "--device",
        device,
        "--cutoff-end-mode",
        cutoff_end_mode,
        "--language",
        case.language,
        "--format-type",
        case.format_type,
        "--out",
        str(adjusted_path),
        "--report",
        str(report_path),
    ]
    if disable_acoustic_heuristics:
        cmd.append("--disable-acoustic-heuristics")
    cmd.extend(["--heuristic-weight", str(float(heuristic_weight))])
    cmd.extend(config.args)
    return cmd, report_path, adjusted_path


def summarize_matrix(
    *,
    cases: Sequence[MatrixCase],
    configs: Sequence[MatrixConfig],
    report_map: Mapping[tuple[str, str], Mapping[str, Any]],
    cases_path: str,
    out_dir: str,
    model: str,
) -> dict[str, Any]:
    summaries = []
    for config in configs:
        case_rows = []
        for case in cases:
            report = report_map[(config.name, case.case_id)]
            case_rows.append(summarize_case_report(case, report))
        summaries.append(summarize_config(config, case_rows))
    return {
        "cases_path": cases_path,
        "out_dir": out_dir,
        "model": model,
        "configs": summaries,
        "selection_hints": build_selection_hints(summaries),
    }


def summarize_case_report(case: MatrixCase, report: Mapping[str, Any]) -> dict[str, Any]:
    diff = report.get("diff") if isinstance(report.get("diff"), Mapping) else {}
    anchor = diff.get("anchor_abs") if isinstance(diff.get("anchor_abs"), Mapping) else {}
    delta = anchor.get("delta") if isinstance(anchor.get("delta"), Mapping) else {}
    pre_abs = delta.get("pre_abs") if isinstance(delta.get("pre_abs"), Mapping) else {}
    by_family = delta.get("by_alias_family") if isinstance(delta.get("by_alias_family"), Mapping) else {}
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    adjusted_by_family = (
        summary.get("adjusted_by_family_label") if isinstance(summary.get("adjusted_by_family_label"), Mapping) else {}
    )
    score_maps = report.get("score_maps") if isinstance(report.get("score_maps"), Mapping) else {}
    after = diff.get("after") if isinstance(diff.get("after"), Mapping) else {}
    out = {
        "case_id": case.case_id,
        "rows_matched": int(after.get("rows_matched") or 0),
        "adjusted_rows": int(summary.get("adjusted_rows") or 0),
        "requested_wavs": int(score_maps.get("requested_wavs") or 0),
        "loaded_wavs": int(score_maps.get("loaded_wavs") or 0),
        "pre_abs_mae_delta": _metric(pre_abs, "MAE_ms_delta"),
        "pre_abs_median_delta": _metric(pre_abs, "abs_median_ms_delta"),
        "pre_abs_p90_delta": _metric(pre_abs, "abs_p90_ms_delta"),
    }
    for family in ("cv", "vc", "cv_head", "vv", "vcv"):
        family_delta = by_family.get(family) if isinstance(by_family.get(family), Mapping) else {}
        family_pre = family_delta.get("pre_abs") if isinstance(family_delta.get("pre_abs"), Mapping) else {}
        family_adjusted = adjusted_by_family.get(family) if isinstance(adjusted_by_family.get(family), Mapping) else {}
        out[f"{family}_mae_delta"] = _metric(family_pre, "MAE_ms_delta")
        out[f"{family}_p90_delta"] = _metric(family_pre, "abs_p90_ms_delta")
        out[f"{family}_adjusted_rows"] = sum(int(value or 0) for value in family_adjusted.values())
    return out


def summarize_config(config: MatrixConfig, case_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mae_values = [_float(row.get("pre_abs_mae_delta")) for row in case_rows]
    p90_values = [_float(row.get("pre_abs_p90_delta")) for row in case_rows]
    family = {}
    for family_name in ("cv", "vc", "cv_head", "vv", "vcv"):
        family_values = [
            _float(row.get(f"{family_name}_mae_delta"))
            for row in case_rows
            if row.get(f"{family_name}_mae_delta") is not None
        ]
        family_p90 = [
            _float(row.get(f"{family_name}_p90_delta"))
            for row in case_rows
            if row.get(f"{family_name}_p90_delta") is not None
        ]
        family_adjusted_rows = sum(int(row.get(f"{family_name}_adjusted_rows") or 0) for row in case_rows)
        if family_values:
            family[family_name] = {
                "mean_mae_delta": round(sum(family_values) / len(family_values), 3),
                "worst_mae_delta": round(max(family_values), 3),
                "improved_cases": sum(1 for value in family_values if value < 0.0),
                "nonworse_cases": sum(1 for value in family_values if value <= 0.0),
                "worst_p90_delta": round(max(family_p90), 3) if family_p90 else None,
                "adjusted_rows": family_adjusted_rows,
            }
    return {
        "config": config.name,
        "args": list(config.args),
        "cases": list(case_rows),
        "mean_mae_delta": round(sum(mae_values) / len(mae_values), 3) if mae_values else 0.0,
        "worst_mae_delta": round(max(mae_values), 3) if mae_values else 0.0,
        "improved_cases": sum(1 for value in mae_values if value < 0.0),
        "nonworse_cases": sum(1 for value in mae_values if value <= 0.0),
        "mean_p90_delta": round(sum(p90_values) / len(p90_values), 3) if p90_values else 0.0,
        "worst_p90_delta": round(max(p90_values), 3) if p90_values else 0.0,
        "by_family": family,
    }


def build_selection_hints(config_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_cases = max((len(item.get("cases", []) or []) for item in config_summaries), default=0)
    if total_cases <= 0:
        return {
            "safe_overall": [],
            "safe_p90": [],
            "safe_family_overall": {},
            "safe_family_p90": {},
            "ranked_p90_aware": [],
        }
    safe_overall = []
    safe_p90 = []
    safe_family_overall: dict[str, list[str]] = {}
    safe_family_p90: dict[str, list[str]] = {}
    ranked_p90_aware = []
    for item in config_summaries:
        config_name = str(item.get("config") or "")
        if int(item.get("nonworse_cases") or 0) == total_cases and float(item.get("worst_mae_delta") or 0.0) <= 0.0:
            safe_overall.append(config_name)
            if float(item.get("worst_p90_delta") or 0.0) <= 0.0:
                safe_p90.append(config_name)
        by_family = item.get("by_family") if isinstance(item.get("by_family"), Mapping) else {}
        for family_name, family_summary in by_family.items():
            if not isinstance(family_summary, Mapping):
                continue
            if int(family_summary.get("adjusted_rows") or 0) <= 0:
                continue
            if (
                int(family_summary.get("nonworse_cases") or 0) == total_cases
                and float(family_summary.get("worst_mae_delta") or 0.0) <= 0.0
            ):
                safe_family_overall.setdefault(str(family_name), []).append(config_name)
                if float(family_summary.get("worst_p90_delta") or 0.0) <= 0.0:
                    safe_family_p90.setdefault(str(family_name), []).append(config_name)
        ranked_p90_aware.append(_p90_aware_candidate_row(item))
    ranked_p90_aware.sort(
        key=lambda row: (
            float(row.get("p90_aware_score") or 0.0),
            -float(row.get("mean_improvement_ms") or 0.0),
            str(row.get("config") or ""),
        )
    )
    return {
        "safe_overall": safe_overall,
        "safe_p90": safe_p90,
        "safe_family_overall": safe_family_overall,
        "safe_family_p90": safe_family_p90,
        "ranked_p90_aware": ranked_p90_aware,
    }


def _p90_aware_candidate_row(item: Mapping[str, Any]) -> dict[str, Any]:
    config_name = str(item.get("config") or "")
    mean_mae_delta = float(item.get("mean_mae_delta") or 0.0)
    worst_mae_delta = float(item.get("worst_mae_delta") or 0.0)
    worst_p90_delta = float(item.get("worst_p90_delta") or 0.0)
    by_family = item.get("by_family") if isinstance(item.get("by_family"), Mapping) else {}
    adjusted_family_p90: dict[str, float] = {}
    for family_name, family_summary in by_family.items():
        if not isinstance(family_summary, Mapping):
            continue
        if int(family_summary.get("adjusted_rows") or 0) <= 0:
            continue
        p90 = family_summary.get("worst_p90_delta")
        if p90 is not None:
            adjusted_family_p90[str(family_name)] = float(p90)
    max_adjusted_family_p90 = max(adjusted_family_p90.values(), default=0.0)
    mean_improvement = max(0.0, -mean_mae_delta)
    p90_tail_penalty = max(0.0, worst_p90_delta) + (0.25 * max(0.0, max_adjusted_family_p90))
    mae_regression_penalty = max(0.0, worst_mae_delta) * 5.0
    # Lower is better. Keep the score interpretable: p90 tail dominates,
    # while mean improvement breaks ties between similarly safe candidates.
    score = p90_tail_penalty + mae_regression_penalty - (0.10 * mean_improvement)
    return {
        "config": config_name,
        "mean_mae_delta": round(mean_mae_delta, 3),
        "worst_mae_delta": round(worst_mae_delta, 3),
        "worst_p90_delta": round(worst_p90_delta, 3),
        "max_adjusted_family_p90_delta": round(max_adjusted_family_p90, 3),
        "p90_tail_penalty": round(p90_tail_penalty, 3),
        "mean_improvement_ms": round(mean_improvement, 3),
        "p90_aware_score": round(score, 3),
    }


def run_matrix(
    *,
    cases: Sequence[MatrixCase],
    configs: Sequence[MatrixConfig],
    model: str,
    out_dir: Path,
    device: str,
    cutoff_end_mode: str,
    python_exe: str,
    disable_acoustic_heuristics: bool,
    heuristic_weight: float,
    repo_root: Path,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    reports: dict[tuple[str, str], Mapping[str, Any]] = {}
    for config in configs:
        for case in cases:
            cmd, report_path, _adjusted_path = build_apply_command(
                python_exe=python_exe,
                case=case,
                config=config,
                model=model,
                out_dir=out_dir,
                device=device,
                cutoff_end_mode=cutoff_end_mode,
                disable_acoustic_heuristics=disable_acoustic_heuristics,
                heuristic_weight=heuristic_weight,
            )
            report_path.parent.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(
                cmd,
                cwd=str(repo_root),
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if completed.returncode != 0:
                tail = str(completed.stderr or "")[-2400:]
                raise RuntimeError(f"{config.name}/{case.case_id} failed with code {completed.returncode}:\n{tail}")
            reports[(config.name, case.case_id)] = _read_json(str(report_path))
    return reports


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, help="Cases JSON with case_id/gold/pred/wav_dir entries.")
    parser.add_argument("--model", required=True, help="Phoneme-boundary checkpoint path.")
    parser.add_argument("--out-dir", required=True, help="Matrix output directory.")
    parser.add_argument("--configs", default="", help="Optional configs JSON. Defaults to --preset.")
    parser.add_argument("--preset", choices=("safety_sweep", "default_only"), default="safety_sweep")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cutoff-end-mode", choices=("raw", "relative", "utau", "compatible"), default="utau")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for single-case evaluator.")
    parser.add_argument("--disable-acoustic-heuristics", action="store_true")
    parser.add_argument("--heuristic-weight", type=float, default=0.28)
    parser.add_argument("--skip-missing", action="store_true", help="Drop cases with missing gold/pred/wav_dir paths.")
    parser.add_argument("--summary-name", default="matrix_summary.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cases = load_cases(args.cases)
    issues = validate_cases(cases, repo_root=REPO_ROOT)
    if issues and not bool(args.skip_missing):
        raise SystemExit(json.dumps({"missing_case_paths": issues}, ensure_ascii=False, indent=2))
    if issues:
        missing_ids = {str(item.get("case_id")) for item in issues}
        cases = [case for case in cases if case.case_id not in missing_ids]
    configs = load_configs(args.configs, preset=args.preset)
    out_dir = Path(args.out_dir)
    reports = run_matrix(
        cases=cases,
        configs=configs,
        model=args.model,
        out_dir=out_dir,
        device=args.device,
        cutoff_end_mode=args.cutoff_end_mode,
        python_exe=args.python,
        disable_acoustic_heuristics=bool(args.disable_acoustic_heuristics),
        heuristic_weight=float(args.heuristic_weight),
        repo_root=REPO_ROOT,
    )
    summary = summarize_matrix(
        cases=cases,
        configs=configs,
        report_map=reports,
        cases_path=os.path.abspath(args.cases),
        out_dir=os.path.abspath(args.out_dir),
        model=os.path.abspath(args.model),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / args.summary_name
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_path(path: str, *, repo_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def _safe_path_name(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value or "").strip())
    return out or "unnamed"


def _metric(payload: Any, key: str) -> float | None:
    if not isinstance(payload, Mapping) or key not in payload:
        return None
    return _float(payload.get(key))


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _print_text(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(str(text).encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    raise SystemExit(main())
