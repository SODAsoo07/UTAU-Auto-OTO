from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import io
import json
import math
import os
import random
import re
import traceback
from collections import defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from core.coarse_crnn.alias_role import classify_alias_role, normalize_role
from core.coarse_crnn.audio import load_wav_mono, log_mel_spectrogram
from core.coarse_crnn.boundary_generator import generate_oto_with_boundary_decoder
from core.coarse_crnn.boundary_targets import (
    _infer_alias_type as _boundary_alias_type,
)
from core.coarse_crnn.boundary_targets import (
    _infer_transition_type as _boundary_transition_type,
)
from core.coarse_crnn.lang import normalize_language
from core.oto_file_utils import parse_oto_line, read_text_with_fallback
from ml.scripts.coarse_crnn.auto_oto import (
    DEFAULT_MODEL_PATH,
    VARIANT_PRESETS,
    apply_env_preset,
    restore_env,
)


ANCHOR_KEYS: tuple[str, ...] = (
    "offset_abs",
    "overlap_abs",
    "pre_abs",
    "cons_abs",
    "cutoff_abs",
)
TRANSITION_ROLES: frozenset[str] = frozenset({"vc", "vv", "v-cv"})
SILENCE_ALIASES: frozenset[str] = frozenset({"r", "br", "pau", "sil", "rest"})


@dataclass(frozen=True)
class VoicebankCase:
    language: str
    format_type: str
    bank_name: str
    bank_dir: str
    gt_oto_path: str
    wav_count: int

    @property
    def group_key(self) -> str:
        return f"{self.language}/{self.format_type}"

    @property
    def case_id(self) -> str:
        return f"{self.language}/{self.format_type}/{self.bank_name}"


@dataclass(frozen=True)
class OtoAnchorsAbs:
    offset_abs: float
    overlap_abs: float
    pre_abs: float
    cons_abs: float
    cutoff_abs: float


@dataclass(frozen=True)
class OtoRow:
    wav_rel: str
    wav_norm: str
    alias: str
    occ_idx: int
    line_index: int
    role: str
    n_like: bool
    offset: float
    cons: float
    cutoff: float
    pre: float
    ovl: float
    anchors: OtoAnchorsAbs


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Sample voicebanks from dataset_staged by language/format and compare "
            "auto-generated oto.ini against reference oto.ini with visual error plots."
        )
    )
    ap.add_argument("--dataset-staged", default="dataset_staged")
    ap.add_argument("--per-group", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260516)
    ap.add_argument(
        "--variant",
        default="F3",
        choices=sorted(VARIANT_PRESETS.keys()),
        help="Auto-OTO env preset applied during generation.",
    )
    ap.add_argument("--model", default="", help=f"Boundary scorer model path. Default: {DEFAULT_MODEL_PATH}")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-banks", type=int, default=0, help="Optional hard cap for sampled banks.")
    ap.add_argument("--max-plots-per-bank", type=int, default=8)
    ap.add_argument("--weak-percentile", type=float, default=20.0)
    ap.add_argument("--skip-generation", action="store_true", help="Use existing auto OTO file instead of generating.")
    ap.add_argument(
        "--existing-auto-name",
        default="oto_auto_ml.ini",
        help="Existing auto OTO filename used when --skip-generation is on.",
    )
    ap.add_argument("--out-root", default=os.path.join("ml", "eval", "oto_visual_benchmark"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    staged_root = os.path.abspath(str(args.dataset_staged))
    if not os.path.isdir(staged_root):
        raise SystemExit(f"dataset_staged not found: {staged_root}")
    model_path = _resolve_model_path(str(args.model))

    cases = discover_voicebank_cases(staged_root)
    if not cases:
        raise SystemExit("No valid voicebanks found (need oto.ini + wav files).")

    selected = sample_cases_by_group(
        cases=cases,
        per_group=max(1, int(args.per_group)),
        seed=int(args.seed),
    )
    if int(args.max_banks) > 0:
        selected = selected[: int(args.max_banks)]
    if not selected:
        raise SystemExit("No sampled voicebanks.")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.abspath(os.path.join(str(args.out_root), f"{stamp}_sample{len(selected)}_{args.variant.lower()}"))
    os.makedirs(run_root, exist_ok=True)
    banks_root = os.path.join(run_root, "banks")
    os.makedirs(banks_root, exist_ok=True)

    run_meta = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset_staged": staged_root,
        "variant": str(args.variant),
        "model_path": model_path,
        "device": str(args.device),
        "per_group": int(args.per_group),
        "seed": int(args.seed),
        "max_banks": int(args.max_banks),
        "max_plots_per_bank": int(args.max_plots_per_bank),
        "weak_percentile": float(args.weak_percentile),
        "skip_generation": bool(args.skip_generation),
        "existing_auto_name": str(args.existing_auto_name),
        "sampled_case_ids": [case.case_id for case in selected],
    }
    _write_json(os.path.join(run_root, "run_meta.json"), run_meta)

    reports: list[dict[str, Any]] = []
    for idx, case in enumerate(selected, start=1):
        if bool(args.verbose):
            _safe_console(f"[{idx}/{len(selected)}] {case.case_id}")
        case_report = run_case_benchmark(
            case=case,
            case_index=idx,
            case_count=len(selected),
            banks_root=banks_root,
            variant=str(args.variant),
            model_path=model_path,
            device=str(args.device),
            weak_percentile=float(args.weak_percentile),
            max_plots=max(1, int(args.max_plots_per_bank)),
            skip_generation=bool(args.skip_generation),
            existing_auto_name=str(args.existing_auto_name),
            verbose=bool(args.verbose),
        )
        reports.append(case_report)

    summary = {
        "run_meta": run_meta,
        "case_reports": reports,
        "aggregate": aggregate_reports(reports),
    }
    summary_path = os.path.join(run_root, "summary.json")
    _write_json(summary_path, summary)
    write_bank_summary_csv(os.path.join(run_root, "bank_summary.csv"), reports)
    write_index_html(os.path.join(run_root, "index.html"), summary)

    _safe_console(json.dumps({"run_root": run_root, "summary_json": summary_path}, ensure_ascii=False, indent=2))
    return 0


def discover_voicebank_cases(dataset_staged_root: str) -> list[VoicebankCase]:
    root = Path(dataset_staged_root)
    out: list[VoicebankCase] = []
    for lang_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        lang = normalize_language(lang_dir.name)
        if lang not in {"korean", "japanese"}:
            continue
        for fmt_dir in sorted([p for p in lang_dir.iterdir() if p.is_dir()]):
            fmt = str(fmt_dir.name).strip().lower()
            for bank_dir in sorted([p for p in fmt_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
                gt_oto = bank_dir / "oto.ini"
                if not gt_oto.is_file():
                    continue
                wav_count = len(list(bank_dir.glob("*.wav")))
                if wav_count <= 0:
                    continue
                out.append(
                    VoicebankCase(
                        language=str(lang),
                        format_type=fmt,
                        bank_name=bank_dir.name,
                        bank_dir=str(bank_dir.resolve()),
                        gt_oto_path=str(gt_oto.resolve()),
                        wav_count=int(wav_count),
                    )
                )
    return out


def sample_cases_by_group(*, cases: list[VoicebankCase], per_group: int, seed: int) -> list[VoicebankCase]:
    rng = random.Random(int(seed))
    grouped: dict[tuple[str, str], list[VoicebankCase]] = defaultdict(list)
    for case in cases:
        grouped[(case.language, case.format_type)].append(case)
    selected: list[VoicebankCase] = []
    for key in sorted(grouped.keys()):
        rows = sorted(grouped[key], key=lambda item: item.bank_name.lower())
        if len(rows) <= int(per_group):
            selected.extend(rows)
            continue
        picked_idx = sorted(rng.sample(range(len(rows)), int(per_group)))
        selected.extend([rows[i] for i in picked_idx])
    selected.sort(key=lambda item: (item.language, item.format_type, item.bank_name.lower()))
    return selected


def run_case_benchmark(
    *,
    case: VoicebankCase,
    case_index: int,
    case_count: int,
    banks_root: str,
    variant: str,
    model_path: str,
    device: str,
    weak_percentile: float,
    max_plots: int,
    skip_generation: bool,
    existing_auto_name: str,
    verbose: bool,
) -> dict[str, Any]:
    case_dir = os.path.join(
        banks_root,
        f"{case.language}__{case.format_type}",
        _slugify(case.bank_name),
    )
    os.makedirs(case_dir, exist_ok=True)
    auto_oto_path = os.path.join(case_dir, "auto_oto.ini")
    generation = {
        "status": "skipped" if skip_generation else "pending",
        "processed": 0,
        "total": 0,
        "errors": [],
        "source_mode": "existing_auto" if skip_generation else "generated",
        "source_path": "",
    }

    try:
        if skip_generation:
            candidate = os.path.join(case.bank_dir, str(existing_auto_name))
            if not os.path.isfile(candidate):
                raise FileNotFoundError(f"existing auto OTO missing: {candidate}")
            generation["status"] = "ok"
            generation["source_path"] = os.path.abspath(candidate)
            auto_oto_path = os.path.abspath(candidate)
        else:
            if verbose:
                _safe_console(
                    f"  - generate ({case_index}/{case_count}) {case.case_id} -> {auto_oto_path}",
                )
            processed, total, errors = generate_auto_oto_for_case(
                case=case,
                out_path=auto_oto_path,
                variant=variant,
                model_path=model_path,
                device=device,
            )
            generation["processed"] = int(processed)
            generation["total"] = int(total)
            generation["errors"] = list(errors)[:100]
            generation["status"] = "ok" if int(processed) > 0 and not errors else "error"
            generation["source_path"] = os.path.abspath(auto_oto_path)
    except Exception as exc:
        generation["status"] = "exception"
        generation["errors"] = [f"{exc}", traceback.format_exc(limit=3)]

    gt_rows = read_oto_rows(case.gt_oto_path, language=case.language)
    auto_rows = read_oto_rows(auto_oto_path, language=case.language) if os.path.isfile(auto_oto_path) else {}

    eval_payload = evaluate_oto_pair(
        gt_rows=gt_rows,
        auto_rows=auto_rows,
        bank_dir=case.bank_dir,
        language=case.language,
        weak_percentile=weak_percentile,
    )
    top_rows = select_plot_rows(eval_payload.get("row_errors") or [], top_k=max_plots)
    plots = render_case_plots(
        case=case,
        case_dir=case_dir,
        row_errors=top_rows,
        gt_rows=gt_rows,
        auto_rows=auto_rows,
    )

    case_report = {
        "case_id": case.case_id,
        "language": case.language,
        "format_type": case.format_type,
        "bank_name": case.bank_name,
        "bank_dir": case.bank_dir,
        "gt_oto_path": case.gt_oto_path,
        "auto_oto_path": os.path.abspath(auto_oto_path) if auto_oto_path else "",
        "generation": generation,
        "eval": eval_payload.get("summary") or {},
        "plot_count": len(plots),
        "plots": plots,
        "artifacts": {
            "case_dir": case_dir,
            "case_eval_json": os.path.join(case_dir, "eval.json"),
        },
    }
    _write_json(os.path.join(case_dir, "eval.json"), case_report)
    return case_report


def generate_auto_oto_for_case(
    *,
    case: VoicebankCase,
    out_path: str,
    variant: str,
    model_path: str,
    device: str,
) -> tuple[int, int, list[str]]:
    preset = dict(VARIANT_PRESETS.get(str(variant), {}) or {})
    restore = apply_env_preset(preset)
    try:
        processed, total, errors = generate_oto_with_boundary_decoder(
            wav_dir=str(case.bank_dir),
            out_path=str(out_path),
            source_oto_path=str(case.gt_oto_path),
            language=str(case.language),
            format_type=str(case.format_type),
            model_path=str(model_path),
            device=str(device),
            alias_suffix="",
            callback=None,
            special_aliases=None,
        )
        return int(processed), int(total), list(errors or [])
    finally:
        restore_env(restore)


def evaluate_oto_pair(
    *,
    gt_rows: dict[tuple[str, str, int], OtoRow],
    auto_rows: dict[tuple[str, str, int], OtoRow],
    bank_dir: str,
    language: str,
    weak_percentile: float,
) -> dict[str, Any]:
    common_keys = sorted(set(gt_rows.keys()) & set(auto_rows.keys()))
    wav_db = compute_wav_rms_db_by_name(bank_dir=bank_dir, used_wavs={k[0] for k in common_keys})
    weak_threshold_db = _percentile(list(wav_db.values()), float(weak_percentile)) if wav_db else -999.0

    row_errors: list[dict[str, Any]] = []
    for key in common_keys:
        gt = gt_rows[key]
        auto = auto_rows[key]
        wav_level_db = float(wav_db.get(key[0], -999.0))
        weak = bool(wav_level_db <= weak_threshold_db)
        err = build_row_error(
            gt=gt,
            auto=auto,
            language=language,
            weak=weak,
            wav_rms_db=wav_level_db,
        )
        row_errors.append(err)

    summary = build_error_summary(row_errors=row_errors, weak_threshold_db=weak_threshold_db)
    return {
        "summary": summary,
        "row_errors": row_errors,
    }


def read_oto_rows(path: str, *, language: str) -> dict[tuple[str, str, int], OtoRow]:
    out: dict[tuple[str, str, int], OtoRow] = {}
    counters: dict[tuple[str, str], int] = defaultdict(int)
    text = read_text_with_fallback(path)
    if not text:
        return out
    for line_idx, raw in enumerate(text.splitlines()):
        parsed = parse_oto_line(raw)
        if not parsed:
            continue
        wav_rel = str(parsed.get("wav", "") or "").strip()
        alias = str(parsed.get("alias", "") or "").strip()
        if not wav_rel or not alias:
            continue
        base = (os.path.basename(wav_rel).strip().lower(), alias)
        occ = counters[base]
        counters[base] = occ + 1
        role = infer_alias_role(alias=alias, language=language)
        n_like = is_n_like_alias(alias=alias, language=language)
        row = OtoRow(
            wav_rel=wav_rel,
            wav_norm=base[0],
            alias=alias,
            occ_idx=int(occ),
            line_index=int(line_idx),
            role=role,
            n_like=bool(n_like),
            offset=float(parsed.get("offset", 0.0) or 0.0),
            cons=float(parsed.get("cons", 0.0) or 0.0),
            cutoff=float(parsed.get("cutoff", 0.0) or 0.0),
            pre=float(parsed.get("pre", 0.0) or 0.0),
            ovl=float(parsed.get("ovl", 0.0) or 0.0),
            anchors=to_absolute_anchors(parsed),
        )
        out[(base[0], alias, int(occ))] = row
    return out


def infer_alias_role(*, alias: str, language: str) -> str:
    lang = normalize_language(language) or "korean"
    alias_type = _boundary_alias_type(alias, language=lang)
    trans_type = _boundary_transition_type(alias, language=lang)
    role = classify_alias_role(
        lang,
        alias,
        alias_type=alias_type,
        transition_type=trans_type,
        is_special=False,
    )
    return normalize_role(role)


def is_n_like_alias(*, alias: str, language: str) -> bool:
    text = str(alias or "").strip()
    if not text:
        return False
    low = text.lower()
    lang = normalize_language(language) or ""
    if lang == "japanese":
        if ("ん" in text) or ("ン" in text):
            return True
        return bool(re.search(r"(?:^|[\s_\-'])n(?:$|[\s_\-'])", low))
    if lang == "korean":
        if bool(re.search(r"(?:^|[\s_\-'])(?:n|ng|nn|ny|nj)(?:$|[\s_\-'])", low)):
            return True
        tokens = [tok for tok in re.split(r"[\s_\-']+", low) if tok]
        return any(tok in {"n", "ng", "nn", "ny", "nj"} for tok in tokens)
    return bool(re.search(r"(?:^|[\s_\-'])n(?:$|[\s_\-'])", low))


def to_absolute_anchors(parsed: dict[str, Any]) -> OtoAnchorsAbs:
    offset = max(0.0, float(parsed.get("offset", 0.0) or 0.0))
    pre_rel = max(0.0, float(parsed.get("pre", 0.0) or 0.0))
    ovl_rel = max(0.0, float(parsed.get("ovl", 0.0) or 0.0))
    cons_rel = max(0.0, float(parsed.get("cons", 0.0) or 0.0))
    cutoff_raw = float(parsed.get("cutoff", 0.0) or 0.0)
    pre_abs = offset + pre_rel
    ovl_abs = offset + min(pre_rel, ovl_rel)
    cons_abs = offset + max(pre_rel, cons_rel)
    cutoff_abs = offset + (abs(cutoff_raw) if cutoff_raw < 0.0 else cutoff_raw)
    return OtoAnchorsAbs(
        offset_abs=float(offset),
        overlap_abs=float(ovl_abs),
        pre_abs=float(pre_abs),
        cons_abs=float(cons_abs),
        cutoff_abs=float(cutoff_abs),
    )


def build_row_error(
    *,
    gt: OtoRow,
    auto: OtoRow,
    language: str,
    weak: bool,
    wav_rms_db: float,
) -> dict[str, Any]:
    deltas = {
        key: float(getattr(auto.anchors, key) - getattr(gt.anchors, key))
        for key in ANCHOR_KEYS
    }
    abs_err = {f"{key}_abs_err": abs(val) for key, val in deltas.items()}
    role = str(auto.role or gt.role or infer_alias_role(alias=auto.alias, language=language))
    n_like = bool(auto.n_like or gt.n_like or is_n_like_alias(alias=auto.alias, language=language))
    transition = bool(role in TRANSITION_ROLES)
    pre_cons_gap_auto = float(max(0.0, auto.cons - auto.pre))
    pre_cons_gap_gt = float(max(0.0, gt.cons - gt.pre))
    return {
        "wav_norm": auto.wav_norm,
        "wav_rel": auto.wav_rel,
        "alias": auto.alias,
        "occ_idx": int(auto.occ_idx),
        "line_index": int(auto.line_index),
        "role": role,
        "n_like": bool(n_like),
        "transition_role": transition,
        "weak_wav": bool(weak),
        "wav_rms_db": float(wav_rms_db),
        **deltas,
        **abs_err,
        "pre_cons_gap_auto": pre_cons_gap_auto,
        "pre_cons_gap_gt": pre_cons_gap_gt,
        "pre_cons_gap_delta": float(pre_cons_gap_auto - pre_cons_gap_gt),
    }


def build_error_summary(*, row_errors: list[dict[str, Any]], weak_threshold_db: float) -> dict[str, Any]:
    overall = aggregate_row_errors(row_errors)
    transition_rows = [row for row in row_errors if bool(row.get("transition_role", False))]
    n_like_rows = [row for row in row_errors if bool(row.get("n_like", False))]
    weak_rows = [row for row in row_errors if bool(row.get("weak_wav", False))]
    role_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in row_errors:
        role_groups[str(row.get("role", "other"))].append(row)
    by_role = {
        role: aggregate_row_errors(rows)
        for role, rows in sorted(role_groups.items(), key=lambda item: item[0])
    }
    return {
        "rows_compared": int(len(row_errors)),
        "weak_threshold_db": float(weak_threshold_db),
        "overall": overall,
        "transition_only": aggregate_row_errors(transition_rows),
        "n_like_only": aggregate_row_errors(n_like_rows),
        "weak_wav_only": aggregate_row_errors(weak_rows),
        "by_role": by_role,
    }


def aggregate_row_errors(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    out: dict[str, Any] = {"count": int(len(rows))}
    for key in ANCHOR_KEYS:
        arr = np.asarray([float(row.get(key, 0.0) or 0.0) for row in rows], dtype=np.float64)
        abs_arr = np.abs(arr)
        out[f"{key}_bias_ms"] = float(np.mean(arr))
        out[f"{key}_mae_ms"] = float(np.mean(abs_arr))
        out[f"{key}_p90_abs_ms"] = float(np.quantile(abs_arr, 0.90))
    pre_abs_arr = np.asarray([float(row.get("pre_abs", 0.0) or 0.0) for row in rows], dtype=np.float64)
    pre_abs_err = np.abs(pre_abs_arr)
    out["pre_abs_le_20ms_rate"] = float(np.mean(pre_abs_err <= 20.0))
    out["pre_abs_le_40ms_rate"] = float(np.mean(pre_abs_err <= 40.0))
    out["pre_abs_ge_80ms_rate"] = float(np.mean(pre_abs_err >= 80.0))
    gap_auto = np.asarray([float(row.get("pre_cons_gap_auto", 0.0) or 0.0) for row in rows], dtype=np.float64)
    gap_gt = np.asarray([float(row.get("pre_cons_gap_gt", 0.0) or 0.0) for row in rows], dtype=np.float64)
    out["pre_cons_gap_auto_mean_ms"] = float(np.mean(gap_auto))
    out["pre_cons_gap_gt_mean_ms"] = float(np.mean(gap_gt))
    out["pre_cons_gap_tight_rate_auto_le_20ms"] = float(np.mean(gap_auto <= 20.0))
    return out


def compute_wav_rms_db_by_name(*, bank_dir: str, used_wavs: set[str]) -> dict[str, float]:
    wav_paths = list(Path(bank_dir).rglob("*.wav"))
    out: dict[str, float] = {}
    for wav_path in wav_paths:
        key = wav_path.name.strip().lower()
        if used_wavs and key not in used_wavs:
            continue
        try:
            samples, _sr, _dur = load_wav_mono(str(wav_path), target_sr=16000)
            if samples.size <= 0:
                out[key] = -999.0
                continue
            rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)) + 1e-12))
            out[key] = float(20.0 * math.log10(max(rms, 1e-8)))
        except Exception:
            out[key] = -999.0
    return out


def select_plot_rows(row_errors: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    if not row_errors:
        return []
    def _score(row: dict[str, Any]) -> float:
        score = (
            float(row.get("pre_abs_abs_err", 0.0) or 0.0) * 2.0
            + float(row.get("cons_abs_abs_err", 0.0) or 0.0) * 1.2
            + float(row.get("offset_abs_abs_err", 0.0) or 0.0) * 0.6
        )
        if bool(row.get("transition_role", False)):
            score += 20.0
        if bool(row.get("n_like", False)):
            score += 18.0
        if bool(row.get("weak_wav", False)):
            score += 12.0
        return score
    rows = list(row_errors)
    rows.sort(key=_score, reverse=True)
    return rows[: max(1, int(top_k))]


def render_case_plots(
    *,
    case: VoicebankCase,
    case_dir: str,
    row_errors: list[dict[str, Any]],
    gt_rows: dict[tuple[str, str, int], OtoRow],
    auto_rows: dict[tuple[str, str, int], OtoRow],
) -> list[dict[str, Any]]:
    if not row_errors:
        return []
    wav_lookup = build_wav_lookup(case.bank_dir)
    plot_dir = os.path.join(case_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(row_errors, start=1):
        key = (str(row["wav_norm"]), str(row["alias"]), int(row["occ_idx"]))
        gt = gt_rows.get(key)
        auto = auto_rows.get(key)
        if gt is None or auto is None:
            continue
        wav_path = wav_lookup.get(str(row["wav_norm"]))
        if not wav_path:
            continue
        filename = (
            f"{idx:02d}_line{int(row['line_index']):04d}_"
            f"{_slugify(str(row['wav_norm']))}_{_slugify(str(row['alias']))}.svg"
        )
        plot_path = os.path.join(plot_dir, filename)
        ok = render_row_svg(
            wav_path=wav_path,
            gt=gt,
            auto=auto,
            row_error=row,
            out_path=plot_path,
        )
        if not ok:
            continue
        out.append(
            {
                "rank": int(idx),
                "wav_norm": str(row["wav_norm"]),
                "alias": str(row["alias"]),
                "occ_idx": int(row["occ_idx"]),
                "role": str(row["role"]),
                "n_like": bool(row["n_like"]),
                "weak_wav": bool(row["weak_wav"]),
                "pre_abs_abs_err": float(row.get("pre_abs_abs_err", 0.0) or 0.0),
                "cons_abs_abs_err": float(row.get("cons_abs_abs_err", 0.0) or 0.0),
                "offset_abs_abs_err": float(row.get("offset_abs_abs_err", 0.0) or 0.0),
                "plot_path": os.path.abspath(plot_path),
                "plot_rel": os.path.relpath(plot_path, case_dir),
            }
        )
    return out


def build_wav_lookup(bank_dir: str) -> dict[str, str]:
    out: dict[str, str] = {}
    # Prefer top-level wavs first.
    for wav in sorted(Path(bank_dir).glob("*.wav"), key=lambda p: p.name.lower()):
        key = wav.name.strip().lower()
        out[key] = str(wav.resolve())
    for wav in sorted(Path(bank_dir).rglob("*.wav"), key=lambda p: (len(p.parts), p.name.lower())):
        key = wav.name.strip().lower()
        if key not in out:
            out[key] = str(wav.resolve())
    return out


def render_row_svg(
    *,
    wav_path: str,
    gt: OtoRow,
    auto: OtoRow,
    row_error: dict[str, Any],
    out_path: str,
) -> bool:
    try:
        samples, sr, duration_sec = load_wav_mono(str(wav_path), target_sr=16000)
    except Exception:
        return False
    if samples.size <= 0 or sr <= 0:
        return False
    duration_ms = float(duration_sec) * 1000.0
    window_start_ms, window_end_ms = resolve_view_window_ms(gt=gt, auto=auto, duration_ms=duration_ms)

    spec_uri = build_spectrogram_data_uri(
        samples=samples,
        sr=int(sr),
        start_ms=window_start_ms,
        end_ms=window_end_ms,
        width=1140,
        height=260,
    )
    waveform_points = build_waveform_points(
        samples=samples,
        sr=int(sr),
        start_ms=window_start_ms,
        end_ms=window_end_ms,
        x0=70.0,
        y0=370.0,
        width=1140.0,
        height=220.0,
    )

    def x_map(ms: float) -> float:
        span = max(1.0, float(window_end_ms) - float(window_start_ms))
        t = (float(ms) - float(window_start_ms)) / span
        return 70.0 + max(0.0, min(1.0, t)) * 1140.0

    lines: list[str] = []
    lines.append("<svg xmlns='http://www.w3.org/2000/svg' width='1280' height='760' viewBox='0 0 1280 760'>")
    lines.append("<rect width='100%' height='100%' fill='#ffffff'/>")
    lines.append("<text x='70' y='28' font-size='16' font-family='Segoe UI, sans-serif' fill='#111111'>")
    lines.append(
        escape(
            f"{os.path.basename(wav_path)} | alias={auto.alias} | role={row_error.get('role')} | "
            f"n_like={bool(row_error.get('n_like'))} | weak={bool(row_error.get('weak_wav'))}"
        )
    )
    lines.append("</text>")
    lines.append("<text x='70' y='50' font-size='12' font-family='Consolas, monospace' fill='#333333'>")
    lines.append(
        escape(
            f"pre_err={float(row_error.get('pre_abs_abs_err', 0.0)):.1f}ms, "
            f"cons_err={float(row_error.get('cons_abs_abs_err', 0.0)):.1f}ms, "
            f"offset_err={float(row_error.get('offset_abs_abs_err', 0.0)):.1f}ms, "
            f"wav_rms={float(row_error.get('wav_rms_db', -999.0)):.1f}dB"
        )
    )
    lines.append("</text>")

    # Spectrogram panel.
    lines.append("<rect x='70' y='70' width='1140' height='260' fill='#f7f7f7' stroke='#d0d0d0' stroke-width='1'/>")
    if spec_uri:
        lines.append(
            f"<image x='70' y='70' width='1140' height='260' preserveAspectRatio='none' href='{spec_uri}'/>"
        )
    lines.append("<text x='74' y='86' font-size='11' font-family='Segoe UI, sans-serif' fill='#ffffff'>Spectrogram</text>")

    # Waveform panel.
    lines.append("<rect x='70' y='370' width='1140' height='220' fill='#fbfbfb' stroke='#d0d0d0' stroke-width='1'/>")
    if waveform_points:
        lines.append(f"<polyline points='{waveform_points}' fill='none' stroke='#4f4f4f' stroke-width='1'/>")
    lines.append("<line x1='70' y1='480' x2='1210' y2='480' stroke='#b8b8b8' stroke-width='1'/>")
    lines.append("<text x='74' y='386' font-size='11' font-family='Segoe UI, sans-serif' fill='#606060'>Waveform</text>")

    # Boundary lines.
    anchors_gt = {
        "offset": gt.anchors.offset_abs,
        "pre": gt.anchors.pre_abs,
        "cons": gt.anchors.cons_abs,
        "cutoff": gt.anchors.cutoff_abs,
    }
    anchors_auto = {
        "offset": auto.anchors.offset_abs,
        "pre": auto.anchors.pre_abs,
        "cons": auto.anchors.cons_abs,
        "cutoff": auto.anchors.cutoff_abs,
    }
    for name, t_ms in anchors_gt.items():
        alpha = "0.45" if name in {"offset", "cutoff"} else "0.90"
        x = x_map(float(t_ms))
        lines.append(
            f"<line x1='{x:.2f}' y1='70' x2='{x:.2f}' y2='590' "
            f"stroke='#d9534f' stroke-width='1.3' stroke-dasharray='6,4' opacity='{alpha}'/>"
        )
    for name, t_ms in anchors_auto.items():
        alpha = "0.45" if name in {"offset", "cutoff"} else "0.90"
        x = x_map(float(t_ms))
        lines.append(
            f"<line x1='{x:.2f}' y1='70' x2='{x:.2f}' y2='590' "
            f"stroke='#2f6fd6' stroke-width='1.3' opacity='{alpha}'/>"
        )

    lines.append("<text x='74' y='610' font-size='11' font-family='Segoe UI, sans-serif' fill='#d9534f'>GT (dashed)</text>")
    lines.append("<text x='180' y='610' font-size='11' font-family='Segoe UI, sans-serif' fill='#2f6fd6'>AUTO (solid)</text>")
    lines.append("<text x='70' y='640' font-size='11' font-family='Consolas, monospace' fill='#444444'>")
    lines.append(escape(f"window_ms=[{window_start_ms:.1f}, {window_end_ms:.1f}]"))
    lines.append("</text>")
    lines.append("</svg>")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return True


def resolve_view_window_ms(*, gt: OtoRow, auto: OtoRow, duration_ms: float) -> tuple[float, float]:
    lo = min(
        float(gt.anchors.offset_abs),
        float(auto.anchors.offset_abs),
        float(gt.anchors.pre_abs),
        float(auto.anchors.pre_abs),
        float(gt.anchors.cons_abs),
        float(auto.anchors.cons_abs),
    )
    hi = max(
        float(gt.anchors.cutoff_abs),
        float(auto.anchors.cutoff_abs),
        float(gt.anchors.pre_abs),
        float(auto.anchors.pre_abs),
        float(gt.anchors.cons_abs),
        float(auto.anchors.cons_abs),
    )
    start = max(0.0, lo - 140.0)
    end = min(float(duration_ms), hi + 180.0)
    if end - start < 260.0:
        center = (start + end) * 0.5
        start = max(0.0, center - 140.0)
        end = min(float(duration_ms), center + 140.0)
    if end <= start:
        end = min(float(duration_ms), start + 300.0)
    return float(start), float(end)


def build_spectrogram_data_uri(
    *,
    samples: np.ndarray,
    sr: int,
    start_ms: float,
    end_ms: float,
    width: int,
    height: int,
) -> str:
    try:
        feats, hop_sec = log_mel_spectrogram(samples, int(sr), n_mels=64, frame_ms=25.0, hop_ms=10.0)
    except Exception:
        return ""
    if feats.size <= 0:
        return ""
    frame_hop_ms = float(hop_sec) * 1000.0
    i0 = max(0, int(math.floor(float(start_ms) / max(frame_hop_ms, 1e-6))))
    i1 = min(int(feats.shape[0]), int(math.ceil(float(end_ms) / max(frame_hop_ms, 1e-6))) + 1)
    if i1 <= i0:
        return ""
    chunk = np.asarray(feats[i0:i1, :], dtype=np.float32).T  # [mels, frames]
    if chunk.size <= 0:
        return ""
    p5 = float(np.quantile(chunk, 0.05))
    p95 = float(np.quantile(chunk, 0.95))
    if p95 <= p5 + 1e-6:
        p95 = p5 + 1e-6
    norm = np.clip((chunk - p5) / (p95 - p5), 0.0, 1.0)
    img_rgb = mel_to_rgb_image(norm)
    if img_rgb.size <= 0:
        return ""
    pil = Image.fromarray(img_rgb, mode="RGB")
    pil = pil.resize((max(8, int(width)), max(8, int(height))), resample=Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/png;base64," + payload


def mel_to_rgb_image(norm: np.ndarray) -> np.ndarray:
    # Input shape: [mels, frames], values in [0,1].
    arr = np.asarray(norm, dtype=np.float32)
    if arr.ndim != 2 or arr.size <= 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    # Reverse rows so low-frequency bins appear near the bottom.
    arr = np.flipud(arr)
    r = np.clip((1.80 * arr) - 0.20, 0.0, 1.0)
    g = np.clip(1.50 - (np.abs(arr - 0.45) * 3.00), 0.0, 1.0)
    b = np.clip(1.60 * (1.00 - arr), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=2)
    return np.asarray(np.round(rgb * 255.0), dtype=np.uint8)


def build_waveform_points(
    *,
    samples: np.ndarray,
    sr: int,
    start_ms: float,
    end_ms: float,
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> str:
    if samples.size <= 1 or sr <= 0:
        return ""
    i0 = max(0, int(round((float(start_ms) / 1000.0) * float(sr))))
    i1 = min(int(samples.size), int(round((float(end_ms) / 1000.0) * float(sr))))
    if i1 <= i0:
        return ""
    seg = np.asarray(samples[i0:i1], dtype=np.float32)
    if seg.size <= 1:
        return ""
    max_points = 1800
    step = max(1, int(math.ceil(float(seg.size) / float(max_points))))
    seg = seg[::step]
    t = np.linspace(float(start_ms), float(end_ms), num=seg.size, endpoint=False)

    x = x0 + ((t - float(start_ms)) / max(1.0, float(end_ms) - float(start_ms))) * float(width)
    y_center = y0 + (float(height) * 0.5)
    y = y_center - (np.clip(seg, -1.0, 1.0) * (float(height) * 0.42))
    points = " ".join(f"{float(px):.2f},{float(py):.2f}" for px, py in zip(x, y))
    return points


def build_case_row(case_report: dict[str, Any]) -> dict[str, Any]:
    eval_summary = dict(case_report.get("eval") or {})
    overall = dict(eval_summary.get("overall") or {})
    transition = dict(eval_summary.get("transition_only") or {})
    n_like = dict(eval_summary.get("n_like_only") or {})
    weak = dict(eval_summary.get("weak_wav_only") or {})
    return {
        "case_id": str(case_report.get("case_id", "")),
        "language": str(case_report.get("language", "")),
        "format_type": str(case_report.get("format_type", "")),
        "bank_name": str(case_report.get("bank_name", "")),
        "gen_status": str((case_report.get("generation") or {}).get("status", "")),
        "rows_compared": int(eval_summary.get("rows_compared", 0) or 0),
        "overall_pre_mae_ms": float(overall.get("pre_abs_mae_ms", 0.0) or 0.0),
        "overall_pre_bias_ms": float(overall.get("pre_abs_bias_ms", 0.0) or 0.0),
        "transition_pre_mae_ms": float(transition.get("pre_abs_mae_ms", 0.0) or 0.0),
        "transition_pre_bias_ms": float(transition.get("pre_abs_bias_ms", 0.0) or 0.0),
        "n_like_pre_mae_ms": float(n_like.get("pre_abs_mae_ms", 0.0) or 0.0),
        "weak_pre_mae_ms": float(weak.get("pre_abs_mae_ms", 0.0) or 0.0),
        "weak_threshold_db": float(eval_summary.get("weak_threshold_db", 0.0) or 0.0),
        "plot_count": int(case_report.get("plot_count", 0) or 0),
    }


def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [build_case_row(report) for report in reports]
    ok = [row for row in rows if row["gen_status"] == "ok"]
    out: dict[str, Any] = {
        "case_count": int(len(rows)),
        "generation_ok_count": int(sum(1 for row in rows if row["gen_status"] == "ok")),
        "rows_compared_total": int(sum(int(row["rows_compared"]) for row in rows)),
    }
    for key in (
        "overall_pre_mae_ms",
        "transition_pre_mae_ms",
        "n_like_pre_mae_ms",
        "weak_pre_mae_ms",
    ):
        vals = [float(row[key]) for row in ok if float(row.get(key, 0.0) or 0.0) > 0.0]
        out[f"{key}_mean"] = float(np.mean(vals)) if vals else 0.0
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['language']}/{row['format_type']}"].append(row)
    by_group: dict[str, Any] = {}
    for group_key in sorted(grouped.keys()):
        g_rows = grouped[group_key]
        g_ok = [row for row in g_rows if row["gen_status"] == "ok"]
        by_group[group_key] = {
            "case_count": int(len(g_rows)),
            "generation_ok_count": int(len(g_ok)),
            "rows_compared_total": int(sum(int(row["rows_compared"]) for row in g_rows)),
            "overall_pre_mae_ms_mean": float(
                np.mean([float(row["overall_pre_mae_ms"]) for row in g_ok]) if g_ok else 0.0
            ),
            "transition_pre_mae_ms_mean": float(
                np.mean([float(row["transition_pre_mae_ms"]) for row in g_ok]) if g_ok else 0.0
            ),
            "n_like_pre_mae_ms_mean": float(
                np.mean([float(row["n_like_pre_mae_ms"]) for row in g_ok]) if g_ok else 0.0
            ),
            "weak_pre_mae_ms_mean": float(
                np.mean([float(row["weak_pre_mae_ms"]) for row in g_ok]) if g_ok else 0.0
            ),
        }
    out["by_group"] = by_group
    return out


def write_bank_summary_csv(path: str, reports: list[dict[str, Any]]) -> None:
    rows = [build_case_row(report) for report in reports]
    fieldnames = [
        "case_id",
        "language",
        "format_type",
        "bank_name",
        "gen_status",
        "rows_compared",
        "overall_pre_mae_ms",
        "overall_pre_bias_ms",
        "transition_pre_mae_ms",
        "transition_pre_bias_ms",
        "n_like_pre_mae_ms",
        "weak_pre_mae_ms",
        "weak_threshold_db",
        "plot_count",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_index_html(path: str, summary: dict[str, Any]) -> None:
    rows = list(summary.get("case_reports") or [])
    aggregate = dict(summary.get("aggregate") or {})
    run_meta = dict(summary.get("run_meta") or {})

    lines: list[str] = []
    lines.append("<!doctype html>")
    lines.append("<html lang='en'><head><meta charset='utf-8'/>")
    lines.append("<meta name='viewport' content='width=device-width, initial-scale=1'/>")
    lines.append("<title>OTO Visual Benchmark</title>")
    lines.append(
        "<style>"
        "body{font-family:Segoe UI,Arial,sans-serif;margin:18px;color:#222}"
        "h1,h2,h3{margin:0.6em 0 0.35em}"
        "table{border-collapse:collapse;width:100%;margin:10px 0}"
        "th,td{border:1px solid #ddd;padding:6px 8px;font-size:13px;text-align:left}"
        "th{background:#f3f5f7}"
        ".case{border:1px solid #d6dbe1;border-radius:8px;padding:12px;margin:16px 0}"
        ".meta{font-size:13px;color:#444}"
        ".plots img{max-width:100%;border:1px solid #d8d8d8;margin:8px 0}"
        ".ok{color:#117733;font-weight:600}.err{color:#aa3333;font-weight:600}"
        "</style></head><body>"
    )
    lines.append("<h1>Dataset-Staged OTO Visual Benchmark</h1>")
    lines.append("<div class='meta'>")
    lines.append(f"<div>created_at: {escape(str(run_meta.get('created_at', '')))}</div>")
    lines.append(f"<div>dataset_staged: {escape(str(run_meta.get('dataset_staged', '')))}</div>")
    lines.append(
        f"<div>variant/model: {escape(str(run_meta.get('variant', '')))} / "
        f"{escape(os.path.basename(str(run_meta.get('model_path', ''))))}</div>"
    )
    lines.append(f"<div>sampled_cases: {len(list(run_meta.get('sampled_case_ids') or []))}</div>")
    lines.append("</div>")

    lines.append("<h2>Aggregate</h2>")
    lines.append("<table><tbody>")
    for key in (
        "case_count",
        "generation_ok_count",
        "rows_compared_total",
        "overall_pre_mae_ms_mean",
        "transition_pre_mae_ms_mean",
        "n_like_pre_mae_ms_mean",
        "weak_pre_mae_ms_mean",
    ):
        lines.append(f"<tr><th>{escape(key)}</th><td>{escape(str(aggregate.get(key, '')))}</td></tr>")
    lines.append("</tbody></table>")

    lines.append("<h2>Per Case Summary</h2>")
    lines.append("<table><thead><tr>")
    headers = [
        "case_id",
        "gen_status",
        "rows_compared",
        "overall_pre_mae_ms",
        "transition_pre_mae_ms",
        "n_like_pre_mae_ms",
        "weak_pre_mae_ms",
        "plot_count",
    ]
    for h in headers:
        lines.append(f"<th>{escape(h)}</th>")
    lines.append("</tr></thead><tbody>")
    for report in rows:
        row = build_case_row(report)
        lines.append("<tr>")
        for h in headers:
            val = row.get(h, "")
            if h == "gen_status":
                cls = "ok" if str(val) == "ok" else "err"
                lines.append(f"<td class='{cls}'>{escape(str(val))}</td>")
            else:
                lines.append(f"<td>{escape(str(val))}</td>")
        lines.append("</tr>")
    lines.append("</tbody></table>")

    for report in rows:
        row = build_case_row(report)
        case_dir = str(((report.get("artifacts") or {}).get("case_dir", "")) or "")
        case_title = str(report.get("case_id", ""))
        lines.append("<div class='case'>")
        lines.append(f"<h3>{escape(case_title)}</h3>")
        lines.append(
            f"<div class='meta'>status=<span class='{'ok' if row['gen_status']=='ok' else 'err'}'>{escape(str(row['gen_status']))}</span> "
            f"| rows={row['rows_compared']} | overall_pre_mae={row['overall_pre_mae_ms']:.2f}ms | "
            f"transition_pre_mae={row['transition_pre_mae_ms']:.2f}ms | "
            f"n_like_pre_mae={row['n_like_pre_mae_ms']:.2f}ms | weak_pre_mae={row['weak_pre_mae_ms']:.2f}ms</div>"
        )
        lines.append("<div class='plots'>")
        for plot in list(report.get("plots") or []):
            rel = str(plot.get("plot_rel", "") or "")
            if not rel:
                continue
            # Make image path relative to index.html.
            abs_plot = os.path.join(case_dir, rel)
            rel_to_root = os.path.relpath(abs_plot, os.path.dirname(path))
            caption = (
                f"{plot.get('rank')} | {plot.get('wav_norm')} | {plot.get('alias')} | role={plot.get('role')} | "
                f"pre_err={float(plot.get('pre_abs_abs_err', 0.0)):.1f}ms"
            )
            lines.append(f"<div><div class='meta'>{escape(caption)}</div>")
            lines.append(f"<img src='{escape(rel_to_root)}' alt='{escape(caption)}'/></div>")
        lines.append("</div>")
        lines.append("</div>")

    lines.append("</body></html>")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _percentile(values: list[float], q_percent: float) -> float:
    if not values:
        return -999.0
    q = max(0.0, min(100.0, float(q_percent))) / 100.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.quantile(arr, q))


def _resolve_model_path(model_path: str) -> str:
    requested = str(model_path or "").strip()
    if requested:
        full = os.path.abspath(requested)
        if not os.path.isfile(full):
            raise SystemExit(f"model not found: {full}")
        return full
    default = os.path.abspath(DEFAULT_MODEL_PATH)
    if not os.path.isfile(default):
        raise SystemExit(f"default model not found: {default} (set --model)")
    return default


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _slugify(text: str) -> str:
    base = str(text or "").strip().lower()
    base = re.sub(r"[^a-z0-9]+", "_", base)
    base = base.strip("_")
    return base or "item"


def _safe_console(text: str) -> None:
    msg = str(text or "")
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        safe = msg.encode("cp932", errors="replace").decode("cp932", errors="replace")
        print(safe, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
