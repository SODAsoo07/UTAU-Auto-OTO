from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.format_type_utils import normalize_format_type, normalize_language_name
from core.oto_ml_policy import alias_family_to_alias_types, normalize_alias_family
from core.oto_ml.features.mel_patches import resolve_rawmel_cache_dir
from core.oto_ml_collection import build_datasets_from_candidates
from core.oto_ml_collection_discovery import discover_training_candidates_from_dataset_root
from core.oto_ml_coupled import evaluate_coupled_bundle, train_coupled_bundle, train_coupled_bundle_rawmel
from core.oto_ml_lightgbm import evaluate_lightgbm_selector_bundle, train_lightgbm_selector_bundle
from core.oto_ml_selector import build_selector_dataset_csv_from_delta_dataset
from core.runtime_encoding import bootstrap_utf8_runtime


def _split_csv_tokens(raw: str) -> List[str]:
    return [v.strip() for v in str(raw or "").split(",") if v.strip()]


def _parse_languages(raw: str) -> List[str]:
    tokens = _split_csv_tokens(raw)
    if not tokens:
        tokens = ["korean", "japanese"]
    out: List[str] = []
    for token in tokens:
        t = str(token).strip().lower()
        if t in {"all", "both", "*"}:
            out.extend(["korean", "japanese"])
            continue
        lang = normalize_language_name(t)
        if lang in {"korean", "japanese"}:
            out.append(lang)
    # preserve order, unique
    seen = set()
    dedup = []
    for lang in out:
        if lang in seen:
            continue
        seen.add(lang)
        dedup.append(lang)
    return dedup


def _parse_format_filter(raw: str) -> Dict[str, Optional[set]]:
    """
    grammar:
      - 'auto' / '' / '*' : no explicit filter per language
      - 'cv,cvc'          : shared filter for both languages
      - 'korean=cv,cvc;japanese=cvvc,vcv'
    """
    text = str(raw or "").strip()
    if not text or text.lower() in {"auto", "*", "all"}:
        return {}

    out: Dict[str, Optional[set]] = {}
    if ";" in text or "=" in text:
        chunks = [c.strip() for c in text.split(";") if c.strip()]
        for chunk in chunks:
            if "=" not in chunk:
                continue
            k, v = chunk.split("=", 1)
            lang = normalize_language_name(k.strip().lower())
            if lang not in {"korean", "japanese"}:
                continue
            fmt_set = set()
            for f in _split_csv_tokens(v):
                nf = normalize_format_type(lang, f)
                if nf:
                    fmt_set.add(nf)
            out[lang] = fmt_set
        return out

    shared = set()
    for f in _split_csv_tokens(text):
        for lang in ("korean", "japanese"):
            nf = normalize_format_type(lang, f)
            if nf:
                shared.add(nf)
    if shared:
        out["korean"] = set(shared)
        out["japanese"] = set(shared)
    return out


def _default_min_mapping_conf(language: str, fmt: str) -> float:
    lang = normalize_language_name(language)
    f = normalize_format_type(lang, fmt) or "general"
    if lang == "korean":
        table = {
            "cv": 0.55,
            "cvc": 0.75,
            "cvvc": 0.78,
            "vcv": 0.72,
            "general": 0.50,
        }
        return float(table.get(f, 0.50))
    table = {
        "cv": 0.55,
        "cvc": 0.60,
        "cvvc": 0.70,
        "vcv": 0.68,
        "general": 0.50,
    }
    return float(table.get(f, 0.50))


def _collect_jobs_from_built(built_result: Dict[str, object]) -> List[Tuple[str, str, str, int]]:
    jobs: Dict[Tuple[str, str], Tuple[str, int]] = {}
    for item in list(built_result.get("candidates") or []):
        try:
            status = str(getattr(item, "status", "") or "")
            language = str(getattr(item, "language", "") or "").strip().lower()
            fmt = str(getattr(item, "format_type", "") or "").strip().lower()
            out_csv = str(getattr(item, "out_csv", "") or "")
            saved_rows = int(getattr(item, "saved_rows", 0) or 0)
        except Exception:
            continue
        if status != "built" or not out_csv:
            continue
        key = (language, fmt)
        prev = jobs.get(key)
        if prev is None:
            jobs[key] = (out_csv, max(saved_rows, 0))
        else:
            jobs[key] = (out_csv, int(prev[1]) + max(saved_rows, 0))
    out: List[Tuple[str, str, str, int]] = []
    for (language, fmt), (out_csv, rows) in sorted(jobs.items()):
        if out_csv:
            out.append((language, fmt, out_csv, int(rows)))
    return out


def _parse_subset_targets(raw: str) -> set:
    out = set()
    for token in _split_csv_tokens(raw):
        text = str(token or "").strip().lower()
        if "/" not in text:
            continue
        lang_raw, fmt_raw = text.split("/", 1)
        lang = normalize_language_name(lang_raw)
        fmt = normalize_format_type(lang, fmt_raw) or str(fmt_raw).strip().lower()
        if lang and fmt:
            out.add((lang, fmt))
    return out


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_bool(value) -> bool:
    s = str(value or "").strip().lower()
    return s in {"1", "true", "yes", "on", "y", "t"}


def _blank_attach_rates(eval_summary: Dict[str, object]) -> Dict[str, float]:
    kpi = dict((eval_summary or {}).get("blank_attach_kpi") or {})
    zones = dict((eval_summary or {}).get("blank_attach_kpi_pitch_zones") or {})
    high_zone = dict(zones.get("high") or {})
    return {
        "overall": _as_float(kpi.get("pred_attach_rate"), 0.0),
        "head": _as_float(kpi.get("head_pred_attach_rate"), 0.0),
        "vv": _as_float(kpi.get("vv_pred_attach_rate"), 0.0),
        "delta": _as_float(kpi.get("pred_minus_gold_rate"), 0.0),
        "high_pitch_head": _as_float(high_zone.get("head_pred_attach_rate"), 0.0),
    }


def _blank_attach_delta_rate(eval_summary: Dict[str, object]) -> float:
    kpi = dict((eval_summary or {}).get("blank_attach_kpi") or {})
    return _as_float(kpi.get("pred_minus_gold_rate"), 0.0)


def _blank_attach_high_pitch_head_rate(eval_summary: Dict[str, object]) -> float:
    zones = dict((eval_summary or {}).get("blank_attach_kpi_pitch_zones") or {})
    high = dict(zones.get("high") or {})
    return _as_float(high.get("head_pred_attach_rate"), 0.0)


def _ensure_blank_attach_eval_fields(eval_summary: Dict[str, object]) -> Dict[str, object]:
    out = dict(eval_summary or {})
    kpi = dict(out.get("blank_attach_kpi") or {})
    kpi.setdefault("enabled", False)
    kpi.setdefault("pred_attach_rate", 0.0)
    kpi.setdefault("head_pred_attach_rate", 0.0)
    kpi.setdefault("vv_pred_attach_rate", 0.0)
    kpi.setdefault("primary_kpi_name", "head_pred_attach_rate")
    kpi.setdefault("primary_kpi_value", float(kpi.get("head_pred_attach_rate", 0.0) or 0.0))
    out["blank_attach_kpi"] = kpi
    if not isinstance(out.get("kpi_primary"), dict):
        out["kpi_primary"] = {
            "name": str(kpi.get("primary_kpi_name", "head_pred_attach_rate")),
            "value": float(kpi.get("primary_kpi_value", 0.0) or 0.0),
        }
    zone_kpi = dict(out.get("blank_attach_kpi_pitch_zones") or {})
    high_zone = dict(zone_kpi.get("high") or {})
    high_zone.setdefault("enabled", False)
    high_zone.setdefault("head_pred_attach_rate", 0.0)
    high_zone.setdefault("pred_minus_gold_rate", 0.0)
    zone_kpi["high"] = high_zone
    out["blank_attach_kpi_pitch_zones"] = zone_kpi
    return out


def _check_blank_attach_gate(
    eval_summary: Dict[str, object],
    *,
    max_overall: float,
    max_head: float,
    max_vv: float,
    max_delta_rate: float = -1.0,
    max_high_pitch_head: float = -1.0,
) -> Tuple[bool, List[str], Dict[str, float]]:
    rates = _blank_attach_rates(eval_summary or {})
    delta_rate = _blank_attach_delta_rate(eval_summary or {})
    high_pitch_head_rate = _blank_attach_high_pitch_head_rate(eval_summary or {})
    reasons: List[str] = []
    if float(max_overall) >= 0.0 and rates["overall"] > float(max_overall):
        reasons.append(
            f"overall_blank_attach_rate {rates['overall']:.4f} > {float(max_overall):.4f}"
        )
    if float(max_head) >= 0.0 and rates["head"] > float(max_head):
        reasons.append(
            f"head_blank_attach_rate {rates['head']:.4f} > {float(max_head):.4f}"
        )
    if float(max_vv) >= 0.0 and rates["vv"] > float(max_vv):
        reasons.append(
            f"vv_blank_attach_rate {rates['vv']:.4f} > {float(max_vv):.4f}"
        )
    if float(max_delta_rate) >= 0.0 and delta_rate > float(max_delta_rate):
        reasons.append(
            f"blank_attach_pred_minus_gold_rate {delta_rate:.4f} > {float(max_delta_rate):.4f}"
        )
    if float(max_high_pitch_head) >= 0.0 and high_pitch_head_rate > float(max_high_pitch_head):
        reasons.append(
            f"high_pitch_head_blank_attach_rate {high_pitch_head_rate:.4f} > {float(max_high_pitch_head):.4f}"
        )
    return len(reasons) == 0, reasons, rates


def _parse_gate_value(payload: Dict[str, object], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key not in payload:
            continue
        try:
            return float(payload.get(key))
        except Exception:
            continue
    return None


def _normalize_gate_entry(payload: object) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(payload, dict):
        return out
    overall = _parse_gate_value(payload, ("max_blank_attach_rate", "max_overall", "overall"))
    head = _parse_gate_value(payload, ("max_head_blank_attach_rate", "max_head", "head"))
    vv = _parse_gate_value(payload, ("max_vv_blank_attach_rate", "max_vv", "vv"))
    delta = _parse_gate_value(payload, ("max_blank_attach_delta_rate", "max_delta", "delta"))
    high_head = _parse_gate_value(
        payload,
        ("max_high_pitch_head_blank_attach_rate", "max_high_pitch_head", "high_pitch_head"),
    )
    if overall is not None:
        out["max_blank_attach_rate"] = float(overall)
    if head is not None:
        out["max_head_blank_attach_rate"] = float(head)
    if vv is not None:
        out["max_vv_blank_attach_rate"] = float(vv)
    if delta is not None:
        out["max_blank_attach_delta_rate"] = float(delta)
    if high_head is not None:
        out["max_high_pitch_head_blank_attach_rate"] = float(high_head)
    return out


def _load_blank_attach_gate_profile(path: str) -> Dict[str, object]:
    raw_path = str(path or "").strip()
    if not raw_path:
        return {"path": "", "default": {}, "overrides": {}}
    profile_path = os.path.abspath(raw_path)
    if not os.path.isfile(profile_path):
        raise FileNotFoundError(profile_path)
    with open(profile_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise RuntimeError(f"Invalid gate profile JSON (expected object): {profile_path}")
    default_entry = _normalize_gate_entry(raw.get("default", raw.get("defaults", {})))
    overrides: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, value in dict(raw.get("overrides") or {}).items():
        token = str(key or "").strip().lower()
        if "/" not in token:
            continue
        lang_raw, fmt_raw = token.split("/", 1)
        lang = normalize_language_name(lang_raw)
        fmt = normalize_format_type(lang, fmt_raw) or str(fmt_raw).strip().lower()
        if not lang or not fmt:
            continue
        overrides[(lang, fmt)] = _normalize_gate_entry(value)
    return {"path": profile_path, "default": default_entry, "overrides": overrides}


def _resolve_blank_attach_gate_thresholds(
    *,
    language: str,
    format_type: str,
    cli_overall: float,
    cli_head: float,
    cli_vv: float,
    cli_delta: float,
    cli_high_pitch_head: float,
    profile: Dict[str, object],
) -> Dict[str, float]:
    out = {
        "max_blank_attach_rate": float(cli_overall),
        "max_head_blank_attach_rate": float(cli_head),
        "max_vv_blank_attach_rate": float(cli_vv),
        "max_blank_attach_delta_rate": float(cli_delta),
        "max_high_pitch_head_blank_attach_rate": float(cli_high_pitch_head),
    }

    default_entry = dict((profile or {}).get("default") or {})
    for key in out.keys():
        default_value = default_entry.get(key)
        if default_value is None:
            continue
        try:
            out[key] = float(default_value)
        except Exception:
            continue

    overrides = dict((profile or {}).get("overrides") or {})
    override_entry = dict(overrides.get((str(language or "").strip().lower(), str(format_type or "").strip().lower())) or {})
    for key in out.keys():
        override_value = override_entry.get(key)
        if override_value is None:
            continue
        try:
            out[key] = float(override_value)
        except Exception:
            continue
    return out


def _read_backend_from_model_meta(model_dir: str) -> str:
    meta_path = os.path.join(str(model_dir or ""), "model_meta.json")
    if not os.path.isfile(meta_path):
        return ""
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return str(payload.get("backend", "") or "").strip().lower()
    except Exception:
        return ""


def _discover_lightgbm_model_dir(model_root: str, language: str, format_type: str) -> str:
    root = os.path.join(str(model_root or ""), str(language or "").strip().lower(), str(format_type or "").strip().lower())
    if not os.path.isdir(root):
        return ""
    preferred: List[str] = []
    fallback: List[str] = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        backend = _read_backend_from_model_meta(path)
        if backend == "lightgbm":
            preferred.append(path)
        elif backend == "":
            fallback.append(path)
    if preferred:
        for path in preferred:
            if "lightgbm" in os.path.basename(path).lower():
                return path
        return preferred[0]
    if fallback:
        for path in fallback:
            if "lightgbm" in os.path.basename(path).lower():
                return path
        return fallback[0]
    return ""


def _selector_dataset_path(model_root: str, language: str, format_type: str, dataset_csv: str) -> str:
    base = os.path.splitext(os.path.basename(str(dataset_csv or "")))[0] or "dataset"
    file_name = f"{base}.{str(language).lower()}.{str(format_type).lower()}.selector.csv"
    out_dir = os.path.join(str(model_root or ""), "_selector_datasets")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, file_name)


def _selector_family_dataset_path(
    model_root: str,
    language: str,
    format_type: str,
    dataset_csv: str,
    alias_family: str,
) -> str:
    base = os.path.splitext(os.path.basename(str(dataset_csv or "")))[0] or "dataset"
    fam = normalize_alias_family(alias_family) or "general"
    file_name = f"{base}.{str(language).lower()}.{str(format_type).lower()}.selector.{fam}.csv"
    out_dir = os.path.join(str(model_root or ""), "_selector_datasets")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, file_name)


def _parse_selector_families(raw: str) -> List[str]:
    out: List[str] = []
    for token in _split_csv_tokens(raw):
        fam = normalize_alias_family(token)
        if fam:
            out.append(fam)
    if not out:
        out = ["cv", "vc", "vcv", "vowel"]
    seen = set()
    dedup: List[str] = []
    for fam in out:
        if fam in seen:
            continue
        seen.add(fam)
        dedup.append(fam)
    return dedup


def _subset_suffix(min_conf: float, max_blank_risk: float, require_blank_flag_clear: bool) -> str:
    bits: List[str] = []
    if float(min_conf) >= 0.0:
        bits.append(f"conf{int(round(float(min_conf) * 1000.0))}")
    if float(max_blank_risk) >= 0.0:
        bits.append(f"blank{int(round(float(max_blank_risk) * 1000.0))}")
    if require_blank_flag_clear:
        bits.append("blankflag0")
    return "_".join(bits) if bits else "all"


def _materialize_subset_csv(
    *,
    dataset_csv: str,
    model_root: str,
    language: str,
    format_type: str,
    min_conf: float,
    max_blank_risk: float,
    require_blank_flag_clear: bool,
) -> Tuple[str, Dict[str, object]]:
    min_conf_v = float(min_conf)
    max_blank_v = float(max_blank_risk)
    need_filter = (min_conf_v >= 0.0) or (max_blank_v >= 0.0) or bool(require_blank_flag_clear)
    meta = {
        "enabled": bool(need_filter),
        "min_mapping_confidence": min_conf_v,
        "max_blank_risk": max_blank_v,
        "require_blank_flag_clear": bool(require_blank_flag_clear),
    }
    if not need_filter:
        meta["rows_in"] = -1
        meta["rows_out"] = -1
        meta["keep_ratio"] = 1.0
        return dataset_csv, meta

    subset_dir = os.path.join(model_root, "_subsets")
    os.makedirs(subset_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(dataset_csv))[0]
    suffix = _subset_suffix(min_conf_v, max_blank_v, bool(require_blank_flag_clear))
    out_csv = os.path.join(subset_dir, f"{base_name}.{language}.{format_type}.{suffix}.csv")

    rows_in = 0
    rows_out = 0
    with open(dataset_csv, "r", encoding="utf-8-sig", newline="") as src, open(
        out_csv, "w", encoding="utf-8-sig", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        if not reader.fieldnames:
            raise RuntimeError(f"subset filter failed: no CSV header ({dataset_csv})")
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            rows_in += 1
            if min_conf_v >= 0.0:
                conf = _as_float(row.get("mapping_confidence"), 0.0)
                if conf < min_conf_v:
                    continue
            if max_blank_v >= 0.0:
                blank_risk = _as_float(row.get("blank_risk_score"), 0.0)
                if blank_risk > max_blank_v:
                    continue
            if require_blank_flag_clear and _as_bool(row.get("blank_risk_flag")):
                continue
            writer.writerow(row)
            rows_out += 1

    if rows_out <= 0:
        raise RuntimeError(
            f"subset filter removed all rows: {dataset_csv} (min_conf={min_conf_v}, "
            f"max_blank_risk={max_blank_v}, require_blank_flag_clear={bool(require_blank_flag_clear)})"
        )
    meta["rows_in"] = int(rows_in)
    meta["rows_out"] = int(rows_out)
    meta["keep_ratio"] = float(rows_out) / float(max(1, rows_in))
    meta["out_csv"] = out_csv
    return out_csv, meta


def main():
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(description="Batch coupled training by language/format matrix.")
    ap.add_argument("--dataset-root", required=True, help="Staged dataset root (contains language/format folders)")
    ap.add_argument("--workspace-root", default=os.path.join(ROOT, "ml_workspace"))
    ap.add_argument("--model-root", default=os.path.join(ROOT, "ML_models"))
    ap.add_argument("--languages", default="korean,japanese", help="korean,japanese|both|all")
    ap.add_argument(
        "--formats",
        default="auto",
        help="auto | cv,cvc,... | korean=cv,cvc;japanese=cvvc,vcv",
    )
    ap.add_argument(
        "--auto-oto-policy",
        default="require",
        choices=["require", "generate-temp", "generate-persist"],
        help="Policy when auto oto is missing during dataset build",
    )
    ap.add_argument("--backend", default="coupled_nn_v1", help="coupled_nn_v1 or coupled_nn_v2_rawmel")
    ap.add_argument("--rawmel-cache", default="", help="Raw mel cache dir (v2 only). Empty=auto per job")
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda")
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--batch-size", type=int, default=192)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--min-confidence", type=float, default=0.50)
    ap.add_argument("--group-column", default="voicebank_id")
    ap.add_argument("--alias-types", default="")
    ap.add_argument("--alias-family", default="")
    ap.add_argument(
        "--subset-targets",
        default="",
        help="Optional comma list: korean/vcv,japanese/cvvc. Empty=apply subset filter to all jobs when enabled.",
    )
    ap.add_argument(
        "--subset-min-mapping-confidence",
        type=float,
        default=-1.0,
        help="If >=0, filter dataset rows by mapping_confidence >= value before training/eval.",
    )
    ap.add_argument(
        "--subset-max-blank-risk",
        type=float,
        default=-1.0,
        help="If >=0, filter dataset rows by blank_risk_score <= value before training/eval.",
    )
    ap.add_argument(
        "--subset-require-blank-flag-clear",
        action="store_true",
        help="When set, rows with blank_risk_flag=true are excluded before training/eval.",
    )
    ap.add_argument(
        "--min-mapping-confidence",
        type=float,
        default=-1.0,
        help="If <0, use per-language/format defaults",
    )
    ap.add_argument("--skip-build", action="store_true", help="Skip dataset build; train from existing CSVs only")
    ap.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip coupled model training step (use existing model_dir for eval/selector).",
    )
    ap.add_argument(
        "--skip-selector-train",
        action="store_true",
        help="Skip selector model training step (dataset build/eval only for selector path).",
    )
    ap.add_argument("--skip-eval", action="store_true", help="Skip evaluation step")
    ap.add_argument(
        "--max-blank-attach-rate",
        type=float,
        default=-1.0,
        help="If >=0, fail job when eval blank_attach_kpi.pred_attach_rate exceeds this value.",
    )
    ap.add_argument(
        "--max-head-blank-attach-rate",
        type=float,
        default=-1.0,
        help="If >=0, fail job when eval blank_attach_kpi.head_pred_attach_rate exceeds this value.",
    )
    ap.add_argument(
        "--max-vv-blank-attach-rate",
        type=float,
        default=-1.0,
        help="If >=0, fail job when eval blank_attach_kpi.vv_pred_attach_rate exceeds this value.",
    )
    ap.add_argument(
        "--max-blank-attach-delta-rate",
        type=float,
        default=-1.0,
        help="If >=0, fail job when eval blank_attach_kpi.pred_minus_gold_rate exceeds this value.",
    )
    ap.add_argument(
        "--max-high-pitch-head-blank-attach-rate",
        type=float,
        default=-1.0,
        help="If >=0, fail job when eval blank_attach_kpi_pitch_zones.high.head_pred_attach_rate exceeds this value.",
    )
    ap.add_argument(
        "--blank-attach-gate-profile",
        default="",
        help="Optional JSON profile for per-language/format gate thresholds.",
    )
    ap.add_argument(
        "--train-selector",
        action="store_true",
        help="Train selector bundle per job from the coupled delta dataset.",
    )
    ap.add_argument(
        "--selector-objective",
        default="auto",
        choices=["auto", "pointwise", "ranking", "pairwise"],
        help="Selector objective mode.",
    )
    ap.add_argument("--selector-group-column", default="voicebank_id")
    ap.add_argument("--selector-num-boost-round", type=int, default=400)
    ap.add_argument("--selector-early-stopping-rounds", type=int, default=40)
    ap.add_argument(
        "--selector-model-root",
        default="",
        help="Optional root path for selector model outputs. Empty=existing lightgbm route if found, else current job model dir.",
    )
    ap.add_argument(
        "--selector-min-mapping-confidence",
        type=float,
        default=-1.0,
        help="If >=0, selector dataset rows are filtered by mapping_confidence >= value. If <0, uses job min_mapping_confidence.",
    )
    ap.add_argument("--selector-require-train-keep", action="store_true")
    ap.add_argument("--selector-exclude-nuclei-fallback", action="store_true")
    ap.add_argument(
        "--selector-split-by-alias-family",
        action="store_true",
        help="Train selector bundles per alias_family (cv/vc/vcv/vowel) under families/<family>.",
    )
    ap.add_argument(
        "--selector-families",
        default="cv,vc,vcv,vowel",
        help="Comma list for --selector-split-by-alias-family (default: cv,vc,vcv,vowel).",
    )
    ap.add_argument("--require-all", action="store_true", help="Fail if any selected language/format job fails or missing")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    workspace_root = os.path.abspath(args.workspace_root)
    model_root = os.path.abspath(args.model_root)
    os.makedirs(workspace_root, exist_ok=True)
    os.makedirs(model_root, exist_ok=True)

    languages = _parse_languages(args.languages)
    if not languages:
        raise SystemExit("No valid languages selected.")
    format_filter = _parse_format_filter(args.formats)

    candidates = discover_training_candidates_from_dataset_root(
        dataset_root,
        auto_oto_policy=args.auto_oto_policy,
    )
    selected = []
    for c in candidates:
        lang = normalize_language_name(getattr(c, "language", ""))
        fmt = normalize_format_type(lang, getattr(c, "format_type", "")) or str(getattr(c, "format_type", "")).strip().lower()
        if lang not in languages:
            continue
        wanted = format_filter.get(lang)
        if isinstance(wanted, set) and wanted and fmt not in wanted:
            continue
        selected.append(c)

    if not selected:
        raise SystemExit("No candidates matched the selected language/format filters.")

    summary: Dict[str, object] = {
        "dataset_root": dataset_root,
        "workspace_root": workspace_root,
        "model_root": model_root,
        "languages": languages,
        "format_filter": {k: sorted(v) for k, v in format_filter.items()},
        "selected_candidates": len(selected),
        "jobs": [],
        "failures": [],
    }

    jobs: List[Tuple[str, str, str, int]] = []
    if args.skip_build:
        # fallback: infer expected CSV path by selected language/format combos
        combos = sorted({(normalize_language_name(getattr(c, "language", "")), normalize_format_type(getattr(c, "language", ""), getattr(c, "format_type", "")) or str(getattr(c, "format_type", "")).strip().lower()) for c in selected})
        for lang, fmt in combos:
            csv_path = os.path.join(workspace_root, "datasets", lang, f"dataset_{lang}_{fmt}.csv")
            if os.path.isfile(csv_path):
                jobs.append((lang, fmt, csv_path, -1))
            elif args.require_all:
                raise SystemExit(f"Missing dataset CSV for {lang}/{fmt}: {csv_path}")
    else:
        built = build_datasets_from_candidates(
            selected,
            workspace_root,
            auto_oto_policy=args.auto_oto_policy,
            progress_callback=lambda message: print(message, flush=True),
        )
        jobs = _collect_jobs_from_built(built)
        summary["build_summary"] = built.get("summary", {})
        if not jobs:
            raise SystemExit("No datasets were built from selected candidates.")

    alias_types = [v.strip() for v in str(args.alias_types).split(",") if v.strip()]
    backend = str(args.backend or "coupled_nn_v1").strip().lower()
    subset_targets = _parse_subset_targets(str(args.subset_targets or ""))
    subset_min_conf = float(args.subset_min_mapping_confidence)
    subset_max_blank = float(args.subset_max_blank_risk)
    subset_blank_flag_clear = bool(args.subset_require_blank_flag_clear)
    subset_enabled = (
        (subset_min_conf >= 0.0)
        or (subset_max_blank >= 0.0)
        or subset_blank_flag_clear
    )
    max_blank_attach = float(args.max_blank_attach_rate)
    max_head_blank_attach = float(args.max_head_blank_attach_rate)
    max_vv_blank_attach = float(args.max_vv_blank_attach_rate)
    max_blank_attach_delta = float(args.max_blank_attach_delta_rate)
    max_high_pitch_head_blank_attach = float(args.max_high_pitch_head_blank_attach_rate)
    selector_enabled = bool(args.train_selector)
    selector_objective = str(args.selector_objective or "auto").strip().lower()
    selector_group_column = str(args.selector_group_column or "voicebank_id").strip()
    selector_num_boost_round = int(args.selector_num_boost_round)
    selector_early_stopping_rounds = int(args.selector_early_stopping_rounds)
    selector_model_root = os.path.abspath(str(args.selector_model_root).strip()) if str(args.selector_model_root or "").strip() else ""
    selector_min_mapping_confidence = float(args.selector_min_mapping_confidence)
    selector_require_train_keep = bool(args.selector_require_train_keep)
    selector_exclude_nuclei_fallback = bool(args.selector_exclude_nuclei_fallback)
    selector_split_by_alias_family = bool(args.selector_split_by_alias_family)
    selector_families = _parse_selector_families(str(args.selector_families or ""))
    gate_profile = _load_blank_attach_gate_profile(str(args.blank_attach_gate_profile or ""))
    summary["blank_attach_gate_profile"] = {
        "path": str(gate_profile.get("path", "") or ""),
        "default": dict(gate_profile.get("default") or {}),
        "override_keys": sorted(
            [
                f"{lang}/{fmt}"
                for (lang, fmt) in dict(gate_profile.get("overrides") or {}).keys()
            ]
        ),
    }
    summary["selector"] = {
        "enabled": selector_enabled,
        "objective": selector_objective,
        "group_column": selector_group_column,
        "num_boost_round": selector_num_boost_round,
        "early_stopping_rounds": selector_early_stopping_rounds,
        "model_root": selector_model_root,
        "min_mapping_confidence": selector_min_mapping_confidence,
        "require_train_keep": selector_require_train_keep,
        "exclude_nuclei_fallback": selector_exclude_nuclei_fallback,
        "split_by_alias_family": selector_split_by_alias_family,
        "families": list(selector_families),
        "skip_selector_train": bool(args.skip_selector_train),
        "pipeline_mode": "two_stage_per_format",
    }

    for lang, fmt, dataset_csv, built_rows in jobs:
        job = {
            "language": lang,
            "format_type": fmt,
            "dataset_csv": dataset_csv,
            "built_rows": int(built_rows),
            "status": "pending",
        }
        summary["jobs"].append(job)

        out_dir = os.path.join(model_root, lang, fmt, ("v2_coupled_rawmel" if backend == "coupled_nn_v2_rawmel" else "v1_coupled"))
        eval_report = os.path.join(out_dir, "eval_summary.json")
        job["model_dir"] = out_dir
        job["eval_report"] = eval_report

        mmc = float(args.min_mapping_confidence)
        if mmc < 0.0:
            mmc = _default_min_mapping_conf(lang, fmt)
        job["min_mapping_confidence"] = float(mmc)

        dataset_for_job = dataset_csv
        if subset_enabled and (not subset_targets or (lang, fmt) in subset_targets):
            dataset_for_job, subset_meta = _materialize_subset_csv(
                dataset_csv=dataset_csv,
                model_root=model_root,
                language=lang,
                format_type=fmt,
                min_conf=subset_min_conf,
                max_blank_risk=subset_max_blank,
                require_blank_flag_clear=subset_blank_flag_clear,
            )
            job["subset"] = subset_meta
        job["dataset_csv_used"] = dataset_for_job
        gate_thresholds = _resolve_blank_attach_gate_thresholds(
            language=lang,
            format_type=fmt,
            cli_overall=max_blank_attach,
            cli_head=max_head_blank_attach,
            cli_vv=max_vv_blank_attach,
            cli_delta=max_blank_attach_delta,
            cli_high_pitch_head=max_high_pitch_head_blank_attach,
            profile=gate_profile,
        )
        job["blank_attach_gate_thresholds"] = gate_thresholds

        if args.dry_run:
            job["status"] = "dry_run"
            continue

        try:
            os.makedirs(out_dir, exist_ok=True)
            if backend == "coupled_nn_v2_rawmel":
                rawmel_cache_hint = str(args.rawmel_cache or "").strip()
                if not rawmel_cache_hint:
                    rawmel_cache_hint = os.path.join(workspace_root, "rawmel_cache_noml_auto")
                if rawmel_cache_hint and not os.path.isdir(rawmel_cache_hint):
                    rawmel_cache_hint = ""
                rawmel_cache = resolve_rawmel_cache_dir(
                    language=lang,
                    format_type=fmt,
                    root_hint=rawmel_cache_hint,
                    extra_roots=[
                        os.path.join(workspace_root, "rawmel_cache_noml_auto"),
                        os.path.join(workspace_root, "rawmel_cache"),
                    ],
                )
                if rawmel_cache:
                    job["rawmel_cache"] = rawmel_cache
                # rawmel backend eval requires cache even when coupled training is skipped.
                if (not rawmel_cache) and ((not args.skip_train) or (not args.skip_eval)):
                    raise RuntimeError(f"rawmel cache not found for {lang}/{fmt}")

            if not args.skip_train:
                if backend == "coupled_nn_v2_rawmel":
                    rawmel_cache = str(job.get("rawmel_cache", "") or "")
                    if not rawmel_cache:
                        raise RuntimeError(f"rawmel cache not found for {lang}/{fmt}")
                    train_meta = train_coupled_bundle_rawmel(
                        language=lang,
                        format_type=fmt,
                        dataset_csv=dataset_for_job,
                        out_dir=out_dir,
                        rawmel_cache_dir=rawmel_cache,
                        group_column=args.group_column,
                        alias_types=alias_types,
                        alias_family=str(args.alias_family or ""),
                        min_mapping_confidence=float(mmc),
                        device=args.device,
                        epochs=int(args.epochs),
                        batch_size=int(args.batch_size),
                        learning_rate=float(args.learning_rate),
                        min_confidence=float(args.min_confidence),
                        progress_every=100,
                    )
                else:
                    train_meta = train_coupled_bundle(
                        language=lang,
                        format_type=fmt,
                        dataset_csv=dataset_for_job,
                        out_dir=out_dir,
                        group_column=args.group_column,
                        alias_types=alias_types,
                        alias_family=str(args.alias_family or ""),
                        min_mapping_confidence=float(mmc),
                        device=args.device,
                        epochs=int(args.epochs),
                        batch_size=int(args.batch_size),
                        learning_rate=float(args.learning_rate),
                        min_confidence=float(args.min_confidence),
                        progress_every=100,
                    )
                job["train_meta"] = train_meta

            if not args.skip_eval:
                eval_summary = evaluate_coupled_bundle(
                    model_dir=out_dir,
                    dataset_csv=dataset_for_job,
                    language=lang,
                    format_type=fmt,
                    device=args.device,
                    rawmel_cache_dir=str(job.get("rawmel_cache", "") or ""),
                )
                eval_summary = _ensure_blank_attach_eval_fields(eval_summary)
                os.makedirs(os.path.dirname(eval_report), exist_ok=True)
                with open(eval_report, "w", encoding="utf-8") as f:
                    json.dump(eval_summary, f, ensure_ascii=False, indent=2)
                job["eval_summary"] = eval_summary
                gate_ok, gate_reasons, gate_rates = _check_blank_attach_gate(
                    eval_summary,
                    max_overall=float(gate_thresholds.get("max_blank_attach_rate", -1.0)),
                    max_head=float(gate_thresholds.get("max_head_blank_attach_rate", -1.0)),
                    max_vv=float(gate_thresholds.get("max_vv_blank_attach_rate", -1.0)),
                    max_delta_rate=float(gate_thresholds.get("max_blank_attach_delta_rate", -1.0)),
                    max_high_pitch_head=float(gate_thresholds.get("max_high_pitch_head_blank_attach_rate", -1.0)),
                )
                job["blank_attach_rates"] = gate_rates
                if gate_reasons:
                    job["blank_attach_gate"] = {"ok": bool(gate_ok), "reasons": gate_reasons}
                if not gate_ok:
                    raise RuntimeError("blank-attach gate failed: " + "; ".join(gate_reasons))

            if selector_enabled:
                selector_min_conf = (
                    float(selector_min_mapping_confidence)
                    if float(selector_min_mapping_confidence) >= 0.0
                    else float(mmc)
                )
                auto_lightgbm_dir = _discover_lightgbm_model_dir(model_root, lang, fmt)
                selector_model_dir = ""
                selector_model_dir_reason = ""
                if selector_model_root:
                    selector_model_dir = os.path.join(selector_model_root, lang, fmt, "v1_selector")
                    selector_model_dir_reason = "selector_model_root"
                elif auto_lightgbm_dir:
                    selector_model_dir = auto_lightgbm_dir
                    selector_model_dir_reason = "auto_detected_lightgbm_route"
                else:
                    selector_model_dir = out_dir
                    selector_model_dir_reason = "fallback_job_model_dir"
                os.makedirs(selector_model_dir, exist_ok=True)

                runtime_notice = ""
                if selector_model_dir_reason == "fallback_job_model_dir":
                    runtime_notice = (
                        "No lightgbm route detected. Selector model was saved in coupled model dir; "
                        "runtime auto-load may skip it unless a lightgbm route is configured."
                    )

                if selector_split_by_alias_family:
                    family_reports: List[Dict[str, object]] = []
                    trained_family_count = 0
                    for family in selector_families:
                        family_alias_types = [v for v in alias_family_to_alias_types(family) if str(v).strip()]
                        if alias_types:
                            alias_type_filter = {str(v).strip().lower() for v in alias_types if str(v).strip()}
                            family_alias_types = [
                                v for v in family_alias_types if str(v).strip().lower() in alias_type_filter
                            ]
                        family_report: Dict[str, object] = {
                            "alias_family": str(family),
                            "alias_types": list(family_alias_types),
                            "status": "pending",
                            "train_meta": {},
                            "eval_summary": {},
                        }
                        if not family_alias_types:
                            family_report["status"] = "skipped"
                            family_report["skip_reason"] = "no_alias_types_for_family"
                            family_reports.append(family_report)
                            continue

                        family_dataset_csv = _selector_family_dataset_path(
                            model_root=model_root,
                            language=lang,
                            format_type=fmt,
                            dataset_csv=dataset_for_job,
                            alias_family=family,
                        )
                        family_dataset_summary = build_selector_dataset_csv_from_delta_dataset(
                            delta_dataset_csv=dataset_for_job,
                            out_csv=family_dataset_csv,
                            language=lang,
                            format_type=fmt,
                            alias_types=family_alias_types,
                            require_train_keep=selector_require_train_keep,
                            min_mapping_confidence=float(selector_min_conf),
                            exclude_nuclei_fallback=selector_exclude_nuclei_fallback,
                        )
                        family_rows = int(family_dataset_summary.get("rows", 0) or 0)
                        family_report["dataset_csv"] = family_dataset_csv
                        family_report["dataset_rows"] = family_rows
                        family_report["dataset_groups"] = int(family_dataset_summary.get("groups", 0) or 0)
                        if family_rows < 16:
                            family_report["status"] = "skipped"
                            family_report["skip_reason"] = "dataset_too_small"
                            family_reports.append(family_report)
                            continue

                        family_model_dir = os.path.join(selector_model_dir, "families", str(family))
                        os.makedirs(family_model_dir, exist_ok=True)
                        family_report["model_dir"] = family_model_dir
                        try:
                            if not args.skip_selector_train:
                                selector_train_meta = train_lightgbm_selector_bundle(
                                    language=lang,
                                    format_type=fmt,
                                    selector_dataset_csv=family_dataset_csv,
                                    out_dir=family_model_dir,
                                    group_column=selector_group_column,
                                    objective=selector_objective,
                                    num_boost_round=selector_num_boost_round,
                                    early_stopping_rounds=selector_early_stopping_rounds,
                                    alias_family=str(family),
                                )
                                family_report["train_meta"] = selector_train_meta
                            if not args.skip_eval:
                                selector_eval_summary = evaluate_lightgbm_selector_bundle(
                                    model_dir=family_model_dir,
                                    selector_dataset_csv=family_dataset_csv,
                                    language=lang,
                                    format_type=fmt,
                                )
                                family_report["eval_summary"] = selector_eval_summary
                                selector_eval_report = os.path.join(family_model_dir, "selector_eval_summary.json")
                                with open(selector_eval_report, "w", encoding="utf-8") as f:
                                    json.dump(selector_eval_summary, f, ensure_ascii=False, indent=2)
                                family_report["eval_report"] = selector_eval_report
                            family_report["status"] = "ok"
                            trained_family_count += 1
                        except Exception as family_exc:
                            family_report["status"] = "failed"
                            family_report["error"] = str(family_exc)
                        family_reports.append(family_report)

                    if trained_family_count <= 0:
                        raise RuntimeError(
                            f"selector family split produced no trainable families for {lang}/{fmt} "
                            f"(required rows>=16)."
                        )

                    selector_report = {
                        "enabled": True,
                        "objective": selector_objective,
                        "split_by_alias_family": True,
                        "model_dir": selector_model_dir,
                        "model_dir_reason": selector_model_dir_reason,
                        "families": family_reports,
                        "families_ok": int(sum(1 for it in family_reports if str(it.get('status')) == "ok")),
                        "families_failed": int(sum(1 for it in family_reports if str(it.get('status')) == "failed")),
                        "families_skipped": int(sum(1 for it in family_reports if str(it.get('status')) == "skipped")),
                    }
                    if runtime_notice:
                        selector_report["runtime_notice"] = runtime_notice
                else:
                    selector_dataset_csv = _selector_dataset_path(
                        model_root=model_root,
                        language=lang,
                        format_type=fmt,
                        dataset_csv=dataset_for_job,
                    )
                    selector_dataset_summary = build_selector_dataset_csv_from_delta_dataset(
                        delta_dataset_csv=dataset_for_job,
                        out_csv=selector_dataset_csv,
                        language=lang,
                        format_type=fmt,
                        require_train_keep=selector_require_train_keep,
                        min_mapping_confidence=float(selector_min_conf),
                        exclude_nuclei_fallback=selector_exclude_nuclei_fallback,
                    )
                    selector_dataset_rows = int(selector_dataset_summary.get("rows", 0) or 0)
                    if selector_dataset_rows < 16:
                        raise RuntimeError(
                            f"selector dataset too small for {lang}/{fmt}: rows={selector_dataset_rows} (<16)"
                        )
                    selector_report = {
                        "enabled": True,
                        "objective": selector_objective,
                        "dataset_csv": selector_dataset_csv,
                        "dataset_rows": selector_dataset_rows,
                        "dataset_groups": int(selector_dataset_summary.get("groups", 0) or 0),
                        "model_dir": selector_model_dir,
                        "model_dir_reason": selector_model_dir_reason,
                        "train_meta": {},
                        "eval_summary": {},
                    }
                    if runtime_notice:
                        selector_report["runtime_notice"] = runtime_notice
                    if not args.skip_selector_train:
                        selector_train_meta = train_lightgbm_selector_bundle(
                            language=lang,
                            format_type=fmt,
                            selector_dataset_csv=selector_dataset_csv,
                            out_dir=selector_model_dir,
                            group_column=selector_group_column,
                            objective=selector_objective,
                            num_boost_round=selector_num_boost_round,
                            early_stopping_rounds=selector_early_stopping_rounds,
                            alias_family=str(args.alias_family or ""),
                        )
                        selector_report["train_meta"] = selector_train_meta
                    if not args.skip_eval:
                        selector_eval_summary = evaluate_lightgbm_selector_bundle(
                            model_dir=selector_model_dir,
                            selector_dataset_csv=selector_dataset_csv,
                            language=lang,
                            format_type=fmt,
                        )
                        selector_report["eval_summary"] = selector_eval_summary
                        selector_eval_report = os.path.join(selector_model_dir, "selector_eval_summary.json")
                        with open(selector_eval_report, "w", encoding="utf-8") as f:
                            json.dump(selector_eval_summary, f, ensure_ascii=False, indent=2)
                        selector_report["eval_report"] = selector_eval_report
                job["selector"] = selector_report

            job["status"] = "ok"
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = str(exc)
            summary["failures"].append(
                {
                    "language": lang,
                    "format_type": fmt,
                    "dataset_csv": dataset_csv,
                    "error": str(exc),
                }
            )
            if args.require_all:
                break

    summary["job_count"] = len(summary["jobs"])
    summary["ok_count"] = sum(1 for j in summary["jobs"] if j.get("status") in {"ok", "dry_run"})
    summary["failed_count"] = len(summary["failures"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.require_all and summary["failed_count"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
