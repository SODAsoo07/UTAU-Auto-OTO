from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from sklearn.model_selection import GroupShuffleSplit
except Exception:  # pragma: no cover
    GroupShuffleSplit = None

import textgrid

from core.format_type_utils import normalize_format_type
from core.ja_lab_generator import split_ja_romaji_syllable
from core.ja_mapping_v2 import build_ja_cv_anchor_score_grid
from core.ja_oto_mapping import (
    _alias_to_ja_cv_target,
    _clean_phone_mark,
    _is_nucleus_phone,
    _ja_soft_cv_match_level,
    _normalize_ja_syllable_token,
    classify_ja_alias,
)
from core.kr_mapping_v2 import build_kr_cv_anchor_score_grid
from core.kr_oto_mapping import _alias_to_cv_target
from core.kr_oto_rules import _cv_match_score, _kr_cv_kernel, _normalize_cv_match_token, classify_alias, normalize_ipa_mark
from core.lab_generator import decompose_hangul_to_roman
from core.mapping_supervised_runtime import (
    MAPPING_SUPERVISED_FEATURE_NAMES,
    MAPPING_SUPERVISED_FEATURE_VERSION,
)
from core.oto_mapping_plan import build_monotonic_index_plan
from core.oto_ml_collection_discovery import discover_training_candidates_from_dataset_root
from core.oto_ml_features import parse_oto_rows
from core.oto_normalization import normalize_wav_key


DATASET_FIELDNAMES = [
    "group_id",
    "language",
    "format_type",
    "voicebank_id",
    "wav",
    "alias",
    "line_index",
    "target_idx",
    "cand_idx",
    "label",
    "manual_anchor_ms",
    "heuristic_score",
    "manual_score",
    "target_token",
    "candidate_token",
    *MAPPING_SUPERVISED_FEATURE_NAMES,
]


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _candidate_anchor_ms(syl: Dict[str, object], *, vowel_check_fn=None, clean_mark_fn=None) -> float:
    phones = list((syl or {}).get("phones") or [])
    start_ms = _safe_float((syl or {}).get("start_time"), 0.0) * 1000.0
    if not phones:
        return float(start_ms)
    first_any = None
    first_vowel = None
    for p in phones:
        mark = str(getattr(p, "mark", "") or "")
        clean = clean_mark_fn(mark) if clean_mark_fn else mark
        p_start = _safe_float(getattr(p, "minTime", 0.0), 0.0) * 1000.0
        if first_any is None and clean not in {"", "sil", "sp", "spn", "pau"}:
            first_any = p_start
        if first_vowel is None and vowel_check_fn and vowel_check_fn(clean):
            first_vowel = p_start
    if first_vowel is not None:
        return float(first_vowel)
    if first_any is not None:
        return float(first_any)
    return float(start_ms)


def _find_textgrid_for_wav(tg_map: Dict[str, str], wav_name: str) -> str:
    key = normalize_wav_key(str(os.path.splitext(wav_name or "")[0] or ""))
    return str(tg_map.get(key, "") or "")


def _build_textgrid_map(tg_dir: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not tg_dir or not os.path.isdir(tg_dir):
        return out
    for dp, _, fns in os.walk(tg_dir):
        for fn in fns:
            if not str(fn).lower().endswith(".textgrid"):
                continue
            path = os.path.join(dp, fn)
            key = normalize_wav_key(os.path.splitext(fn)[0])
            if key and key not in out:
                out[key] = path
    return out


def _extract_tiers(tg_path: str):
    tg = textgrid.TextGrid.fromFile(tg_path)
    phone_tier = None
    word_tier = None
    for t in tg:
        if not isinstance(t, textgrid.IntervalTier):
            continue
        name = str(getattr(t, "name", "") or "").strip().lower()
        if phone_tier is None and ("phone" in name or name == "phoneme"):
            phone_tier = t
        if word_tier is None and ("word" in name or "mora" in name):
            word_tier = t
    return word_tier, phone_tier


def _build_syllables_info(language: str, word_tier, phone_tier) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    if word_tier is None:
        return out
    ph_intervals = list(phone_tier or [])
    lang = str(language or "").strip().lower()
    for w in word_tier:
        mark = str(getattr(w, "mark", "") or "").strip()
        if not mark:
            continue
        w_start = _safe_float(getattr(w, "minTime", 0.0), 0.0)
        w_end = _safe_float(getattr(w, "maxTime", 0.0), 0.0)
        s_phones = [
            p
            for p in ph_intervals
            if _safe_float(getattr(p, "minTime", 0.0), 0.0) >= (w_start - 0.01)
            and _safe_float(getattr(p, "maxTime", 0.0), 0.0) <= (w_end + 0.01)
        ]
        if lang == "korean":
            roman_parts: List[str] = []
            for ch in mark:
                roman_parts.extend(decompose_hangul_to_roman(ch))
            roman_raw = "".join(roman_parts).lower()
            out.append(
                {
                    "word": mark,
                    "roman": roman_raw,
                    "roman_cv": _kr_cv_kernel(roman_raw),
                    "start_time": w_start,
                    "end_time": w_end,
                    "phones": s_phones,
                }
            )
        else:
            out.append(
                {
                    "word": mark,
                    "roman": _normalize_ja_syllable_token(mark),
                    "start_time": w_start,
                    "end_time": w_end,
                    "phones": s_phones,
                }
            )
    return out


def _manual_target_score(language: str, target_tok: str, syl: Dict[str, object], manual_anchor_ms: float) -> Tuple[float, str]:
    lang = str(language or "").strip().lower()
    if lang == "korean":
        cand_tok = _normalize_cv_match_token(syl.get("roman_cv") or syl.get("roman") or syl.get("word") or "")
        token_score = float(_cv_match_score(target_tok, cand_tok))
        anchor = _candidate_anchor_ms(
            syl,
            vowel_check_fn=lambda m: str(normalize_ipa_mark(m) or "").strip().lower() in {"a", "i", "u", "e", "o"},
            clean_mark_fn=lambda m: str(m or "").strip().lower(),
        )
    else:
        cand_tok = _normalize_ja_syllable_token(syl.get("roman") or syl.get("word") or "")
        soft = int(_ja_soft_cv_match_level(target_tok, cand_tok) or 0) if cand_tok else 0
        token_score = float(soft * 42.0)
        anchor = _candidate_anchor_ms(syl, vowel_check_fn=_is_nucleus_phone, clean_mark_fn=_clean_phone_mark)

    blank = max(0.0, min(1.0, _safe_float((syl or {}).get("blank_confidence"), 0.0)))
    time_diff = abs(float(anchor) - float(manual_anchor_ms))
    score = token_score - (time_diff * 0.10) - (blank * 20.0)
    return float(score), str(cand_tok or "")


def _build_records_from_rows(language: str, rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    lang = str(language or "").strip().lower()
    out: List[Dict[str, object]] = []
    for row in sorted(rows, key=lambda r: int(_safe_float(r.get("line_index"), 0.0))):
        alias = str(row.get("alias", "") or "").strip()
        if not alias:
            continue
        if lang == "korean":
            alias_type = str(classify_alias(alias, None) or "").strip().lower()
            target_tok = str(_alias_to_cv_target(alias, alias_type) or "")
        else:
            alias_type = str(classify_ja_alias(alias, None) or "").strip().lower()
            target_tok = str(_alias_to_ja_cv_target(alias, alias_type) or "")
        if alias_type not in {"cv", "cv_head", "vcv", "mono"}:
            continue
        if not target_tok:
            continue
        offset = _safe_float(row.get("offset"), 0.0)
        pre = _safe_float(row.get("pre"), 0.0)
        out.append(
            {
                "alias": alias,
                "alias_type": alias_type,
                "target_token": target_tok,
                "line_index": int(_safe_float(row.get("line_index"), 0.0)),
                "manual_anchor_ms": float(offset + pre),
            }
        )
    return out


def _ensure_feature_row(row: Dict[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name in MAPPING_SUPERVISED_FEATURE_NAMES:
        out[name] = _safe_float(row.get(name), 0.0)
    return out


def build_mapping_supervised_dataset(
    *,
    dataset_root: str,
    out_csv: str,
    language: str,
    format_types: Optional[Sequence[str]] = None,
    max_voicebanks: int = 0,
    auto_oto_policy: str = "generate-temp",
) -> Dict[str, object]:
    lang = str(language or "").strip().lower()
    if lang not in {"korean", "japanese"}:
        raise ValueError(f"Unsupported language: {language}")
    selected_formats = {
        normalize_format_type(lang, fmt)
        for fmt in (format_types or [])
        if str(fmt or "").strip()
    }
    selected_formats.discard("")

    candidates = discover_training_candidates_from_dataset_root(dataset_root, auto_oto_policy=auto_oto_policy)
    ready = [
        c
        for c in candidates
        if str(c.status or "").strip().lower() == "ready"
        and str(c.language or "").strip().lower() == lang
        and (not selected_formats or str(c.format_type or "").strip().lower() in selected_formats)
    ]
    if max_voicebanks > 0:
        ready = ready[: int(max_voicebanks)]

    rows_out: List[Dict[str, object]] = []
    stats = {
        "language": lang,
        "dataset_root": dataset_root,
        "voicebanks_total": len(ready),
        "voicebanks_used": 0,
        "files_used": 0,
        "groups_total": 0,
        "rows_total": 0,
        "positive_rows": 0,
        "negative_rows": 0,
        "skipped_no_textgrid": 0,
        "skipped_parse_error": 0,
        "skipped_empty_plan": 0,
        "format_filter": sorted(selected_formats),
    }

    for cand in ready:
        tg_map = _build_textgrid_map(cand.tg_dir)
        if not tg_map:
            stats["skipped_no_textgrid"] += 1
            continue
        try:
            manual_rows = parse_oto_rows(cand.manual_oto, language=lang)
        except Exception:
            stats["skipped_parse_error"] += 1
            continue
        by_wav: Dict[str, List[Dict[str, object]]] = {}
        for r in manual_rows:
            wav = str(r.get("wav", "") or "").strip()
            if not wav:
                continue
            by_wav.setdefault(wav, []).append(r)
        if not by_wav:
            continue

        used_in_vb = False
        for wav_name, wav_rows in by_wav.items():
            tg_path = _find_textgrid_for_wav(tg_map, wav_name)
            if not tg_path or not os.path.isfile(tg_path):
                stats["skipped_no_textgrid"] += 1
                continue
            try:
                word_tier, phone_tier = _extract_tiers(tg_path)
                syllables_info = _build_syllables_info(lang, word_tier, phone_tier)
            except Exception:
                stats["skipped_parse_error"] += 1
                continue
            if not syllables_info:
                continue

            records = _build_records_from_rows(lang, wav_rows)
            expected_tokens = [str(r["target_token"]) for r in records]
            if not expected_tokens or len(syllables_info) < len(expected_tokens):
                continue

            if lang == "korean":
                grid = build_kr_cv_anchor_score_grid(expected_tokens, syllables_info, use_mel=True)
            else:
                grid = build_ja_cv_anchor_score_grid(expected_tokens, syllables_info, use_mel=True)
            score_rows = list(grid.get("score_rows") or [])
            feature_rows = list(grid.get("feature_rows") or [])
            if not score_rows or not feature_rows:
                stats["skipped_empty_plan"] += 1
                continue

            manual_score_rows: List[List[float]] = []
            cand_tokens_rows: List[List[str]] = []
            for i, rec in enumerate(records):
                target_tok = str(rec.get("target_token", "") or "")
                manual_anchor_ms = _safe_float(rec.get("manual_anchor_ms"), 0.0)
                row_scores: List[float] = []
                row_tokens: List[str] = []
                for syl in syllables_info:
                    s, cand_tok = _manual_target_score(lang, target_tok, syl, manual_anchor_ms)
                    row_scores.append(float(s))
                    row_tokens.append(cand_tok)
                manual_score_rows.append(row_scores)
                cand_tokens_rows.append(row_tokens)

            label_plan = build_monotonic_index_plan(manual_score_rows) or []
            if len(label_plan) != len(records):
                label_plan = [int(max(range(len(row)), key=lambda idx: row[idx])) for row in manual_score_rows]

            wav_norm = normalize_wav_key(os.path.splitext(wav_name)[0])
            for i, rec in enumerate(records):
                if i >= len(score_rows) or i >= len(feature_rows):
                    continue
                heur_row = list(score_rows[i] or [])
                feat_row = list(feature_rows[i] or [])
                man_row = list(manual_score_rows[i] or [])
                tok_row = list(cand_tokens_rows[i] or [])
                if not heur_row or len(heur_row) != len(feat_row):
                    continue
                label_idx = int(max(0, min(int(label_plan[i]), len(heur_row) - 1)))
                group_id = f"{cand.voicebank_id}|{wav_norm}|{int(rec['line_index'])}|{i}"
                for j in range(len(heur_row)):
                    feat = _ensure_feature_row(feat_row[j] if j < len(feat_row) else {})
                    row = {
                        "group_id": group_id,
                        "language": lang,
                        "format_type": str(cand.format_type or ""),
                        "voicebank_id": str(cand.voicebank_id or ""),
                        "wav": wav_name,
                        "alias": str(rec["alias"]),
                        "line_index": int(rec["line_index"]),
                        "target_idx": int(i),
                        "cand_idx": int(j),
                        "label": 1 if j == label_idx else 0,
                        "manual_anchor_ms": float(rec["manual_anchor_ms"]),
                        "heuristic_score": float(heur_row[j]),
                        "manual_score": float(man_row[j] if j < len(man_row) else 0.0),
                        "target_token": str(rec["target_token"]),
                        "candidate_token": str(tok_row[j] if j < len(tok_row) else ""),
                    }
                    row.update(feat)
                    rows_out.append(row)

            stats["files_used"] += 1
            used_in_vb = True
            stats["groups_total"] += int(len(records))

        if used_in_vb:
            stats["voicebanks_used"] += 1

    stats["rows_total"] = len(rows_out)
    stats["positive_rows"] = int(sum(1 for r in rows_out if int(r.get("label", 0)) == 1))
    stats["negative_rows"] = int(max(0, stats["rows_total"] - stats["positive_rows"]))

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)
    stats["out_csv"] = out_csv
    return stats


def _group_split(df):
    if GroupShuffleSplit is not None and "group_id" in df.columns and df["group_id"].nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, va_idx = next(splitter.split(df, groups=df["group_id"]))
        return tr_idx, va_idx
    split_at = max(1, int(len(df) * 0.8))
    tr_idx = list(range(split_at))
    va_idx = list(range(split_at, len(df))) or list(range(max(0, split_at - 1), len(df)))
    return tr_idx, va_idx


def _top1_hit_rate(df, pred_col: str) -> float:
    if len(df) <= 0 or "group_id" not in df.columns:
        return 0.0
    picked = (
        df.sort_values(["group_id", pred_col], ascending=[True, False])
        .groupby("group_id", as_index=False)
        .head(1)
    )
    if len(picked) <= 0:
        return 0.0
    return float(pd.to_numeric(picked["label"], errors="coerce").fillna(0).mean())


def train_mapping_supervised_model(
    *,
    dataset_csv: str,
    out_dir: str,
    language: str,
    format_type: str,
    num_boost_round: int = 700,
    early_stopping_rounds: int = 60,
) -> Dict[str, object]:
    if lgb is None:
        raise RuntimeError("lightgbm is required")
    if pd is None:
        raise RuntimeError("pandas is required")
    if not os.path.isfile(dataset_csv):
        raise FileNotFoundError(dataset_csv)

    df = pd.read_csv(dataset_csv, low_memory=False)
    if len(df) < 16:
        raise RuntimeError("Not enough rows to train mapping supervised model (need >= 16 rows).")
    for name in MAPPING_SUPERVISED_FEATURE_NAMES:
        if name not in df.columns:
            df[name] = 0.0
        df[name] = pd.to_numeric(df[name], errors="coerce").fillna(0.0)
    if "label" not in df.columns:
        raise RuntimeError("Dataset is missing label column.")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    if float(df["label"].sum()) <= 0.0:
        raise RuntimeError("Dataset has no positive samples.")

    tr_idx, va_idx = _group_split(df)
    tr = df.iloc[tr_idx].copy()
    va = df.iloc[va_idx].copy()
    if len(tr) <= 0 or len(va) <= 0:
        raise RuntimeError("Train/valid split failed.")

    x_tr = tr[MAPPING_SUPERVISED_FEATURE_NAMES]
    x_va = va[MAPPING_SUPERVISED_FEATURE_NAMES]
    y_tr = tr["label"].astype(float)
    y_va = va["label"].astype(float)

    pos = float(max(1.0, float(y_tr.sum())))
    neg = float(max(1.0, float(len(y_tr) - y_tr.sum())))
    scale_pos_weight = float(neg / pos)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "min_data_in_leaf": 40,
        "verbosity": -1,
        "scale_pos_weight": scale_pos_weight,
    }
    tr_ds = lgb.Dataset(x_tr, label=y_tr)
    va_ds = lgb.Dataset(x_va, label=y_va, reference=tr_ds)

    booster = lgb.train(
        params,
        tr_ds,
        num_boost_round=int(max(20, num_boost_round)),
        valid_sets=[va_ds],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(int(max(10, early_stopping_rounds)), verbose=False)],
    )

    va_pred = booster.predict(x_va, num_iteration=booster.best_iteration)
    eval_df = va.copy()
    eval_df["_pred"] = va_pred
    eval_df["_heur"] = pd.to_numeric(eval_df.get("heuristic_score"), errors="coerce").fillna(0.0)
    eval_df["_manual"] = pd.to_numeric(eval_df.get("manual_score"), errors="coerce").fillna(0.0)

    top1_model = _top1_hit_rate(eval_df, "_pred")
    top1_heur = _top1_hit_rate(eval_df, "_heur")
    top1_manual_proxy = _top1_hit_rate(eval_df, "_manual")

    os.makedirs(out_dir, exist_ok=True)
    model_txt = os.path.join(out_dir, "model.txt")
    booster.save_model(model_txt)

    schema = {
        "feature_version": MAPPING_SUPERVISED_FEATURE_VERSION,
        "feature_names": list(MAPPING_SUPERVISED_FEATURE_NAMES),
    }
    schema_path = os.path.join(out_dir, "feature_schema.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    meta = {
        "backend": "mapping_supervised_lightgbm_v1",
        "language": str(language or "").strip().lower(),
        "format_type": str(format_type or "").strip().lower(),
        "feature_version": MAPPING_SUPERVISED_FEATURE_VERSION,
        "feature_names": list(MAPPING_SUPERVISED_FEATURE_NAMES),
        "dataset_csv": dataset_csv,
        "train_rows": int(len(tr)),
        "valid_rows": int(len(va)),
        "groups_train": int(tr["group_id"].nunique()) if "group_id" in tr.columns else 0,
        "groups_valid": int(va["group_id"].nunique()) if "group_id" in va.columns else 0,
        "positive_train": float(y_tr.sum()),
        "positive_valid": float(y_va.sum()),
        "scale_pos_weight": float(scale_pos_weight),
        "num_boost_round": int(num_boost_round),
        "early_stopping_rounds": int(early_stopping_rounds),
        "best_iteration": int(getattr(booster, "best_iteration", 0) or 0),
        "valid_top1_model": float(top1_model),
        "valid_top1_heuristic": float(top1_heur),
        "valid_top1_manual_proxy": float(top1_manual_proxy),
        "model_txt": model_txt,
        "feature_schema_json": schema_path,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


__all__ = [
    "DATASET_FIELDNAMES",
    "build_mapping_supervised_dataset",
    "train_mapping_supervised_model",
]
