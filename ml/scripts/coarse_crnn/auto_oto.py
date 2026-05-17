"""One-shot Auto-OTO generation for a single voicebank.

Wraps the boundary scorer + decoder + residual pipeline behind a single CLI
that handles source-OTO discovery, language inference, model defaults,
backup of an existing `oto.ini`, and the env-variable preset for whichever
tuning variant the user wants. Removes the need to author a manifest just to
process one voicebank.

Typical use::

    python -m ml.scripts.coarse_crnn.auto_oto C:/path/to/my_voicebank

That picks variant `F2` by default (current best per the listening cycle:
LWC + full residual coverage + extended cons/tail caps) and writes the result
back into the voicebank's `oto.ini`. The existing `oto.ini` is **overwritten
without a backup** by default — users typically keep their own manual backups.
Pass ``--backup`` to write a timestamped copy first.

Override anything that needs overriding::

    python -m ml.scripts.coarse_crnn.auto_oto C:/path/to/vb `
        --variant F1 `
        --language japanese `
        --model C:/custom/scorer.pt `
        --source-oto C:/path/to/reclist.ini `
        --output C:/somewhere/else/oto.ini `
        --backup `
        --dry-run

For the multi-voicebank A/B evaluation cycle (manifests, sample CSVs, HTML
labeler, bias diagnostic), keep using `build_ab_listening_set`,
`install_voicebank_oto`, `build_listening_html`, etc. This script is the
short path for actually generating an OTO you intend to use.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import sys
from typing import Any

from core.coarse_crnn.boundary_generator import generate_oto_with_boundary_decoder
from core.coarse_crnn.lang import infer_language_from_path, normalize_language


VARIANT_PRESETS: dict[str, dict[str, str]] = {
    # No env overrides — pipeline defaults, residual conditional on, LWC off.
    "baseline": {},
    # F-1 (LWC) + residual full coverage (Task 3 finding, 2026-05-15).
    "F1": {
        "UTOA_BOUNDARY_LWC_ENABLE": "on",
        "UTOA_BOUNDARY_RESIDUAL_ENABLE": "on",
        "UTOA_BOUNDARY_RESIDUAL_CONDITIONAL_ENABLE": "on",
        "UTOA_BOUNDARY_RESIDUAL_MODEL_DIR": "ml_workspace/models/coarse_crnn/boundary_residual_v3_slotfix",
        "UTOA_BOUNDARY_RESIDUAL_TARGETS": "delta_overlap,delta_cutoff",
    },
    # F-2: F-1 + extended consonant/tail caps (Quick Win A+B from listening cycle).
    "F2": {
        "UTOA_BOUNDARY_LWC_ENABLE": "on",
        "UTOA_BOUNDARY_RESIDUAL_ENABLE": "on",
        "UTOA_BOUNDARY_RESIDUAL_CONDITIONAL_ENABLE": "on",
        "UTOA_BOUNDARY_RESIDUAL_MODEL_DIR": "ml_workspace/models/coarse_crnn/boundary_residual_v3_slotfix",
        "UTOA_BOUNDARY_RESIDUAL_TARGETS": "delta_overlap,delta_cutoff",
        "UTOA_BOUNDARY_CONS_MIN_MS": "40",
        "UTOA_BOUNDARY_CONS_MAX_MS": "140",
        "UTOA_BOUNDARY_TAIL_MIN_MS": "60",
        "UTOA_BOUNDARY_TAIL_MAX_MS": "220",
        "UTOA_BOUNDARY_MAX_SPAN_MS": "600",
    },
    # F-3: F-2 + filename-token anchor timeline (Phase H option C). Addresses
    # the catastrophic slot mis-mapping observed on 8-mora KO CV-VC wavs
    # ("발음이 뒤죽박죽" listening report 2026-05-16) by using the filename's
    # apostrophe-separated syllable tokens to anchor each slot, instead of
    # linearly partitioning the active region.
    "F3": {
        "UTOA_BOUNDARY_LWC_ENABLE": "on",
        "UTOA_BOUNDARY_RESIDUAL_ENABLE": "on",
        "UTOA_BOUNDARY_RESIDUAL_CONDITIONAL_ENABLE": "on",
        "UTOA_BOUNDARY_RESIDUAL_MODEL_DIR": "ml_workspace/models/coarse_crnn/boundary_residual_v3_slotfix",
        "UTOA_BOUNDARY_RESIDUAL_TARGETS": "delta_overlap,delta_cutoff",
        "UTOA_BOUNDARY_CONS_MIN_MS": "40",
        "UTOA_BOUNDARY_CONS_MAX_MS": "140",
        "UTOA_BOUNDARY_TAIL_MIN_MS": "60",
        "UTOA_BOUNDARY_TAIL_MAX_MS": "220",
        "UTOA_BOUNDARY_MAX_SPAN_MS": "600",
        "UTOA_BOUNDARY_ANCHOR_TIMELINE": "filename",
        # Two-token transition aliases (`u k`, `i p`, etc.) on 8-mora
        # tense+coda wavs collapsed to slot 0 in the 2026-05-17 val gate
        # (mis_mapping_rate 0.697). Compound matcher resolves them to the
        # bare-CV → next-CV adjacency in the filename token list.
        "UTOA_BOUNDARY_COMPOUND_TOKEN_SLOT_MATCH_ENABLE": "on",
        # Per-consonant-class runtime offset shift — all zeroed after the
        # 2026-05-17 source-OTO regeneration. Earlier "voiced stops −200ms
        # early / sonorants +150ms late" diagnostic was driven by row-shift
        # rollback to a broken source-OTO; with a correct source the model's
        # raw output sits within ±100ms for ~74% of VC rows and within ±50ms
        # for the majority of CV/-cv/v-cv. Class-shift values of
        # (200/-150/100/120) drop ±50% match from ~84% to ~25%.
        # Keep knobs exposed so future model checkpoints can re-tune via
        # env override (apply_env_preset now respects pre-existing env).
        "UTOA_BOUNDARY_SHIFT_VOICED_STOP_MS": "0",
        "UTOA_BOUNDARY_SHIFT_SONORANT_MS": "0",
        "UTOA_BOUNDARY_SHIFT_AFFRICATE_MS": "0",
        "UTOA_BOUNDARY_SHIFT_FRICATIVE_MS": "0",
        "UTOA_BOUNDARY_SHIFT_LIQUID_R_MS": "0",
        # Glide multiplier still in place for safety — no-op when all class
        # shifts are 0, but preserves correct scaling if any class shift is
        # re-enabled via env override.
        "UTOA_BOUNDARY_SHIFT_GLIDE_MULTIPLIER": "0.5",
        # VC role shift kept at 0 after the 2026-05-17 sweep against
        # regenerated source-OTO: VC_MS=0 → onset VC meanΔ=+45ms / ±200ms
        # 90%; VC_MS=400 → meanΔ=+314ms / ±200ms 25%. The earlier "−500 to
        # −800 ms early" diagnostic that motivated +400 was an artifact of
        # the broken source-OTO (compounded with row-shift rollback). Keep
        # the knob exposed so future model checkpoints can re-tune.
        "UTOA_BOUNDARY_SHIFT_VC_MS": "0",
        # Coda VC per articulation class. The v16 single −40 knob fixed the
        # +41 ms aggregate mean but over-corrected sonorant codas (which
        # were already perfect via source rollback) while under-correcting
        # plosive codas (still +40~+148 ms late). v17 sweep 0/−60/−90/−120/
        # −150 on plosive showed −150 wins on coda ±50% (77.1%), ±100%
        # (90.0%), and plosive ±100% (71.4%). Class-specific:
        #   - Sonorant (L/M/N/NG/R/ng): 0 — source rollback already correct
        #   - Plosive (K/T/P): −150 — model's V→tense-coda boundary biased late
        #   - Aspirate (H): 0 — near-centered, individual wav outliers only
        "UTOA_BOUNDARY_SHIFT_VC_CODA_SONORANT_MS": "0",
        "UTOA_BOUNDARY_SHIFT_VC_CODA_PLOSIVE_MS": "-150",
        "UTOA_BOUNDARY_SHIFT_VC_CODA_ASPIRATE_MS": "0",
        # VC rows drifting to next-CV onset was still frequent in listening.
        # Keep VC centers away from the far-right edge of the slot and
        # re-center slightly earlier than the old default.
        "UTOA_BOUNDARY_VC_TARGET_RATIO": "0.58",
        "UTOA_BOUNDARY_VC_CLAMP_RIGHT_RATIO": "0.80",
        "UTOA_BOUNDARY_VC_LATE_PENALTY_START_RATIO": "0.76",
        "UTOA_BOUNDARY_VC_LATE_PENALTY_WEIGHT": "1.35",
        # Prevent preutterance/fixed-end near-collapse on transition rows.
        "UTOA_BOUNDARY_TRANSITION_PRE_CONS_MIN_GAP_MS": "24",
        "UTOA_BOUNDARY_TRANSITION_PRE_CONS_MAX_GAP_MS": "72",
        # Keep residual-adjusted transition preutterance inside anchor envelope.
        "UTOA_BOUNDARY_TRANSITION_ENVELOPE_GUARD_ENABLE": "on",
        "UTOA_BOUNDARY_VC_PRE_MAX_RATIO": "0.80",
        # Row-shift rollback intentionally disabled (2026-05-17): source-OTO
        # numerical values must not influence inference output. boundary_
        # generator skips the entire guard path now (empty source dicts),
        # so these flags are inert — kept here for backward-compat with
        # operators who pass UTOA_BOUNDARY_ROW_SHIFT_ROLLBACK_ENABLE=on from
        # the shell (apply_env_preset respects pre-existing env, but the
        # generator ignores the read regardless).
        "UTOA_BOUNDARY_ROW_SHIFT_ROLLBACK_ENABLE": "off",
        # Weak/n-like robustness.
        "UTOA_BOUNDARY_WEAK_AUDIO_RELIABILITY_MAX": "0.42",
        "UTOA_BOUNDARY_LWC_WEAK_THRESHOLD_RELAX": "0.06",
        "UTOA_BOUNDARY_NLIKE_POSTERIOR_CONS_BONUS": "0.45",
        "UTOA_BOUNDARY_NLIKE_POSTERIOR_VOWEL_PENALTY": "0.20",
        # BPM beat-grid prior (2026-05-17): infrastructure ready but
        # disabled by default. KO TABI 8mora diagnostic showed snap REGRESSES
        # accuracy (ALL ±50% 30.9 → 27.3%) because lead-in detection + tail-
        # buffer assumption produce beat grid with 50–150 ms cumulative
        # error — snap then pulls already-correct rows to wrong beats. The
        # acoustic BPM detection itself is accurate (120 BPM at 0.9 conf for
        # TABI bw wav) but phase calibration is the bottleneck.
        # Opt-in for voicebanks recorded with stable BGM + consistent lead-
        # in (will be re-enabled per-voicebank when verified clean):
        #   UTOA_BOUNDARY_BPM_PRIOR_ENABLE=on
        #   UTOA_BOUNDARY_BPM_MS=120     # manual override (more reliable)
        #   UTOA_BOUNDARY_BPM_LEAD_IN_MS=250
        "UTOA_BOUNDARY_BPM_PRIOR_ENABLE": "off",
        "UTOA_BOUNDARY_BPM_RANGE_MIN": "100",
        "UTOA_BOUNDARY_BPM_RANGE_MAX": "150",
        "UTOA_BOUNDARY_BPM_SNAP_TOLERANCE_MS": "120",
        "UTOA_BOUNDARY_BPM_MIN_CONFIDENCE": "0.5",
    },
}

DEFAULT_MODEL_PATH = os.path.join(
    "ml_workspace", "models", "coarse_crnn", "oto_boundary_scorer_v3_slotfix.pt"
)
SOURCE_OTO_CANDIDATES: tuple[str, ...] = (
    "source_oto.ini",
    "reclist.ini",
    "oto.template.ini",
    "oto.source.ini",
)
BACKUP_SUFFIX_PREFIX = "oto.ini.autooto.bak."


def resolve_source_oto(voicebank: str, explicit: str = "") -> str:
    if explicit:
        candidate = os.path.abspath(explicit)
        if not os.path.isfile(candidate):
            raise SystemExit(f"--source-oto file not found: {candidate}")
        return candidate
    for name in SOURCE_OTO_CANDIDATES:
        candidate = os.path.join(voicebank, name)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise SystemExit(
        f"source OTO not found in {voicebank} (tried: {', '.join(SOURCE_OTO_CANDIDATES)}). "
        "Use --source-oto to specify."
    )


def resolve_language(voicebank: str, requested: str) -> str:
    supported = {"korean", "japanese"}
    if requested and requested.strip().lower() != "auto":
        norm = normalize_language(requested)
        if norm in supported:
            return norm
        raise SystemExit(f"unsupported --language value: {requested!r} (supported: korean, japanese, auto)")
    detected = infer_language_from_path(voicebank)
    if detected in supported:
        return detected
    raise SystemExit(
        f"cannot infer language from {voicebank}. Pass --language korean or --language japanese."
    )


def resolve_model_path(requested: str) -> str:
    if requested:
        path = os.path.abspath(requested)
        if not os.path.isfile(path):
            raise SystemExit(f"--model file not found: {path}")
        return path
    default = os.path.abspath(DEFAULT_MODEL_PATH)
    if not os.path.isfile(default):
        raise SystemExit(
            f"default model not found at {default}. Pass --model to override."
        )
    return default


def resolve_output_path(voicebank: str, explicit: str = "") -> str:
    if explicit:
        return os.path.abspath(explicit)
    return os.path.abspath(os.path.join(voicebank, "oto.ini"))


def backup_existing_oto(output_path: str, timestamp: str) -> str | None:
    if not os.path.isfile(output_path):
        return None
    backup_name = f"{BACKUP_SUFFIX_PREFIX}{timestamp}"
    backup_path = os.path.join(os.path.dirname(output_path), backup_name)
    shutil.copy2(output_path, backup_path)
    return backup_path


def apply_env_preset(preset: dict[str, str]) -> dict[str, str | None]:
    """Apply env overrides and return a restore map for the caller to undo with `restore_env`.

    Preset values do NOT overwrite env vars already set by the caller — this
    lets ``UTOA_BOUNDARY_SHIFT_VC_MS=200 python auto_oto.py --variant F3 ...``
    actually override the F3 preset's baked-in value, instead of being silently
    clobbered. The shell-set env takes precedence; the preset only fills in
    keys the caller hasn't already provided.
    """
    restore: dict[str, str | None] = {}
    for key, value in preset.items():
        key_s = str(key)
        if key_s in os.environ:
            continue
        restore[key_s] = os.environ.get(key_s)
        os.environ[key_s] = str(value)
    return restore


def restore_env(restore: dict[str, str | None]) -> None:
    for key, value in restore.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _log_callback(verbose: bool):
    if not verbose:
        return None

    def _emit(text: str) -> None:
        sys.stdout.write(f"{text}\n")
        sys.stdout.flush()

    return _emit


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate oto.ini for a single voicebank using the boundary scorer pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("voicebank", help="Voicebank wav directory (also where oto.ini lands by default).")
    ap.add_argument(
        "--variant",
        default="F2",
        choices=sorted(VARIANT_PRESETS.keys()),
        help="Tuning preset: baseline (raw pipeline), F1 (LWC + full residual), F2 (F1 + extended cons/tail caps), F3 (F2 + filename anchors).",
    )
    ap.add_argument(
        "--language",
        default="auto",
        help="Voicebank language: auto / korean / japanese (auto = infer from path).",
    )
    ap.add_argument(
        "--model",
        default="",
        help=f"Boundary scorer checkpoint. Default: {DEFAULT_MODEL_PATH}",
    )
    ap.add_argument("--source-oto", default="", help="Override source OTO path (default: <voicebank>/source_oto.ini).")
    ap.add_argument("--output", default="", help="Output OTO path (default: <voicebank>/oto.ini).")
    ap.add_argument("--alias-suffix", default="", help="Alias suffix appended to every emitted entry.")
    ap.add_argument("--special-aliases", default="", help="Comma-separated alias list to treat as 'special' role.")
    ap.add_argument("--device", default="auto", help="Torch device (auto/cpu/cuda).")
    ap.add_argument(
        "--backup",
        action="store_true",
        help="Back up the existing oto.ini to oto.ini.autooto.bak.<timestamp> before overwriting. Default off — users typically keep their own manual backups.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Resolve paths and print plan without generating.")
    ap.add_argument("--verbose", action="store_true", help="Forward boundary-pipeline log lines to stdout.")
    ap.add_argument("--summary-json", default="", help="Optional path to write a small JSON summary.")
    args = ap.parse_args()

    voicebank = os.path.abspath(args.voicebank)
    if not os.path.isdir(voicebank):
        raise SystemExit(f"voicebank directory not found: {voicebank}")

    source_oto = resolve_source_oto(voicebank, args.source_oto)
    language = resolve_language(voicebank, args.language)
    model_path = resolve_model_path(args.model)
    output_path = resolve_output_path(voicebank, args.output)
    preset = VARIANT_PRESETS[args.variant]
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    special = {item.strip() for item in str(args.special_aliases or "").split(",") if item.strip()}

    plan = {
        "voicebank": voicebank,
        "source_oto": source_oto,
        "language": language,
        "model": model_path,
        "output": output_path,
        "variant": args.variant,
        "env_overrides": preset,
        "would_backup": (
            os.path.join(os.path.dirname(output_path), f"{BACKUP_SUFFIX_PREFIX}{timestamp}")
            if (args.backup and os.path.isfile(output_path))
            else None
        ),
    }
    print(f"[auto-oto] voicebank   = {voicebank}")
    print(f"[auto-oto] source_oto  = {source_oto}")
    print(f"[auto-oto] language    = {language}")
    print(f"[auto-oto] model       = {model_path}")
    print(f"[auto-oto] output      = {output_path}")
    print(f"[auto-oto] variant     = {args.variant}  (env: {len(preset)} overrides)")
    if plan["would_backup"]:
        print(f"[auto-oto] backup      = {plan['would_backup']}")
    elif os.path.isfile(output_path):
        print("[auto-oto] backup      = (off - existing oto.ini will be overwritten; pass --backup to keep a copy)")

    if args.dry_run:
        print("[auto-oto] dry-run, nothing written")
        if args.summary_json:
            _write_summary(args.summary_json, {"plan": plan, "dry_run": True})
        return 0

    backup_path: str | None = None
    if args.backup:
        backup_path = backup_existing_oto(output_path, timestamp)
        if backup_path:
            print(f"[auto-oto] backed up existing oto.ini → {backup_path}")

    restore = apply_env_preset(preset)
    try:
        processed, total, errors = generate_oto_with_boundary_decoder(
            wav_dir=voicebank,
            out_path=output_path,
            source_oto_path=source_oto,
            language=language,
            format_type="",
            model_path=model_path,
            device=str(args.device),
            alias_suffix=str(args.alias_suffix or ""),
            callback=_log_callback(bool(args.verbose)),
            special_aliases=special,
        )
    finally:
        restore_env(restore)

    summary: dict[str, Any] = {
        "voicebank": voicebank,
        "source_oto": source_oto,
        "language": language,
        "variant": args.variant,
        "model": model_path,
        "output": output_path,
        "backup": backup_path,
        "processed": int(processed),
        "total": int(total),
        "errors": list(errors)[:50],
    }
    print(f"[auto-oto] processed {processed}/{total}  errors={len(errors)}  wrote={output_path}")
    if errors:
        print(f"[auto-oto] first error: {errors[0]}")

    if args.summary_json:
        _write_summary(args.summary_json, summary)

    return 0 if (processed > 0 and not errors) else 1


def _write_summary(path: str, payload: dict[str, Any]) -> None:
    full = os.path.abspath(path)
    os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    with open(full, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.exit(main())
