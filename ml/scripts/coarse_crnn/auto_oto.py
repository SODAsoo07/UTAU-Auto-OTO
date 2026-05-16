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
        "UTOA_BOUNDARY_RESIDUAL_CONDITIONAL_ENABLE": "off",
        "UTOA_BOUNDARY_RESIDUAL_MODEL_DIR": "ml_workspace/models/coarse_crnn/boundary_residual_v3_slotfix",
        "UTOA_BOUNDARY_RESIDUAL_TARGETS": "delta_offset,delta_preutterance,delta_overlap,delta_cutoff",
    },
    # F-2: F-1 + extended consonant/tail caps (Quick Win A+B from listening cycle).
    "F2": {
        "UTOA_BOUNDARY_LWC_ENABLE": "on",
        "UTOA_BOUNDARY_RESIDUAL_ENABLE": "on",
        "UTOA_BOUNDARY_RESIDUAL_CONDITIONAL_ENABLE": "off",
        "UTOA_BOUNDARY_RESIDUAL_MODEL_DIR": "ml_workspace/models/coarse_crnn/boundary_residual_v3_slotfix",
        "UTOA_BOUNDARY_RESIDUAL_TARGETS": "delta_offset,delta_preutterance,delta_overlap,delta_cutoff",
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
        "UTOA_BOUNDARY_RESIDUAL_CONDITIONAL_ENABLE": "off",
        "UTOA_BOUNDARY_RESIDUAL_MODEL_DIR": "ml_workspace/models/coarse_crnn/boundary_residual_v3_slotfix",
        "UTOA_BOUNDARY_RESIDUAL_TARGETS": "delta_offset,delta_preutterance,delta_overlap,delta_cutoff",
        "UTOA_BOUNDARY_CONS_MIN_MS": "40",
        "UTOA_BOUNDARY_CONS_MAX_MS": "140",
        "UTOA_BOUNDARY_TAIL_MIN_MS": "60",
        "UTOA_BOUNDARY_TAIL_MAX_MS": "220",
        "UTOA_BOUNDARY_MAX_SPAN_MS": "600",
        "UTOA_BOUNDARY_ANCHOR_TIMELINE": "filename",
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
    """Apply env overrides and return a restore map for the caller to undo with `restore_env`."""
    restore: dict[str, str | None] = {}
    for key, value in preset.items():
        restore[str(key)] = os.environ.get(str(key))
        os.environ[str(key)] = str(value)
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
