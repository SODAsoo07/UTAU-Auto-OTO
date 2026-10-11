from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .alias_generation import generate_bank_alias_rows
from .audio import analyze_wav, file_sha256
from .domain import RenderedOtoRow
from .evaluation import WavDirectoryIndex
from .oto_io import parse_oto, resolve_output_encoding, write_lines_atomic
from .oto_style import OtoStyle
from .planning import (
    fallback_source_rows,
    infer_slot_count,
    plan_aliases,
)
from .project_runtime import generate_project
from .renderer import force_review, render_alias
from .timing_profile import BUILTIN_PROFILE_ID, load_timing_profile, select_timing_profile
from .tempo import MAX_BPM, MIN_BPM, TempoEstimate, beats_for_period, bpm_to_period_ms, estimate_tempo
from .voicebank_project_adapter import VoicebankReviewRequired, build_routed_voicebank_project, parse_filename_phone_slots


DEVELOPMENT_ALIGNMENT_PROFILES = {
    "occurrence_v1": {
        "feature_backend_override": "numpy_multiband_pitch_v2",
        "use_occurrence_priors": True,
        "occurrence_prior_scope": "supported_vcv",
        "occurrence_prior_min_coverage": 0.60,
        "occurrence_prior_weight_scale": 0.75,
        "occurrence_rhythm_tracking": "adaptive_interval",
        "internal_gap_policy": "prior_strict_30ms",
    },
    "isolated_sonorant_v1": {"sustain_isolated_sonorants": True},
    "cvvc_slot_rhythm_v1": {"sustain_isolated_sonorants": True, "cvvc_slot_rhythm": True},
    "cvvc_slot_rhythm_v2": {
        "sustain_isolated_sonorants": True, "cvvc_slot_rhythm": True, "cvvc_nasal_coda_boundary": True,
    },
    "cvvc_slot_rhythm_v3": {
        "sustain_isolated_sonorants": True, "cvvc_slot_rhythm": True, "cvvc_nasal_coda_boundary": True,
        "cvvc_slot_grid_refit": True,
    },
    # v4: v3 slot rhythm + nasal coda onset from an audio band cue, applied to
    # every Korean vowel->nasal coda (not only rhythm slots).
    "cvvc_slot_rhythm_v4": {
        "sustain_isolated_sonorants": True, "cvvc_slot_rhythm": True, "cvvc_slot_grid_refit": True,
        "korean_nasal_coda_onset": True,
    },
    # v5: v4 + ㄹ coda onset from the same file's vowel length before nasal codas.
    "cvvc_slot_rhythm_v5": {
        "sustain_isolated_sonorants": True, "cvvc_slot_rhythm": True, "cvvc_slot_grid_refit": True,
        "korean_nasal_coda_onset": True, "korean_lateral_coda_rhythm": True,
    },
    # v6: v5 + nasal codas with a weak band step (after eu/i) start after the
    # same rhythm vowel length, when the syllable has a consonant onset.
    "cvvc_slot_rhythm_v6": {
        "sustain_isolated_sonorants": True, "cvvc_slot_rhythm": True, "cvvc_slot_grid_refit": True,
        "korean_nasal_coda_onset": True, "korean_lateral_coda_rhythm": True,
        "korean_nasal_coda_rhythm_fallback": True,
    },
    # v7: v6 + stop/affricate closures start where the level drops 15 dB.
    "cvvc_slot_rhythm_v7": {
        "sustain_isolated_sonorants": True, "cvvc_slot_rhythm": True, "cvvc_slot_grid_refit": True,
        "korean_nasal_coda_onset": True, "korean_lateral_coda_rhythm": True,
        "korean_nasal_coda_rhythm_fallback": True, "korean_stop_closure_onset": True,
    },
    # v8: v7 + phrase-final vowels/codas end where voicing stops (before breath, fry, pause or file end).
    "cvvc_slot_rhythm_v8": {
        "sustain_isolated_sonorants": True, "cvvc_slot_rhythm": True, "cvvc_slot_grid_refit": True,
        "korean_nasal_coda_onset": True, "korean_lateral_coda_rhythm": True,
        "korean_nasal_coda_rhythm_fallback": True, "korean_stop_closure_onset": True,
        "korean_ending_voicing_offset": True,
    },
    # v9: v8 + codas without two acoustic nasal codas in the file (ㄹ-only lists,
    # weak nasals) start after 0.255 of the syllable period.
    "cvvc_slot_rhythm_v9": {
        "sustain_isolated_sonorants": True, "cvvc_slot_rhythm": True, "cvvc_slot_grid_refit": True,
        "korean_nasal_coda_onset": True, "korean_lateral_coda_rhythm": True,
        "korean_nasal_coda_rhythm_fallback": True, "korean_stop_closure_onset": True,
        "korean_ending_voicing_offset": True, "korean_coda_period_ratio": True,
    },
    # Pause-aware gaps alone (any format): gaps cost by activity, so pauses and
    # tail silence are skipped instead of stretching phones.
    "pause_aware_v1": {"pause_aware_gaps": True},
    # v10: v9 + pause-aware gaps.
    "cvvc_slot_rhythm_v10": {
        "sustain_isolated_sonorants": True, "cvvc_slot_rhythm": True, "cvvc_slot_grid_refit": True,
        "korean_nasal_coda_onset": True, "korean_lateral_coda_rhythm": True,
        "korean_nasal_coda_rhythm_fallback": True, "korean_stop_closure_onset": True,
        "korean_ending_voicing_offset": True, "korean_coda_period_ratio": True, "pause_aware_gaps": True,
    },
    # v11: v10 + syllable position fixes. Two-pass sonorant identity: cepstral templates per
    # vowel/nasal/liquid learned from the bank's first pass weight the second pass of multi-syllable
    # files with adjacent sonorants (weight 0.7). A nasal or liquid that is a syllable of its own
    # (Korean list '-M', '-L', Japanese moraic n) may be held like a vowel.
    "cvvc_slot_rhythm_v11": {
        "sustain_isolated_sonorants": True, "cvvc_slot_rhythm": True, "cvvc_slot_grid_refit": True,
        "korean_nasal_coda_onset": True, "korean_lateral_coda_rhythm": True,
        "korean_nasal_coda_rhythm_fallback": True, "korean_stop_closure_onset": True,
        "korean_ending_voicing_offset": True, "korean_coda_period_ratio": True, "pause_aware_gaps": True,
        "sonorant_identity_weight": 0.7, "syllabic_sonorant_holds": True,
    },
    # v12: v11 + the coarse decoder pass searches without beam pruning (a pruned coarse pass
    # could lock the refine windows onto a path off by whole syllables), and a pause written in
    # the file name (e.g. "hing'-'al") becomes a silence state anchored to an interior silence.
    "cvvc_slot_rhythm_v12": {
        "sustain_isolated_sonorants": True, "cvvc_slot_rhythm": True, "cvvc_slot_grid_refit": True,
        "korean_nasal_coda_onset": True, "korean_lateral_coda_rhythm": True,
        "korean_nasal_coda_rhythm_fallback": True, "korean_stop_closure_onset": True,
        "korean_ending_voicing_offset": True, "korean_coda_period_ratio": True, "pause_aware_gaps": True,
        "sonorant_identity_weight": 0.7, "syllabic_sonorant_holds": True,
        "exhaustive_coarse_search": True, "explicit_pause_tokens": True,
    },
    # v13: v12 + the decoder searches the same objective without beam or coarse pass
    # (vectorised exact search). The pruned search missed the best path in 168 of 1000 files
    # of the comparison banks (syllable_diag_20261011).
    "cvvc_slot_rhythm_v13": {
        "sustain_isolated_sonorants": True, "cvvc_slot_rhythm": True, "cvvc_slot_grid_refit": True,
        "korean_nasal_coda_onset": True, "korean_lateral_coda_rhythm": True,
        "korean_nasal_coda_rhythm_fallback": True, "korean_stop_closure_onset": True,
        "korean_ending_voicing_offset": True, "korean_coda_period_ratio": True, "pause_aware_gaps": True,
        "sonorant_identity_weight": 0.7, "syllabic_sonorant_holds": True,
        "explicit_pause_tokens": True, "exact_hsmm_search": True,
    },
    # v14: v13 + activity floor for recordings sung almost end to end (their 15th-percentile level
    # is speech, which cut the first syllables off the activity region: JP VCV `_みみみゅ…`).
    "cvvc_slot_rhythm_v14": {
        "sustain_isolated_sonorants": True, "cvvc_slot_rhythm": True, "cvvc_slot_grid_refit": True,
        "korean_nasal_coda_onset": True, "korean_lateral_coda_rhythm": True,
        "korean_nasal_coda_rhythm_fallback": True, "korean_stop_closure_onset": True,
        "korean_ending_voicing_offset": True, "korean_coda_period_ratio": True, "pause_aware_gaps": True,
        "sonorant_identity_weight": 0.7, "syllabic_sonorant_holds": True,
        "explicit_pause_tokens": True, "exact_hsmm_search": True, "dense_speech_activity_floor": True,
    },
    # v15: v14 + a third identity pass (templates learned again from the pass-2 timelines).
    "cvvc_slot_rhythm_v15": {
        "sustain_isolated_sonorants": True, "cvvc_slot_rhythm": True, "cvvc_slot_grid_refit": True,
        "korean_nasal_coda_onset": True, "korean_lateral_coda_rhythm": True,
        "korean_nasal_coda_rhythm_fallback": True, "korean_stop_closure_onset": True,
        "korean_ending_voicing_offset": True, "korean_coda_period_ratio": True, "pause_aware_gaps": True,
        "sonorant_identity_weight": 0.7, "sonorant_identity_passes": 3, "syllabic_sonorant_holds": True,
        "explicit_pause_tokens": True, "exact_hsmm_search": True, "dense_speech_activity_floor": True,
    },
    # v16: v15 + identity templates for every speech phone (consonant class and glides too).
    "cvvc_slot_rhythm_v16": {
        "sustain_isolated_sonorants": True, "cvvc_slot_rhythm": True, "cvvc_slot_grid_refit": True,
        "korean_nasal_coda_onset": True, "korean_lateral_coda_rhythm": True,
        "korean_nasal_coda_rhythm_fallback": True, "korean_stop_closure_onset": True,
        "korean_ending_voicing_offset": True, "korean_coda_period_ratio": True, "pause_aware_gaps": True,
        "sonorant_identity_weight": 0.7, "sonorant_identity_passes": 3, "sonorant_identity_scope": "speech",
        "syllabic_sonorant_holds": True, "explicit_pause_tokens": True, "exact_hsmm_search": True,
        "dense_speech_activity_floor": True,
    },
}


@dataclass(frozen=True)
class GenerationOptions:
    bank: Path
    output: Path
    base_oto: Path | None = None
    recursive: bool = False
    output_encoding: str = "preserve"
    review_threshold: float = 0.45
    allow_heuristic_clean: bool = False
    max_files: int | None = None
    force: bool = False
    allow_in_place_output: bool = False
    format_hint: str | None = None
    language_hint: str | None = None
    timing_profile: Path | None = None
    use_builtin_profiles: bool = True
    backend: str = "legacy_heuristic"
    development_alignment_profile: str | None = None
    occurrence_ranker_model: Path | None = None
    occurrence_ranker_strength: float = 0.0
    occurrence_ranker_provider: str = "auto"
    development_occurrence_override: Path | None = None
    recordings_manifest: Path | None = None
    exclude_wavs: tuple[str, ...] = ()
    # "exclude_row" leaves unbindable base-OTO rows out of oto.ini and lists
    # them in unmapped_aliases.csv instead of stopping the whole bank.
    unmapped_alias_policy: str = "fail"
    # "consonant" ends VC rows inside the next consonant (listening-validated
    # vc_consonant_end); "template" keeps the template expressions.
    vc_end_policy: str = "template"
    # "vowel_end" puts '-' release preutterances at the vowel's end (hand-made reference convention).
    release_pre_policy: str = "template"
    # Without a base oto.ini, write the format's alias list from the file names
    # (alias_generation.py) instead of one CV row per syllable.
    generate_missing_aliases: bool = False
    # Advanced settings: overlap ratios, preutterance shifts, fixed/cutoff margins (oto_style.py).
    oto_style: OtoStyle | None = None
    # Recording tempo. A BPM given by the user wins; otherwise estimate_bpm measures
    # it from steady list files. Either becomes a syllable-grid prior in the decoder
    # (alignment bpm_grid_v1), skipped per file when the pace does not follow it.
    bpm: float | None = None
    estimate_bpm: bool = False
    # Diagnostic files next to oto.ini (routed_hsmm only). Turning them off never changes the OTO rows.
    write_generation_report: bool = True
    write_row_report: bool = True
    write_timelines: bool = True
    write_project_manifest: bool = True
    write_unmapped_aliases_csv: bool = True


DIAGNOSTIC_FILE_OPTIONS = (
    "write_generation_report",
    "write_row_report",
    "write_timelines",
    "write_project_manifest",
    "write_unmapped_aliases_csv",
)
EXCLUDED_ALIAS_SAMPLE_LIMIT = 10


def inspect_voicebank(bank: Path, base_oto: Path | None = None, recursive: bool = False) -> dict[str, Any]:
    bank = bank.resolve()
    wavs = _discover_wavs(bank, recursive)
    payload: dict[str, Any] = {
        "bank": str(bank),
        "wav_count": len(wavs),
        "recursive": recursive,
        "base_oto": str(base_oto.resolve()) if base_oto else None,
    }
    if base_oto:
        rows, encoding, newline = parse_oto(base_oto.resolve())
        wav_index = WavDirectoryIndex(bank, recursive=recursive)
        groups, unresolved = _group_base_rows_by_identity(rows, wav_index)
        mapped = sum(1 for wav in wavs if wav_index.identity_for_path(wav) in groups)
        payload.update(
            {
                "base_oto_rows": len(rows),
                "base_oto_groups": len(groups),
                "base_oto_encoding": encoding,
                "base_oto_newline": "crlf" if newline == "\r\n" else "lf",
                "mapped_wavs": mapped,
                "unmapped_wavs": len(wavs) - mapped,
                "unused_base_groups": max(0, len(groups) - mapped) + len(unresolved),
            }
        )
    return payload


def generate_voicebank(options: GenerationOptions) -> dict[str, Any]:
    backend = options.backend.strip().casefold()
    if options.development_alignment_profile and backend != "routed_hsmm":
        raise ValueError("development alignment profiles require backend=routed_hsmm")
    if options.unmapped_alias_policy != "fail" and backend != "routed_hsmm":
        raise ValueError("unmapped_alias_policy requires backend=routed_hsmm")
    if options.vc_end_policy != "template" and backend != "routed_hsmm":
        raise ValueError("vc_end_policy requires backend=routed_hsmm")
    if options.release_pre_policy != "template" and backend != "routed_hsmm":
        raise ValueError("release_pre_policy requires backend=routed_hsmm")
    if not all(getattr(options, name) for name in DIAGNOSTIC_FILE_OPTIONS) and backend != "routed_hsmm":
        raise ValueError("turning diagnostic files off requires backend=routed_hsmm")
    _validate_occurrence_ranker_options(options, backend=backend)
    _validate_development_occurrence_override(options, backend=backend)
    if backend == "routed_hsmm":
        return _generate_voicebank_routed(options)
    if backend != "legacy_heuristic":
        raise ValueError(f"unsupported generation backend: {options.backend}")
    return _generate_voicebank_legacy(options)


def _generate_voicebank_legacy(options: GenerationOptions) -> dict[str, Any]:
    started = time.perf_counter()
    bank = options.bank.resolve()
    output = options.output.resolve()
    if not bank.is_dir():
        raise FileNotFoundError(f"보이스뱅크 폴더가 없습니다: {bank}")
    _validate_output_location(bank, output, allow_in_place=options.allow_in_place_output)
    if output.exists() and any(output.iterdir()) and not options.force:
        raise FileExistsError(f"출력 폴더가 비어 있지 않습니다. --force가 필요합니다: {output}")
    output.mkdir(parents=True, exist_ok=True)
    timing_profiles, timing_profile_sha256 = load_timing_profile(
        options.timing_profile,
        include_builtin=options.use_builtin_profiles,
    )
    wavs, excluded_wavs = _selected_wavs(bank, options)
    if not wavs:
        raise ValueError(f"WAV를 찾지 못했습니다: {bank}")
    wav_index = WavDirectoryIndex(bank, recursive=options.recursive)

    base_rows = []
    source_encoding: str | None = None
    source_newline = "\r\n"
    base_sha256: str | None = None
    base_groups: dict[str, list] = {}
    unresolved_base_groups: list[str] = []
    if options.base_oto:
        base_path = options.base_oto.resolve()
        base_rows, source_encoding, source_newline = parse_oto(base_path)
        base_sha256 = file_sha256(base_path)
        base_groups, unresolved_base_groups = _group_base_rows_by_identity(base_rows, wav_index)
    encoding = resolve_output_encoding(options.output_encoding, source_encoding)

    all_rows: list[RenderedOtoRow] = []
    row_timing_sources: list[str] = []
    timelines: list[dict[str, Any]] = []
    mapped_keys: set[str] = set()
    mapping_warnings: list[dict[str, str]] = []
    rows_by_directory: dict[str, list[RenderedOtoRow]] = defaultdict(list)
    for wav in wavs:
        relative = wav.relative_to(bank)
        key = wav_index.identity_for_path(wav)
        source_rows = base_groups.get(key)
        if source_rows:
            mapped_keys.add(key)
        else:
            source_rows = fallback_source_rows(wav.name)
            mapping_warnings.append({"wav": relative.as_posix(), "reason": "base_alias_plan_missing_filename_fallback"})
        slot_count = infer_slot_count(wav.name, source_rows, options.format_hint)
        decoder_mode = _select_decoder_mode(
            wav,
            language_hint=options.language_hint,
            format_hint=options.format_hint,
        )
        timeline = analyze_wav(wav, slot_count, decoder_mode=decoder_mode)
        timeline_payload = timeline.to_dict()
        timeline_payload["wav"] = relative.as_posix()
        timeline_payload["alias_source"] = "base_oto_alias_only" if key in base_groups else "filename_fallback"
        timeline_payload["slot_count"] = slot_count
        timelines.append(timeline_payload)
        planned = plan_aliases(source_rows, slot_count, options.format_hint, wav.name)
        for plan in planned:
            profile_entry = select_timing_profile(
                timing_profiles,
                language=options.language_hint,
                format_type=options.format_hint,
                role=plan.role,
            )
            row = render_alias(
                relative.name,
                timeline,
                plan,
                review_threshold=options.review_threshold,
                timing_profile=profile_entry,
                format_hint=options.format_hint,
            )
            if key not in base_groups:
                row = force_review(row, "filename_fallback_alias_plan")
            if not options.allow_heuristic_clean:
                row = force_review(row, "uncalibrated_acoustic_heuristic")
            all_rows.append(row)
            row_timing_sources.append(
                "audio_plus_dataset_timing_profile" if profile_entry else "audio_only_no_base_oto_timing"
            )
            directory_key = relative.parent.as_posix()
            rows_by_directory[directory_key].append(row)

    timeline_dir = output / "timelines"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    for payload in timelines:
        safe_name = _safe_timeline_name(payload["wav"], payload["wav_sha256"])
        _write_json(timeline_dir / safe_name, payload, force=options.force)

    output_paths: list[dict[str, str]] = []
    for directory_key, rows in sorted(rows_by_directory.items()):
        target_dir = output if directory_key in {"", "."} else output / "oto_by_directory" / directory_key
        canonical = target_dir / "oto.ini"
        clean = target_dir / "oto.clean.ini"
        review = target_dir / "oto.review.ini"
        write_lines_atomic(canonical, (row.to_line() for row in rows), encoding=encoding, newline=source_newline)
        write_lines_atomic(clean, (row.to_line() for row in rows if row.split == "clean"), encoding=encoding, newline=source_newline)
        write_lines_atomic(review, (row.to_line() for row in rows if row.split == "review"), encoding=encoding, newline=source_newline)
        output_paths.append(
            {
                "source_directory": directory_key,
                "oto": str(canonical),
                "clean": str(clean),
                "review": str(review),
            }
        )

    write_lines_atomic(
        output / "row_report.jsonl",
        (
            json.dumps(
                {
                    **row.to_report(),
                    "timing_source": timing_source,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            for row, timing_source in zip(all_rows, row_timing_sources)
        ),
        encoding="utf-8",
        newline="\n",
    )
    split_counts = Counter(row.split for row in all_rows)
    role_counts = Counter(row.role for row in all_rows)
    reason_counts = Counter(reason for row in all_rows for reason in row.reasons)
    profiled_row_count = sum(source == "audio_plus_dataset_timing_profile" for source in row_timing_sources)
    timing_profile_source = (
        str(options.timing_profile.resolve())
        if options.timing_profile
        else BUILTIN_PROFILE_ID if profiled_row_count else None
    )
    report = {
        "schema_version": 1,
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "app_version": __version__,
        "bank": str(bank),
        "output": str(output),
        "wav_count": len(wavs),
        "generated_rows": len(all_rows),
        "clean_rows": split_counts.get("clean", 0),
        "review_rows": split_counts.get("review", 0),
        "average_confidence": sum(row.confidence for row in all_rows) / max(1, len(all_rows)),
        "confidence_calibrated": False,
        "role_counts": dict(sorted(role_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "mapping_warnings": mapping_warnings,
        "mapped_wavs": len(mapped_keys),
        "unmapped_wavs": len(wavs) - len(mapped_keys),
        "unused_base_groups": [*sorted(set(base_groups) - mapped_keys), *sorted(unresolved_base_groups)],
        "output_encoding": encoding,
        "source_newline": "crlf" if source_newline == "\r\n" else "lf",
        "output_paths": output_paths,
        "format_hint": options.format_hint,
        "language_hint": options.language_hint,
        "generation_backend": "legacy_heuristic",
        "in_place_output": output == bank,
        "elapsed_seconds": time.perf_counter() - started,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "provenance": {
            "generation_backend": "legacy_heuristic",
            "transcript_source": "original_filename_only",
            "alias_source": "base_oto_alias_only" if options.base_oto else "filename_fallback",
            "label_source": "acoustic_heuristic_v1",
            "timing_source": "audio_plus_dataset_timing_profile" if profiled_row_count else "audio_only_no_base_oto_timing",
            "timing_profile": timing_profile_source,
            "timing_profile_sha256": timing_profile_sha256,
            "builtin_profile_enabled": options.use_builtin_profiles,
            "profiled_row_count": profiled_row_count,
            "base_oto": str(options.base_oto.resolve()) if options.base_oto else None,
            "base_oto_sha256": base_sha256,
            "base_oto_timing_reused": False,
            "heuristic_clean_allowed": options.allow_heuristic_clean,
        },
    }
    _write_json(output / "generation_report.json", report, force=True)
    _write_json(
        output / "project_manifest.json",
        {
            "schema_version": 1,
            "bank": str(bank),
            "wavs": [
                {
                    "wav": payload["wav"],
                    "wav_sha256": payload["wav_sha256"],
                    "slot_count": payload["slot_count"],
                    "transcript_source": "original_filename_only",
                    "alias_source": payload["alias_source"],
                    "label_source": "acoustic_heuristic_v1",
                }
                for payload in timelines
            ],
        },
        force=True,
    )
    return report


def _generate_voicebank_routed(options: GenerationOptions) -> dict[str, Any]:
    started = time.perf_counter()
    bank = options.bank.resolve()
    output = options.output.resolve()
    if not bank.is_dir():
        raise FileNotFoundError(bank)
    _validate_output_location(bank, output, allow_in_place=options.allow_in_place_output)
    if output.exists() and any(output.iterdir()) and not options.force:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    wavs, excluded_wavs = _selected_wavs(bank, options)
    if not wavs:
        raise ValueError(f"no WAV files found: {bank}")
    wav_index = WavDirectoryIndex(bank, recursive=options.recursive)

    source_encoding: str | None = None
    source_newline = "\r\n"
    base_sha256: str | None = None
    base_groups: dict[str, list] = {}
    unresolved_base_groups: list[str] = []
    if options.base_oto:
        base_path = options.base_oto.resolve()
        base_rows, source_encoding, source_newline = parse_oto(base_path)
        base_sha256 = file_sha256(base_path)
        base_groups, unresolved_base_groups = _group_base_rows_by_identity(base_rows, wav_index)
    resolve_output_encoding(options.output_encoding, source_encoding)

    mapped_keys: set[str] = set()
    base_rows_by_wav: dict[str, tuple] = {}
    for wav in wavs:
        key = wav_index.identity_for_path(wav)
        rows = tuple(base_groups.get(key, ()))
        if rows:
            mapped_keys.add(key)
        base_rows_by_wav[wav.relative_to(bank).as_posix()] = rows
    generated_alias_wavs = 0
    alias_generation_skipped: list[dict[str, str]] = []
    if options.generate_missing_aliases and not options.base_oto:
        if not options.format_hint:
            raise ValueError("alias generation needs a format (cv, cvc, cvvc or vcv)")
        by_directory: dict[str, list[str]] = defaultdict(list)
        for relative in base_rows_by_wav:
            by_directory[str(Path(relative).parent)].append(relative)
        for relatives in by_directory.values():
            generated, skipped = generate_bank_alias_rows(
                sorted(relatives), format_hint=options.format_hint, language_hint=options.language_hint
            )
            alias_generation_skipped.extend(skipped)
            for relative, rows in generated.items():
                if rows:
                    base_rows_by_wav[relative] = tuple(rows)
                    generated_alias_wavs += 1
        # Files whose names cannot be read are left out and listed, like unbindable OTO rows.
        unreadable = {item["wav"] for item in alias_generation_skipped}
        if unreadable:
            wavs = [wav for wav in wavs if wav.relative_to(bank).as_posix() not in unreadable]
            for relative in unreadable:
                base_rows_by_wav.pop(relative, None)
            if not wavs:
                raise ValueError("no WAV file name could be read for alias generation")

    build = build_routed_voicebank_project(
        bank=bank,
        wavs=wavs,
        base_rows_by_wav=base_rows_by_wav,
        format_hint=options.format_hint,
        language_hint=options.language_hint,
        recursive=options.recursive,
        output_encoding=options.output_encoding,
        evidence_provider=(options.occurrence_ranker_provider if options.occurrence_ranker_model is not None else "cuda"),
        recordings_manifest=options.recordings_manifest,
        unmapped_alias_policy=options.unmapped_alias_policy,
        vc_end_policy=options.vc_end_policy,
        release_pre_policy=options.release_pre_policy,
    )
    tempo = resolve_recording_tempo(options, bank, wavs)
    project_manifest_path = output / "project_manifest.json" if options.write_project_manifest else None
    voicebank_input_hashes = {
        **dict(build.input_hashes),
        "base_oto_sha256": base_sha256,
    }
    development_profile = options.development_alignment_profile
    if development_profile is not None and development_profile not in DEVELOPMENT_ALIGNMENT_PROFILES:
        raise ValueError(f"unsupported development alignment profile: {development_profile}")
    alignment_options = dict(
        DEVELOPMENT_ALIGNMENT_PROFILES[development_profile]
        if development_profile is not None
        else {}
    )
    if tempo.get("syllable_period_ms") is not None:
        alignment_options["syllable_period_ms"] = float(tempo["syllable_period_ms"])
    occurrence_ranker = _occurrence_ranker_provenance(options)
    manual_occurrence_override = _load_development_occurrence_override(options.development_occurrence_override)
    if manual_occurrence_override is not None:
        alignment_options.update(
            {
                "manual_occurrence_start_anchors_by_wav": manual_occurrence_override["anchors_by_wav"],
                "manual_occurrence_anchor_provenance": manual_occurrence_override["provenance"],
            }
        )
    if occurrence_ranker["requested"]:
        alignment_options.update(
            {
                "occurrence_ranker_model": options.occurrence_ranker_model.resolve(),
                "occurrence_ranker_strength": options.occurrence_ranker_strength,
            }
        )
    report = generate_project(
        build.project,
        oto_style=options.oto_style,
        output_oto=output / "oto.ini",
        force=options.force,
        source_encoding=source_encoding,
        output_newline=source_newline,
        alignment_options=alignment_options,
        write_generation_report=options.write_generation_report,
        write_row_report=options.write_row_report,
        write_timelines=options.write_timelines,
        provenance_context={
            "generation_backend": "routed_hsmm",
            "route_id": build.route_id,
            "voicebank": str(bank),
            "project_manifest": str(project_manifest_path) if project_manifest_path is not None else None,
            "base_oto": str(options.base_oto.resolve()) if options.base_oto else None,
            "base_oto_sha256": base_sha256,
            "base_oto_timing_reused": False,
            "voicebank_input_hashes": voicebank_input_hashes,
            "excluded_wavs": excluded_wavs,
            "mapped_wavs": len(mapped_keys),
            "unmapped_wavs": len(wavs) - len(mapped_keys),
            "unused_base_groups": [
                *sorted(set(base_groups) - mapped_keys),
                *sorted(unresolved_base_groups),
            ],
            "heuristic_clean_allowed": False,
            "development_alignment_profile": development_profile,
            "occurrence_ranker": occurrence_ranker,
            "unmapped_alias_policy": options.unmapped_alias_policy,
            "vc_end_policy": options.vc_end_policy,
            "release_pre_policy": options.release_pre_policy,
            "oto_style": options.oto_style.to_dict() if options.oto_style is not None else None,
            "recording_tempo": tempo,
            "excluded_alias_rows": len(build.excluded_rows),
            "manual_occurrence_override": (
                manual_occurrence_override["provenance"] if manual_occurrence_override is not None else None
            ),
        },
    )
    manifest = {
        **dict(build.manifest),
        "base_oto": str(options.base_oto.resolve()) if options.base_oto else None,
        "base_oto_sha256": base_sha256,
        "mapped_wavs": len(mapped_keys),
        "unmapped_wavs": len(wavs) - len(mapped_keys),
        "unused_base_groups": [
            *sorted(set(base_groups) - mapped_keys),
            *sorted(unresolved_base_groups),
        ],
        "occurrence_ranker": occurrence_ranker,
        "excluded_wavs": excluded_wavs,
        "manual_occurrence_override": (
            manual_occurrence_override["provenance"] if manual_occurrence_override is not None else None
        ),
    }
    if project_manifest_path is not None:
        _write_json(project_manifest_path, manifest, force=True)
    report.update(
        {
            "generation_backend": "routed_hsmm",
            "bank": str(bank),
            "output": str(output),
            "in_place_output": output == bank,
            "wav_count": len(wavs),
            "excluded_wavs": excluded_wavs,
            "mapped_wavs": len(mapped_keys),
            "unmapped_wavs": len(wavs) - len(mapped_keys),
            "project_manifest": str(project_manifest_path) if project_manifest_path is not None else None,
            "elapsed_seconds": time.perf_counter() - started,
            "excluded_alias_rows": len(build.excluded_rows),
            "excluded_alias_samples": [
                {"wav": row["wav"], "alias": row["alias"], "reason": row["reason"]}
                for row in build.excluded_rows[:EXCLUDED_ALIAS_SAMPLE_LIMIT]
            ],
            "generated_alias_wavs": generated_alias_wavs,
            "recording_tempo": tempo,
            "alias_generation_skipped": alias_generation_skipped,
        }
    )
    report["diagnostic_files"] = {
        **dict(report.get("diagnostic_files", {})),
        "project_manifest": options.write_project_manifest,
        "unmapped_aliases_csv": options.write_unmapped_aliases_csv,
    }
    if build.excluded_rows and options.write_unmapped_aliases_csv:
        excluded_csv = output / "unmapped_aliases.csv"
        with excluded_csv.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["wav", "alias", "source_index", "reason", "detail", "written_to_oto"])
            writer.writeheader()
            writer.writerows(build.excluded_rows)
        report["excluded_alias_rows_csv"] = str(excluded_csv)
    return report


def resolve_recording_tempo(options: GenerationOptions, bank: Path, wavs: list[Path]) -> dict[str, Any]:
    """Syllable period for the decoder from the user's BPM and/or the bank estimate."""

    estimate: TempoEstimate | None = None
    if options.estimate_bpm:
        estimate = estimate_tempo(
            wavs,
            lambda wav: filename_syllable_count(wav.name, options.format_hint, options.language_hint),
        )
    payload: dict[str, Any] = {
        "user_bpm": options.bpm,
        "estimate_enabled": bool(options.estimate_bpm),
        "estimate": estimate.to_dict() if estimate is not None else None,
        "syllable_period_ms": None,
        "source": "off",
    }
    if options.bpm is not None:
        bpm = float(options.bpm)
        if not MIN_BPM <= bpm <= MAX_BPM:
            raise ValueError(f"BPM must be between {MIN_BPM:g} and {MAX_BPM:g}")
        per_beat = beats_for_period(estimate.syllable_period_ms, bpm) if estimate is not None and estimate.ok else 1.0
        payload.update(syllable_period_ms=bpm_to_period_ms(bpm, per_beat), syllables_per_beat=per_beat, source="user_bpm")
    elif estimate is not None and estimate.ok:
        payload.update(syllable_period_ms=estimate.syllable_period_ms, syllables_per_beat=estimate.syllables_per_beat,
                       source="estimated")
    return payload


def filename_syllable_count(wav_name: str, format_hint: str | None, language_hint: str | None) -> int | None:
    """Sung syllables named by a file, or None when the name cannot be read or holds a pause."""

    if not format_hint:
        return None
    try:
        slots = parse_filename_phone_slots(wav_name, format_hint=format_hint, language_hint=language_hint)
    except VoicebankReviewRequired:
        return None
    special = {"exh", "vf", "sil", "q", "pau", "br", "breath"}
    regular = [slot for slot in slots if not all(phone in special for phone in slot)]
    if any(all(phone in special for phone in slot) for slot in slots[:-1]):
        return None
    return len(regular)


def _validate_output_location(bank: Path, output: Path, *, allow_in_place: bool) -> None:
    if output == bank:
        if not allow_in_place:
            raise ValueError("원본 보호를 위해 보이스뱅크 직접 저장 허용이 필요합니다")
        return
    if output.is_relative_to(bank):
        raise ValueError("원본 보호를 위해 output은 보이스뱅크 하위 폴더가 될 수 없습니다")


def _validate_occurrence_ranker_options(options: GenerationOptions, *, backend: str) -> None:
    model = options.occurrence_ranker_model
    strength = float(options.occurrence_ranker_strength)
    if model is None and strength == 0.0:
        return
    if backend != "routed_hsmm":
        raise ValueError("occurrence ranker requires backend=routed_hsmm")
    if model is None:
        raise ValueError("occurrence_ranker_model is required when occurrence_ranker_strength is set")
    if options.occurrence_ranker_provider.strip().casefold() not in {"cuda", "cpu", "auto"}:
        raise ValueError("occurrence_ranker_provider must be cuda, cpu, or auto")
    if not model.resolve().is_file():
        raise FileNotFoundError(f"occurrence ranker model is missing: {model.resolve()}")
    if not 0.0 < strength <= 0.35:
        raise ValueError("occurrence_ranker_strength must be between 0 and 0.35")


def _selected_wavs(bank: Path, options: GenerationOptions) -> tuple[list[Path], list[dict[str, str]]]:
    all_wavs = _discover_wavs(bank, options.recursive)
    by_relative = {path.relative_to(bank).as_posix(): path for path in all_wavs}
    requested = {str(value).replace("\\", "/").lstrip("./") for value in options.exclude_wavs if str(value).strip()}
    unknown = sorted(requested - set(by_relative))
    if unknown:
        raise ValueError("excluded WAV is not present in the selected bank: " + ", ".join(unknown))
    excluded = [
        {"wav": relative, "wav_sha256": file_sha256(path)}
        for relative, path in sorted(by_relative.items())
        if relative in requested
    ]
    wavs = [path for relative, path in sorted(by_relative.items()) if relative not in requested]
    if options.max_files is not None:
        wavs = wavs[: max(0, options.max_files)]
    return wavs, excluded


def _validate_development_occurrence_override(options: GenerationOptions, *, backend: str) -> None:
    artifact = options.development_occurrence_override
    if artifact is None:
        return
    if backend != "routed_hsmm":
        raise ValueError("development occurrence override requires backend=routed_hsmm")
    if not artifact.resolve().is_file():
        raise FileNotFoundError(f"development occurrence override is missing: {artifact.resolve()}")


def _load_development_occurrence_override(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read development occurrence override: {resolved}") from exc
    if not isinstance(payload, dict):
        raise ValueError("development occurrence override must be a JSON object")
    if payload.get("candidate_kind") != "auto_oto_hsmm_exact_recording_manual_occurrence_override_development_only":
        raise ValueError("development occurrence override has an unexpected candidate_kind")
    if payload.get("candidate_status") != "ready_for_development_constrained_redecode":
        raise ValueError("development occurrence override is not ready for constrained re-decode")
    if any(bool(payload.get(key)) for key in ("trainable", "promotion_eligible")) or not bool(payload.get("review_only")):
        raise ValueError("development occurrence override must remain review-only")
    rows = payload.get("overrides")
    if not isinstance(rows, list) or not rows:
        raise ValueError("development occurrence override has no state anchors")
    anchors_by_wav: dict[str, dict[str, float]] = {}
    hashes_by_wav: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("development occurrence override contains an invalid row")
        wav = str(row.get("wav") or "").strip()
        state_id = str(row.get("state_instance_id") or "").strip()
        wav_sha256 = str(row.get("wav_sha256") or "").strip()
        try:
            start_ms = float(row["reviewed_start_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("development occurrence override has an invalid reviewed_start_ms") from exc
        if not wav or not state_id or not wav_sha256 or start_ms < 0.0:
            raise ValueError("development occurrence override has an incomplete state anchor")
        if wav in hashes_by_wav and hashes_by_wav[wav] != wav_sha256:
            raise ValueError(f"development occurrence override has inconsistent WAV hashes: {wav}")
        current = anchors_by_wav.setdefault(wav, {})
        if state_id in current:
            raise ValueError(f"development occurrence override has duplicate state anchors: {wav}/{state_id}")
        hashes_by_wav[wav] = wav_sha256
        current[state_id] = start_ms
    return {
        "anchors_by_wav": anchors_by_wav,
        "provenance": {
            "candidate_path": str(resolved),
            "candidate_sha256": file_sha256(resolved),
            "candidate_kind": payload["candidate_kind"],
            "candidate_status": payload["candidate_status"],
            "review_only": True,
            "trainable": False,
            "promotion_eligible": False,
            "anchor_wav_count": len(anchors_by_wav),
            "anchor_state_count": sum(len(values) for values in anchors_by_wav.values()),
            "wav_sha256_by_wav": hashes_by_wav,
        },
    }


def _occurrence_ranker_provenance(options: GenerationOptions) -> dict[str, object]:
    model = options.occurrence_ranker_model
    if model is None:
        return {"requested": False, "model_path": None, "model_sha256": None, "strength": 0.0}
    resolved = model.resolve()
    return {
        "requested": True,
        "model_path": str(resolved),
        "model_sha256": file_sha256(resolved),
        "strength": float(options.occurrence_ranker_strength),
        "provider": options.occurrence_ranker_provider.strip().casefold(),
    }


def _group_base_rows_by_identity(rows, wav_index: WavDirectoryIndex) -> tuple[dict[str, list], list[str]]:
    groups: dict[str, list] = defaultdict(list)
    unresolved: set[str] = set()
    for row in rows:
        try:
            identity = wav_index.identity(row.source_wav)
        except (FileNotFoundError, ValueError):
            unresolved.add(row.source_wav)
            continue
        groups[identity].append(row)
    return dict(groups), sorted(unresolved)


def _select_decoder_mode(
    wav: Path,
    *,
    language_hint: str | None,
    format_hint: str | None,
) -> str:
    language = (language_hint or "").casefold()
    format_type = (format_hint or "").casefold()
    if language != "korean":
        return "local"
    if format_type == "cvc":
        stem = wav.stem
        return "monotonic" if "_" in stem and "'" not in stem else "local"
    return "monotonic"


def _discover_wavs(bank: Path, recursive: bool) -> list[Path]:
    iterator = bank.rglob("*.wav") if recursive else bank.glob("*.wav")
    return sorted((path for path in iterator if path.is_file()), key=lambda path: path.relative_to(bank).as_posix().casefold())


def _safe_timeline_name(relative_wav: str, digest: str) -> str:
    stem = Path(relative_wav).stem
    safe = "".join(character if character.isalnum() or character in "-_." else "_" for character in stem)
    return f"{safe[:80]}__{digest[:10]}.timeline.json"


def _write_json(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(path)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    write_lines_atomic(path, text.splitlines(), encoding="utf-8", newline="\n", backup_existing=path.exists())
