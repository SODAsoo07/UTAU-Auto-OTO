"""Profile the HSMM OTO generation pipeline to identify CPU bottlenecks.

Usage:
    python scripts/profile_hsmm_pipeline.py <wav_dir> [--max-wavs N] [--language LANG]

Outputs:
    - Console: top 40 cumulative-time functions
    - scripts/profile_output.prof: full pstats binary (viewable with snakeviz)
"""
from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_pipeline(wav_dir: str, max_wavs: int, language: str, format_type: str):
    from pathlib import Path
    from core.mfa_free_oto.features import extract_features
    from core.mfa_free_oto.runtime_inference import predict_wav
    from core.mfa_free_oto.row_plan import build_filename_slots, build_filename_template_rows
    from core.mfa_free_oto.hsmm_adapter import decode_filename_slots_with_hsmm
    from core.mfa_free_oto.evidence_pack import (
        build_acoustic_evidence_pack,
        candidate_priors_from_evidence_pack,
    )
    from core.mfa_free_oto.oto_adapter import (
        OtoAdapterConfig,
        adapt_template_row,
        assign_template_row_anchors,
        timeline_expected_slots_for_template_rows,
    )
    from core.mfa_free_oto.workflow import _event_source_for_oto, _expected_phones

    wav_files = sorted(Path(wav_dir).glob("*.wav"))[:max_wavs]
    total = len(wav_files)
    print(f"Profiling {total} WAV files from: {wav_dir}")
    print(f"Language: {language}, Format: {format_type}")
    print("-" * 60)

    timings = {
        "extract_features": 0.0,
        "predict_wav": 0.0,
        "hsmm_decode": 0.0,
        "evidence_pack": 0.0,
        "assign_anchors": 0.0,
        "adapt_rows": 0.0,
    }

    for i, wav_path in enumerate(wav_files):
        row_plan_slots = build_filename_slots(wav_path.name, language=language, format_type=format_type)
        template_group, row_plan_phones, _ = build_filename_template_rows(
            wav_path.name, language=language, format_type=format_type,
        )
        expected_phones = list(row_plan_phones or _expected_phones(wav_path.name, [], filename_slots=row_plan_slots))
        expected_slots = (
            timeline_expected_slots_for_template_rows(template_group, expected_phones, language=language)
            if template_group and expected_phones else None
        )

        # 1) Feature extraction (pyworld)
        t0 = time.perf_counter()
        features = extract_features(wav_path, encoder="acoustic_world_v1")
        timings["extract_features"] += time.perf_counter() - t0

        # 2) Full predict_wav (includes features again - measure total)
        t0 = time.perf_counter()
        prediction = predict_wav(
            wav_path,
            expected_phones=expected_phones,
            expected_slots=expected_slots,
            encoder="acoustic_world_v1",
            use_slot_viterbi=True,
            language=language,
        )
        timings["predict_wav"] += time.perf_counter() - t0

        # 3) HSMM decode
        prediction_events = list(_event_source_for_oto(prediction))
        if row_plan_slots:
            t0 = time.perf_counter()
            candidate_priors = candidate_priors_from_evidence_pack(
                build_acoustic_evidence_pack(
                    prediction.posterior,
                    wav_name=wav_path.name,
                    expected_phones=expected_phones,
                    filename_slots=row_plan_slots,
                    runtime_events=prediction_events,
                ),
                row_plan_slots,
            )
            hsmm = decode_filename_slots_with_hsmm(
                prediction.posterior,
                row_plan_slots,
                event_priors=(*prediction_events, *candidate_priors),
                language=language,
            )
            timings["hsmm_decode"] += time.perf_counter() - t0

        # 4) Evidence pack
        t0 = time.perf_counter()
        evidence_pack = build_acoustic_evidence_pack(
            prediction.posterior,
            wav_name=wav_path.name,
            expected_phones=expected_phones,
            filename_slots=row_plan_slots,
            runtime_events=prediction_events,
        )
        timings["evidence_pack"] += time.perf_counter() - t0

        # 5) Anchor assignment
        if template_group:
            decoded_source = list(hsmm.events) if (row_plan_slots and hsmm and hsmm.ok) else prediction_events
            t0 = time.perf_counter()
            row_anchors = assign_template_row_anchors(
                prediction.posterior,
                decoded_source,
                template_group,
                min_score=0.02,
                use_source_timing_prior=False,
                expected_phones=expected_phones,
                language=language,
            )
            timings["assign_anchors"] += time.perf_counter() - t0

            # 6) Row adaptation
            adapter_config = OtoAdapterConfig(
                mode="template-preserve", language=language,
                format_type=format_type, alias_type="auto",
            )
            t0 = time.perf_counter()
            for tpl, anchor in zip(template_group, row_anchors):
                adapt_template_row(tpl, anchor, file_duration_ms=features.duration_ms, config=adapter_config)
            timings["adapt_rows"] += time.perf_counter() - t0

        if (i + 1) % 10 == 0 or i == total - 1:
            print(f"  [{i+1}/{total}] {wav_path.name}")

    print("\n" + "=" * 60)
    print("STAGE TIMING SUMMARY (seconds)")
    print("=" * 60)
    total_time = sum(timings.values())
    for stage, elapsed in sorted(timings.items(), key=lambda x: -x[1]):
        pct = 100.0 * elapsed / total_time if total_time > 0 else 0
        per_wav = elapsed / total if total > 0 else 0
        print(f"  {stage:<20s} {elapsed:8.2f}s  ({pct:5.1f}%)  [{per_wav*1000:.1f} ms/wav]")
    print(f"  {'TOTAL':<20s} {total_time:8.2f}s")
    print(f"\n  WAV count: {total}")
    if total > 0:
        print(f"  Avg per WAV: {total_time/total*1000:.1f} ms")
        print(f"  Throughput: {total/total_time:.1f} WAV/s" if total_time > 0 else "")


def main():
    parser = argparse.ArgumentParser(description="Profile HSMM OTO pipeline")
    parser.add_argument("wav_dir", help="Path to voicebank WAV folder")
    parser.add_argument("--max-wavs", type=int, default=50, help="Max WAVs to process (default: 50)")
    parser.add_argument("--language", default="japanese", help="Language (default: japanese)")
    parser.add_argument("--format-type", default="CV", help="Format type (default: CV)")
    parser.add_argument("--full-profile", action="store_true", help="Also run cProfile for detailed callgraph")
    args = parser.parse_args()

    if not os.path.isdir(args.wav_dir):
        print(f"ERROR: WAV directory not found: {args.wav_dir}")
        sys.exit(1)

    # Stage-level timing (always)
    _run_pipeline(args.wav_dir, args.max_wavs, args.language, args.format_type)

    # Full cProfile (optional, for snakeviz)
    if args.full_profile:
        print("\n\nRunning cProfile (full callgraph)...")
        prof_path = os.path.join(os.path.dirname(__file__), "profile_output.prof")
        profiler = cProfile.Profile()
        profiler.enable()
        _run_pipeline(args.wav_dir, args.max_wavs, args.language, args.format_type)
        profiler.disable()
        profiler.dump_stats(prof_path)
        print(f"\nFull profile saved to: {prof_path}")
        print("View with: snakeviz scripts/profile_output.prof\n")
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")
        stats.print_stats(40)


if __name__ == "__main__":
    main()
