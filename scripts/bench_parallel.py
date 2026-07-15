"""Benchmark sequential vs parallel HSMM OTO generation."""
from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WAV_DIR = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\oyh57\SODAsoo1\VocalSynth\UTAU\Singer\SODAsoo_CVVC_Test\SODA_D4"
MAX_WAVS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
LANGUAGE = "japanese"
FORMAT_TYPE = "CVVC"


def run_benchmark(max_workers: int | None, label: str):
    from core.mfa_free_oto.workflow import generate_no_mfa_oto_with_model_context
    from pathlib import Path

    wav_files = sorted(Path(WAV_DIR).glob("*.wav"))[:MAX_WAVS]
    actual_count = len(wav_files)

    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "bench_output.ini")
        messages = []

        def cb(msg):
            messages.append(msg)

        t0 = time.perf_counter()
        report = generate_no_mfa_oto_with_model_context(
            wav_dir=WAV_DIR,
            out_path=out_path,
            language=LANGUAGE,
            format_type=FORMAT_TYPE,
            encoder="acoustic_world_v1",
            use_slot_viterbi=True,
            use_hsmm_decoder=True,
            max_workers=max_workers,
            callback=cb,
        )
        elapsed = time.perf_counter() - t0

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Time:       {elapsed:.2f}s")
    print(f"  WAVs:       {actual_count} (dir has {len(list(Path(WAV_DIR).glob('*.wav')))} total)")
    print(f"  Rows:       {report.processed}")
    print(f"  Per WAV:    {elapsed/actual_count*1000:.0f} ms")
    print(f"  Throughput: {actual_count/elapsed:.2f} WAV/s")
    print(f"  Confidence: {report.confidence:.3f}")
    print(f"  Errors:     {len(report.errors)}")
    if report.errors:
        for e in report.errors[:3]:
            print(f"    - {e}")
    return elapsed


if __name__ == "__main__":
    print(f"WAV dir: {WAV_DIR}")
    print(f"Max WAVs: {MAX_WAVS}")
    print(f"CPU count: {os.cpu_count()}")

    t_seq = run_benchmark(max_workers=1, label="SEQUENTIAL (max_workers=1)")
    t_par = run_benchmark(max_workers=None, label=f"PARALLEL (max_workers=auto={max(1, (os.cpu_count() or 1)-1)})")

    print(f"\n{'='*60}")
    print(f"  SPEEDUP: {t_seq/t_par:.2f}x")
    print(f"  Saved:   {t_seq - t_par:.1f}s")
    print(f"{'='*60}")
