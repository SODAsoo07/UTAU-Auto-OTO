"""Benchmark template-based (source_oto) path: sequential vs parallel."""
from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WAV_DIR = r"C:\Users\oyh57\SODAsoo1\VocalSynth\UTAU\Singer\SODAsoo_CVVC_Test\SODA_D4"
SOURCE_OTO = r"C:\Users\oyh57\SODAsoo1\VocalSynth\UTAU\Singer\SODAsoo_CVVC_Test\SODA_D4\oto.ini"
LANGUAGE = "japanese"
FORMAT_TYPE = "CVVC"


def run(max_workers, label):
    from core.mfa_free_oto.workflow import generate_no_mfa_oto_with_model_context

    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "bench_output.ini")
        t0 = time.perf_counter()
        report = generate_no_mfa_oto_with_model_context(
            wav_dir=WAV_DIR,
            out_path=out_path,
            source_oto_path=SOURCE_OTO,
            language=LANGUAGE,
            format_type=FORMAT_TYPE,
            encoder="acoustic_world_v1",
            use_slot_viterbi=True,
            use_hsmm_decoder=True,
            max_workers=max_workers,
        )
        elapsed = time.perf_counter() - t0
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Time:       {elapsed:.2f}s")
    print(f"  Rows:       {report.processed}")
    print(f"  Errors:     {len(report.errors)}")
    if report.errors:
        for e in list(report.errors)[:5]:
            print(f"    - {e}")
    return elapsed


if __name__ == "__main__":
    print(f"WAV dir: {WAV_DIR}")
    print(f"Source OTO: {SOURCE_OTO}")
    print(f"CPU count: {os.cpu_count()}")

    t_seq = run(1, "SEQUENTIAL (template path)")
    t_par = run(None, f"PARALLEL (template path, workers={max(1,(os.cpu_count() or 1)-1)})")

    print(f"\n{'='*60}")
    print(f"  SPEEDUP: {t_seq/t_par:.2f}x")
    print(f"  Saved:   {t_seq - t_par:.1f}s")
    print(f"{'='*60}")
