Legacy scripts kept for reference after the mapping core v2 refactor.

These files are not used by the current runtime, UI, or test path.
They were tied to the older mapping-threshold tuning workflow and are
kept only as historical utilities or for one-off analysis.

Moved here:
- analyze_kr_mapping_jumps.py
- build_kr_vcv_profile_from_oto.py
- tune_alignment_thresholds.py

Not moved:
- Runtime-facing modules under `core/`
- Batch comparison helpers that are still useful with v2
- SOFA-related code, because the current app still imports it
