# coarse_crnn workflow

This workflow is intentionally separate from the existing `sequence` aligner.

The same audio/CRNN code also has an OTO-anchor mode. For UTAU voicebanks this
is the preferred direction: train directly against `oto.ini` parameters instead
of using coarse phone boundaries as the final target.

Build a Korean/Japanese manifest:

```powershell
python -m ml.scripts.coarse_crnn.build_manifest `
  --dataset-staged dataset_staged `
  --public-root "C:\Users\oyh57\SODAsoo1\VocalSynth\Data\PublicData" `
  --out ml_workspace\coarse_crnn\manifest.jsonl
```

Train a small CPU-friendly CRNN:

```powershell
python -m ml.scripts.coarse_crnn.train `
  --manifest ml_workspace\coarse_crnn\manifest.jsonl `
  --out ml_workspace\models\coarse_crnn\coarse_crnn.pt `
  --device cpu
```

For actual local training, prefer the existing `.venv` and CUDA when available:

```powershell
.\.venv\Scripts\Activate.ps1
python -m ml.scripts.coarse_crnn.build_clean_manifest `
  --manifest ml_workspace\coarse_crnn\manifest_full.jsonl `
  --out ml_workspace\coarse_crnn\manifest_clean.jsonl
python -m ml.scripts.coarse_crnn.train `
  --manifest ml_workspace\coarse_crnn\manifest_clean.jsonl `
  --out ml_workspace\models\coarse_crnn\coarse_crnn_clean_boundary.pt `
  --device cuda `
  --batch-size 16 `
  --max-frames 1200 `
  --boundary-head `
  --augment `
  --class-balance
```

Run one alignment:

```powershell
python -m ml.scripts.coarse_crnn.align `
  --audio path\to\sample.wav `
  --language ja `
  --model ml_workspace\models\coarse_crnn\coarse_crnn.pt `
  --out output\coarse_crnn
```

Evaluate alignment quality against gold TextGrid/timed lab rows:

```powershell
python -m ml.scripts.coarse_crnn.build_eval_splits `
  --manifest ml_workspace\coarse_crnn\manifest_full.jsonl `
  --out-dir ml_workspace\coarse_crnn\eval_splits
python -m ml.scripts.coarse_crnn.evaluate_alignment `
  --manifest ml_workspace\coarse_crnn\eval_splits\eval_all.jsonl `
  --model ml_workspace\models\coarse_crnn\coarse_crnn.pt `
  --device cuda `
  --source dataset_staged `
  --max-items 0 `
  --max-per-language 0 `
  --out ml_workspace\coarse_crnn\alignment_eval.json
```

Build and train the OTO parameter predictor:

```powershell
.\.venv\Scripts\Activate.ps1
python -m ml.scripts.coarse_crnn.build_oto_manifest `
  --dataset-staged dataset_staged `
  --out ml_workspace\coarse_crnn\oto_manifest.jsonl
python -m ml.scripts.coarse_crnn.build_oto_splits `
  --manifest ml_workspace\coarse_crnn\oto_manifest.jsonl `
  --out-dir ml_workspace\coarse_crnn\oto_splits
python -m ml.scripts.coarse_crnn.train_oto `
  --manifest ml_workspace\coarse_crnn\oto_splits\oto_train.jsonl `
  --val-manifest ml_workspace\coarse_crnn\oto_splits\oto_val.jsonl `
  --out ml_workspace\models\coarse_crnn\oto_anchor_crnn.pt `
  --device cuda `
  --batch-size 16 `
  --max-frames 1200
python -m ml.scripts.coarse_crnn.evaluate_oto `
  --manifest ml_workspace\coarse_crnn\oto_splits\oto_test.jsonl `
  --model ml_workspace\models\coarse_crnn\oto_anchor_crnn.pt `
  --device cuda `
  --max-items 0 `
  --out ml_workspace\coarse_crnn\oto_eval.json
```

Predict one `oto.ini` row:

```powershell
python -m ml.scripts.coarse_crnn.predict_oto `
  --audio path\to\sample.wav `
  --alias "ka" `
  --language japanese `
  --format-type cv `
  --model ml_workspace\models\coarse_crnn\oto_anchor_crnn.pt
```

Runtime-style OTO evaluation (with generator-like base fallback):

```powershell
$env:PYTHONPATH='.'
python ml/scripts/coarse_crnn/evaluate_oto_runtime.py `
  --model ml_workspace/models/coarse_crnn/oto_anchor_crnn_active_context_focusboost_rand3000_e2.pt `
  --max-items 300 `
  --seed 20260512 `
  --device cpu `
  --use-row-base-fallback `
  --preserve-audio-groups `
  --out ml_workspace/coarse_crnn/oto_runtime_eval_300.json
```

Audio-candidate snap risk gate (recommended default):

```powershell
# safer default: apply candidate snap only when prediction is risky
$env:UTOA_OTO_CRNN_AUDIO_CANDIDATE_REQUIRE_RISKY='1'

# A/B check only: force old aggressive behavior
$env:UTOA_OTO_CRNN_AUDIO_CANDIDATE_REQUIRE_RISKY='0'
```

CVVC snap jump caps (recommended default):

```powershell
# Limit over-forward snap jumps that often cause one-step mis-mapping.
$env:UTOA_OTO_CRNN_AUDIO_CANDIDATE_CVVC_HEAD_MAX_FORWARD_DELTA_MS='23'
$env:UTOA_OTO_CRNN_AUDIO_CANDIDATE_CVVC_CV_MAX_FORWARD_DELTA_MS='32'
$env:UTOA_OTO_CRNN_AUDIO_CANDIDATE_CVVC_CVTAIL_MAX_FORWARD_DELTA_MS='19'
```

Role-adaptive fallback + noisy audio robustness (recommended defaults):

```powershell
# Reduce over-triggered low-confidence fallback for transition-heavy roles.
$env:UTOA_OTO_CRNN_LOW_CONF_ROLE_ADAPT_ENABLE='1'
$env:UTOA_OTO_CRNN_LOW_CONF_ROLE_RELAX_ROLES='vc,v-cv,vv'
$env:UTOA_OTO_CRNN_LOW_CONF_ROLE_RELAX_CONF_DELTA='0.04'
$env:UTOA_OTO_CRNN_LOW_CONF_ROLE_RELAX_ERROR_DELTA_MS='90'
$env:UTOA_OTO_CRNN_LOW_CONF_ROLE_RELAX_EXTRA_VOTES='1'

# In low-volume / high-noise takes, pick the strongest sustained activity run
# instead of the earliest short burst.
$env:UTOA_OTO_CRNN_ACTIVITY_NOISE_ROBUST_ENABLE='1'
$env:UTOA_OTO_CRNN_ACTIVITY_PICK_STRONGEST_RUN='1'
$env:UTOA_OTO_CRNN_SKIP_LEADING_SILENCE_PICK_STRONGEST_RUN='1'

# Auto row-activity shift (experimental, default OFF):
# - enable only when a target bank shows real front-blank collapse
# - keep OFF for normal banks to avoid sequence jitter
$env:UTOA_OTO_CRNN_ACTIVITY_ROW_SHIFT_AUTO_LOW_SNR_ENABLE='1'
$env:UTOA_OTO_CRNN_ACTIVITY_ROW_SHIFT_AUTO_FRONT_EARLY_ENABLE='1'
$env:UTOA_OTO_CRNN_ACTIVITY_ROW_SHIFT_AUTO_EARLY_MARGIN_MS='70'
$env:UTOA_OTO_CRNN_ACTIVITY_ROW_SHIFT_AUTO_ROLES='cv,-cv,special'
```

Base OTO fallback policy (production recommendation):

```powershell
# Default production-safe mode: do not trust base oto absolute timings.
$env:UTOA_OTO_CRNN_USE_BASE_OTO_FALLBACK='0'

# Only for controlled A/B/debug when source oto quality is verified.
$env:UTOA_OTO_CRNN_USE_BASE_OTO_FALLBACK='1'

# If fallback is enabled, apply it only on wavs whose filename/alias row order
# quality passes the minimum ratio gate.
$env:UTOA_OTO_CRNN_BASE_FALLBACK_QUALITY_GATE_ENABLE='1'
$env:UTOA_OTO_CRNN_BASE_FALLBACK_MIN_ORDER_RATIO='0.42'
$env:UTOA_OTO_CRNN_BASE_FALLBACK_MIN_EXPECTED_TOKENS='4'

# CVVC-specific relaxed gate (uses core alias-token order score).
$env:UTOA_OTO_CRNN_BASE_FALLBACK_MIN_ORDER_RATIO_CVVC='0.34'
$env:UTOA_OTO_CRNN_BASE_FALLBACK_MIN_EXPECTED_TOKENS_CVVC='4'
```

Wav-level monotonic row-order alignment (recommended defaults):

```powershell
# Align all rows in one wav to onset candidates under monotonic order constraints.
$env:UTOA_OTO_CRNN_ROW_ORDER_ALIGN_ENABLE='1'

# Conservative guard: if movement/jump is too large, keep original prediction.
$env:UTOA_OTO_CRNN_ROW_ORDER_MAX_SHIFT_MS='260'
$env:UTOA_OTO_CRNN_ROW_ORDER_MAX_SKIP_POINTS='2'
$env:UTOA_OTO_CRNN_ROW_ORDER_MAX_STEP_GAP_MS='420'
```

Filename vs alias order validation:

```powershell
$env:PYTHONPATH='.'
python ml/scripts/coarse_crnn/validate_filename_alias_order.py `
  --source-oto path/to/base_oto.ini `
  --out ml_workspace/coarse_crnn/filename_alias_order_report.json `
  --min-ratio 0.60
```

Practical A/B compare for generated OTO files:

```powershell
$env:PYTHONPATH='.'
python ml/scripts/coarse_crnn/compare_generated_oto_ab.py `
  --a-oto dataset_staged/korean/cvvc/SODAsoo-TABI_KO_CV-VC_8mora/oto.crnn_roworder_gate.ini `
  --b-oto dataset_staged/korean/cvvc/SODAsoo-TABI_KO_CV-VC_8mora/oto.crnn_roworder_nogate.ini `
  --source-oto dataset_staged/korean/cvvc/SODAsoo-TABI_KO_CV-VC_8mora/oto.ini `
  --wav-dir dataset_staged/korean/cvvc/SODAsoo-TABI_KO_CV-VC_8mora `
  --out ml_workspace/coarse_crnn/oto_ab_practical_compare_tabi_gate_vs_nogate.json
```

- Reports file-level rates for:
  - `front_silence_rate`
  - `sequence_anomaly_rate` (source-independent)
  - `one_step_shift_rate`
  - `mis_mapping_rate`
- If source oto timing is low quality, treat `one_step_shift_rate`/`mis_mapping_rate` as relative signals only.

Recent patch notes (2026-05-13):

- Base OTO fallback is now opt-in in runtime generator path.
  - Default: `UTOA_OTO_CRNN_USE_BASE_OTO_FALLBACK=0`
  - Reason: many source oto files have unreliable absolute timing.
- Base OTO fallback quality gate was added for mixed-quality datasets.
  - When fallback is enabled, wavs with low filename-vs-alias order ratio are blocked from using base fallback.
  - Defaults: `UTOA_OTO_CRNN_BASE_FALLBACK_QUALITY_GATE_ENABLE=1`, `UTOA_OTO_CRNN_BASE_FALLBACK_MIN_ORDER_RATIO=0.42`.
- CVVC-aware order score path was added.
  - CVVC-like wavs use core alias tokens (mono-syllable rows) and support-weighted order score.
  - Default CVVC threshold: `UTOA_OTO_CRNN_BASE_FALLBACK_MIN_ORDER_RATIO_CVVC=0.34`.
- Low-confidence fallback trigger was made role-adaptive (`vc`, `v-cv`, `vv` relax defaults).
- Noise-robust activity detection was added for low-volume/high-noise recordings.
  - Strongest sustained run selection is enabled for both activity profile and leading-silence trim.
- Wav-level monotonic row-order alignment was added.
  - Rows in the same wav are jointly aligned to onset candidates under monotonic constraints.
  - Conservative hold guards prevent over-jumps (`MAX_SHIFT_MS`, `MAX_SKIP_POINTS`, `MAX_STEP_GAP_MS`).
- New validator script:
  - `ml/scripts/coarse_crnn/validate_filename_alias_order.py`
  - Detects filename token order vs alias order mismatch candidates.
- Row-activity auto shift remained experimental after A/B:
  - `Achu_CVVC` low-SNR-only auto: no measurable gain/loss vs OFF.
  - `Achu_CVVC` front-early auto: slight sequence-anomaly regression.
  - Production default remains OFF; enable per-bank only when front-blank collapse is confirmed.
- CVVC snap jump cap A/B:
  - `Achu_CVVC`: slight improvement (`sequence_anomaly`, `one_step_shift`, `mis_mapping` all down).
  - `TABI`: `one_step_shift`/`mis_mapping` small down, sequence anomaly small up.
  - Keep the cap defaults on; treat as low-risk conservative guard.

Observed evaluation trend on `holdout_tabi` (full 951, roleorder_e2):

- Before this patch series: fallback dependency (`pre50 ON-OFF`) was high (`~0.22`).
- After patch series: fallback dependency dropped to about `0.05`.
  - ON/OFF report: `ml_workspace/coarse_crnn/compare_runtime_onoff_holdout_tabi_full951_roleorder_e2_after_patch.json`

Recommended next execution order:

1. Run filename-order validator on the source oto and quarantine high-mismatch wav files.
2. Regenerate OTO with row-order alignment on and base OTO fallback off.
3. Re-run ON/OFF runtime eval on the same manifest/model and compare by role (`-cv`, `vv`, `cv-` first).
