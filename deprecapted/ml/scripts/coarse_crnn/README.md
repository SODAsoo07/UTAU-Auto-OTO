# coarse_crnn workflow

This workflow is intentionally separate from the existing `sequence` aligner.

Deprecated OTO note: the direct-parameter OTO CRNN stack has moved under
`core/coarse_crnn/deprecated/direct_param/`. The old `core.coarse_crnn.oto_*`
import paths are compatibility wrappers only. New OTO model work should target
a frame-level boundary scorer plus wav-level monotonic decoder instead of direct
`offset/consonant/cutoff/preutterance/overlap` regression.

Deprecated script note: direct-parameter training/eval CLIs were moved under
`ml/scripts/coarse_crnn/deprecated/direct_param/`. The old top-level script
paths now fail closed with a deprecation error so they are not used by accident.

The same audio/CRNN code also has a deprecated OTO-anchor mode that trained
directly against `oto.ini` parameters. Keep it for reproduction only; the next
production OTO direction is boundary scoring plus decoder-side parameter
construction.

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

Build and train the active boundary-decoder OTO scorer:

```powershell
.\.venv\Scripts\Activate.ps1
$env:UTOA_BOUNDARY_PHONE_AUX_IDENTITY_LANGS='korean,japanese'
python -m ml.scripts.coarse_crnn.build_oto_manifest `
  --dataset-staged dataset_staged `
  --out ml_workspace\coarse_crnn\oto_manifest.jsonl
python -m ml.scripts.coarse_crnn.build_oto_splits `
  --manifest ml_workspace\coarse_crnn\oto_manifest.jsonl `
  --out-dir ml_workspace\coarse_crnn\oto_splits
python -m ml.scripts.coarse_crnn.train_boundary_oto `
  --manifest ml_workspace\coarse_crnn\oto_splits_full\oto_train.jsonl `
  --out ml_workspace\models\coarse_crnn\oto_boundary_scorer_phone_family_aux_20260518.pt `
  --device cuda `
  --batch-size 8 `
  --max-frames 1600 `
  --enable-phone-aux-heads `
  --enable-phone-family-heads `
  --init-from ml_workspace\models\coarse_crnn\oto_boundary_scorer_v5_codabridge_20260517.pt `
  --freeze-non-phone-aux `
  --cvs-loss-weight 0.08 `
  --consonant-id-loss-weight 0.0 `
  --vowel-id-loss-weight 0.0 `
  --consonant-family-loss-weight 0.05 `
  --vowel-nucleus-loss-weight 0.05 `
  --vowel-glide-loss-weight 0.03 `
  --phone-aux-class-balanced `
  --lr 5e-4
python -m ml.scripts.coarse_crnn.evaluate_boundary_manifest `
  --manifest ml_workspace\coarse_crnn\oto_splits_full\oto_val.jsonl `
  --model ml_workspace\models\coarse_crnn\oto_boundary_scorer_phone_family_aux_20260518.pt `
  --device cuda `
  --out ml_workspace\coarse_crnn\boundary_eval_phone_aux_val.json
python -m ml.scripts.coarse_crnn.evaluate_phone_aux_targets `
  --manifest ml_workspace\coarse_crnn\oto_splits_full\oto_val.jsonl `
  --model ml_workspace\models\coarse_crnn\oto_boundary_scorer_phone_family_aux_20260518.pt `
  --device cuda `
  --identity-languages korean,japanese `
  --out ml_workspace\coarse_crnn\phone_family_aux_eval_val.json
```

Inspect a CVVC phone-aware pseudo-target before a full training run:

```powershell
python -m ml.scripts.coarse_crnn.plot_phone_aware_debug `
  --manifest ml_workspace\coarse_crnn\oto_splits_full\oto_val.jsonl `
  --wav "C:\path\to\voicebank\_ba'ba'_'bya'bya'_'bwa'bwa.wav" `
  --alias "a b" `
  --line-index 21 `
  --model ml_workspace\models\coarse_crnn\oto_boundary_scorer_v5_codabridge_20260517.pt `
  --device cpu `
  --out ml_workspace\coarse_crnn\phone_aware_debug_v5_val_korean_vc.png
```

The plot shows mel, boundary posterior/target, source OTO row spans, CVS class
targets, and consonant/vowel identity targets. Older checkpoints without
phone-aware aux heads still render boundary posterior plus pseudo-targets; aux
posterior lines appear only after training with `--enable-phone-aux-heads`.

Phone-aware identity targets cover Korean and Japanese by default. Japanese
kana/romaji aliases are normalized into the shared consonant/vowel inventory,
and pitch suffixes such as `C4P`, `A3`, or frame-like vowel suffixes are stripped
before identity lookup. CVS class targets remain language-agnostic. Set
`UTOA_BOUNDARY_PHONE_AUX_IDENTITY_LANGS=korean` to restrict identity loss back to
Korean-only, or `UTOA_BOUNDARY_PHONE_AUX_IDENTITY_LANGS=all` only for an explicit
experiment with a broader identity inventory.

Runtime decoder scoring uses CVS phone class posterior plus the broad family
heads by default when the checkpoint has them. Fine consonant/vowel identity
posterior is diagnostic-only by default (`UTOA_BOUNDARY_PHONE_AUX_ID_WEIGHT=0`).
Tune family contribution with `UTOA_BOUNDARY_PHONE_AUX_FAMILY_WEIGHT`; enable
fine identity only for an explicit A/B run.

`evaluate_phone_aux_targets` now reports `decoder_useful_metrics` first:
CVS consonant/vowel separation, `consonant_family`, `vowel_nucleus`, and
`vowel_glide`. Fine consonant/vowel top-1 accuracy is still reported, but it is
reference-only for runtime promotion decisions.

Model auto-discovery does not pick experimental `phone_aux` checkpoints by
default. Pass the checkpoint explicitly, set `UTOA_BOUNDARY_SCORER_MODEL_PATH`,
or opt in with `UTOA_BOUNDARY_SCORER_ALLOW_EXPERIMENTAL=1`.

Generate a full voicebank OTO with the active decoder:

```powershell
.\.venv\Scripts\Activate.ps1
python -m ml.scripts.coarse_crnn.generate_boundary_oto `
  --wav-dir path\to\voicebank `
  --source-oto path\to\voicebank\source_oto.ini `
  --out path\to\voicebank\oto.ini `
  --language korean `
  --model ml_workspace\models\coarse_crnn\oto_boundary_scorer_phone_aux.pt `
  --device cuda
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

Wav-level sequence boundary alignment (default runtime path):

```powershell
# New boundary-candidate alignment path. This is enabled by default in the
# generator and bypasses the older raw row-order aligner.
$env:UTOA_OTO_CRNN_SEQUENCE_ALIGN_ENABLE='1'

# Conservative defaults verified on val1000 with ctx15_v3.
$env:UTOA_OTO_CRNN_SEQUENCE_ALIGN_ROLES='vc,vv'
$env:UTOA_OTO_CRNN_SEQUENCE_ALIGN_MAX_SHIFT_MS='220'
$env:UTOA_OTO_CRNN_SEQUENCE_ALIGN_ANCHOR_BLEND='0.60'
$env:UTOA_OTO_CRNN_SEQUENCE_ALIGN_BOUNDARY_BLEND='0.45'
$env:UTOA_OTO_CRNN_SEQUENCE_ALIGN_MIN_MOVE_MS='8'
$env:UTOA_OTO_CRNN_SEQUENCE_ALIGN_MODEL_SCORE_BLEND='0.35'

# Keep legacy per-row candidate snap/matcher disabled while the sequence aligner
# owns placement. Enable only for A/B debugging.
$env:UTOA_OTO_CRNN_SEQUENCE_ALIGN_ALLOW_LEGACY_CANDIDATE_POSTPROCESS='0'
```

The default role allowlist intentionally excludes `cv`, `v`, `-cv`, and `v-cv`
anchor-like rows. The current verified default only adjusts `vc`/`vv` boundary
rows, so it can improve boundary timing without relocating the predicted file
position. To test the broader aligner, set
`UTOA_OTO_CRNN_SEQUENCE_ALIGN_ROLES='vc,vv,v-cv'` or `all`, but keep it as an
A/B experiment until it beats the conservative default on a real voicebank.

Filename-slot crop mapping is enabled by default:

```powershell
$env:UTOA_OTO_CRNN_FILENAME_SLOT_ENABLE='1'
$env:UTOA_OTO_CRNN_ALIAS_SLOT_LOCK_ENABLE='1'
$env:UTOA_OTO_CRNN_ALIAS_SLOT_LOCK_MODE='direct'
$env:UTOA_OTO_CRNN_ALIAS_SLOT_LOCK_MAX_SHIFT_MS='3000'
$env:UTOA_OTO_CRNN_ALIAS_SLOT_LOCK_MIN_MOVE_MS='30'
$env:UTOA_OTO_CRNN_ALIAS_SLOT_LOCK_VC_TARGET_RATIO='0.62'
$env:UTOA_OTO_CRNN_ALIAS_SLOT_LOCK_VC_FALLBACK_RATIO='0.62'
$env:UTOA_OTO_CRNN_VC_ANCHOR_ROLES='-cv,cv,v,special'
$env:UTOA_OTO_CRNN_VC_ANCHOR_MIN_CONFIDENCE='0.20'
$env:UTOA_OTO_CRNN_VC_ANCHOR_LEFT_PAD_MS='45'
$env:UTOA_OTO_CRNN_VC_ANCHOR_RIGHT_PAD_MS='60'
$env:UTOA_OTO_CRNN_VC_ANCHOR_MIN_GAP_MS='24'
$env:UTOA_OTO_CRNN_FINAL_RIGHT_GUARD_ENABLE='1'
```

This makes the inference crop use the syllable order parsed from the wav
filename when at least two filename tokens are available. It prevents shuffled
`oto.ini` row order from feeding the model the wrong syllable window while
keeping the output `oto.ini` row order unchanged. VC rows are attached to the
left vowel slot; VV and V-CV rows prefer the right vowel slot when both sides
are visible. Japanese kana filenames and aliases are normalized through the
Japanese filename parser, and Hangul filenames are split per syllable before
slot matching. Korean boundary aliases can be vowel-only (`a k`) while the
filename token is CV-shaped (`ga`), so Korean V/VC/VV/V-CV rows also support
vowel-key slot matching.

Alias-slot lock is the stronger pronunciation guard. If the model predicts an
OTO row near the wrong syllable slot, the decoder moves the whole anchor set
so the absolute `preutterance` lands on the filename slot required by the alias.
This changes `offset` as well as `preutterance`, so it can actually change the
rendered phoneme source instead of only nudging timing inside the same wrong
audio region. `direct` is the production default; set
`UTOA_OTO_CRNN_ALIAS_SLOT_LOCK_MODE='relative'` only to reproduce the older
nearest-slot translation behavior for A/B tests.

VC rows use a role-specific target. A VC alias such as `a k` is not locked to
the left syllable onset; it is locked near the transition from the left vowel
tail to the next onset. The default target is
`vowel_end + (next_onset - vowel_end) * 0.62`, with the same ratio used between
neighboring filename slots when no stable vowel segment is available.

When usable CV/V anchors exist in the same wav, VC placement is constrained by
that anchor timeline first. Non-VC anchor rows are locked before VC rows; then a
VC row at slot `i` searches only between anchor `i` and the next available
anchor. This prevents a VC row from drifting to a later onset elsewhere in the
file. If the anchor pair is unavailable or too narrow, the older VC slot fallback
above is used.

Final right-boundary guard is enabled by default after all slot/sequence
movement. It clamps `consonant` and `cutoff` by alias role so rows moved to the
correct source syllable do not keep an overlong fixed/tail region that can sound
stretched.

Boundary-slot experimental model guard:

```powershell
# Boundary-slot/heavy models are blocked by default in the generator path.
# Enable only for controlled A/B tests, not production generation.
$env:UTOA_OTO_CRNN_ALLOW_EXPERIMENTAL_MODEL='1'

# Boundary-slot graph postprocess is also default OFF after heavy real-bank
# testing showed large parameter misplacement rates.
$env:UTOA_OTO_CRNN_BOUNDARY_SLOT_GRAPH_ENABLE='0'
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

Dataset-staged visual benchmark (sample 3 banks per language/format):

```powershell
# Full run: auto-generate OTO for sampled banks, compare vs reference oto.ini,
# and export per-row waveform+spectrogram SVG overlays.
.\.venv\Scripts\python.exe -m ml.scripts.coarse_crnn.benchmark_dataset_staged_oto_visual `
  --dataset-staged dataset_staged `
  --per-group 3 `
  --variant F3 `
  --out-root ml/eval/oto_visual_benchmark `
  --max-plots-per-bank 8 `
  --verbose
```

Fast dry-style check using existing `oto_auto_ml.ini` (no generation):

```powershell
.\.venv\Scripts\python.exe -m ml.scripts.coarse_crnn.benchmark_dataset_staged_oto_visual `
  --dataset-staged dataset_staged `
  --per-group 1 `
  --max-banks 1 `
  --skip-generation `
  --existing-auto-name oto_auto_ml.ini `
  --out-root ml/eval/oto_visual_benchmark `
  --verbose
```

Outputs:

- `summary.json`: full metrics (`overall`, `transition_only`, `n_like_only`, `weak_wav_only`, `by_role`)
- `bank_summary.csv`: per-bank summary table
- `index.html`: visual report index
- `banks/<language>__<format>/<bank>/plots/*.svg`: waveform+spectrogram overlays with GT vs AUTO boundaries

Boundary-manifest release gates (catastrophic shift focused):

```powershell
.\.venv\Scripts\python.exe -m ml.scripts.coarse_crnn.evaluate_boundary_manifest `
  --manifest ml_workspace/coarse_crnn/oto_splits_full/oto_eval.jsonl `
  --model ml_workspace/models/coarse_crnn/oto_boundary_scorer_v3_slotfix.pt `
  --out ml_workspace/coarse_crnn/boundary_eval_gate.json `
  --gate-max-mis-mapping-rate 0.18 `
  --gate-max-one-step-shift-rate 0.12 `
  --gate-max-vc-pre-ge-80ms-rate 0.45
```

- Gate fail returns exit code `1` and writes all metrics to `--out`.
- New shift metrics: `mis_mapping_rate`, `one_step_shift_rate`, `vc_pre_ge_80ms_rate`.

Boundary-scorer training (target normalization + n-like/weak weighting):

```powershell
.\.venv\Scripts\python.exe -m ml.scripts.coarse_crnn.train_boundary_oto `
  --manifest ml_workspace/coarse_crnn/oto_splits_full/oto_train.jsonl `
  --out ml_workspace/models/coarse_crnn/oto_boundary_scorer_v3_slotfix.pt `
  --normalize-target-fields `
  --drop-invalid-target-rows `
  --drop-wav-backward-rows `
  --n-like-weight 0.80 `
  --weak-wav-weight 0.65 `
  --weak-wav-rms-db-threshold -32 `
  --clean-summary ml_workspace/coarse_crnn/boundary_manifest_clean_summary.json
```

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

Recent patch notes (2026-05-14):

- Real-bank heavy boundary-slot test showed about 80% alias-level parameter misplacement, so this model family is no longer a deployment candidate.
- Generator now blocks boundary-slot/heavy experimental checkpoints by default. Set `UTOA_OTO_CRNN_ALLOW_EXPERIMENTAL_MODEL=1` only for controlled A/B tests.
- Boundary-slot graph postprocess default changed to OFF (`UTOA_OTO_CRNN_BOUNDARY_SLOT_GRAPH_ENABLE=0`) because it can move many rows when the learned slot ordering is wrong.
- OTO eval reports now include `position_bad_rate_100ms` and `position_bad_rate_250ms`; the 250 ms rate is also part of the default gate via `--gate-max-position-bad-rate-250ms`.
- `compare_oto_eval.py` now includes `position_bad_rate_250ms`, so a model that only improves `preutterance_acc` but misplaces full parameters is visible in A/B summaries.
- Alias-slot lock default mode is now `direct`: instead of translating from the nearest predicted slot, it forces absolute `preutterance` onto the filename slot required by the alias. This is intentionally stronger because the previous relative mode still left many rows in the wrong audible syllable region.
- Alias-slot lock now uses a VC-specific transition target. VC rows no longer land on the left syllable onset; they land near the vowel-tail-to-next-onset transition.
- Korean filename-slot matching now has a vowel-key fallback for V/VC/VV/V-CV rows, so aliases like `a k` can map to filename syllables like `ga` instead of falling back to raw row order.
- Final right-boundary guard now runs after slot/sequence movement by default to reduce overlong/stretched timing after relocation.
- VC alias-slot lock now uses the already locked CV/V anchor timeline when possible. VC rows are constrained between the left anchor and next anchor before falling back to the older slot-based VC target.

Last edited: 2026-05-14 15:32 KST
