# SOFA UTAU KR Workspace

This workspace is isolated from the default SOFA path to avoid mixing baseline and KR-tuned experiments.

## Directory policy

- Custom SOFA repo root (runtime default for Korean): `Auto_OTO/.sofa/SOFA_UTAU_KR_v1`
- Baseline SOFA stays untouched: `Auto_OTO/.sofa/SOFA`
- Training/eval workspace root: `Auto_OTO/sofa_utau_kr`

## Subdirectories

- `data/`: converted `weak_label` and `full_label` datasets
- `ckpt/`: fine-tuned checkpoints
- `reports/`: run/evaluation logs, confidence summaries
- `ab_tests/`: listening test sheets and decisions
- `scripts/`: data conversion and evaluation helpers

## Scripts

- `scripts/build_weak_labels_from_oto.py`  
  Convert manual `oto.ini` to SOFA `weak_label/*/transcriptions.csv` (`name,ph_seq`) with skip reports.

- `scripts/build_full_labels_from_textgrid.py`  
  Convert gold TextGrid phones tier to SOFA `full_label/*/transcriptions.csv` (`name,ph_seq,ph_dur`) with duration normalization.

- `scripts/run_sofa_evaluate.py`  
  Wrapper around SOFA `evaluate.py` that accumulates CSV/JSON run reports.
