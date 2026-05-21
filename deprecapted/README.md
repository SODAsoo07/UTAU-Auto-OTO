# Deprecated Runtime Graveyard

This folder contains retired AutoOTO code that must not be imported by active runtime, UI, training, or evaluation workflows.

Moved here:

- `core/coarse_crnn/`
- `core/cvn/` and root CVN compatibility wrappers
- `ml/scripts/coarse_crnn/`
- CVN training wrappers
- CRNN/CVN-specific tests and dev scripts

The active replacement path is `core/model_context` plus the MFA and sequence-aligner workflows.
