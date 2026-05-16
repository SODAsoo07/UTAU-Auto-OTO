# Test Scripts

Utilities in this folder are not runtime entrypoints.
They are test/support tools separated from `scripts/`.

- `alignment/`: sample assembly and visual alignment checks
- `benchmark/`: benchmark-only comparison tools
- `evaluate/`: evaluation helpers not used by runtime path
- `smoke/`: install/runtime smoke checks
- `training/`: experimental training preprocess/baseline helpers

Rules:

- Keep runtime/build entrypoints under `scripts/`.
- Keep test-only utilities under `tests/scripts/`.
