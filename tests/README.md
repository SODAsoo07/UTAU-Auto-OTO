# Tests

`tests/` contains pytest suites and test/support scripts.

- `tests/scripts/`: non-runtime helper scripts for testing/smoke/benchmark
- `tests/scripts/training/`: experimental training preprocess/baseline helpers
- `tests/scripts/evaluate/`: evaluation/export helper scripts
- `ml/tests/`: ML unit tests

Rules:

- Keep operational entrypoints under `scripts/`.
- Keep test-only tools under `tests/scripts/<purpose>/`.
