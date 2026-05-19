# Deprecated Direct-Parameter OTO CRNN

This folder contains the legacy OTO CRNN stack that directly regresses UTAU OTO
parameters (`offset`, `consonant`, `cutoff`, `preutterance`, `overlap`).

Status: deprecated.

Use this code only for compatibility, old checkpoint reproduction, and A/B
comparison. New OTO work should use a frame-level boundary scorer plus a
wav-level monotonic decoder.

The old `core.coarse_crnn.oto_*` root import paths are intentionally closed.
Import this package directly only when reproducing the deprecated stack.
