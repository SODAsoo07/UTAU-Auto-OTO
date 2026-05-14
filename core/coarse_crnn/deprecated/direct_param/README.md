# Deprecated Direct-Parameter OTO CRNN

This folder contains the legacy OTO CRNN stack that directly regresses UTAU OTO
parameters (`offset`, `consonant`, `cutoff`, `preutterance`, `overlap`).

Status: deprecated.

Use this code only for compatibility, old checkpoint reproduction, and A/B
comparison. New OTO work should use a frame-level boundary scorer plus a
wav-level monotonic decoder.

Compatibility wrappers remain at the old `core.coarse_crnn.oto_*` import paths
so existing UI, CLI, and tests keep running while the new boundary-scorer stack
is developed.

