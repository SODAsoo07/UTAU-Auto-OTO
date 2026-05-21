"""MFA-free OTO proof-of-concept runtime primitives.

This package intentionally does not import the deprecated coarse CRNN or
phoneme-boundary stacks. Training and evaluation CLIs live under ml/scripts;
the core package only contains reusable manifest, feature, decoding, metrics,
and lightweight inference helpers.
"""

from .types import (
    EVENT_LABELS,
    FRAME_LABELS,
    DecodedEvent,
    FramePosterior,
    SuccessCriteria,
)
from .slot_viterbi import ExpectedSlot, SlotAssignment, SlotViterbiResult, assign_slots_viterbi
from .runtime_inference import RuntimePrediction, predict_wav
from .workflow import (
    BoundaryEvidence,
    NoMfaWorkflowGuard,
    NoMfaRowResult,
    NoMfaWorkflowReport,
    generate_no_mfa_oto_with_model_context,
)

__all__ = [
    "DecodedEvent",
    "EVENT_LABELS",
    "FRAME_LABELS",
    "FramePosterior",
    "ExpectedSlot",
    "SlotAssignment",
    "SlotViterbiResult",
    "SuccessCriteria",
    "assign_slots_viterbi",
    "RuntimePrediction",
    "predict_wav",
    "BoundaryEvidence",
    "NoMfaRowResult",
    "NoMfaWorkflowGuard",
    "NoMfaWorkflowReport",
    "generate_no_mfa_oto_with_model_context",
]
