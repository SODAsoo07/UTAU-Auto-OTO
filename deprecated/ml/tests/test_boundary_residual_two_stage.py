from __future__ import annotations

from types import SimpleNamespace

from core.coarse_crnn.boundary_residual import (
    BoundaryResidualBundle,
    apply_residual_to_decoded_row,
    apply_boundary_residual_deltas,
    predict_boundary_stage1_role_bias,
)
from core.coarse_crnn.boundary_types import AbsoluteOtoAnchors, OtoRowSpec


def _dummy_bundle() -> BoundaryResidualBundle:
    return BoundaryResidualBundle(
        model_dir=".",
        boosters={},
        feature_names=(),
        categorical_features=(),
        delta_clips={},
        active_targets=(),
        stage1_role_bias={
            "delta_offset": {"default": 0.0, "vc": -8.0},
            "delta_preutterance": {"default": 0.0, "vc": -4.0},
            "delta_overlap": {"default": 0.0, "vc": -2.0},
            "delta_cutoff": {"default": 0.0, "vc": -12.0},
        },
        meta={},
    )


def test_stage1_role_bias_resolves_role_specific_values():
    bundle = _dummy_bundle()
    vc = predict_boundary_stage1_role_bias(bundle, {"role": "vc"})
    cv = predict_boundary_stage1_role_bias(bundle, {"role": "cv"})
    assert float(vc["delta_offset"]) == -8.0
    assert float(vc["delta_preutterance"]) == -4.0
    assert float(cv["delta_offset"]) == 0.0


def test_apply_boundary_residual_deltas_supports_offset_and_invariants():
    base = {
        "offset": 100.0,
        "consonant": 140.0,
        "cutoff": -190.0,
        "preutterance": 80.0,
        "overlap": 50.0,
    }
    out, audit = apply_boundary_residual_deltas(
        params=base,
        deltas={
            "delta_offset": -15.0,
            "delta_preutterance": -20.0,
            "delta_overlap": +40.0,  # should be clamped to preutterance
            "delta_cutoff": -10.0,
        },
    )
    assert bool(audit["changed"]) is True
    assert float(out["offset"]) == 85.0
    assert float(out["preutterance"]) == 60.0
    assert float(out["overlap"]) <= float(out["preutterance"])
    assert float(out["consonant"]) >= float(out["preutterance"]) + 1.0


def test_apply_residual_to_decoded_row_clamps_vc_pre_inside_anchor_envelope(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_TRANSITION_ENVELOPE_GUARD_ENABLE", "on")
    monkeypatch.setenv("UTOA_BOUNDARY_VC_PRE_MAX_RATIO", "0.80")

    bundle = BoundaryResidualBundle(
        model_dir=".",
        boosters={},
        feature_names=(),
        categorical_features=(),
        delta_clips={},
        active_targets=(),
        stage1_role_bias={
            "delta_offset": {"default": 0.0, "vc": 0.0},
            # Push preutterance very late; guard should pull it back.
            "delta_preutterance": {"default": 0.0, "vc": 180.0},
            "delta_overlap": {"default": 0.0, "vc": 0.0},
            "delta_cutoff": {"default": 0.0, "vc": 0.0},
        },
        meta={},
    )

    spec = OtoRowSpec(
        wav_name="x.wav",
        wav_path="x.wav",
        alias="a k",
        role="vc",
        slot_index=0,
        slot_count=2,
        prev_alias="a",
        next_alias="ka",
        language="korean",
        format_type="cvvc",
        line_index=1,
        duration_ms=700.0,
        source_params={},
    )
    row = SimpleNamespace(
        spec=spec,
        selected_time_ms=240.0,
        fallback_used=False,
        quality_score=0.7,
        anchors=AbsoluteOtoAnchors(
            offset_abs=160.0,
            overlap_abs=190.0,
            pre_abs=220.0,
            consonant_abs=260.0,
            cutoff_abs=340.0,
            confidence=0.8,
            reason="test",
        ),
    )

    params, _audit = apply_residual_to_decoded_row(
        row=row,
        anchor_timeline_ms=[100.0, 300.0],
        params_pred={
            "offset": 160.0,
            "consonant": 100.0,
            "cutoff": -180.0,
            "preutterance": 60.0,
            "overlap": 30.0,
        },
        bundle=bundle,
    )
    pre_abs = float(params["offset"]) + float(params["preutterance"])
    # VC envelope max with left=100, right=300, ratio=0.80 -> <= 260ms.
    assert pre_abs <= 260.0 + 1e-3
