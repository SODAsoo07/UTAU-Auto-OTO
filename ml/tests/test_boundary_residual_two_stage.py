from __future__ import annotations

from core.coarse_crnn.boundary_residual import (
    BoundaryResidualBundle,
    apply_boundary_residual_deltas,
    predict_boundary_stage1_role_bias,
)


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
