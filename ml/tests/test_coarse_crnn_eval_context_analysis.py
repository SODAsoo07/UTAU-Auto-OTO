from __future__ import annotations

from ml.scripts.coarse_crnn.analyze_oto_eval_contexts import analyze_eval_contexts


def test_analyze_eval_contexts_ranks_weak_contexts_and_examples():
    payload = {
        "files": [
            _row("cvvc", "cv", "other", "vb_a", "a", pre=120, offset=400, consonant=150, cutoff=900),
            _row("cvvc", "cv", "other", "vb_a", "b", pre=80, offset=300, consonant=100, cutoff=700),
            _row("cvvc", "vc", "multi", "vb_b", "c", pre=20, offset=40, consonant=30, cutoff=60),
            _row("cvc", "cv", "cc", "vb_c", "d", pre=500, offset=500, consonant=500, cutoff=500),
        ]
    }

    result = analyze_eval_contexts(payload, format_type="cvvc", min_files=1, top_contexts=2, top_examples=2)

    assert result["evaluated_files"] == 3
    assert result["top_contexts"][0]["key"] == "cvvc|cv|other"
    assert result["top_contexts"][0]["files"] == 2
    assert result["top_contexts"][0]["preutterance_acc_50ms"] == 0.0
    assert result["top_contexts"][1]["key"] == "cvvc|vc|multi"
    assert result["examples"][0]["alias"] == "a"


def _row(
    fmt: str,
    alias_type: str,
    transition: str,
    voicebank: str,
    alias: str,
    *,
    pre: float,
    offset: float,
    consonant: float,
    cutoff: float,
) -> dict:
    return {
        "format_type": fmt,
        "alias_type": alias_type,
        "transition_type": transition,
        "voicebank_id": voicebank,
        "alias": alias,
        "alias_role": alias_type,
        "param_abs_errors_ms": {
            "preutterance": pre,
            "offset": offset,
            "consonant": consonant,
            "cutoff_abs": cutoff,
            "overlap": 10.0,
        },
    }
