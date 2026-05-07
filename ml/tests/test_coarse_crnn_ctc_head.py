"""CTC alignment head smoke tests (3-A scaffold).

These tests cover the minimum surface area of the CTC head and its loss path.
They are intentionally tiny — they should run on CPU in a few seconds and
do not exercise the full training loop.
"""
from __future__ import annotations


def test_ctc_phone_vocab_has_blank_at_zero():
    from core.coarse_crnn.oto_ctc import CTC_BLANK_ID, CTC_PHONE_VOCAB, CTC_VOCAB_SIZE

    assert CTC_BLANK_ID == 0
    assert CTC_PHONE_VOCAB[0] == "<blank>"
    assert CTC_PHONE_VOCAB[1] == "<unk>"
    assert CTC_VOCAB_SIZE == len(CTC_PHONE_VOCAB)


def test_oto_config_vocab_size_default_matches_ctc_module():
    """A-1 sync: appending tokens to CTC_PHONE_VOCAB must auto-resize the head.

    If a future change to oto_ctc.CTC_PHONE_VOCAB adds entries, the model
    config's default ctc_phone_vocab_size must follow without manual edits.
    """
    from core.coarse_crnn.oto_ctc import CTC_VOCAB_SIZE
    from core.coarse_crnn.oto_model import OtoCrnnConfig

    cfg = OtoCrnnConfig()
    assert cfg.ctc_phone_vocab_size == CTC_VOCAB_SIZE


def test_oto_config_from_dict_fills_missing_vocab_size():
    """Old checkpoints saved before A-1 must load with the current vocab size."""
    from core.coarse_crnn.oto_ctc import CTC_VOCAB_SIZE
    from core.coarse_crnn.oto_model import OtoCrnnConfig

    payload = {"n_mels": 8, "hidden": 4, "conv_channels": 6}
    cfg = OtoCrnnConfig.from_dict(payload)
    assert cfg.ctc_phone_vocab_size == CTC_VOCAB_SIZE


def test_phone_ids_from_alias_returns_known_tokens():
    from core.coarse_crnn.oto_ctc import CTC_UNK_ID, phone_ids_from_alias

    ids = phone_ids_from_alias("ka", language="japanese")
    # Should produce at least one phone, and not be entirely unk.
    assert len(ids) >= 1
    assert any(i != CTC_UNK_ID for i in ids)


def test_phone_ids_empty_for_blank_alias():
    from core.coarse_crnn.oto_ctc import phone_ids_from_alias

    assert phone_ids_from_alias("", language="korean") == []
    assert phone_ids_from_alias("   ", language="korean") == []


def test_oto_model_emits_ctc_logits_when_enabled():
    import torch

    from core.coarse_crnn.oto_ctc import CTC_VOCAB_SIZE
    from core.coarse_crnn.oto_model import OtoCrnnConfig, build_oto_model

    cfg = OtoCrnnConfig(
        n_mels=8,
        hidden=4,
        conv_channels=6,
        cond_dim=3,
        enable_ctc_head=True,
        ctc_phone_vocab_size=CTC_VOCAB_SIZE,
    )
    model = build_oto_model(cfg)
    x = torch.randn(2, 11, 8)
    out = model(x, torch.tensor([0, 1]), torch.tensor([0, 2]))

    assert out.get("ctc_logits") is not None
    assert out["ctc_logits"].shape == (2, 11, CTC_VOCAB_SIZE)


def test_oto_model_omits_ctc_logits_by_default():
    import torch

    from core.coarse_crnn.oto_model import OtoCrnnConfig, build_oto_model

    cfg = OtoCrnnConfig(n_mels=8, hidden=4, conv_channels=6, cond_dim=3)
    model = build_oto_model(cfg)
    x = torch.randn(2, 11, 8)
    out = model(x, torch.tensor([0, 1]), torch.tensor([0, 2]))

    # Old-checkpoint compat: when the head is off, the dict still has the key
    # but its value is None so callers can branch cheaply.
    assert out.get("ctc_logits") is None


def test_ctc_alignment_loss_is_zero_when_no_head():
    import torch

    from core.coarse_crnn.oto_training import OtoTrainConfig, _ctc_alignment_loss

    nn = __import__("torch.nn", fromlist=["functional"])
    cfg = OtoTrainConfig(ctc_loss_weight=0.3)
    mask = torch.ones((2, 8), dtype=torch.float32)
    phone_targets = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]], dtype=torch.long)
    phone_lengths = torch.tensor([3, 2], dtype=torch.long)

    loss = _ctc_alignment_loss(None, phone_targets, phone_lengths, mask, nn, cfg)
    assert float(loss.detach().cpu().item()) == 0.0


def test_ctc_alignment_loss_zero_for_empty_targets():
    import torch

    from core.coarse_crnn.oto_ctc import CTC_VOCAB_SIZE
    from core.coarse_crnn.oto_training import OtoTrainConfig, _ctc_alignment_loss

    nn = __import__("torch.nn", fromlist=["functional"])
    cfg = OtoTrainConfig(ctc_loss_weight=0.3)
    ctc_logits = torch.randn(2, 8, CTC_VOCAB_SIZE)
    mask = torch.ones((2, 8), dtype=torch.float32)
    phone_targets = torch.zeros((2, 1), dtype=torch.long)
    phone_lengths = torch.zeros((2,), dtype=torch.long)  # all empty

    loss = _ctc_alignment_loss(ctc_logits, phone_targets, phone_lengths, mask, nn, cfg)
    assert float(loss.detach().cpu().item()) == 0.0


def test_ctc_alignment_loss_finite_for_valid_batch():
    import torch

    from core.coarse_crnn.oto_ctc import CTC_VOCAB_SIZE
    from core.coarse_crnn.oto_training import OtoTrainConfig, _ctc_alignment_loss

    nn = __import__("torch.nn", fromlist=["functional"])
    cfg = OtoTrainConfig(ctc_loss_weight=0.3)
    ctc_logits = torch.randn(2, 16, CTC_VOCAB_SIZE)
    mask = torch.ones((2, 16), dtype=torch.float32)
    # Two rows: first has 3-phone target, second has 2-phone target (both valid).
    phone_targets = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]], dtype=torch.long)
    phone_lengths = torch.tensor([3, 2], dtype=torch.long)

    loss = _ctc_alignment_loss(ctc_logits, phone_targets, phone_lengths, mask, nn, cfg)
    value = float(loss.detach().cpu().item())
    assert value > 0.0
    assert torch.isfinite(loss)


def test_ctc_warmup_active_boundary_conditions():
    """Warmup activation matches a strict half-open interval [0, warmup_steps)."""
    from core.coarse_crnn.oto_training import _ctc_warmup_active

    # Disabled when warmup_steps <= 0.
    assert _ctc_warmup_active(0, 0) is False
    assert _ctc_warmup_active(50, 0) is False
    assert _ctc_warmup_active(0, -5) is False

    # Active inside the warmup window.
    assert _ctc_warmup_active(0, 100) is True
    assert _ctc_warmup_active(50, 100) is True
    assert _ctc_warmup_active(99, 100) is True

    # First step at or beyond the boundary deactivates warmup.
    assert _ctc_warmup_active(100, 100) is False
    assert _ctc_warmup_active(150, 100) is False


def test_train_step_loss_uses_ctc_only_during_warmup():
    """Warmup phase must skip anchor regression and only emit CTC gradient."""
    import torch

    from core.coarse_crnn.oto_ctc import CTC_VOCAB_SIZE
    from core.coarse_crnn.oto_training import OtoTrainConfig, _train_step_loss

    nn = __import__("torch.nn", fromlist=["functional"])
    cfg = OtoTrainConfig(
        ctc_loss_weight=0.3,
        ctc_warmup_steps=10,
        # Set non-trivial anchor weights so the post-warmup branch differs.
        heatmap_loss_weight=1.0,
        scalar_loss_weight=0.55,
    )

    torch.manual_seed(1337)
    outputs = {
        "heatmap_logits": torch.randn(2, 8, 5),
        "scalar_logits": torch.randn(2, 5),
        "confidence_logits": torch.randn(2),
        "scalar_logvar": torch.randn(2, 5),
        "ctc_logits": torch.randn(2, 8, CTC_VOCAB_SIZE),
    }
    heat = torch.zeros((2, 8, 5))
    scalar = torch.zeros((2, 5))
    weight = torch.ones((2,))
    mask = torch.ones((2, 8))
    phone_targets = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]], dtype=torch.long)
    phone_lengths = torch.tensor([3, 2], dtype=torch.long)

    common = dict(
        heat=heat,
        scalar=scalar,
        weight=weight,
        mask=mask,
        nn=nn,
        cfg=cfg,
        relative_scalar=False,
        phone_targets=phone_targets,
        phone_lengths=phone_lengths,
    )

    # Warmup active: should equal ctc_loss * lambda (only CTC term).
    loss_warmup = _train_step_loss(outputs, global_step=0, **common)
    # Post-warmup: full multi-task loss differs from CTC-only.
    loss_post = _train_step_loss(outputs, global_step=20, **common)

    assert torch.isfinite(loss_warmup) and torch.isfinite(loss_post)
    # Both should be positive.
    assert float(loss_warmup) > 0.0
    assert float(loss_post) > 0.0
    # And clearly different magnitudes (anchor losses kick in post-warmup).
    assert abs(float(loss_warmup) - float(loss_post)) > 1e-3


def test_train_step_loss_warmup_disabled_matches_oto_loss():
    """With warmup_steps=0 the dispatcher must equal _oto_loss bit-for-bit."""
    import torch

    from core.coarse_crnn.oto_ctc import CTC_VOCAB_SIZE
    from core.coarse_crnn.oto_training import OtoTrainConfig, _oto_loss, _train_step_loss

    nn = __import__("torch.nn", fromlist=["functional"])
    cfg = OtoTrainConfig(ctc_loss_weight=0.3, ctc_warmup_steps=0)
    torch.manual_seed(7)
    outputs = {
        "heatmap_logits": torch.randn(2, 8, 5),
        "scalar_logits": torch.randn(2, 5),
        "confidence_logits": torch.randn(2),
        "scalar_logvar": torch.randn(2, 5),
        "ctc_logits": torch.randn(2, 8, CTC_VOCAB_SIZE),
    }
    heat = torch.zeros((2, 8, 5))
    scalar = torch.zeros((2, 5))
    weight = torch.ones((2,))
    mask = torch.ones((2, 8))
    phone_targets = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]], dtype=torch.long)
    phone_lengths = torch.tensor([3, 2], dtype=torch.long)

    via_dispatch = _train_step_loss(
        outputs, heat, scalar, weight, mask, nn, cfg,
        relative_scalar=False,
        phone_targets=phone_targets, phone_lengths=phone_lengths,
        global_step=0,
    )
    via_full = _oto_loss(
        outputs, heat, scalar, weight, mask, nn, cfg,
        relative_scalar=False,
        phone_targets=phone_targets, phone_lengths=phone_lengths,
    )
    assert torch.allclose(via_dispatch, via_full)


def test_ctc_alignment_loss_skips_overlong_targets():
    """If a row's target length exceeds its audio frames, that row is dropped.

    Mirrors the safety guard in _ctc_alignment_loss: CTC requires
    input_length >= target_length.
    """
    import torch

    from core.coarse_crnn.oto_ctc import CTC_VOCAB_SIZE
    from core.coarse_crnn.oto_training import OtoTrainConfig, _ctc_alignment_loss

    nn = __import__("torch.nn", fromlist=["functional"])
    cfg = OtoTrainConfig(ctc_loss_weight=0.3)
    ctc_logits = torch.randn(2, 4, CTC_VOCAB_SIZE)
    mask = torch.zeros((2, 4), dtype=torch.float32)
    # Row 0 has 4 active frames; row 1 has 2 active frames.
    mask[0, :4] = 1.0
    mask[1, :2] = 1.0
    # Target: row 0 has 2 phones (fits), row 1 has 5 phones (does not fit).
    phone_targets = torch.tensor([[2, 3, 0, 0, 0], [4, 5, 6, 7, 8]], dtype=torch.long)
    phone_lengths = torch.tensor([2, 5], dtype=torch.long)

    loss = _ctc_alignment_loss(ctc_logits, phone_targets, phone_lengths, mask, nn, cfg)
    # Loss should still be finite and non-negative because row 0 still fits.
    assert torch.isfinite(loss)
    assert float(loss.detach().cpu().item()) > 0.0
