from __future__ import annotations


def test_oto_model_forward_shapes():
    import torch

    from core.coarse_crnn.oto_model import OtoCrnnConfig, build_oto_model

    cfg = OtoCrnnConfig(n_mels=8, hidden=4, conv_channels=6, cond_dim=3)
    model = build_oto_model(cfg)
    x = torch.randn(2, 11, 8)
    out = model(x, torch.tensor([0, 1]), torch.tensor([0, 2]))

    assert out["heatmap_logits"].shape == (2, 11, len(cfg.anchor_names))
    assert out["scalar_logits"].shape == (2, len(cfg.anchor_names))
    assert out["confidence_logits"].shape == (2,)


def test_oto_model_forward_accepts_row_context():
    import torch

    from core.coarse_crnn.oto_model import OtoCrnnConfig, build_oto_model

    cfg = OtoCrnnConfig(n_mels=8, hidden=4, conv_channels=6, cond_dim=3, numeric_context_dim=2)
    model = build_oto_model(cfg)
    x = torch.randn(2, 11, 8)
    context = torch.zeros((2, cfg.numeric_context_dim), dtype=torch.float32)
    out = model(
        x,
        torch.tensor([0, 1]),
        torch.tensor([0, 2]),
        context,
        torch.tensor([0, 3]),
        torch.tensor([0, 1]),
        torch.tensor([1, 2]),
        torch.tensor([2, 3]),
        torch.tensor([1, 2]),
        torch.tensor([2, 3]),
    )

    assert out["heatmap_logits"].shape == (2, 11, len(cfg.anchor_names))


def test_oto_model_forward_accepts_format_residual_heads():
    import torch

    from core.coarse_crnn.oto_model import OtoCrnnConfig, build_oto_model

    cfg = OtoCrnnConfig(
        n_mels=8,
        hidden=4,
        conv_channels=6,
        cond_dim=3,
        enable_format_residual_heads=True,
    )
    model = build_oto_model(cfg)
    x = torch.randn(2, 11, 8)
    out = model(x, torch.tensor([0, 1]), torch.tensor([0, 2]))

    assert out["heatmap_logits"].shape == (2, 11, len(cfg.anchor_names))
    assert out["scalar_logits"].shape == (2, len(cfg.anchor_names))


def test_oto_model_forward_accepts_shared_heads_config():
    import torch

    from core.coarse_crnn.oto_model import OtoCrnnConfig, build_oto_model

    cfg = OtoCrnnConfig(
        n_mels=8,
        hidden=4,
        conv_channels=6,
        cond_dim=3,
        enable_format_residual_heads=False,
    )
    model = build_oto_model(cfg)
    x = torch.randn(2, 11, 8)
    out = model(x, torch.tensor([0, 1]), torch.tensor([0, 2]))

    assert out["heatmap_logits"].shape == (2, 11, len(cfg.anchor_names))
    assert out["scalar_logits"].shape == (2, len(cfg.anchor_names))


def test_oto_model_config_defaults_to_relative_params_for_new_training():
    from core.coarse_crnn.oto_model import OtoCrnnConfig, uses_relative_param_head

    cfg = OtoCrnnConfig()

    assert cfg.scalar_target_mode == "relative_params"
    assert cfg.target_window_formats == ("vcv", "cvvc")
    assert cfg.target_window_frame_overrides == ("vcv=240", "cvvc=360")
    assert cfg.cvvc_target_window_alias_types == ("vc", "vv")
    assert "cvvc=0.25" in cfg.right_boundary_prior_blends
    assert uses_relative_param_head(cfg)


def test_oto_model_config_keeps_old_checkpoints_absolute():
    from core.coarse_crnn.oto_model import OtoCrnnConfig, uses_relative_param_head

    cfg = OtoCrnnConfig.from_dict({"n_mels": 8, "hidden": 4, "conv_channels": 6})

    assert cfg.scalar_target_mode == "absolute_anchors"
    assert cfg.target_window_formats == ("vcv",)
    assert cfg.target_window_frame_overrides == ()
    assert "vc" in cfg.cvvc_target_window_alias_types
    assert cfg.right_boundary_prior_blends == ()
    assert not uses_relative_param_head(cfg)


def test_oto_model_config_defaults_role_features_off_for_old_payloads():
    from core.coarse_crnn.oto_model import OtoCrnnConfig

    cfg = OtoCrnnConfig.from_dict({"n_mels": 8, "hidden": 4, "conv_channels": 6})

    assert cfg.alias_roles == ()
    assert cfg.enable_alias_role_embedding is False
    assert cfg.enable_extra_alias_flags is False


def test_oto_model_role_embedding_branch_only_when_enabled():
    import torch

    from core.coarse_crnn.oto_model import OtoCrnnConfig, build_oto_model

    cfg_off = OtoCrnnConfig(n_mels=8, hidden=4, conv_channels=6, cond_dim=3)
    cfg_on = OtoCrnnConfig(
        n_mels=8,
        hidden=4,
        conv_channels=6,
        cond_dim=3,
        enable_alias_role_embedding=True,
        enable_extra_alias_flags=True,
    )

    model_off = build_oto_model(cfg_off)
    model_on = build_oto_model(cfg_on)

    assert model_off.alias_role_emb is None
    assert model_off.extra_flags_proj is None
    assert model_on.alias_role_emb is not None
    assert model_on.extra_flags_proj is not None

    x = torch.randn(2, 11, 8)
    out = model_on(
        x,
        torch.tensor([0, 1]),
        torch.tensor([0, 2]),
        None,
        torch.tensor([0, 3]),
        torch.tensor([0, 1]),
        torch.tensor([1, 2]),
        torch.tensor([2, 3]),
        torch.tensor([1, 2]),
        torch.tensor([2, 3]),
        alias_role_ids=torch.tensor([0, 5]),
        prev_alias_role_ids=torch.tensor([1, 2]),
        next_alias_role_ids=torch.tensor([3, 4]),
        extra_alias_flags=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )

    assert out["scalar_logits"].shape == (2, len(cfg_on.anchor_names))


def test_oto_model_alias_role_id_normalizes_legacy_names():
    from core.coarse_crnn.oto_model import OtoCrnnConfig, alias_role_id

    cfg = OtoCrnnConfig()

    cv_idx = alias_role_id(cfg, "cv")
    head_idx = alias_role_id(cfg, "cv_head")  # legacy alias should map to -cv
    fallback_idx = alias_role_id(cfg, "totally-unknown")

    roles = list(cfg.alias_roles)
    assert cv_idx == roles.index("cv")
    assert head_idx == roles.index("-cv")
    assert fallback_idx == roles.index("other")
