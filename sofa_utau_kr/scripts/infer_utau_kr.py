#!/usr/bin/env python
"""
UTAU KR infer wrapper for SOFA.

Adds:
- explicit sofa repo path injection
- AP detector attribute override via json config
- UTAU-friendly defaults
"""

import json
import os
import pathlib
import sys

import click
import lightning as pl
import torch


def _append_repo_to_path(repo_dir: str) -> None:
    if repo_dir and repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)


def _load_ap_config(path: str) -> dict:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"ap config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("ap config must be a json object")
    return data


def _apply_ap_config(detector, cfg: dict) -> None:
    if not cfg:
        return
    for key, value in cfg.items():
        if hasattr(detector, key):
            setattr(detector, key, value)


@click.command()
@click.option("--sofa_repo_dir", required=True, type=str, help="Path to SOFA repo root")
@click.option("--ckpt", "-c", required=True, type=str, help="Path to checkpoint")
@click.option("--folder", "-f", default="segments", type=str, help="Input folder")
@click.option("--mode", "-m", default="force", type=click.Choice(["force", "match"]))
@click.option("--g2p", "-g", default="Dictionary", type=str, help="G2P class")
@click.option(
    "--ap_detector",
    "-a",
    default="LoudnessSpectralcentroidAPDetector",
    type=str,
    help="AP detector class",
)
@click.option("--ap_config", default="", type=str, help="Optional AP detector config json")
@click.option("--in_format", "-if", default="lab", type=str)
@click.option("--out_formats", "-of", default="textgrid", type=str)
@click.option("--save_confidence", "-sc", is_flag=True, default=False, show_default=True)
@click.option(
    "--dictionary",
    "-d",
    default="dictionary/opencpop-extension.txt",
    type=str,
    help="Dictionary path when g2p=Dictionary",
)
def main(
    sofa_repo_dir,
    ckpt,
    folder,
    mode,
    g2p,
    ap_detector,
    ap_config,
    in_format,
    out_formats,
    save_confidence,
    **kwargs,
):
    _append_repo_to_path(sofa_repo_dir)

    import modules.AP_detector  # noqa: WPS433
    import modules.g2p  # noqa: WPS433
    from modules.utils.export_tool import Exporter  # noqa: WPS433
    from modules.utils.post_processing import post_processing  # noqa: WPS433
    from train import LitForcedAlignmentTask  # noqa: WPS433

    g2p_name = g2p if g2p.endswith("G2P") else f"{g2p}G2P"
    g2p_class = getattr(modules.g2p, g2p_name)
    grapheme_to_phoneme = g2p_class(**kwargs)
    grapheme_to_phoneme.set_in_format(in_format)
    dataset = grapheme_to_phoneme.get_dataset(pathlib.Path(folder).rglob("*.wav"))

    detector_name = ap_detector if ap_detector.endswith("APDetector") else f"{ap_detector}APDetector"
    detector_class = getattr(modules.AP_detector, detector_name)
    detector = detector_class(**kwargs)
    _apply_ap_config(detector, _load_ap_config(ap_config))

    out_format_list = [item.strip().lower() for item in out_formats.split(",") if item.strip()]
    if save_confidence and "confidence" not in out_format_list:
        out_format_list.append("confidence")

    torch.set_grad_enabled(False)
    model = LitForcedAlignmentTask.load_from_checkpoint(ckpt)
    model.set_inference_mode(mode)
    trainer = pl.Trainer(logger=False)
    predictions = trainer.predict(model, dataloaders=dataset, return_predictions=True)

    predictions = detector.process(predictions)
    predictions, log = post_processing(predictions)
    exporter = Exporter(predictions, log)
    exporter.export(out_format_list)
    print("Output files are saved to the same folder as the input wav files.")


if __name__ == "__main__":
    main()

