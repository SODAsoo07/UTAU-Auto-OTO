import argparse
import os
import shutil
import sys
from typing import List, Tuple

import torch
import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ml.wfl_asr import BIOPhonemeTagger


class WhisperEncoderWrapper(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_features)
        return out.last_hidden_state


class WFLHeadWrapper(torch.nn.Module):
    def __init__(self, model: BIOPhonemeTagger):
        super().__init__()
        self.lang_emb = model.lang_emb
        self.lang_proj = model.lang_proj
        self.bilstm = model.bilstm
        self.conformer_layers = model.conformer_layers
        self.dilated_conv_stack = model.dilated_conv_stack if model.enable_dilated_conv else None
        self.classifier = model.classifier
        self.boundary_offset_head = model.boundary_offset_head

    def forward(self, hidden_states: torch.Tensor, lang_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lang_embed = self.lang_emb(lang_id)
        lang_embed = lang_embed.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
        hidden_states = torch.cat([hidden_states, lang_embed], dim=-1)
        hidden_states = self.lang_proj(hidden_states)

        if self.bilstm is not None:
            hidden_states, _ = self.bilstm(hidden_states)
        out = hidden_states
        for layer in self.conformer_layers:
            out = layer(out)
        if self.dilated_conv_stack is not None:
            out = self.dilated_conv_stack(out.transpose(1, 2)).transpose(1, 2)

        logits = self.classifier(out)
        offsets = self.boundary_offset_head(out.transpose(1, 2)).transpose(1, 2)
        return logits, offsets


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_labels(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _extract_state_dict(payload):
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    return payload


def export_onnx(
    config_path: str,
    checkpoint_path: str,
    phonemes_path: str,
    out_dir: str,
    opset: int = 17,
    dynamic: bool = True,
) -> None:
    config = load_config(config_path)
    labels = load_labels(phonemes_path)
    model = BIOPhonemeTagger(config, labels)
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(payload)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if str(model.encoder_type).lower() != "whisper":
        raise ValueError("This export script is for Whisper encoder only.")

    os.makedirs(out_dir, exist_ok=True)

    encoder_wrapper = WhisperEncoderWrapper(model.encoder)
    head_wrapper = WFLHeadWrapper(model)

    sample_rate = int(config["data"].get("sample_rate", 16000))
    hop_length = 160
    sample_frames = int(30.0 * float(sample_rate) / float(hop_length))
    input_features = torch.randn(1, 80, sample_frames, dtype=torch.float32)
    hidden_states = torch.randn(1, sample_frames // 2, model.encoder.config.d_model, dtype=torch.float32)
    lang_id = torch.zeros(1, dtype=torch.long)

    encoder_path = os.path.join(out_dir, "encoder.onnx")
    head_path = os.path.join(out_dir, "head.onnx")

    encoder_dynamic_axes = None
    head_dynamic_axes = None
    if dynamic:
        encoder_dynamic_axes = {"input_features": {0: "batch", 2: "frames"}, "hidden_states": {0: "batch", 1: "frames"}}
        head_dynamic_axes = {
            "hidden_states": {0: "batch", 1: "frames"},
            "lang_id": {0: "batch"},
            "logits": {0: "batch", 1: "frames"},
            "offsets": {0: "batch", 1: "frames"},
        }

    torch.onnx.export(
        encoder_wrapper,
        (input_features,),
        encoder_path,
        input_names=["input_features"],
        output_names=["hidden_states"],
        dynamic_axes=encoder_dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )

    torch.onnx.export(
        head_wrapper,
        (hidden_states, lang_id),
        head_path,
        input_names=["hidden_states", "lang_id"],
        output_names=["logits", "offsets"],
        dynamic_axes=head_dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )


def _copy_asset(src: str, dst_dir: str) -> None:
    if not src or not os.path.exists(src):
        return
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export WFL-ASR Whisper encoder/head to ONNX.")
    parser.add_argument("--config", required=True, help="Path to config_infer.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to model .pt checkpoint")
    parser.add_argument("--phonemes", required=True, help="Path to phonemes.txt")
    parser.add_argument("--langs", required=False, default="", help="Path to langs.txt (optional copy)")
    parser.add_argument("--merge-map", required=False, default="", help="Path to phoneme_merge_map.json (optional copy)")
    parser.add_argument("--out-dir", required=True, help="Output directory for ONNX models")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--static", action="store_true", help="Disable dynamic axes")
    parser.add_argument("--copy-assets", action="store_true", help="Copy config/labels into output dir")
    args = parser.parse_args()

    export_onnx(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        phonemes_path=args.phonemes,
        out_dir=args.out_dir,
        opset=args.opset,
        dynamic=not args.static,
    )

    if args.copy_assets:
        _copy_asset(args.config, args.out_dir)
        _copy_asset(args.phonemes, args.out_dir)
        _copy_asset(args.langs, args.out_dir)
        _copy_asset(args.merge_map, args.out_dir)

    print(f"Exported ONNX models to: {args.out_dir}")


if __name__ == "__main__":
    main()
