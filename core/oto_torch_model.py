from __future__ import annotations

import math
from typing import Dict, List

import torch
from torch import nn


TARGET_DIM = 5


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.block(x)


class OtoTorchRegressor(nn.Module):
    def __init__(
        self,
        numeric_dim: int,
        categorical_vocab_sizes: Dict[str, int],
        categorical_feature_names: List[str],
        voicebank_vocab_size: int,
        mel_bins: int = 80,
        window_frames: int = 160,
    ):
        super().__init__()
        self.numeric_dim = int(numeric_dim)
        self.categorical_feature_names = list(categorical_feature_names)
        self.voicebank_vocab_size = int(voicebank_vocab_size)
        self.mel_bins = int(mel_bins)
        self.window_frames = int(window_frames)

        self.audio_branch = nn.Sequential(
            ConvBlock(1, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.audio_out = 128

        self.cat_embeddings = nn.ModuleDict()
        self.cat_dims: Dict[str, int] = {}
        for name in self.categorical_feature_names:
            vocab_size = max(2, int(categorical_vocab_sizes.get(name, 2)))
            dim = 16 if name == "voicebank_id" else min(8, max(4, int(math.ceil(math.sqrt(vocab_size)))))
            self.cat_dims[name] = dim
            self.cat_embeddings[name] = nn.Embedding(vocab_size, dim)

        tab_in = self.numeric_dim + sum(self.cat_dims.values())
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tab_in, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.SiLU(),
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.audio_out + 64, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, TARGET_DIM),
        )

    def forward(self, mel_window, numeric_features, categorical_features):
        if mel_window.dim() == 3:
            mel_window = mel_window.unsqueeze(1)
        audio = self.audio_branch(mel_window)
        audio = audio.flatten(1)

        cat_vectors = []
        for name in self.categorical_feature_names:
            if name not in categorical_features:
                raise KeyError(f"Missing categorical feature tensor: {name}")
            cat_vectors.append(self.cat_embeddings[name](categorical_features[name]))
        tab_input = torch.cat([numeric_features] + cat_vectors, dim=1)
        tab = self.tabular_mlp(tab_input)
        fused = torch.cat([audio, tab], dim=1)
        return self.fusion_mlp(fused)
