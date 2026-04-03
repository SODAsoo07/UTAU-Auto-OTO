import torch
import torch.nn as nn
import torchaudio
from transformers import WhisperFeatureExtractor, WhisperModel, WavLMModel, WavLMConfig, Wav2Vec2FeatureExtractor


class FeedForwardModule(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerBlock(nn.Module):
    def __init__(self, dim, heads=4, ff_expansion=4, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(dim, ff_expansion, dropout)
        self.ff2 = FeedForwardModule(dim, ff_expansion, dropout)
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        self.conv = nn.Sequential(
            nn.Conv1d(dim, 2 * dim, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, kernel_size=conv_kernel, padding=conv_kernel // 2),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.ln1(x + attn_out)
        x_ln = self.ln2(x)
        x_conv = self.conv(x_ln.transpose(1, 2)).transpose(1, 2)
        if x.size(1) != x_conv.size(1):
            min_len = min(x.size(1), x_conv.size(1))
            x = x[:, :min_len]
            x_conv = x_conv[:, :min_len]
        x = x + x_conv
        x = x + 0.5 * self.ff2(x)
        return x


class BIOPhonemeTagger(nn.Module):
    def __init__(self, config, label_list):
        super().__init__()
        encoder_type = config["model"]["encoder_type"].lower()
        model_name = config["model"]["whisper_model"] if encoder_type == "whisper" else config["model"]["wavlm_model"]

        self.encoder_type = encoder_type
        self.freeze_encoder = config["model"].get("freeze_encoder", False)
        self.enable_bilstm = config["model"].get("enable_bilstm", True)

        self.enable_dilated_conv = config["model"].get("enable_dilated_conv", True)
        self.dilated_conv_depth = config["model"].get("dilated_conv_depth", 2)
        self.dilated_conv_kernel = config["model"].get("dilated_conv_kernel", 3)

        if encoder_type == "whisper":
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
            self.encoder = WhisperModel.from_pretrained(model_name).encoder
            hidden_size = self.encoder.config.d_model
        elif encoder_type == "wavlm":
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            wavlm_config = WavLMConfig.from_pretrained(model_name)
            wavlm_config.apply_spec_augment = False
            wavlm_config.mask_time_prob = 0.0
            wavlm_config.mask_time_length = 0
            self.encoder = WavLMModel.from_pretrained(model_name, config=wavlm_config)
            hidden_size = self.encoder.config.hidden_size
        elif encoder_type in ("none", "null"):
            self.encoder = None
            self.feature_extractor = None
            self.mel_extractor = torchaudio.transforms.MelSpectrogram(
                sample_rate=config["data"]["sample_rate"],
                n_fft=400,
                hop_length=int(config["data"].get("frame_duration", 0.02) * config["data"]["sample_rate"]),
                n_mels=config["data"].get("n_mels", 80),
            )
            hidden_size = self.mel_extractor.n_mels
        else:
            raise ValueError("Unsupported encoder type. Use 'whisper', 'wavlm', or 'none'.")

        self.lang_emb_dim = config["model"].get("lang_emb_dim", 64)
        self.lang_emb = nn.Embedding(config["model"]["num_languages"], self.lang_emb_dim)
        self.lang_proj = nn.Linear(hidden_size + self.lang_emb_dim, hidden_size)

        if self.freeze_encoder and self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if self.enable_bilstm:
            self.bilstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size // 2,
                num_layers=config["model"].get("bilstm_num_layer", 1),
                batch_first=True,
                bidirectional=True,
            )
        else:
            self.bilstm = None

        self.conformer_layers = nn.ModuleList(
            [
                ConformerBlock(
                    dim=hidden_size,
                    heads=config["model"].get("conformer_heads", 4),
                    ff_expansion=config["model"].get("conformer_ff_expansion", 4),
                    conv_kernel=config["model"].get("conformer_kernel_size", 31),
                    dropout=config["model"].get("conformer_dropout", 0.1),
                )
                for _ in range(config["model"].get("num_conformer_layers", 2))
            ]
        )

        if self.enable_dilated_conv:
            convs = []
            for i in range(self.dilated_conv_depth):
                dilation = 2 ** i
                padding = dilation * (self.dilated_conv_kernel - 1) // 2
                convs.append(
                    nn.Conv1d(
                        hidden_size,
                        hidden_size,
                        kernel_size=self.dilated_conv_kernel,
                        dilation=dilation,
                        padding=padding,
                    )
                )
                convs.append(nn.ReLU())
            self.dilated_conv_stack = nn.Sequential(*convs)

        self.classifier = nn.Linear(hidden_size, len(label_list))

        self.boundary_offset_head = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_size, 2, kernel_size=1),
            nn.Sigmoid(),
        )

        self.label_list = label_list
        self.label2id = {label: i for i, label in enumerate(label_list)}
        self.id2label = {i: label for label, i in self.label2id.items()}

    def forward(self, input_values, lang_id=None, max_label_len=None):
        if self.encoder_type in ("none", "null"):
            hidden_states = self.mel_extractor(input_values).transpose(1, 2).to(input_values.device)
        elif self.encoder_type == "whisper":
            features = self.feature_extractor(
                input_values.cpu().numpy(), sampling_rate=16000, return_tensors="pt"
            )
            input_features = features["input_features"].to(input_values.device)
            encoder_outputs = self.encoder(input_features)
            hidden_states = encoder_outputs.last_hidden_state
        elif self.encoder_type == "wavlm":
            features = self.feature_extractor(
                input_values.cpu().numpy(), sampling_rate=16000, return_tensors="pt"
            )
            input_features = features["input_values"].to(input_values.device)
            hidden_states = self.encoder(input_features).last_hidden_state
        else:
            raise ValueError("Unsupported encoder_type")

        if max_label_len is not None:
            current_len = hidden_states.size(1)
            if current_len > max_label_len:
                hidden_states = hidden_states[:, :max_label_len, :]
            elif current_len < max_label_len:
                pad_len = max_label_len - current_len
                padding = torch.zeros(
                    hidden_states.size(0),
                    pad_len,
                    hidden_states.size(2),
                    device=hidden_states.device,
                )
                hidden_states = torch.cat([hidden_states, padding], dim=1)

        if lang_id is not None:
            lang_embed = self.lang_emb(lang_id)
            lang_embed = lang_embed.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
            hidden_states = torch.cat([hidden_states, lang_embed], dim=-1)
            hidden_states = self.lang_proj(hidden_states)

        if self.enable_bilstm and self.bilstm is not None:
            hidden_states, _ = self.bilstm(hidden_states)
        out = hidden_states

        for layer in self.conformer_layers:
            out = layer(out)

        if self.enable_dilated_conv:
            out = self.dilated_conv_stack(out.transpose(1, 2)).transpose(1, 2)

        logits = self.classifier(out)
        offsets = self.boundary_offset_head(out.transpose(1, 2)).transpose(1, 2)
        return logits, offsets

    def decode_predictions(self, logits):
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids

    def id_to_label(self, ids):
        return [[self.id2label[i.item()] for i in seq] for seq in ids]
