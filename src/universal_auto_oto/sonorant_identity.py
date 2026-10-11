"""Bank-level spectral identity for sonorant phones (development option).

The heuristic HSMM emission scores phone roles (vowel-like, periodic, nasal ...) but not phone
identity, so between two adjacent sonorants (a-i, u-n, r-a) the boundary follows the duration
prior. This module learns one cepstral template per sonorant phone from a bank's first-pass
timelines and turns it into per-frame factors that lower a sonorant state's score on frames that
look like another sonorant of the same recording.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf

from .domain import PhoneDefinition

CEPSTRUM_HOP_MS = 5.0
CEPSTRUM_WINDOW_MS = 30.0
CEPSTRUM_BANDS = 26
CEPSTRUM_COEFFICIENTS = 12
TEMPLATE_MIN_SEGMENT_MS = 40.0
TEMPLATE_MIN_FRAMES = 20
DEFAULT_WEIGHT = 0.5
DEFAULT_TEMPERATURE = 1.0


def is_sonorant(definition: PhoneDefinition | None) -> bool:
    if definition is None:
        return False
    features = definition.features
    return bool(features.get("syllabic") or features.get("vowel_like") or features.get("sonorant") is True)


IDENTITY_SCOPES = ("sonorant", "speech")
NON_SPEECH_MANNERS = frozenset({"silence", "breath", "vocal_fry"})


def in_identity_scope(definition: PhoneDefinition | None, scope: str = "sonorant") -> bool:
    """Phones that get templates: sonorants, or with scope "speech" every speech phone
    (consonants included, so lenis / fortis / aspirated and glides are told apart too)."""
    if scope == "sonorant":
        return is_sonorant(definition)
    if scope == "speech":
        return definition is not None and (
            is_sonorant(definition) or definition.features.get("manner") not in NON_SPEECH_MANNERS)
    raise ValueError(f"unknown identity scope: {scope}")


def has_adjacent_sonorants(phones: Sequence[str], inventory: Mapping[str, PhoneDefinition],
                           scope: str = "sonorant") -> bool:
    return any(in_identity_scope(inventory.get(a), scope) and in_identity_scope(inventory.get(b), scope)
               for a, b in zip(phones, phones[1:]))


def cepstra(wav_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Mel cepstra c1..c12 (level-free spectral shape) on a 5 ms grid: (times_ms, frames)."""
    audio, sample_rate = sf.read(str(wav_path), always_2d=True)
    signal = audio.mean(axis=1)
    hop = max(1, int(round(sample_rate * CEPSTRUM_HOP_MS / 1000.0)))
    window = max(hop, int(round(sample_rate * CEPSTRUM_WINDOW_MS / 1000.0)))
    if len(signal) < window:
        signal = np.pad(signal, (0, window - len(signal)))
    frames = np.lib.stride_tricks.sliding_window_view(signal, window)[::hop] * np.hamming(window)
    fft_size = 1 << (window - 1).bit_length()
    power = np.abs(np.fft.rfft(frames, fft_size, axis=1)) ** 2
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)  # noqa: E731
    edges = 700.0 * (10.0 ** (np.linspace(mel(80.0), mel(min(7000.0, sample_rate / 2.0 - 1.0)), CEPSTRUM_BANDS + 2) / 2595.0) - 1.0)
    bank = np.zeros((CEPSTRUM_BANDS, len(freqs)))
    for band in range(CEPSTRUM_BANDS):
        low, centre, high = edges[band], edges[band + 1], edges[band + 2]
        bank[band] = np.clip(np.minimum((freqs - low) / (centre - low), (high - freqs) / (high - centre)), 0.0, None)
    log_mel = np.log(power @ bank.T + 1e-10)
    index = np.arange(CEPSTRUM_BANDS)
    dct = np.cos(np.pi / CEPSTRUM_BANDS * (index[None, :] + 0.5) * np.arange(1, CEPSTRUM_COEFFICIENTS + 1)[:, None])
    times = (np.arange(len(frames)) * hop + window / 2.0) / sample_rate * 1000.0
    return times, log_mel @ dct.T


def cepstra_at(wav_path: Path, times_ms: Sequence[float]) -> np.ndarray:
    grid, frames = cepstra(wav_path)
    index = np.clip(np.searchsorted(grid, np.asarray(times_ms, dtype=np.float64)), 0, len(grid) - 1)
    return frames[index]


@dataclass(frozen=True)
class SonorantTemplates:
    means: Mapping[str, np.ndarray]
    variances: Mapping[str, np.ndarray]
    frame_counts: Mapping[str, int]
    scope: str = "sonorant"

    @classmethod
    def build(cls, items: Iterable[tuple[Path, Sequence[tuple[str, float, float]]]],
              inventory: Mapping[str, PhoneDefinition], scope: str = "sonorant") -> "SonorantTemplates":
        """Templates from the middle half of every decoded sonorant segment of at least 40 ms."""
        frames: dict[str, list[np.ndarray]] = {}
        for wav_path, segments in items:
            wanted = [(phone, start, end) for phone, start, end in segments
                      if in_identity_scope(inventory.get(phone), scope) and end - start >= TEMPLATE_MIN_SEGMENT_MS]
            if not wanted:
                continue
            grid, cep = cepstra(wav_path)
            for phone, start, end in wanted:
                quarter = 0.25 * (end - start)
                mask = (grid >= start + quarter) & (grid <= end - quarter)
                if mask.any():
                    frames.setdefault(phone, []).append(cep[mask])
        means, variances, counts = {}, {}, {}
        for phone, chunks in sorted(frames.items()):
            values = np.concatenate(chunks)
            if len(values) < TEMPLATE_MIN_FRAMES:
                continue
            variance = values.var(axis=0)
            means[phone] = values.mean(axis=0)
            variances[phone] = variance + 1e-3 + 0.05 * float(variance.mean())
            counts[phone] = int(len(values))
        return cls(means=means, variances=variances, frame_counts=counts, scope=scope)

    def log_likelihood(self, phone: str, frames: np.ndarray) -> np.ndarray:
        mean, variance = self.means[phone], self.variances[phone]
        return -0.5 * (((frames - mean) ** 2 / variance) + np.log(2.0 * np.pi * variance)).mean(axis=1)

    def sha256(self) -> str:
        payload = {phone: [np.round(self.means[phone], 6).tolist(), np.round(self.variances[phone], 6).tolist()]
                   for phone in sorted(self.means)}
        if self.scope != "sonorant":
            payload["__scope__"] = self.scope
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def summary(self) -> dict[str, object]:
        return {"phones": sorted(self.means), "frame_counts": dict(self.frame_counts), "sha256": self.sha256(),
                **({"scope": self.scope} if self.scope != "sonorant" else {})}

    def factors(self, phones: Iterable[str], frames: np.ndarray, *, weight: float = DEFAULT_WEIGHT,
                temperature: float = DEFAULT_TEMPERATURE) -> dict[str, np.ndarray]:
        """Per-frame score factors in [1 - weight, 1] for the recording's own sonorant phones.

        A frame keeps its full score for the phone it matches best among those phones and loses
        up to ``weight`` for phones it matches clearly worse. With a single phone there is
        nothing to compare and no factor is returned.
        """
        labels = sorted({phone for phone in phones if phone in self.means})
        if len(labels) < 2:
            return {}
        likelihood = np.stack([self.log_likelihood(phone, frames) for phone in labels])
        relative = np.exp((likelihood - likelihood.max(axis=0, keepdims=True)) / max(1e-6, float(temperature)))
        return {phone: (1.0 - weight) + weight * relative[row] for row, phone in enumerate(labels)}
