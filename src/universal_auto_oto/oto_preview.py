"""Read-only OTO preview: spectrogram and waveform of a row's WAV with its OTO parameters drawn on top.

Rows are colored by the clean/review split (oto.clean.ini / oto.review.ini next to the previewed
oto.ini) and, when row_report.jsonl is present, by the fitted row risk (row_risk.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import stft

from .waveform_view import WaveformViewport, visible_waveform_columns
from .oto_io import oto_cutoff_end_abs, read_text_auto

SPECTROGRAM_MAX_HZ = 8000.0
SPECTROGRAM_RANGE_DB = 80.0
SPECTROGRAM_HOP_MS = 4.0
WAVEFORM_ENVELOPE_COLUMNS = 32768
ROW_CONTEXT_MS = 120.0
HIGH_RISK = 0.5  # rows above this probability of a large pre error are drawn red, not orange

# UTAU setParam convention: blanks shaded blue, fixed range pink, pre red, overlap green.
PARAM_COLORS = {
    "blank": "#3b5b8c",
    "fixed": "#c2185b",
    "preutterance": "#ff5252",
    "overlap": "#69f0ae",
    "offset": "#90caf9",
    "cutoff": "#90caf9",
}
SPLIT_COLORS = {"clean": "#2e7d32", "review": "#ef6c00", "high_risk": "#c62828", "unknown": "#546e7a"}
SPLIT_TEXT_COLORS = {"clean": "#81c784", "review": "#ffb74d", "high_risk": "#ff6e6e", "unknown": "#90a4ae"}
# The generator UI's dark palette (oto_generator_ui.PANEL_BG etc.).
DARK = {
    "bg": "#1F232B",
    "field": "#262B35",
    "border": "#303744",
    "text": "#FFFFFF",
    "hint": "#DCE9F8",
    "primary": "#2F7EB8",
    "primary_hover": "#286C9D",
    "select": "#2F5F86",
    "canvas": "#101418",
}
# Magma-like stops for the spectrogram colormap (position, RGB).
COLORMAP_STOPS = (
    (0.00, (0, 0, 4)),
    (0.25, (59, 15, 112)),
    (0.50, (140, 41, 129)),
    (0.70, (222, 73, 104)),
    (0.85, (254, 159, 109)),
    (1.00, (252, 253, 191)),
)


@dataclass(frozen=True)
class OtoPreviewRow:
    line_number: int  # 1-based line in the oto file
    wav: str
    alias: str
    offset: float
    consonant: float
    cutoff: float
    preutterance: float
    overlap: float

    @property
    def key(self) -> tuple[str, str]:
        return self.wav.casefold(), self.alias


@dataclass(frozen=True)
class RowStatus:
    split: str  # clean / review / unknown
    risk: float | None = None  # P(|pre error| > 80 ms) when the row went through row_risk.py
    confidence: float | None = None
    reasons: tuple[str, ...] = ()

    @property
    def color_key(self) -> str:
        if self.split == "review" and self.risk is not None and self.risk > HIGH_RISK:
            return "high_risk"
        return self.split if self.split in SPLIT_COLORS else "unknown"

    @property
    def label(self) -> str:
        names = {"clean": "clean", "review": "review", "high_risk": "review!", "unknown": "-"}
        return names[self.color_key]


def _number(value: str) -> float:
    try:
        return float(value.strip())
    except ValueError:
        return 0.0


def parse_oto_rows(path: Path) -> list[OtoPreviewRow]:
    text, _, _ = read_text_auto(path)
    rows: list[OtoPreviewRow] = []
    for index, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        wav, payload = line.split("=", 1)
        if not wav.strip():
            continue
        fields = payload.split(",")
        values = [_number(item) for item in fields[1:6]] + [0.0] * (5 - len(fields[1:6]))
        rows.append(OtoPreviewRow(index, wav.strip(), fields[0].strip(), *values))
    return rows


def occurrence_keys(rows: Iterable[OtoPreviewRow]) -> list[tuple[str, str, int]]:
    """(wav, alias, n): the n-th row with this wav and alias. Stable across oto/clean/review files,
    which keep the generator's row order."""

    seen: dict[tuple[str, str], int] = defaultdict(int)
    keys = []
    for row in rows:
        keys.append((*row.key, seen[row.key]))
        seen[row.key] += 1
    return keys


def _companion(oto_path: Path, part: str) -> Path:
    return oto_path.with_name(f"{oto_path.stem}.{part}{oto_path.suffix}")


def load_row_statuses(oto_path: Path, rows: Sequence[OtoPreviewRow]) -> list[RowStatus]:
    """Status per row, in order. Split comes from the .clean/.review companions when they cover
    exactly the previewed rows; risk and reasons from row_report.jsonl beside them."""

    keys = occurrence_keys(rows)
    split: dict[tuple[str, str, int], str] = {}
    part = oto_path.stem.rsplit(".", 1)[-1] if "." in oto_path.stem else ""
    if part in ("clean", "review"):  # the split file itself: every row has that split
        base_path = oto_path.with_name(oto_path.stem.rsplit(".", 1)[0] + oto_path.suffix)
        split = {key: part for key in keys}
    else:
        part, base_path = "", oto_path
    clean_path, review_path = _companion(oto_path, "clean"), _companion(oto_path, "review")
    if not part and clean_path.is_file() and review_path.is_file():
        for name, path in (("clean", clean_path), ("review", review_path)):
            try:
                companion_rows = parse_oto_rows(path)
            except (OSError, ValueError, UnicodeError):
                companion_rows = []
            for key in occurrence_keys(companion_rows):
                split[key] = name
        if set(split) != set(keys):  # stale or hand-edited companions: do not guess
            split = {}

    report: dict[tuple[str, str, int], Mapping[str, object]] = {}
    report_path = oto_path.with_name("row_report.jsonl")
    if report_path.is_file():
        seen: dict[tuple[str, str], int] = defaultdict(int)
        try:
            lines = report_path.read_text(encoding="utf-8").splitlines()
            entries = [json.loads(line) for line in lines if line.strip()]
        except (OSError, ValueError):
            entries = []
        base_name = base_path.name.casefold()
        for entry in entries:
            target = str(entry.get("output_oto") or "").casefold()
            if target and Path(target).name != base_name:
                continue
            if part and entry.get("split") != part:
                continue
            pair = (str(entry.get("wav", "")).casefold(), str(entry.get("alias", "")))
            report[(*pair, seen[pair])] = entry
            seen[pair] += 1
        if not set(keys) <= set(report):
            report = {}

    statuses = []
    for key in keys:
        entry = report.get(key, {})
        reasons = tuple(str(item) for item in entry.get("reasons", ()) or ())
        confidence = entry.get("confidence")
        confidence = float(confidence) if isinstance(confidence, (int, float)) else None
        # row_risk.py stores confidence = 1 - risk on the rows it scored.
        risked = any(reason.startswith("row_risk_") for reason in reasons)
        risk = 1.0 - confidence if risked and confidence is not None else None
        row_split = split.get(key) or str(entry.get("split") or "unknown")
        statuses.append(RowStatus(row_split, risk, confidence, reasons))
    return statuses


@dataclass(frozen=True)
class OtoMarkers:
    offset: float
    fixed_end: float
    cutoff_end: float
    preutterance: float
    overlap: float


def oto_markers(row: OtoPreviewRow, duration_ms: float) -> OtoMarkers:
    return OtoMarkers(
        offset=row.offset,
        fixed_end=row.offset + row.consonant,
        cutoff_end=oto_cutoff_end_abs(duration_ms=duration_ms, offset=row.offset, cutoff=row.cutoff),
        preutterance=row.offset + row.preutterance,
        overlap=row.offset + row.overlap,
    )


def row_viewport(row: OtoPreviewRow, duration_ms: float, *, context_ms: float = ROW_CONTEXT_MS) -> WaveformViewport:
    markers = oto_markers(row, duration_ms)
    times = [markers.offset, markers.fixed_end, markers.cutoff_end, markers.preutterance, markers.overlap]
    start = max(0.0, min(times) - context_ms)
    end = min(duration_ms, max(times) + context_ms)
    span = max(1.0, end - start)
    return WaveformViewport(duration_ms, duration_ms / span, start)


@dataclass(frozen=True)
class PreviewAudio:
    samples: np.ndarray  # mono float32
    sample_rate: int
    envelope: np.ndarray  # (columns, 2) min/max
    spectrogram_db: np.ndarray  # (bins, frames), 0 dB = loudest, floor -SPECTROGRAM_RANGE_DB
    frame_times_ms: np.ndarray

    @property
    def duration_ms(self) -> float:
        return 1000.0 * len(self.samples) / float(self.sample_rate)


def waveform_envelope(mono: np.ndarray, columns: int = WAVEFORM_ENVELOPE_COLUMNS) -> np.ndarray:
    if not len(mono):
        return np.zeros((0, 2), dtype=np.float32)
    block = max(1, int(np.ceil(len(mono) / columns)))
    padded = int(np.ceil(len(mono) / block)) * block
    values = np.pad(mono, (0, padded - len(mono))).reshape(-1, block)
    return np.column_stack((values.min(axis=1), values.max(axis=1))).astype(np.float32)


def spectrogram_db(mono: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    window = 1 << int(np.ceil(np.log2(max(64.0, sample_rate * 0.025))))  # ~25 ms
    hop = max(1, int(round(sample_rate * SPECTROGRAM_HOP_MS / 1000.0)))
    if len(mono) < window:
        mono = np.pad(mono, (0, window - len(mono)))
    freqs, times, values = stft(mono, fs=sample_rate, nperseg=window, noverlap=window - hop,
                                boundary=None, padded=False)
    keep = freqs <= min(SPECTROGRAM_MAX_HZ, sample_rate / 2.0)
    power = np.abs(values[keep]) ** 2
    db = 10.0 * np.log10(power + 1e-12)
    db = np.maximum(db - db.max(), -SPECTROGRAM_RANGE_DB).astype(np.float32)
    return db, (times * 1000.0).astype(np.float64)


def load_preview_audio(path: Path) -> PreviewAudio:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    envelope = waveform_envelope(mono / peak if peak > 0 else mono)
    db, times = spectrogram_db(mono, int(sample_rate))
    return PreviewAudio(mono, int(sample_rate), envelope, db, times)


def _colormap_lut() -> np.ndarray:
    positions = np.asarray([stop[0] for stop in COLORMAP_STOPS])
    colors = np.asarray([stop[1] for stop in COLORMAP_STOPS], dtype=np.float64)
    grid = np.linspace(0.0, 1.0, 256)
    return np.stack([np.interp(grid, positions, colors[:, channel]) for channel in range(3)], axis=1).astype(np.uint8)


COLORMAP_LUT = _colormap_lut()


def spectrogram_rgb(audio: PreviewAudio, viewport: WaveformViewport, width: int, height: int) -> np.ndarray:
    """(height, width, 3) uint8 image of the visible span, low frequencies at the bottom."""

    width, height = max(1, int(width)), max(1, int(height))
    db, times = audio.spectrogram_db, audio.frame_times_ms
    if db.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    pixel_times = viewport.start_ms + (np.arange(width) + 0.5) / width * viewport.span_ms
    columns = np.clip(np.searchsorted(times, pixel_times), 0, db.shape[1] - 1)
    rows = np.clip(((height - 1 - np.arange(height) + 0.5) / height * db.shape[0]).astype(np.int64), 0, db.shape[0] - 1)
    levels = ((db[np.ix_(rows, columns)] / SPECTROGRAM_RANGE_DB + 1.0) * 255.0).clip(0, 255).astype(np.uint8)
    return COLORMAP_LUT[levels]


def ppm_bytes(rgb: np.ndarray) -> bytes:
    height, width = rgb.shape[:2]
    return f"P6 {width} {height} 255\n".encode("ascii") + np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()


def resolve_wav(row: OtoPreviewRow, wav_dirs: Sequence[Path]) -> Path | None:
    for directory in wav_dirs:
        candidate = directory / row.wav
        if candidate.is_file():
            return candidate
    return None


def format_row_details(row: OtoPreviewRow, status: RowStatus, markers: OtoMarkers | None) -> str:
    parts = [
        f"{row.alias}  ({row.wav}, {row.line_number}행)",
        f"offset {row.offset:.1f}  고정 {row.consonant:.1f}  cutoff {row.cutoff:.1f}  "
        f"선행발성 {row.preutterance:.1f}  오버랩 {row.overlap:.1f}",
    ]
    state = f"상태 {status.label}"
    if status.risk is not None:
        state += f"  risk {status.risk:.2f}"
    elif status.confidence is not None:
        state += f"  confidence {status.confidence:.2f}"
    if markers is not None:
        state += f"  |  절대 위치: 선행발성 {markers.preutterance:.1f} ms, 끝 {markers.cutoff_end:.1f} ms"
    parts.append(state)
    shown = [reason for reason in status.reasons if not reason.startswith("row_risk_")]
    if shown:
        parts.append("근거: " + ", ".join(shown))
    return "\n".join(parts)


def status_summary(statuses: Sequence[RowStatus]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for status in statuses:
        counts[status.color_key] += 1
    summary = (f"{len(statuses)}행  clean {counts['clean']}  review {counts['review'] + counts['high_risk']}"
               f" (그중 risk>{HIGH_RISK:.1f} {counts['high_risk']})")
    if statuses and counts["unknown"] == len(statuses):
        summary += "  |  clean/review 파일이 없어 위험도 표시 없음"
    return summary


def apply_dark_style(window) -> None:
    """Dark ttk styling matching the generator UI. The generator draws only customtkinter widgets,
    so switching the shared ttk theme to clam (the one that honors colors on Windows) is safe."""

    from tkinter import ttk

    style = ttk.Style(window)
    style.theme_use("clam")
    bg, field, border, text = DARK["bg"], DARK["field"], DARK["border"], DARK["text"]
    style.configure(".", background=bg, foreground=text, fieldbackground=field, bordercolor=border,
                    lightcolor=border, darkcolor=border, troughcolor=bg, insertcolor=text,
                    selectbackground=DARK["select"], selectforeground=text)
    style.configure("TButton", background=DARK["primary"], foreground=text, borderwidth=0, padding=(8, 4))
    style.map("TButton", background=[("pressed", DARK["primary_hover"]), ("active", DARK["primary_hover"])])
    style.map("TCheckbutton", background=[("active", bg)],
              indicatorcolor=[("selected", DARK["primary"]), ("!selected", field)])
    style.map("TCombobox", fieldbackground=[("readonly", field)], foreground=[("readonly", text)],
              selectbackground=[("readonly", field)], selectforeground=[("readonly", text)],
              arrowcolor=[("!disabled", text)])
    style.configure("Treeview", background=field, fieldbackground=field, foreground=text, rowheight=22,
                    borderwidth=0)
    style.map("Treeview", background=[("selected", DARK["select"])], foreground=[("selected", text)])
    style.configure("Treeview.Heading", background=border, foreground=DARK["hint"], relief="flat")
    style.map("Treeview.Heading", background=[("active", DARK["select"])])
    style.configure("TScrollbar", background=border, arrowcolor=DARK["hint"], borderwidth=0)
    style.configure("Hint.TLabel", foreground=DARK["hint"])
    window.configure(background=bg)
    # The combobox drop-down is a plain Tk listbox.
    window.option_add("*TCombobox*Listbox.background", field)
    window.option_add("*TCombobox*Listbox.foreground", text)
    window.option_add("*TCombobox*Listbox.selectBackground", DARK["select"])


def launch_oto_preview(oto_path: Path, wav_dirs: Sequence[Path] = (), *, master=None):
    """Open the preview window. With `master` it is a Toplevel of that app, otherwise its own Tk root.
    The returned window has `load_oto(oto_path, wav_dirs)` to show another oto.ini in the same window."""

    import tkinter as tk
    from tkinter import messagebox, ttk

    rows: list[OtoPreviewRow] = []
    statuses: list[RowStatus] = []
    search_dirs: list[Path] = []

    window = tk.Toplevel(master) if master is not None else tk.Tk()
    window.geometry("1280x760")
    window.minsize(900, 520)
    apply_dark_style(window)

    state: dict[str, object] = {"audio": None, "audio_path": None, "row": None, "index": None, "viewport": None,
                                "image": None, "pan": None, "temp": None, "oto_path": None}
    cache: dict[Path, PreviewAudio] = {}

    filter_var = tk.StringVar(value="전체")
    search_var = tk.StringVar()
    sort_risk_var = tk.BooleanVar(value=False)
    details_var = tk.StringVar(value="행을 선택하세요.")
    summary_var = tk.StringVar()

    paned = ttk.PanedWindow(window, orient="horizontal")
    paned.pack(fill="both", expand=True)
    left = ttk.Frame(paned, padding=6)
    right = ttk.Frame(paned, padding=6)
    paned.add(left, weight=1)
    paned.add(right, weight=3)

    controls = ttk.Frame(left)
    controls.pack(fill="x")
    ttk.Combobox(controls, textvariable=filter_var, state="readonly", width=10,
                 values=("전체", "review만", "clean만")).pack(side="left")
    ttk.Entry(controls, textvariable=search_var, width=16).pack(side="left", padx=4, fill="x", expand=True)
    ttk.Checkbutton(controls, text="위험도순", variable=sort_risk_var).pack(side="left")
    ttk.Label(left, textvariable=summary_var, style="Hint.TLabel", wraplength=360).pack(fill="x", pady=(4, 4))

    tree_frame = ttk.Frame(left)
    tree_frame.pack(fill="both", expand=True)
    tree = ttk.Treeview(tree_frame, columns=("line", "alias", "wav", "state", "risk"), show="headings",
                        selectmode="browse")
    for column, title, width, anchor in (("line", "#", 44, "e"), ("alias", "별칭", 90, "w"), ("wav", "WAV", 150, "w"),
                                         ("state", "상태", 64, "center"), ("risk", "risk", 50, "e")):
        tree.heading(column, text=title)
        tree.column(column, width=width, anchor=anchor, stretch=column in ("alias", "wav"))
    for key, color in SPLIT_TEXT_COLORS.items():
        tree.tag_configure(key, foreground=color)
    scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    tree.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    toolbar = ttk.Frame(right)
    toolbar.pack(fill="x")
    canvas = tk.Canvas(right, background=DARK["canvas"], highlightthickness=0)
    ttk.Label(right, textvariable=details_var, justify="left", font=("Consolas", 10)).pack(fill="x", side="bottom",
                                                                                          pady=(6, 0))
    hscroll = ttk.Scrollbar(right, orient="horizontal")
    hscroll.pack(fill="x", side="bottom")
    canvas.pack(fill="both", expand=True)

    def visible_indices() -> list[int]:
        query = search_var.get().strip().casefold()
        mode = filter_var.get()
        indices = []
        for index, (row, status) in enumerate(zip(rows, statuses)):
            if mode == "review만" and status.split != "review":
                continue
            if mode == "clean만" and status.split != "clean":
                continue
            if query and query not in row.alias.casefold() and query not in row.wav.casefold():
                continue
            indices.append(index)
        if sort_risk_var.get():
            indices.sort(key=lambda i: -(statuses[i].risk if statuses[i].risk is not None
                                          else 1.0 - (statuses[i].confidence or 1.0)))
        return indices

    def populate(*_args) -> None:
        selected = tree.selection()
        if selected and int(selected[0]) >= len(rows):
            selected = ()
        tree.delete(*tree.get_children())
        for index in visible_indices():
            row, status = rows[index], statuses[index]
            risk = f"{status.risk:.2f}" if status.risk is not None else ""
            tree.insert("", "end", iid=str(index), tags=(status.color_key,),
                        values=(row.line_number, row.alias, row.wav, status.label, risk))
        if selected and tree.exists(selected[0]):
            tree.selection_set(selected[0])
            tree.see(selected[0])

    def current_audio() -> PreviewAudio | None:
        audio = state["audio"]
        return audio if isinstance(audio, PreviewAudio) else None

    def draw(*_args) -> None:
        canvas.delete("all")
        audio, row, viewport = current_audio(), state["row"], state["viewport"]
        width, height = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        if audio is None or not isinstance(row, OtoPreviewRow) or not isinstance(viewport, WaveformViewport):
            canvas.create_text(width / 2, height / 2, text=str(state.get("message") or "행을 선택하세요."),
                               fill="#bac6cc")
            return
        axis = 20
        spec_height = int((height - axis) * 0.6)
        wave_top, wave_bottom = spec_height + 4, height - axis
        image = tk.PhotoImage(data=ppm_bytes(spectrogram_rgb(audio, viewport, width, spec_height)), format="PPM")
        state["image"] = image  # keep a reference or Tk drops the image
        canvas.create_image(0, 0, image=image, anchor="nw")
        center, half = (wave_top + wave_bottom) / 2.0, (wave_bottom - wave_top) / 2.0
        canvas.create_line(0, center, width, center, fill="#39454d")
        for x, low, high in visible_waveform_columns(audio.envelope, viewport, canvas_width=width):
            canvas.create_line(x, center - high * half, x, center - low * half + 1, fill="#8fa3ad")

        def x_of(time_ms: float) -> float:
            return viewport.time_to_x(time_ms, width)

        markers = oto_markers(row, audio.duration_ms)
        stipple = {"stipple": "gray50"}
        canvas.create_rectangle(x_of(0.0), 0, x_of(markers.offset), wave_bottom, fill=PARAM_COLORS["blank"],
                                outline="", **stipple)
        canvas.create_rectangle(x_of(markers.cutoff_end), 0, x_of(audio.duration_ms), wave_bottom,
                                fill=PARAM_COLORS["blank"], outline="", **stipple)
        canvas.create_rectangle(x_of(markers.offset), 0, x_of(markers.fixed_end), wave_bottom,
                                fill=PARAM_COLORS["fixed"], outline="", stipple="gray25")
        for name, time_ms, label, dash in (
            ("offset", markers.offset, "offset", (3, 3)),
            ("cutoff", markers.cutoff_end, "cutoff", (3, 3)),
            ("overlap", markers.overlap, "오버랩", ()),
            ("preutterance", markers.preutterance, "선행발성", ()),
        ):
            x = x_of(time_ms)
            if -2 <= x <= width + 2:
                canvas.create_line(x, 0, x, wave_bottom, fill=PARAM_COLORS[name], width=2, dash=dash)
                text_y = {"preutterance": 4, "overlap": 20, "offset": 36, "cutoff": 36}[name]
                canvas.create_text(x + 3, text_y, text=label, fill=PARAM_COLORS[name], anchor="nw")
        canvas.create_line(0, wave_bottom, width, wave_bottom, fill="#56636b")
        for tick in range(6):
            fraction = tick / 5.0
            x = fraction * width
            anchor = "sw" if tick == 0 else "se" if tick == 5 else "s"
            canvas.create_line(x, wave_bottom, x, wave_bottom + 5, fill="#56636b")
            canvas.create_text(x, height - 1, text=f"{viewport.start_ms + fraction * viewport.span_ms:.0f} ms",
                               fill="#bac6cc", anchor=anchor)
        status = statuses[int(state["index"])]
        canvas.create_rectangle(width - 92, 4, width - 4, 24, fill=SPLIT_COLORS[status.color_key], outline="")
        canvas.create_text(width - 48, 14, text=status.label, fill="#ffffff")
        hscroll.set(*viewport.scrollbar_range)

    def set_viewport(viewport: WaveformViewport) -> None:
        state["viewport"] = viewport
        draw()

    def select_row(_event=None) -> None:
        selection = tree.selection()
        if not selection:
            return
        index = int(selection[0])
        row = rows[index]
        path = resolve_wav(row, search_dirs)
        state.update(row=row, index=index)
        if path is None:
            state.update(audio=None, audio_path=None, message=f"WAV를 찾을 수 없습니다: {row.wav}")
            details_var.set(format_row_details(row, statuses[index], None))
            draw()
            return
        try:
            if path not in cache:
                if len(cache) > 16:
                    cache.clear()
                cache[path] = load_preview_audio(path)
        except Exception as error:  # unreadable WAV: show it, keep the window usable
            state.update(audio=None, audio_path=None, message=f"WAV를 읽을 수 없습니다: {error}")
            draw()
            return
        audio = cache[path]
        state.update(audio=audio, audio_path=path)
        details_var.set(format_row_details(row, statuses[index], oto_markers(row, audio.duration_ms)))
        set_viewport(row_viewport(row, audio.duration_ms))

    def zoom_by(factor: float, anchor_x: float | None = None) -> None:
        viewport = state["viewport"]
        if not isinstance(viewport, WaveformViewport):
            return
        width = max(1, canvas.winfo_width())
        x = width / 2.0 if anchor_x is None else anchor_x
        set_viewport(viewport.zoomed(viewport.zoom * factor, anchor_time_ms=viewport.x_to_time(x, width),
                                     anchor_fraction=max(0.0, min(1.0, x / width))))

    def wheel(event):
        if event.delta:
            steps = max(-4.0, min(4.0, event.delta / 120.0))
            viewport = state["viewport"]
            if event.state & 0x0001 and isinstance(viewport, WaveformViewport):  # Shift: pan
                set_viewport(viewport.shifted(-steps * viewport.span_ms * 0.08))
            else:
                zoom_by(1.25 ** steps, event.x)
        return "break"

    def begin_pan(event) -> None:
        viewport = state["viewport"]
        if isinstance(viewport, WaveformViewport):
            state["pan"] = (float(event.x), viewport.start_ms)

    def pan(event) -> None:
        viewport, origin = state["viewport"], state["pan"]
        if isinstance(viewport, WaveformViewport) and origin is not None:
            x0, start = origin
            delta = -(event.x - x0) / max(1, canvas.winfo_width()) * viewport.span_ms
            set_viewport(WaveformViewport(viewport.duration_ms, viewport.zoom, start + delta))

    def scroll(command: str, value: str, unit: str | None = None) -> None:
        viewport = state["viewport"]
        if not isinstance(viewport, WaveformViewport):
            return
        if command == "moveto":
            set_viewport(viewport.moved_to_fraction(float(value)))
        elif command == "scroll":
            set_viewport(viewport.shifted(float(value) * viewport.span_ms * (0.8 if unit == "pages" else 0.08)))

    def fit_row() -> None:
        audio, row = current_audio(), state["row"]
        if audio is not None and isinstance(row, OtoPreviewRow):
            set_viewport(row_viewport(row, audio.duration_ms))

    def fit_all() -> None:
        audio = current_audio()
        if audio is not None:
            set_viewport(WaveformViewport(audio.duration_ms))

    def play(whole: bool = False) -> None:
        audio, row = current_audio(), state["row"]
        if audio is None or not isinstance(row, OtoPreviewRow):
            return
        try:
            import winsound

            if whole:
                target = state["audio_path"]
            else:
                markers = oto_markers(row, audio.duration_ms)
                start = int(max(0.0, markers.offset) * audio.sample_rate / 1000.0)
                end = int(min(audio.duration_ms, markers.cutoff_end) * audio.sample_rate / 1000.0)
                if end <= start:
                    return
                if state["temp"] is None:
                    state["temp"] = tempfile.TemporaryDirectory(prefix="auto_oto_preview_")
                winsound.PlaySound(None, winsound.SND_PURGE)  # release the previous clip before rewriting it
                target = Path(state["temp"].name) / "row.wav"
                sf.write(str(target), audio.samples[start:end], audio.sample_rate, subtype="PCM_16")
            winsound.PlaySound(str(target), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as error:
            messagebox.showerror("재생 실패", str(error), parent=window)

    def stop() -> None:
        try:
            import winsound

            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

    def close() -> None:
        stop()
        temp = state["temp"]
        if temp is not None:
            temp.cleanup()
        window.destroy()

    for label, command in (("행 재생 (Space)", play), ("전체 재생", lambda: play(True)), ("정지", stop),
                           ("행에 맞춤 (F)", fit_row), ("전체 보기", fit_all),
                           ("확대 +", lambda: zoom_by(1.5)), ("축소 -", lambda: zoom_by(1 / 1.5))):
        ttk.Button(toolbar, text=label, command=command).pack(side="left", padx=(0, 4))
    legend = ttk.Frame(toolbar)
    legend.pack(side="right")
    for name, label in (("preutterance", "선행발성"), ("overlap", "오버랩"), ("fixed", "고정"), ("blank", "블랭크")):
        tk.Label(legend, text="■", fg=PARAM_COLORS[name], bg=DARK["bg"]).pack(side="left")
        ttk.Label(legend, text=label).pack(side="left", padx=(0, 6))

    hscroll.configure(command=scroll)
    tree.bind("<<TreeviewSelect>>", select_row)
    canvas.bind("<Configure>", draw)
    canvas.bind("<MouseWheel>", wheel)
    canvas.bind("<ButtonPress-1>", lambda e: (canvas.focus_set(), begin_pan(e)))
    canvas.bind("<B1-Motion>", pan)
    canvas.bind("<ButtonRelease-1>", lambda _e: state.update(pan=None))
    for widget in (tree, canvas):  # not the window: Space and F must still type in the search box
        widget.bind("<space>", lambda _e: (play(), "break")[1])
        widget.bind("<KeyPress-f>", lambda _e: fit_row())
    filter_var.trace_add("write", populate)
    search_var.trace_add("write", populate)
    sort_risk_var.trace_add("write", populate)
    window.protocol("WM_DELETE_WINDOW", close)

    def load_oto(path: Path, dirs: Sequence[Path] = ()) -> None:
        """Show `path` in this window. Reloading the same file (e.g. after regenerating it) keeps
        the selected row by its line number."""

        path = Path(path).resolve()
        new_rows = parse_oto_rows(path)  # raises before anything changes when the file is unreadable
        new_statuses = load_row_statuses(path, new_rows)
        previous = state["row"] if state["oto_path"] == path else None
        rows[:], statuses[:] = new_rows, new_statuses
        search_dirs[:] = [Path(item).resolve() for item in dirs] + [path.parent]
        cache.clear()  # a regenerated bank may have new or rewritten WAVs (alias generation)
        state.update(oto_path=path, audio=None, audio_path=None, row=None, index=None, viewport=None, message=None)
        window.title(f"OTO 미리보기 - {path}")
        summary_var.set(status_summary(statuses))
        details_var.set("행을 선택하세요.")
        tree.selection_set(())
        populate()
        children = tree.get_children()
        target = None
        if isinstance(previous, OtoPreviewRow):
            target = next((str(i) for i, row in enumerate(rows) if row.line_number == previous.line_number), None)
        if target is None or not tree.exists(target):
            target = children[0] if children else None
        if target is None:
            draw()
            return
        tree.selection_set(target)
        tree.focus(target)
        tree.see(target)

    window.load_oto = load_oto
    try:
        load_oto(oto_path, wav_dirs)
    except Exception:
        window.destroy()
        raise
    if master is None:
        window.mainloop()
    return window


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AutoOTO oto.ini preview (spectrogram + waveform)")
    parser.add_argument("oto", type=Path, help="oto.ini to preview")
    parser.add_argument("--wav-dir", type=Path, action="append", default=[],
                        help="Folder with the WAVs (default: the oto.ini folder); repeatable")
    args = parser.parse_args(argv)
    try:
        launch_oto_preview(args.oto, args.wav_dir)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
