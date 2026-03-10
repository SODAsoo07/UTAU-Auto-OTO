"""Shared UI theme tokens.

This module is the bridge point for design values agreed in Pencil (.pen)
and values consumed by the CustomTkinter implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Palette:
    panel_bg: str = "#1F232B"
    panel_border: str = "#303744"
    header_accent: str = "#64B5F6"

    primary_button_bg: str = "#2F7EB8"
    primary_button_hover: str = "#286C9D"
    primary_button_text: str = "#EAF4FF"

    menu_bg: str = "#2F7EB8"
    menu_button: str = "#286C9D"
    menu_button_hover: str = "#1F5F8A"
    menu_dropdown_bg: str = "#202833"
    menu_dropdown_hover: str = "#2B3643"
    menu_text: str = "#EAF4FF"

    lang_dropdown_default_bg: str = "#ED7A00"
    lang_dropdown_default_button: str = "#CF6800"
    lang_dropdown_default_hover: str = "#B45700"
    lang_dropdown_dropdown_bg: str = "#222A35"
    lang_dropdown_dropdown_hover: str = "#303A47"
    lang_dropdown_text: str = "#FFF5E8"

    input_bg: str = "#1A1F27"
    input_border: str = "#465467"
    hint_text: str = "#7F8A99"
    success_text: str = "#A5D6A7"

    advanced_toggle_bg: str = "#2B313D"
    advanced_toggle_hover: str = "#3A4251"
    advanced_toggle_text: str = "#DCE9F8"
    advanced_toggle_border: str = "#425064"

    danger_button_bg: str = "#FF6B6B"
    danger_button_hover: str = "#EE5A5A"

    neutral_text: str = "gray"


PALETTE = Palette()

LANGUAGE_NOTICE_THEME: Dict[str, Dict[str, str]] = {
    "korean": {
        "fg_color": "#2E4A3F",
        "text_color": "#C8E6C9",
    },
    "japanese": {
        "fg_color": "#4E342E",
        "text_color": "#FFE0B2",
    },
}

LANGUAGE_DROPDOWN_THEME: Dict[str, Dict[str, str]] = {
    "korean": {
        "fg_color": "#2E7D32",
        "button_color": "#1B5E20",
        "button_hover_color": "#145A18",
    },
    "japanese": {
        "fg_color": "#EF6C00",
        "button_color": "#E65100",
        "button_hover_color": "#BF360C",
    },
}
