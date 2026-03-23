"""Shared UI theme tokens.

This module exposes mutable theme tokens used by CustomTkinter UI code.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Tuple


@dataclass
class Palette:
    panel_bg: str = "#EDE9E4"
    panel_border: str = "#B3B0AA"
    header_accent: str = "#394045"

    primary_button_bg: str = "#9CB3C2"
    primary_button_hover: str = "#8DA6B8"
    primary_button_text: str = "#1F2D3A"

    menu_bg: str = "#C7C3B7"
    menu_button: str = "#A69077"
    menu_button_hover: str = "#C7B391"
    menu_dropdown_bg: str = "#EDECE8"
    menu_dropdown_hover: str = "#E6E0C1"
    menu_text: str = "#262620"

    lang_dropdown_default_bg: str = "#CAD5AA"
    lang_dropdown_default_button: str = "#BCC899"
    lang_dropdown_default_hover: str = "#AFBB8B"
    lang_dropdown_dropdown_bg: str = "#EFF3F7"
    lang_dropdown_dropdown_hover: str = "#E2EAF1"
    lang_dropdown_text: str = "#2D3D4E"

    input_bg: str = "#ECE8DD"
    input_border: str = "#CFC8B8"
    hint_text: str = "#4E5866"
    success_text: str = "#3F4A58"

    advanced_toggle_bg: str = "#D7DEE5"
    advanced_toggle_hover: str = "#CAD4DE"
    advanced_toggle_text: str = "#334151"
    advanced_toggle_border: str = "#BCC6D1"

    danger_button_bg: str = "#E4BCBC"
    danger_button_hover: str = "#DDAFAF"

    neutral_text: str = "#2B2D2E"


THEME_PROFILE_LIGHT = "라이트 모드"
THEME_PROFILE_DARK = "다크 모드"
THEME_PROFILE_OPTIONS: Tuple[str, str] = (THEME_PROFILE_LIGHT, THEME_PROFILE_DARK)
DEFAULT_THEME_PROFILE = THEME_PROFILE_DARK

# Backward-compat aliases for previously used labels.
_THEME_PROFILE_ALIASES: Dict[str, str] = {
    "다크 모드": THEME_PROFILE_LIGHT,
    "다크 모드": THEME_PROFILE_DARK,
    "light": THEME_PROFILE_LIGHT,
    "dark": THEME_PROFILE_DARK,
}

_PALETTE_PRESETS: Dict[str, Palette] = {
    THEME_PROFILE_LIGHT: Palette(),
    THEME_PROFILE_DARK: Palette(
        panel_bg="#1F232B",
        panel_border="#303744",
        header_accent="#EAF4FF",
        primary_button_bg="#2F7EB8",
        primary_button_hover="#286C9D",
        primary_button_text="#EAF4FF",
        menu_bg="#2F7EB8",
        menu_button="#286C9D",
        menu_button_hover="#1F5F8A",
        menu_dropdown_bg="#202833",
        menu_dropdown_hover="#2B3643",
        menu_text="#EAF4FF",
        lang_dropdown_default_bg="#ED7A00",
        lang_dropdown_default_button="#CF6800",
        lang_dropdown_default_hover="#B45700",
        lang_dropdown_dropdown_bg="#222A35",
        lang_dropdown_dropdown_hover="#303A47",
        lang_dropdown_text="#FFF5E8",
        input_bg="#1A1F27",
        input_border="#465467",
        hint_text="#FFFFFF",
        success_text="#FFFFFF",
        advanced_toggle_bg="#2B313D",
        advanced_toggle_hover="#3A4251",
        advanced_toggle_text="#DCE9F8",
        advanced_toggle_border="#425064",
        danger_button_bg="#FF6B6B",
        danger_button_hover="#EE5A5A",
        neutral_text="#FFFFFF",
    ),
}

_LANGUAGE_NOTICE_PRESETS: Dict[str, Dict[str, Dict[str, str]]] = {
    THEME_PROFILE_LIGHT: {
        "korean": {"fg_color": "#DDECC5", "text_color": "#25332F"},
        "japanese": {"fg_color": "#F4DDCD", "text_color": "#3B2A25"},
        "english": {"fg_color": "#D6E7FA", "text_color": "#1F3248"},
    },
    THEME_PROFILE_DARK: {
        "korean": {"fg_color": "#2E4A3F", "text_color": "#FFFFFF"},
        "japanese": {"fg_color": "#4E342E", "text_color": "#FFFFFF"},
        "english": {"fg_color": "#1E3A5F", "text_color": "#FFFFFF"},
    },
}

_LANGUAGE_DROPDOWN_PRESETS: Dict[str, Dict[str, Dict[str, str]]] = {
    THEME_PROFILE_LIGHT: {
        "korean": {"fg_color": "#D0E080", "button_color": "#BED168", "button_hover_color": "#B2C45C"},
        "japanese": {"fg_color": "#F6B37A", "button_color": "#EEA163", "button_hover_color": "#E29150"},
        "english": {"fg_color": "#8ED3F4", "button_color": "#7BC6EC", "button_hover_color": "#68BAE3"},
    },
    THEME_PROFILE_DARK: {
        "korean": {"fg_color": "#2E7D32", "button_color": "#1B5E20", "button_hover_color": "#145A18"},
        "japanese": {"fg_color": "#EF6C00", "button_color": "#E65100", "button_hover_color": "#BF360C"},
        "english": {"fg_color": "#1565C0", "button_color": "#0D47A1", "button_hover_color": "#0B3D91"},
    },
}

_THEME_APPEARANCE_MODE: Dict[str, str] = {
    THEME_PROFILE_LIGHT: "light",
    THEME_PROFILE_DARK: "dark",
}

_TAB_OVERLAY_PRESETS: Dict[str, Dict[str, str]] = {
    THEME_PROFILE_LIGHT: {
        "?????": "#EFECE4",
        "다크 모드": "#E9EEF2",
        "?라이트 모드": "#EEEAE0",
        "??": "#E7EBEF",
    },
    THEME_PROFILE_DARK: {
        "?????": "#1E232B",
        "다크 모드": "#212730",
        "?라이트 모드": "#232832",
        "??": "#1A1F26",
    },
}

PALETTE = Palette()
LANGUAGE_NOTICE_THEME: Dict[str, Dict[str, str]] = {}
LANGUAGE_DROPDOWN_THEME: Dict[str, Dict[str, str]] = {}
TAB_OVERLAY_THEME: Dict[str, str] = {}
CURRENT_THEME_PROFILE = DEFAULT_THEME_PROFILE


def normalize_theme_profile_name(profile_name: str, *, default: str = DEFAULT_THEME_PROFILE) -> str:
    name = str(profile_name or "").strip()
    if name in THEME_PROFILE_OPTIONS:
        return name
    return _THEME_PROFILE_ALIASES.get(name.lower(), _THEME_PROFILE_ALIASES.get(name, default))


def _normalize_theme_profile(profile_name: str) -> str:
    return normalize_theme_profile_name(profile_name, default=DEFAULT_THEME_PROFILE)


def get_theme_appearance_mode(profile_name: str = "") -> str:
    profile = _normalize_theme_profile(profile_name or CURRENT_THEME_PROFILE)
    return _THEME_APPEARANCE_MODE.get(profile, "light")


def apply_theme_profile(profile_name: str) -> str:
    global CURRENT_THEME_PROFILE

    profile = _normalize_theme_profile(profile_name)
    preset = _PALETTE_PRESETS[profile]
    for field in fields(PALETTE):
        setattr(PALETTE, field.name, getattr(preset, field.name))

    LANGUAGE_NOTICE_THEME.clear()
    for lang, values in _LANGUAGE_NOTICE_PRESETS[profile].items():
        LANGUAGE_NOTICE_THEME[lang] = dict(values)

    LANGUAGE_DROPDOWN_THEME.clear()
    for lang, values in _LANGUAGE_DROPDOWN_PRESETS[profile].items():
        LANGUAGE_DROPDOWN_THEME[lang] = dict(values)

    TAB_OVERLAY_THEME.clear()
    for tab_name, color in _TAB_OVERLAY_PRESETS[profile].items():
        TAB_OVERLAY_THEME[tab_name] = color

    CURRENT_THEME_PROFILE = profile
    return profile


def get_current_theme_profile() -> str:
    return CURRENT_THEME_PROFILE


apply_theme_profile(DEFAULT_THEME_PROFILE)
