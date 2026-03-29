# -*- coding: utf-8 -*-
import customtkinter as ctk

from ui.theme_tokens import PALETTE


def ui_section(
    parent,
    title: str,
    subtitle: str | None = None,
    *,
    fg_color=None,
    border_color=None,
    border_width=1,
    pad_x=10,
    pad_y=5,
):
    frame = ctk.CTkFrame(
        parent,
        fg_color=PALETTE.panel_bg if fg_color is None else fg_color,
        border_width=border_width,
        border_color=PALETTE.panel_border if border_color is None else border_color,
    )
    frame.pack(fill="x", padx=pad_x, pady=pad_y)
    ctk.CTkLabel(
        frame,
        text=title,
        font=("", 14, "bold"),
        text_color=PALETTE.header_accent,
    ).pack(anchor="w", padx=12, pady=(10, 6))
    if subtitle:
        ctk.CTkLabel(
            frame,
            text=subtitle,
            text_color=PALETTE.hint_text,
        ).pack(anchor="w", padx=12, pady=(0, 6))
    return frame


def ui_help(parent, text: str, *, wrap=730, padx=34, pady=(0, 8)):
    label = ctk.CTkLabel(
        parent,
        text=text,
        text_color=PALETTE.hint_text,
        justify="left",
        wraplength=wrap,
    )
    label.pack(anchor="w", padx=padx, pady=pady)
    return label


def ui_checkbox(
    parent,
    *,
    text: str,
    variable,
    command=None,
    text_color=PALETTE.neutral_text,
    padx=12,
    pady=(2, 2),
    anchor="w",
):
    checkbox = ctk.CTkCheckBox(
        parent,
        text=text,
        text_color=text_color,
        variable=variable,
        command=command,
    )
    checkbox.pack(anchor=anchor, padx=padx, pady=pady)
    return checkbox


def ui_row(parent, *, padx=12, pady=(0, 2)):
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", padx=padx, pady=pady)
    return row


def ui_inline_checkbox(parent, *, text: str, variable, command=None, padx=(0, 0)):
    checkbox = ctk.CTkCheckBox(
        parent,
        text=text,
        variable=variable,
        command=command,
    )
    checkbox.pack(side="left", padx=padx)
    return checkbox
