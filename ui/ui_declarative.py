# -*- coding: utf-8 -*-
import json
import os
from typing import Any

import customtkinter as ctk

from ui.ui_builders import ui_help, ui_row, ui_section
from ui.theme_tokens import PALETTE


def _load_layout(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_var(ctx, name: str, kind: str, default: Any):
    if hasattr(ctx, name):
        return getattr(ctx, name)
    if kind == "bool":
        var = ctk.BooleanVar(value=bool(default))
    elif kind == "int":
        var = ctk.IntVar(value=int(default))
    else:
        var = ctk.StringVar(value=str(default) if default is not None else "")
    setattr(ctx, name, var)
    return var


def _resolve_var(ctx, spec: dict):
    name = spec.get("var")
    if not name:
        return None
    kind = spec.get("var_type", "str")
    default = spec.get("default", "" if kind == "str" else False)
    return _ensure_var(ctx, name, kind, default)


def _resolve_color(value):
    if value is None:
        return None
    if isinstance(value, str) and hasattr(PALETTE, value):
        return getattr(PALETTE, value)
    return value


def _maybe_call(ctx, name: str, *args, **kwargs):
    if not name:
        return None
    fn = getattr(ctx, name, None)
    if callable(fn):
        return fn(*args, **kwargs)
    return None


def _pack(widget, spec: dict | None):
    if widget is None:
        return
    if not spec:
        widget.pack(anchor="w", padx=12, pady=(2, 2))
        return
    kwargs = {}
    for key in ("side", "fill", "expand", "anchor", "padx", "pady"):
        if key in spec:
            kwargs[key] = spec[key]
    widget.pack(**kwargs)


def _build_node(ctx, parent, node: dict, layout_root: dict):
    ntype = str(node.get("type", "") or "").strip().lower()
    if not ntype:
        return None

    if ntype == "section":
        fg_color = node.get("fg_color")
        if fg_color == "overlay":
            fg_color = layout_root.get("_overlay")
        fg_color = _resolve_color(fg_color)
        frame = ui_section(
            parent,
            node.get("title", ""),
            node.get("subtitle"),
            fg_color=fg_color,
            border_color=_resolve_color(node.get("border_color")),
            border_width=node.get("border_width", 1),
            pad_x=node.get("padx", 10),
            pad_y=node.get("pady", 5),
        )
        for child in node.get("children", []) or []:
            _build_node(ctx, frame, child, layout_root)
        return frame

    if ntype == "row":
        row = ui_row(parent, padx=node.get("padx", 12), pady=node.get("pady", (0, 2)))
        for child in node.get("children", []) or []:
            _build_node(ctx, row, child, layout_root)
        return row

    if ntype == "label":
        label = ctk.CTkLabel(
            parent,
            text=node.get("text", ""),
            text_color=_resolve_color(node.get("text_color", PALETTE.neutral_text)),
            justify=node.get("justify", "left"),
            wraplength=node.get("wraplength"),
            font=node.get("font"),
        )
        _pack(label, node.get("pack"))
        return label

    if ntype == "help":
        label = ui_help(
            parent,
            node.get("text", ""),
            wrap=node.get("wraplength", 730),
            padx=node.get("padx", 34),
            pady=node.get("pady", (0, 8)),
        )
        return label

    if ntype == "checkbox":
        var = _resolve_var(ctx, node)
        command = None
        if node.get("handler"):
            command = lambda: _maybe_call(ctx, node.get("handler"))
        elif hasattr(ctx, "_save_config"):
            command = ctx._save_config
        checkbox = ctk.CTkCheckBox(
            parent,
            text=node.get("text", ""),
            text_color=_resolve_color(node.get("text_color", PALETTE.neutral_text)),
            variable=var,
            command=command,
        )
        _pack(checkbox, node.get("pack"))
        return checkbox

    if ntype == "button":
        handler = node.get("handler")
        if node.get("optional") and not callable(getattr(ctx, handler, None)):
            return None
        btn = ctk.CTkButton(
            parent,
            text=node.get("text", ""),
            width=node.get("width", None) or 140,
            command=(lambda: _maybe_call(ctx, handler)) if handler else None,
        )
        _pack(btn, node.get("pack"))
        if node.get("style") == "primary" and hasattr(ctx, "_style_primary_button"):
            ctx._style_primary_button(btn)
        return btn

    if ntype == "option_menu":
        var = _resolve_var(ctx, node)
        values = node.get("values")
        if node.get("values_handler"):
            values = _maybe_call(ctx, node.get("values_handler")) or values
        menu = ctk.CTkOptionMenu(
            parent,
            values=values or [],
            variable=var,
            width=node.get("width", 180),
            command=(lambda _v: _maybe_call(ctx, node.get("handler"))) if node.get("handler") else None,
        )
        _pack(menu, node.get("pack"))
        return menu

    if ntype == "entry":
        var = _resolve_var(ctx, node)
        ent = ctk.CTkEntry(
            parent,
            width=node.get("width", 90),
            textvariable=var,
            placeholder_text=node.get("placeholder"),
        )
        _pack(ent, node.get("pack"))
        if node.get("bind_save") and hasattr(ctx, "_save_config"):
            ent.bind("<FocusOut>", lambda _e: ctx._save_config())
        return ent

    if ntype == "slot":
        name = node.get("name")
        return _maybe_call(ctx, name, parent, layout_root, node)

    return None


def build_declarative(ctx, parent, *, layout_path: str, tab_key: str, context: dict | None = None):
    layout = _load_layout(layout_path)
    if isinstance(context, dict):
        layout["_context"] = context
        if "overlay" in context:
            layout["_overlay"] = context.get("overlay")
    tab = layout.get("tabs", {}).get(tab_key, {})
    for node in tab.get("nodes", []) or []:
        _build_node(ctx, parent, node, layout)
