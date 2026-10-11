"""AutoOTO window icon (assets/AutoOTO.ico), shared by the generator and preview windows."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

ICON_RESOURCE = "assets/AutoOTO.ico"
# CTkToplevel sets the CustomTkinter icon 200 ms after it is created, whatever the app set before.
CTK_TOPLEVEL_ICON_DELAY_MS = 250


def app_icon_path() -> Path | None:
    path = Path(str(files("universal_auto_oto").joinpath(ICON_RESOURCE)))
    return path if path.is_file() else None


def apply_app_icon(window, *, delay_ms: int = 0) -> None:
    """Show the AutoOTO icon on `window` and make it the default of the windows it opens later.
    A missing or unreadable icon leaves the platform default; it never stops the UI."""

    path = app_icon_path()
    if path is None:
        return

    def apply() -> None:
        try:
            window.iconbitmap(str(path))
            window.iconbitmap(default=str(path))
        except Exception:
            pass

    if delay_ms > 0:
        window.after(delay_ms, apply)
    else:
        apply()
