"""Universal Auto OTO acoustic-first runtime."""

__version__ = "5.3.0"


def display_version(version: str = __version__) -> str:
    """The version as shown to users: "5.3.0b1" -> "5.3.0 Beta", "5.3.0b2" -> "5.3.0 Beta 2"."""

    import re

    match = re.fullmatch(r"(\d+(?:\.\d+)*)b(\d+)", version)
    if match is None:
        return version
    return f"{match.group(1)} Beta" + (f" {match.group(2)}" if match.group(2) != "1" else "")

