from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PreparedAutoPair:
    language: str
    format_type: str
    stage_root: str
    work_dir: str
    manual_oto: str
    auto_oto: str = ""
    tg_dir: str = ""
    dict_path: str = ""
    mfa_path: str = ""
    status: str = "pending"
    reason: str = ""

