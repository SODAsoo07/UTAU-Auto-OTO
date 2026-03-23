from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple


def _subprocess_window_kwargs() -> Dict[str, Any]:
    if os.name != "nt":
        return {}
    kwargs: Dict[str, Any] = {}
    try:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        kwargs["startupinfo"] = startupinfo
    except Exception:
        pass
    try:
        kwargs["creationflags"] = int(kwargs.get("creationflags", 0)) | int(
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    except Exception:
        pass
    return kwargs


def _run_command(args: List[str], timeout_sec: int = 12) -> Tuple[int, str, str]:
    try:
        kwargs = _subprocess_window_kwargs()
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=max(1, int(timeout_sec)),
            **kwargs,
        )
        return int(proc.returncode), str(proc.stdout or ""), str(proc.stderr or "")
    except Exception as exc:
        return 999, "", str(exc)


def _parse_version_tuple(text: str) -> Optional[Tuple[int, int]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    m = re.search(r"(\d+)\.(\d+)", raw)
    if not m:
        return None
    try:
        return int(m.group(1)), int(m.group(2))
    except Exception:
        return None


def detect_nvidia_runtime() -> Dict[str, Any]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {
            "nvidia_smi_found": False,
            "nvidia_gpu_present": False,
            "nvidia_cuda_version": "",
            "nvidia_gpu_names": [],
            "nvidia_probe_error": "",
        }

    rc, out, err = _run_command([smi], timeout_sec=12)
    out_text = str(out or "")
    cuda_ver = ""
    m = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", out_text)
    if m:
        cuda_ver = str(m.group(1) or "").strip()

    names: List[str] = []
    rc_names, out_names, _ = _run_command([smi, "--query-gpu=name", "--format=csv,noheader"], timeout_sec=8)
    if rc_names == 0 and str(out_names or "").strip():
        for row in str(out_names or "").splitlines():
            candidate = str(row or "").strip()
            if candidate and candidate not in names:
                names.append(candidate)
    if not names:
        for line in out_text.splitlines():
            if "NVIDIA" not in line:
                continue
            if "NVIDIA-SMI" in line:
                continue
            line_clean = " ".join(line.replace("|", " ").split())
            n = re.search(r"(NVIDIA\s+[A-Za-z0-9\-\s]+?)(?:\s{2,}|$)", line_clean)
            if n:
                candidate = str(n.group(1) or "").strip()
                if candidate and candidate not in names:
                    names.append(candidate)

    gpu_present = bool((rc == 0 and ("NVIDIA" in out_text)) or names)
    return {
        "nvidia_smi_found": True,
        "nvidia_gpu_present": gpu_present,
        "nvidia_cuda_version": cuda_ver,
        "nvidia_gpu_names": names,
        "nvidia_probe_error": "" if rc == 0 else (str(err or out or "").strip()[:400]),
    }


def detect_torch_runtime() -> Dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception as exc:
        return {
            "torch_importable": False,
            "torch_version": "",
            "torch_cuda_version": "",
            "torch_cuda_available": False,
            "torch_error": str(exc),
        }

    cuda_ver = ""
    try:
        cuda_ver = str(getattr(getattr(torch, "version", None), "cuda", "") or "")
    except Exception:
        cuda_ver = ""

    cuda_available = False
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False

    return {
        "torch_importable": True,
        "torch_version": str(getattr(torch, "__version__", "") or ""),
        "torch_cuda_version": cuda_ver,
        "torch_cuda_available": cuda_available,
        "torch_error": "",
    }


def resolve_recommended_cuda_index(cuda_version: str = "") -> Dict[str, str]:
    parsed = _parse_version_tuple(cuda_version)
    if parsed is not None:
        if parsed >= (12, 8):
            return {"cuda_tag": "cu128", "index_url": "https://download.pytorch.org/whl/cu128"}
        if parsed >= (12, 1):
            return {"cuda_tag": "cu121", "index_url": "https://download.pytorch.org/whl/cu121"}
        if parsed >= (11, 8):
            return {"cuda_tag": "cu118", "index_url": "https://download.pytorch.org/whl/cu118"}
    return {"cuda_tag": "cu121", "index_url": "https://download.pytorch.org/whl/cu121"}


def build_recommended_torch_install_command(
    *, python_exe: str = "", cuda_version: str = ""
) -> List[str]:
    py = str(python_exe or sys.executable).strip() or sys.executable
    index = resolve_recommended_cuda_index(cuda_version)
    return [
        py,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "torch",
        "--index-url",
        index["index_url"],
    ]


def format_command_for_display(args: List[str]) -> str:
    cleaned = [str(a) for a in list(args or []) if str(a or "").strip()]
    if not cleaned:
        return ""
    try:
        return subprocess.list2cmdline(cleaned)
    except Exception:
        return " ".join(cleaned)


def collect_cuda_runtime_diagnosis(python_exe: str = "") -> Dict[str, Any]:
    nvidia = detect_nvidia_runtime()
    torch_info = detect_torch_runtime()
    need_cuda_runtime = bool(
        nvidia.get("nvidia_gpu_present")
        and (not torch_info.get("torch_importable") or not torch_info.get("torch_cuda_available"))
    )
    install_cmd = build_recommended_torch_install_command(
        python_exe=python_exe,
        cuda_version=str(nvidia.get("nvidia_cuda_version", "") or ""),
    )
    index_info = resolve_recommended_cuda_index(str(nvidia.get("nvidia_cuda_version", "") or ""))
    return {
        **nvidia,
        **torch_info,
        "need_cuda_runtime": need_cuda_runtime,
        "recommended_cuda_tag": str(index_info["cuda_tag"]),
        "recommended_index_url": str(index_info["index_url"]),
        "recommended_install_command": install_cmd,
        "recommended_install_command_text": format_command_for_display(install_cmd),
    }


def install_torch_cuda_runtime(
    *,
    python_exe: str = "",
    cuda_version: str = "",
    timeout_sec: int = 2400,
) -> Dict[str, Any]:
    cmd = build_recommended_torch_install_command(
        python_exe=python_exe,
        cuda_version=cuda_version,
    )
    rc, out, err = _run_command(cmd, timeout_sec=max(60, int(timeout_sec)))
    return {
        "command": cmd,
        "command_text": format_command_for_display(cmd),
        "returncode": rc,
        "success": rc == 0,
        "stdout_tail": str(out or "")[-1600:],
        "stderr_tail": str(err or "")[-1600:],
    }


def ensure_cuda_runtime_for_nvidia(
    *,
    auto_install: bool = False,
    python_exe: str = "",
    timeout_sec: int = 2400,
) -> Dict[str, Any]:
    before = collect_cuda_runtime_diagnosis(python_exe=python_exe)
    result: Dict[str, Any] = {
        "before": before,
        "after": before,
        "action": "none",
        "install_result": None,
        "success": True,
    }

    if not bool(before.get("nvidia_gpu_present")):
        result["action"] = "skip_no_nvidia"
        return result
    if bool(before.get("torch_cuda_available")):
        result["action"] = "already_ready"
        return result

    result["action"] = "needs_cuda_runtime"
    result["success"] = False
    if not auto_install:
        return result

    install_result = install_torch_cuda_runtime(
        python_exe=python_exe,
        cuda_version=str(before.get("nvidia_cuda_version", "") or ""),
        timeout_sec=timeout_sec,
    )
    after = collect_cuda_runtime_diagnosis(python_exe=python_exe)
    result.update(
        {
            "after": after,
            "install_result": install_result,
            "action": "auto_install_attempted",
            "success": bool(install_result.get("success")) and bool(after.get("torch_cuda_available")),
        }
    )
    return result


def _main() -> int:
    parser = argparse.ArgumentParser(description="NVIDIA/torch CUDA runtime checker and bootstrap helper.")
    parser.add_argument("--auto-install", action="store_true", help="Try torch CUDA runtime install when needed.")
    parser.add_argument("--python-exe", default="", help="Python executable path for pip install command.")
    parser.add_argument("--timeout-sec", type=int, default=2400, help="Install command timeout in seconds.")
    parser.add_argument("--json", action="store_true", help="Print full result as JSON.")
    parser.add_argument("--quiet", action="store_true", help="Suppress human-readable summary logs.")
    args = parser.parse_args()

    result = ensure_cuda_runtime_for_nvidia(
        auto_install=bool(args.auto_install),
        python_exe=str(args.python_exe or "").strip(),
        timeout_sec=max(60, int(args.timeout_sec)),
    )
    before = dict(result.get("before") or {})
    after = dict(result.get("after") or {})

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif not args.quiet:
        print("[CUDA-CHECK] NVIDIA:", bool(before.get("nvidia_gpu_present")))
        print("[CUDA-CHECK] Torch CUDA available (before):", bool(before.get("torch_cuda_available")))
        print("[CUDA-CHECK] Action:", str(result.get("action", "")))
        if bool(before.get("need_cuda_runtime")):
            print("[CUDA-CHECK] Recommended:", str(before.get("recommended_install_command_text", "")))
        if result.get("install_result"):
            install_result = dict(result.get("install_result") or {})
            print("[CUDA-CHECK] Install returncode:", int(install_result.get("returncode", -1)))
            print("[CUDA-CHECK] Torch CUDA available (after):", bool(after.get("torch_cuda_available")))

    if bool(result.get("success")):
        return 0
    if bool(before.get("need_cuda_runtime")) and not bool(args.auto_install):
        return 20
    if bool(result.get("install_result")):
        install_result = dict(result.get("install_result") or {})
        if not bool(install_result.get("success")):
            return 30
        return 31
    return 32


if __name__ == "__main__":
    raise SystemExit(_main())
