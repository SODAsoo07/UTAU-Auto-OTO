import argparse
import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from typing import Optional


APP_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_ASSET_DIR = os.path.join(APP_DIR, "build_assets")
FFMPEG_DIR = os.path.join(BUILD_ASSET_DIR, "ffmpeg")
FFMPEG_BIN_DIR = os.path.join(FFMPEG_DIR, "bin")
FFMPEG_RELEASE_ZIP_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
REQUIRED_FFMPEG_BINARIES = ("ffmpeg.exe", "ffprobe.exe")
# Nuitka often collects vcruntime140*.dll automatically; forcing them can cause duplicate conflicts.
# Keep explicit bundling focused on msvcp140.dll, which is commonly reported as missing.
REQUIRED_MSVC_RUNTIME_DLLS = ("msvcp140.dll",)
REQUIRED_MFA_ACOUSTIC_MODELS = ("korean_mfa", "japanese_mfa")
KOREAN_WHEEL_BUNDLE_DIRNAME = "mfa_ko_wheels"
KOREAN_WHEEL_PACKAGES = ("python-mecab-ko", "jamo", "python-mecab-ko-dic")
KOREAN_WHEEL_NAME_HINTS = ("python_mecab_ko-", "python_mecab_ko_dic-", "jamo-")
KOREAN_WHEEL_TARGET_PYTHON = "310"
MFA_LAUNCHER_FAILURE_MARKERS = (
    "failed to create process",
    "unable to create process",
    "fatal error in launcher",
    "no python at",
)

DEFAULT_APP_NAME = "UTAU_Auto_OTO"
DEFAULT_CHANNEL = "stable"
SUPPORTED_CHANNELS = ("stable", "preview")
CHANNEL_ALIASES = {"default": "stable"}
SUPPORTED_CHANNEL_INPUTS = ("stable", "preview", "default")

DEFAULT_BACKEND = "nuitka"
SUPPORTED_BACKENDS = ("nuitka", "pyinstaller")

RELEASE_DIR_PREFIX = "UTAU_Auto_OTO_Release"

EXCLUDED_MODULES = [
    "torch",
    "torchaudio",
    "torchvision",
    "ml",
]
EXCLUDED_TRAINING_MODULES = [
    "core.ja_oto_autotune",
    "core.oto_ml.coupled.training",
]
RUNTIME_DATA_BASE_PATHS = [
    (os.path.join(APP_DIR, "assets", "profiles"), "assets/profiles"),
    (os.path.join(APP_DIR, "ml", "configs"), "ml/configs"),
]
RELEASE_AUX_FILES = [
    os.path.join(APP_DIR, "setup_mfa.bat"),
    os.path.join(APP_DIR, "requirements.txt"),
    os.path.join(APP_DIR, "requirements-ml.txt"),
    os.path.join(APP_DIR, "release_assets", "먼저 실행.txt"),
    os.path.join(APP_DIR, "release_assets", "설치_도우미.bat"),
]
APP_ICON_CANDIDATES = [
    os.path.join(APP_DIR, "release_assets", "AutoOTO-icon.ico"),
    os.path.join(APP_DIR, "AutoOTO-icon.ico"),
]

args = argparse.Namespace(
    onefile=False,
    allow_unsafe_onefile=False,
    name=DEFAULT_APP_NAME,
    channel=DEFAULT_CHANNEL,
    channels="",
    backend=DEFAULT_BACKEND,
    skip_deps=False,
    bundle_mfa_runtime=False,
    skip_ml_model_bundle=False,
    mfa_runtime_root="",
    require_mfa_models=False,
)


def _configure_console_encoding():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _normalize_channel(channel: str) -> str:
    normalized = (channel or "").strip().lower()
    return CHANNEL_ALIASES.get(normalized, normalized)


def _parse_channels(channel: str, channels: str) -> list[str]:
    raw_values = []
    if str(channels or "").strip():
        raw_values = [part.strip() for part in str(channels).split(",")]
    elif str(channel or "").strip():
        raw_values = [str(channel).strip()]
    resolved = []
    seen = set()
    for raw in raw_values:
        if not raw:
            continue
        normalized = _normalize_channel(raw)
        if normalized not in SUPPORTED_CHANNELS:
            raise SystemExit(
                f"Invalid channel: {raw} (supported: {', '.join(SUPPORTED_CHANNEL_INPUTS)})"
            )
        if normalized in seen:
            continue
        seen.add(normalized)
        resolved.append(normalized)
    if not resolved:
        resolved.append(DEFAULT_CHANNEL)
    return resolved


def _ensure_ffmpeg_bin():
    ffmpeg_exe = os.path.join(FFMPEG_BIN_DIR, "ffmpeg.exe")
    ffprobe_exe = os.path.join(FFMPEG_BIN_DIR, "ffprobe.exe")
    if os.path.exists(ffmpeg_exe) and os.path.exists(ffprobe_exe):
        _validate_ffmpeg_bin(FFMPEG_BIN_DIR)
        print(f"FFmpeg reuse: {FFMPEG_BIN_DIR}")
        return FFMPEG_BIN_DIR

    os.makedirs(BUILD_ASSET_DIR, exist_ok=True)
    tmp_zip = os.path.join(BUILD_ASSET_DIR, "ffmpeg_release_essentials.zip")
    tmp_extract = tempfile.mkdtemp(prefix="ffmpeg_extract_", dir=BUILD_ASSET_DIR)
    try:
        print("Downloading FFmpeg (Windows shared build)...")
        with urllib.request.urlopen(FFMPEG_RELEASE_ZIP_URL, timeout=180) as resp:
            with open(tmp_zip, "wb") as f:
                f.write(resp.read())

        print("Extracting FFmpeg archive...")
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            zf.extractall(tmp_extract)

        source_bin = ""
        for root, dirs, _ in os.walk(tmp_extract):
            if "bin" in dirs and os.path.exists(os.path.join(root, "bin", "ffmpeg.exe")):
                source_bin = os.path.join(root, "bin")
                break
        if not source_bin:
            raise RuntimeError("ffmpeg.exe was not found in archive.")

        if os.path.exists(FFMPEG_DIR):
            shutil.rmtree(FFMPEG_DIR)
        os.makedirs(FFMPEG_DIR, exist_ok=True)
        shutil.copytree(source_bin, FFMPEG_BIN_DIR)
        print(f"FFmpeg prepared: {FFMPEG_BIN_DIR}")
        _validate_ffmpeg_bin(FFMPEG_BIN_DIR)
        return FFMPEG_BIN_DIR
    finally:
        if os.path.exists(tmp_zip):
            try:
                os.remove(tmp_zip)
            except OSError:
                pass
        shutil.rmtree(tmp_extract, ignore_errors=True)


def _validate_ffmpeg_bin(ffmpeg_bin):
    missing = []
    for file_name in REQUIRED_FFMPEG_BINARIES:
        full_path = os.path.join(ffmpeg_bin, file_name)
        if (not os.path.isfile(full_path)) or os.path.getsize(full_path) <= 0:
            missing.append(full_path)
    if missing:
        lines = "\n".join(f"  - {p}" for p in missing)
        raise RuntimeError(f"Required FFmpeg runtime files are missing:\n{lines}")


def _iter_ffmpeg_runtime_files(ffmpeg_bin):
    if not os.path.isdir(ffmpeg_bin):
        return []
    files = []
    for name in sorted(os.listdir(ffmpeg_bin)):
        src = os.path.join(ffmpeg_bin, name)
        if not os.path.isfile(src):
            continue
        low = name.lower()
        if low.endswith(".exe") or low.endswith(".dll"):
            files.append((src, name))
    return files


def _resolve_msvc_runtime_dlls() -> list[str]:
    # Search common system and VC redist locations without expensive full recursion.
    search_dirs: list[str] = []
    env_candidates = [
        os.environ.get("SystemRoot", ""),
        os.environ.get("windir", ""),
        os.environ.get("VCToolsRedistDir", ""),
        os.environ.get("VCINSTALLDIR", ""),
        os.environ.get("VSINSTALLDIR", ""),
    ]
    for base in env_candidates:
        b = str(base or "").strip()
        if not b:
            continue
        search_dirs.extend(
            [
                b,
                os.path.join(b, "System32"),
                os.path.join(b, "SysWOW64"),
            ]
        )
        search_dirs.extend(
            glob.glob(os.path.join(b, "VC", "Redist", "MSVC", "*", "x64", "Microsoft.VC*.CRT"))
        )
        search_dirs.extend(
            glob.glob(os.path.join(b, "Redist", "MSVC", "*", "x64", "Microsoft.VC*.CRT"))
        )

    # Keep order and remove duplicates.
    dedup_dirs = []
    seen_dirs = set()
    for d in search_dirs:
        norm = os.path.normcase(os.path.abspath(d))
        if norm in seen_dirs:
            continue
        seen_dirs.add(norm)
        dedup_dirs.append(d)

    found = {}
    for dll_name in REQUIRED_MSVC_RUNTIME_DLLS:
        for d in dedup_dirs:
            direct = os.path.join(d, dll_name)
            if os.path.isfile(direct):
                found[dll_name.lower()] = direct
                break

    missing = [name for name in REQUIRED_MSVC_RUNTIME_DLLS if name.lower() not in found]
    if missing:
        print(
            "[WARN] Some VC++ runtime DLLs were not found for bundling: "
            + ", ".join(missing)
            + " (target PC may require VC++ Redistributable 2015-2022 x64)."
        )
    return [found[k.lower()] for k in REQUIRED_MSVC_RUNTIME_DLLS if k.lower() in found]


def _parse_args():
    parser = argparse.ArgumentParser(description="Build UTAU Auto OTO distributables.")
    parser.add_argument("--onefile", action="store_true", help="Build onefile executable (unsafe/experimental).")
    parser.add_argument(
        "--allow-unsafe-onefile",
        action="store_true",
        help="Acknowledge onefile runtime risks and allow --onefile build.",
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_APP_NAME,
        help="Executable name.",
    )
    parser.add_argument(
        "--channel",
        default=DEFAULT_CHANNEL,
        choices=SUPPORTED_CHANNEL_INPUTS,
        help="Release channel tag (stable/default or preview).",
    )
    parser.add_argument(
        "--channels",
        default="",
        help="Comma-separated release channels to package in one build (e.g. stable,preview).",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=SUPPORTED_BACKENDS,
        help="Build backend (nuitka or pyinstaller).",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip dependency installation step (useful in CI when already installed).",
    )
    parser.add_argument(
        "--bundle-mfa-runtime",
        action="store_true",
        help="Bundle a preinstalled local MFA runtime into the release payload for offline first-run setup.",
    )
    parser.add_argument(
        "--skip-ml-model-bundle",
        action="store_true",
        help="Do not bundle runtime ML model folders into the release payload.",
    )
    parser.add_argument(
        "--mfa-runtime-root",
        default="",
        help="Path to MFA runtime root (default: %%LOCALAPPDATA%%\\UTAU_Auto_OTO_v3 when --bundle-mfa-runtime is set).",
    )
    parser.add_argument(
        "--require-mfa-models",
        action="store_true",
        help="Require bundled MFA runtime to already contain korean_mfa and japanese_mfa acoustic models.",
    )
    return parser.parse_args()


def _detect_app_version():
    main_path = os.path.join(APP_DIR, "main.py")
    if not os.path.exists(main_path):
        return "0.0.0"
    try:
        with open(main_path, "r", encoding="utf-8") as f:
            text = f.read()
        m = re.search(r'APP_VERSION\s*=\s*"([^"]+)"', text)
        if m:
            return m.group(1).strip()
    except OSError:
        pass
    return "0.0.0"


def _write_release_channel_metadata(target_path, app_name, app_version, channel):
    payload = {
        "app_name": app_name,
        "app_version": app_version,
        "channel": channel,
        "built_at_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _get_release_dir(channel):
    return os.path.join(APP_DIR, f"{RELEASE_DIR_PREFIX}_{channel}")


def _is_valid_mfa_runtime_root(root_path: str) -> bool:
    env_dir = os.path.join(root_path, ".env")
    return bool(_find_mfa_python(env_dir) and _find_mfa_entrypoint(env_dir))


def _is_valid_mfa_env_dir(path: str) -> bool:
    return bool(_find_mfa_python(path) and _find_mfa_entrypoint(path))


def _build_mfa_runtime_candidates(runtime_root: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(path: str) -> None:
        p = str(path or "").strip()
        if not p:
            return
        abs_path = os.path.abspath(p)
        key = abs_path.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(abs_path)

    explicit = str(runtime_root or "").strip()
    if explicit:
        explicit_abs = os.path.abspath(explicit)
        _add(explicit_abs)
        _add(os.path.join(explicit_abs, ".env"))
        return candidates

    local_app_data = str(os.environ.get("LOCALAPPDATA", "") or "").strip()
    default_root = (
        os.path.join(local_app_data, "UTAU_Auto_OTO_v3")
        if local_app_data
        else os.path.join(os.path.expanduser("~"), "AppData", "Local", "UTAU_Auto_OTO_v3")
    )

    # Auto fallback order (when --mfa-runtime-root is not explicitly passed):
    # 1) shared runtime root (%LOCALAPPDATA%\UTAU_Auto_OTO_v3)
    # 2) source checkout local env (./.env)
    # 3) pre-bundled runtime layout (./mfa_runtime_bundle)
    _add(default_root)
    _add(os.path.join(default_root, ".env"))
    _add(os.path.join(APP_DIR, ".env"))
    _add(os.path.join(APP_DIR, ".venv"))
    _add(os.path.join(APP_DIR, "mfa_runtime_bundle"))
    _add(os.path.join(APP_DIR, "mfa_runtime_bundle", ".env"))
    return candidates


def _resolve_mfa_runtime_root(runtime_root: str) -> str:
    candidates = _build_mfa_runtime_candidates(runtime_root)

    for candidate in candidates:
        if not os.path.isdir(candidate):
            continue
        if _is_valid_mfa_runtime_root(candidate) or _is_valid_mfa_env_dir(candidate):
            return candidate

    # No healthy candidate found: return the first existing path for clearer validation error.
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    # Last fallback: keep previous default semantics.
    if candidates:
        return candidates[0]
    return os.path.join(os.path.expanduser("~"), "AppData", "Local", "UTAU_Auto_OTO_v3")


def _find_mfa_entrypoint(env_dir: str) -> str:
    candidates = [
        os.path.join(env_dir, "Scripts", "mfa.exe"),
        os.path.join(env_dir, "Scripts", "mfa.bat"),
        os.path.join(env_dir, "Scripts", "mfa.cmd"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return ""


def _find_mfa_python(env_dir: str) -> str:
    candidates = [
        os.path.join(env_dir, "python.exe"),
        os.path.join(env_dir, "Scripts", "python.exe"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return ""


def _mfa_runtime_subprocess_env(env_dir: str) -> dict:
    env = os.environ.copy()
    env_prefixes = [
        env_dir,
        os.path.join(env_dir, "Library", "mingw-w64", "bin"),
        os.path.join(env_dir, "Library", "usr", "bin"),
        os.path.join(env_dir, "Library", "bin"),
        os.path.join(env_dir, "Scripts"),
        os.path.join(env_dir, "bin"),
    ]
    env["CONDA_PREFIX"] = env_dir
    env["PATH"] = os.pathsep.join(env_prefixes + [env.get("PATH", "")])
    return env


def _check_mfa_python_health(env_dir: str, python_exe: str) -> None:
    commands = [
        [python_exe, "-m", "montreal_forced_aligner.command_line.mfa", "--help"],
        [python_exe, "-m", "montreal_forced_aligner", "--help"],
    ]
    env = _mfa_runtime_subprocess_env(env_dir)
    failure_notes = []
    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=25,
                check=False,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as exc:
            failure_notes.append(f"{' '.join(cmd)} -> {exc}")
            continue
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        lowered = combined.lower()
        if result.returncode == 0 and not any(marker in lowered for marker in MFA_LAUNCHER_FAILURE_MARKERS):
            return
        snippet = "\n".join(line for line in combined.splitlines()[:12] if line.strip())
        failure_notes.append(
            f"{' '.join(cmd)} -> exit={result.returncode}\n{snippet}"
        )
    details = "\n\n".join(failure_notes) if failure_notes else "unknown error"
    raise RuntimeError(
        "MFA runtime python health check failed. "
        "Run setup_mfa.bat on the build machine and retry.\n\n"
        f"{details}"
    )


def _collect_mfa_acoustic_models(runtime_root_abs: str, env_dir: str = "") -> tuple[dict, list[str]]:
    root_candidates = [
        os.path.join(runtime_root_abs, ".mfa_root_ascii"),
        os.path.join(runtime_root_abs, ".mfa_root"),
        os.path.join(runtime_root_abs, ".mfa"),
    ]
    if env_dir:
        root_candidates.extend(
            [
                os.path.join(env_dir, ".mfa_root_ascii"),
                os.path.join(env_dir, ".mfa_root"),
                os.path.join(env_dir, ".mfa"),
            ]
        )
    existing_roots = [path for path in root_candidates if os.path.isdir(path)]
    status = {}
    for model_name in REQUIRED_MFA_ACOUSTIC_MODELS:
        found = ""
        for root in existing_roots:
            patterns = [
                os.path.join(root, "acoustic", f"{model_name}*"),
                os.path.join(root, "extracted_models", "acoustic", f"{model_name}*"),
            ]
            hits = []
            for pattern in patterns:
                hits.extend([p for p in glob.glob(pattern) if os.path.exists(p)])
            if hits:
                found = os.path.abspath(hits[0])
                break
        status[model_name] = found
    return status, existing_roots


def _validate_mfa_runtime_bundle_source(
    runtime_root: str,
    require_models: bool = False,
) -> tuple[str, str, str, dict, list[str]]:
    runtime_root_abs = os.path.abspath(runtime_root)
    env_dir = os.path.join(runtime_root_abs, ".env")
    # Accept both layouts:
    # 1) runtime root containing .env
    # 2) direct env dir path itself
    if not os.path.isdir(env_dir):
        if _find_mfa_python(runtime_root_abs) and _find_mfa_entrypoint(runtime_root_abs):
            env_dir = runtime_root_abs
            runtime_root_abs = os.path.dirname(runtime_root_abs)

    python_exe = _find_mfa_python(env_dir)
    mfa_entry = _find_mfa_entrypoint(env_dir)
    if not os.path.isdir(env_dir):
        raise FileNotFoundError(
            "MFA runtime env directory was not found. "
            f"Expected: {os.path.join(runtime_root_abs, '.env')} "
            f"or direct env dir: {os.path.abspath(runtime_root)}"
        )
    if not python_exe:
        raise FileNotFoundError(
            "MFA runtime python was not found under: "
            f"{os.path.join(env_dir, 'python.exe')} or {os.path.join(env_dir, 'Scripts', 'python.exe')}"
        )
    if not mfa_entry:
        raise FileNotFoundError(
            f"MFA entrypoint was not found under: {os.path.join(env_dir, 'Scripts')}"
        )
    _check_mfa_python_health(env_dir, python_exe)

    model_status, model_roots = _collect_mfa_acoustic_models(runtime_root_abs, env_dir=env_dir)
    if require_models:
        missing = [name for name, path in model_status.items() if not path]
        if missing:
            roots_text = ", ".join(model_roots) if model_roots else "(no model root folder found)"
            raise FileNotFoundError(
                "Required MFA acoustic models are missing from runtime root.\n"
                f"runtime_root={runtime_root_abs}\n"
                f"missing={', '.join(missing)}\n"
                f"searched_roots={roots_text}"
            )
    return runtime_root_abs, env_dir, mfa_entry, model_status, model_roots


def _resolve_validated_mfa_runtime_bundle_source(
    runtime_root: str,
    *,
    require_models: bool = False,
) -> tuple[str, str, str, dict, list[str]]:
    explicit = str(runtime_root or "").strip()
    candidates = _build_mfa_runtime_candidates(runtime_root)
    if explicit:
        # For explicit paths, preserve deterministic behavior and fail fast on that source.
        return _validate_mfa_runtime_bundle_source(
            candidates[0] if candidates else runtime_root,
            require_models=require_models,
        )

    issues: list[str] = []
    for candidate in candidates:
        if not os.path.isdir(candidate):
            issues.append(f"{candidate} -> path not found")
            continue
        try:
            return _validate_mfa_runtime_bundle_source(candidate, require_models=require_models)
        except Exception as exc:
            first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            issues.append(f"{candidate} -> {first_line}")

    detail_text = "\n".join(f"  - {line}" for line in issues) if issues else "  - no candidates were generated"
    raise FileNotFoundError(
        "Unable to find a valid MFA runtime source for bundling.\n"
        "Checked candidates:\n"
        f"{detail_text}\n"
        "Run setup_mfa.bat once on this build machine or pass --mfa-runtime-root explicitly."
    )


def _copy_mfa_runtime_bundle(
    release_dir: str,
    runtime_root: str,
    *,
    require_models: bool = False,
) -> Optional[str]:
    runtime_root_abs, env_dir, _mfa_entry, model_status, model_roots = _resolve_validated_mfa_runtime_bundle_source(
        runtime_root,
        require_models=require_models,
    )
    bundle_dst = os.path.join(release_dir, "mfa_runtime_bundle")
    if os.path.exists(bundle_dst):
        shutil.rmtree(bundle_dst)
    os.makedirs(bundle_dst, exist_ok=True)

    ignore = shutil.ignore_patterns("logs", "__pycache__", "*.log", "*.tmp", "*.temp", "*.pid")

    # Always stage env into bundle/.env so setup_mfa.bat can restore with a fixed structure.
    shutil.copytree(env_dir, os.path.join(bundle_dst, ".env"), dirs_exist_ok=True, ignore=ignore)

    # Optional runtime manager payload.
    micromamba_src = os.path.join(runtime_root_abs, "micromamba")
    if os.path.isdir(micromamba_src):
        shutil.copytree(micromamba_src, os.path.join(bundle_dst, "micromamba"), dirs_exist_ok=True, ignore=ignore)

    # Copy discovered MFA model roots into bundle root (.mfa_root_ascii / .mfa_root / .mfa).
    copied_model_root_names = set()
    for root in model_roots:
        base_name = os.path.basename(os.path.abspath(root))
        if not base_name:
            continue
        dst = os.path.join(bundle_dst, base_name)
        shutil.copytree(root, dst, dirs_exist_ok=True, ignore=ignore)
        copied_model_root_names.add(base_name)

    copied_env_dir = os.path.join(bundle_dst, ".env")
    copied_mfa_entry = _find_mfa_entrypoint(copied_env_dir)
    copied_python = _find_mfa_python(copied_env_dir)
    if (not copied_python) or (not copied_mfa_entry):
        raise RuntimeError(
            "Copied MFA runtime bundle is invalid (missing python.exe or MFA entrypoint)."
        )

    manifest = {
        "source_runtime_root": runtime_root_abs,
        "source_env_dir": env_dir,
        "copied_at_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "mfa_entrypoint": os.path.relpath(copied_mfa_entry, bundle_dst).replace("\\", "/"),
        "acoustic_models": model_status,
        "copied_model_roots": sorted(copied_model_root_names),
        "require_models": bool(require_models),
    }
    with open(os.path.join(bundle_dst, "bundle_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return bundle_dst


def _iter_runtime_data_entries():
    entries = []
    for src, dst in RUNTIME_DATA_BASE_PATHS:
        if os.path.exists(src):
            entries.append((src, dst))
        else:
            print(f"[WARN] Runtime data missing (skip): {src}")

    def _latest_model_meta_mtime(model_root: str) -> float:
        newest = 0.0
        try:
            for root, _dirs, files in os.walk(model_root):
                if "model_meta.json" not in files:
                    continue
                meta_path = os.path.join(root, "model_meta.json")
                try:
                    newest = max(newest, os.path.getmtime(meta_path))
                except OSError:
                    continue
        except Exception:
            newest = 0.0
        if newest > 0.0:
            return newest
        try:
            return float(os.path.getmtime(model_root))
        except OSError:
            return 0.0

    if bool(getattr(args, "skip_ml_model_bundle", False)):
        print("[INFO] Runtime ML model bundle step skipped (--skip-ml-model-bundle).")
        return entries

    model_override = str(os.environ.get("UTOA_BUILD_MODEL_DIR", "") or "").strip()
    model_candidates = []
    if model_override:
        if os.path.isabs(model_override):
            model_candidates.append(model_override)
        else:
            model_candidates.append(os.path.join(APP_DIR, model_override))
    channel_norm = _normalize_channel(getattr(args, "channel", DEFAULT_CHANNEL))
    model_candidates.extend(
        [
            os.path.join(APP_DIR, "ML_models_latest"),
            os.path.join(APP_DIR, f"ML_models_{channel_norm}"),
            os.path.join(APP_DIR, "ML_models"),
        ]
    )

    selected_model_source = ""
    existing_candidates = [(idx, path) for idx, path in enumerate(model_candidates) if os.path.isdir(path)]
    if existing_candidates:
        if model_override:
            selected_model_source = existing_candidates[0][1]
        else:
            ranked = [
                (_latest_model_meta_mtime(path), -idx, path)
                for idx, path in existing_candidates
            ]
            ranked.sort(reverse=True)
            selected_model_source = ranked[0][2]

    if selected_model_source:
        entries.append((selected_model_source, "ML_models"))
        print(f"[INFO] Runtime ML models source: {selected_model_source} -> ML_models")
    else:
        print("[WARN] Runtime ML model directory not found (ML_models*).")

    legacy_asset_models = os.path.join(APP_DIR, "assets", "models", "oto_ml")
    if os.path.isdir(legacy_asset_models):
        entries.append((legacy_asset_models, "assets/models/oto_ml"))
    else:
        print(f"[WARN] Runtime data missing (skip): {legacy_asset_models}")
    return entries


def _resolve_app_icon_path():
    for candidate in APP_ICON_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return ""


def _build_pyinstaller_args(app_name, ffmpeg_bin, app_icon_path="", onefile=False):
    import customtkinter

    ctk_path = os.path.dirname(customtkinter.__file__)
    pyinstaller_args = [
        "main.py",
        f"--name={app_name}",
        "--windowed",
        "--noconfirm",
        "--clean",
        "--onefile" if onefile else "--onedir",
        f"--add-data={ctk_path};customtkinter/",
        f"--add-data={ffmpeg_bin};ffmpeg/bin",
        "--hidden-import=textgrid",
        "--hidden-import=customtkinter",
    ]
    for src, dst in _iter_runtime_data_entries():
        pyinstaller_args.append(f"--add-data={src};{dst}")
    if app_icon_path:
        pyinstaller_args.append(f"--icon={app_icon_path}")
    for module_name in EXCLUDED_MODULES + EXCLUDED_TRAINING_MODULES:
        pyinstaller_args.append(f"--exclude-module={module_name}")
    return pyinstaller_args


def _run_pyinstaller_build(app_name, ffmpeg_bin, app_icon_path="", onefile=False):
    print("Loading PyInstaller...")
    import PyInstaller.__main__

    pyinstaller_args = _build_pyinstaller_args(
        app_name=app_name,
        ffmpeg_bin=ffmpeg_bin,
        app_icon_path=app_icon_path,
        onefile=onefile,
    )
    PyInstaller.__main__.run(pyinstaller_args)

    if onefile:
        exe_path = os.path.join(APP_DIR, "dist", f"{app_name}.exe")
        if not os.path.isfile(exe_path):
            raise FileNotFoundError(f"Built executable not found: {exe_path}")
        return exe_path

    dist_dir = os.path.join(APP_DIR, "dist", app_name)
    if not os.path.isdir(dist_dir):
        raise FileNotFoundError(f"Built app directory not found: {dist_dir}")
    return dist_dir


def _run_nuitka_build(app_name, ffmpeg_bin, app_icon_path="", onefile=False):
    import customtkinter

    ctk_path = os.path.dirname(customtkinter.__file__)
    output_root = os.path.join(APP_DIR, "dist_nuitka")
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)

    env_jobs = str(os.environ.get("UTOA_NUITKA_JOBS", "") or "").strip()
    cpu_jobs = os.cpu_count() or 1
    if env_jobs:
        try:
            cpu_jobs = int(env_jobs)
        except ValueError:
            print(f"[WARN] Invalid UTOA_NUITKA_JOBS={env_jobs!r}; fallback to auto.")
    cpu_jobs = max(1, int(cpu_jobs))
    cmd = [
        sys.executable,
        "-m",
        "nuitka",
        "main.py",
        "--onefile" if onefile else "--standalone",
        "--assume-yes-for-downloads",
        "--enable-plugin=tk-inter",
        "--windows-console-mode=disable",
        f"--output-dir={output_root}",
        f"--output-filename={app_name}.exe",
        "--include-module=textgrid",
        "--include-package=customtkinter",
        "--remove-output",
        f"--jobs={cpu_jobs}",
        f"--nofollow-import-to={','.join(EXCLUDED_MODULES + EXCLUDED_TRAINING_MODULES)}",
    ]

    _validate_ffmpeg_bin(ffmpeg_bin)
    include_entries = [(ctk_path, "customtkinter")]
    include_entries.extend(_iter_runtime_data_entries())
    for src, dst in include_entries:
        if os.path.isdir(src):
            cmd.append(f"--include-data-dir={src}={dst}")
        else:
            dst_norm = str(dst or "").replace("\\", "/").strip()
            if dst_norm in {"", "."}:
                target_path = os.path.basename(src)
            else:
                target_path = f"{dst_norm.rstrip('/')}/{os.path.basename(src)}"
            cmd.append(f"--include-data-files={src}={target_path}")
    ffmpeg_runtime_files = _iter_ffmpeg_runtime_files(ffmpeg_bin)
    for src, name in ffmpeg_runtime_files:
        cmd.append(f"--include-data-files={src}=ffmpeg/bin/{name}")

    msvc_runtime_dlls = _resolve_msvc_runtime_dlls()
    for dll_path in msvc_runtime_dlls:
        dll_name = os.path.basename(dll_path)
        cmd.append(f"--include-data-files={dll_path}={dll_name}")

    if app_icon_path:
        cmd.append(f"--windows-icon-from-ico={app_icon_path}")

    print("Running Nuitka build command:")
    print(" ".join(cmd))
    subprocess.check_call(cmd)

    if onefile:
        expected = os.path.join(output_root, f"{app_name}.exe")
        if os.path.isfile(expected):
            return expected
        exe_candidates = [
            os.path.join(output_root, n)
            for n in os.listdir(output_root)
            if n.lower().endswith(".exe")
        ]
        if len(exe_candidates) == 1:
            return exe_candidates[0]
        raise FileNotFoundError(f"Nuitka onefile executable not found in: {output_root}")

    expected_dist = os.path.join(output_root, f"{app_name}.dist")
    if os.path.isdir(expected_dist):
        _validate_packaged_ffmpeg(expected_dist)
        return expected_dist

    dist_candidates = [
        os.path.join(output_root, n)
        for n in os.listdir(output_root)
        if n.lower().endswith(".dist") and os.path.isdir(os.path.join(output_root, n))
    ]
    if len(dist_candidates) == 1:
        _validate_packaged_ffmpeg(dist_candidates[0])
        return dist_candidates[0]

    raise FileNotFoundError(f"Nuitka standalone directory not found in: {output_root}")


def _validate_packaged_ffmpeg(dist_dir):
    ffmpeg_bin = os.path.join(dist_dir, "ffmpeg", "bin")
    missing = []
    for file_name in REQUIRED_FFMPEG_BINARIES:
        full_path = os.path.join(ffmpeg_bin, file_name)
        if (not os.path.isfile(full_path)) or os.path.getsize(full_path) <= 0:
            missing.append(full_path)
    if missing:
        lines = "\n".join(f"  - {p}" for p in missing)
        raise RuntimeError(f"Nuitka output is missing required FFmpeg files:\n{lines}")


def _collect_korean_wheels(wheel_dir: str) -> list[str]:
    if not os.path.isdir(wheel_dir):
        return []
    wheels = [
        os.path.join(wheel_dir, name)
        for name in sorted(os.listdir(wheel_dir))
        if name.lower().endswith(".whl") and os.path.isfile(os.path.join(wheel_dir, name))
    ]
    return wheels


def _has_required_korean_wheels(wheel_dir: str) -> bool:
    wheels = _collect_korean_wheels(wheel_dir)
    if not wheels:
        return False
    names = [os.path.basename(path).lower() for path in wheels]
    if not all(any(hint in name for name in names) for hint in KOREAN_WHEEL_NAME_HINTS):
        return False
    mecab_wheels = [name for name in names if name.startswith("python_mecab_ko-")]
    if not mecab_wheels:
        return False
    return any(f"cp{KOREAN_WHEEL_TARGET_PYTHON}" in name for name in mecab_wheels)


def _prepare_korean_wheel_bundle() -> str:
    preferred_dir = os.path.join(APP_DIR, KOREAN_WHEEL_BUNDLE_DIRNAME)
    if _has_required_korean_wheels(preferred_dir):
        return preferred_dir

    wheel_dir = os.path.join(BUILD_ASSET_DIR, KOREAN_WHEEL_BUNDLE_DIRNAME)
    if _has_required_korean_wheels(wheel_dir):
        return wheel_dir

    os.makedirs(wheel_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(wheel_dir, "*.whl")):
        try:
            os.remove(stale)
        except OSError:
            pass

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--platform",
        "win_amd64",
        "--implementation",
        "cp",
        "--python-version",
        KOREAN_WHEEL_TARGET_PYTHON,
        "--only-binary=:all:",
        "--dest",
        wheel_dir,
        *KOREAN_WHEEL_PACKAGES,
    ]
    print("Preparing Korean tokenizer offline wheel bundle...")
    print(" ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        tail = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
        tail_lines = "\n".join(line for line in tail.splitlines()[-8:] if line.strip())
        print("[WARN] Failed to download Korean wheel bundle; installer will use online fallback.")
        if tail_lines:
            print(tail_lines)
        return ""
    if not _has_required_korean_wheels(wheel_dir):
        print(
            "[WARN] Korean wheel bundle download completed but required wheels are missing. "
            "Installer will use online fallback."
        )
        return ""
    return wheel_dir


def _copy_korean_wheel_bundle(release_dir: str) -> Optional[str]:
    source_dir = _prepare_korean_wheel_bundle()
    if not source_dir:
        return None
    wheels = _collect_korean_wheels(source_dir)
    if not wheels:
        return None

    dst_dir = os.path.join(release_dir, KOREAN_WHEEL_BUNDLE_DIRNAME)
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

    total_size = 0
    for wheel_path in wheels:
        dst_path = os.path.join(dst_dir, os.path.basename(wheel_path))
        shutil.copy2(wheel_path, dst_path)
        try:
            total_size += os.path.getsize(dst_path)
        except OSError:
            pass
    print(
        "   -> copied Korean offline wheels: "
        f"{len(wheels)} files ({total_size / (1024 * 1024):.1f} MiB)"
    )
    return dst_dir


def _copy_release_outputs(
    app_name,
    channel,
    app_version,
    built_artifact_path,
    onefile=False,
    bundle_mfa_runtime=False,
    mfa_runtime_root="",
    require_mfa_models=False,
):
    release_dir = _get_release_dir(channel)
    if os.path.exists(release_dir):
        shutil.rmtree(release_dir)
    os.makedirs(release_dir, exist_ok=True)

    _write_release_channel_metadata(
        os.path.join(release_dir, "release_channel.json"),
        app_name=app_name,
        app_version=app_version,
        channel=channel,
    )

    if onefile:
        if not os.path.isfile(built_artifact_path):
            raise FileNotFoundError(f"Built executable not found: {built_artifact_path}")
        shutil.copy(built_artifact_path, release_dir)
        print(f"   -> copied: {built_artifact_path}")
    else:
        if not os.path.isdir(built_artifact_path):
            raise FileNotFoundError(f"Built app directory not found: {built_artifact_path}")
        target_dir = os.path.join(release_dir, app_name)
        shutil.copytree(built_artifact_path, target_dir)
        print(f"   -> copied directory: {built_artifact_path}")
        _write_release_channel_metadata(
            os.path.join(target_dir, "release_channel.json"),
            app_name=app_name,
            app_version=app_version,
            channel=channel,
        )

    for extra_path in RELEASE_AUX_FILES:
        if os.path.exists(extra_path):
            shutil.copy(extra_path, release_dir)
            print(f"   -> copied: {os.path.basename(extra_path)}")
    _copy_korean_wheel_bundle(release_dir)
    if bundle_mfa_runtime:
        resolved_runtime_root = _resolve_mfa_runtime_root(mfa_runtime_root)
        print(f"   -> bundling MFA runtime from: {resolved_runtime_root}")
        bundle_dir = _copy_mfa_runtime_bundle(
            release_dir,
            resolved_runtime_root,
            require_models=require_mfa_models,
        )
        if bundle_dir:
            print(f"   -> bundled MFA runtime: {bundle_dir}")
    return release_dir


def _install_build_dependencies(backend):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    req_candidates = [
        os.path.join(APP_DIR, "requirements.txt"),
        os.path.join(APP_DIR, "requirements-ml.txt"),
    ]
    for req_path in req_candidates:
        if os.path.exists(req_path):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])

    if backend == "nuitka":
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nuitka", "ordered-set", "zstandard"])
    elif backend == "pyinstaller":
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])


def main():
    _configure_console_encoding()
    global args
    parsed_args = _parse_args()
    if parsed_args is not None:
        args = parsed_args

    target_channels = _parse_channels(args.channel, args.channels)
    args.channel = target_channels[0]

    if args.onefile and not args.allow_unsafe_onefile:
        raise SystemExit(
            "onefile builds are disabled by default. Use --allow-unsafe-onefile to proceed."
        )
    if args.allow_unsafe_onefile and not args.onefile:
        print("[INFO] --allow-unsafe-onefile is ignored when --onefile is not used.")
    if args.require_mfa_models and not args.bundle_mfa_runtime:
        print("[WARN] --require-mfa-models is ignored unless --bundle-mfa-runtime is set.")

    os.chdir(APP_DIR)
    app_version = _detect_app_version()
    mode_text = "onefile" if args.onefile else "onedir"

    channel_text = ",".join(target_channels)
    print(f"[INFO] channels={channel_text}, backend={args.backend}, version={app_version}")
    if args.bundle_mfa_runtime:
        resolved_runtime_root = _resolve_mfa_runtime_root(args.mfa_runtime_root)
        print(f"[INFO] bundle_mfa_runtime=1, runtime_root={resolved_runtime_root}")
        print(f"[INFO] require_mfa_models={1 if args.require_mfa_models else 0}")
    else:
        print("[INFO] bundle_mfa_runtime=0")
    if args.skip_deps:
        print("[1/5] Skipping dependency install (--skip-deps).")
    else:
        print("[1/5] Installing build dependencies...")
        _install_build_dependencies(args.backend)

    print("[2/5] Preparing FFmpeg runtime...")
    ffmpeg_bin = _ensure_ffmpeg_bin()
    app_icon_path = _resolve_app_icon_path()
    if app_icon_path:
        print(f"[INFO] app_icon={app_icon_path}")
    else:
        print("[WARN] app icon not found; executable will use default icon.")

    print(f"[3/5] Building app with {args.backend}...")
    if args.backend == "nuitka":
        built_artifact = _run_nuitka_build(args.name, ffmpeg_bin, app_icon_path=app_icon_path, onefile=args.onefile)
    else:
        built_artifact = _run_pyinstaller_build(args.name, ffmpeg_bin, app_icon_path=app_icon_path, onefile=args.onefile)

    print(f"[4/5] Packaging release folder ({mode_text})...")
    release_dirs = []
    total_channels = len(target_channels)
    for idx, channel in enumerate(target_channels, start=1):
        print(f"[4/5] Packaging channel {idx}/{total_channels}: {channel}")
        release_dir = _copy_release_outputs(
            args.name,
            channel=channel,
            app_version=app_version,
            built_artifact_path=built_artifact,
            onefile=args.onefile,
            bundle_mfa_runtime=args.bundle_mfa_runtime,
            mfa_runtime_root=args.mfa_runtime_root,
            require_mfa_models=args.require_mfa_models,
        )
        release_dirs.append(release_dir)

    print("[5/5] Build complete.")
    for release_dir in release_dirs:
        print(f"[DONE] release_dir={release_dir}")


if __name__ == "__main__":
    main()
