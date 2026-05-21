from __future__ import annotations


def resolve_torch_device(torch, requested: str):
    text = str(requested or "auto").strip().lower()
    if text in {"auto", "cuda_auto", "gpu"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA was requested ({requested}) but this PyTorch build cannot use CUDA. "
                f"torch={getattr(torch, '__version__', 'unknown')}"
            )
        return torch.device(text)
    return torch.device(text or "cpu")


def pin_memory_enabled(torch, requested: str) -> bool:
    text = str(requested or "auto").strip().lower()
    return bool(
        (text in {"auto", "cuda_auto", "gpu"} or text.startswith("cuda"))
        and torch.cuda.is_available()
    )


def make_grad_scaler(torch, *, enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        try:
            return torch.cuda.amp.GradScaler(enabled=True)
        except Exception:
            return None


class NullAutocast:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def autocast(torch, *, enabled: bool):
    if not enabled:
        return NullAutocast()
    try:
        return torch.amp.autocast("cuda", enabled=True)
    except Exception:
        try:
            return torch.cuda.amp.autocast(enabled=True)
        except Exception:
            return NullAutocast()


_pin_memory_enabled = pin_memory_enabled
_make_grad_scaler = make_grad_scaler
_autocast = autocast


__all__ = [
    "NullAutocast",
    "_autocast",
    "_make_grad_scaler",
    "_pin_memory_enabled",
    "autocast",
    "make_grad_scaler",
    "pin_memory_enabled",
    "resolve_torch_device",
]
