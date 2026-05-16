"""Generation package marker.

Keep this module lightweight to avoid import-time cycles between compatibility
wrappers under `core/*.py` and implementation modules under `core/generation/*`.

Use explicit imports from concrete modules, for example:
- `core.generation.kr.kr_oto_generator`
- `core.generation.ja.ja_oto_generator`
- `core.generation.kr_base_timing`
"""

__all__ = []
