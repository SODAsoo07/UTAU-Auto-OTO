# Auto OTO

Auto OTO is a source-available freeware tool for automatic oto setup and related
voicebank preparation workflows.

## Default Encoding

- The app default text/file encoding is **UTF-8**.

## License Summary

This repository is not an open-source project in the OSI sense.

- The source code is published under the source-available freeware terms in
  [LICENSE](./LICENSE).
- The official bundled models are licensed separately under
  [MODEL_LICENSE.md](./MODEL_LICENSE.md).
- Output created with the official models, such as generated oto settings,
  prepared voicebanks, and songs created with those voicebanks, may be used
  commercially under the terms described in `MODEL_LICENSE.md`.

## What You Can Do

- Use the software for free.
- Read and modify the source code.
- Share original or modified copies for free with the license notices kept
  intact.
- Use the generated outputs in commercial creative work.

## What You Cannot Do

- Sell the software itself or modified versions of the software.
- Charge for access to the software.
- Sell, host, or otherwise commercially exploit the official bundled models.

## Important Notes

- Third-party dependencies remain under their own licenses.
- The official model package and the source code are licensed separately.
- This repository may contain model files and metadata that are covered by
  `MODEL_LICENSE.md`, not by `LICENSE`.
- If you use your own models or your own data, you remain responsible for the
  rights to those materials.

## Draft Status

The files `LICENSE` and `MODEL_LICENSE.md` are project drafts intended to define
the distribution policy of this repository. Review and adjust them before public
release if you want tighter wording for contributor, data, or jurisdiction-
specific issues.

## UI Design Workflow (Pencil MCP)

- Integration workflow draft: `plan/pencil_integration_workflow.md`
- Design asset folder: `design/`
- Shared UI token module: `ui/theme_tokens.py`

## WFL Release Runtime

Release builds include a standalone CPU WFL runtime under
`UTAU_Auto_OTO/wfl_runtime`. Prepare or refresh it before building with:

```powershell
.\.venv310\Scripts\python.exe scripts\build\prepare_wfl_runtime.py
```

The script uses the sibling `WFL_PoC` checkout by default. Override the source
with `--poc-root` or `UTOA_WFL_POC_ROOT`. Standard `build.py` runs validate the
bundle and stop if the WFL Python, encoder, or Korean/Japanese models are
incomplete.
