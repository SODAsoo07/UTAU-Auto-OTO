# Third-Party Notices

This document summarizes third-party software used by Auto OTO.

This file is provided for convenience only and does not replace the original
license texts of each dependency. If any summary in this file conflicts with an
upstream license, the upstream license controls.

## Scope

This notice covers:

- Python packages used by the source code or by optional ML features
- build tools used to produce distributable binaries
- third-party runtime components that may be bundled into Windows releases

This notice does not grant any rights in third-party software beyond the rights
granted by each component's own license.

## Project License Separation

Auto OTO uses separate licensing for:

- project source code and bundled project assets
- official model files
- third-party dependencies

Third-party components remain under their own licenses and are not relicensed
under the Auto OTO project license.

## Runtime and Source Dependencies

### customtkinter

- Upstream: Tom Schimansky / CustomTkinter
- Website: https://customtkinter.tomschimansky.com
- License: MIT License
- Usage: GUI framework

### TextGrid

- Upstream: Kyle Gorman, Max Bane, Morgan Sonderegger
- Package: `TextGrid`
- License: MIT License
- Usage: TextGrid parsing and handling

### NumPy

- Upstream: NumPy Developers
- Website: https://numpy.org
- License: BSD 3-Clause License
- Usage: numeric processing used by validation and ML-related code
- Note: binary distributions of NumPy may include additional third-party runtime
  components under their own notices

### LightGBM

- Upstream: Microsoft / LightGBM contributors
- Website: https://github.com/microsoft/LightGBM
- License: MIT License
- Usage: optional OTO ML inference and training backend

### pandas

- Upstream: pandas contributors
- Website: https://pandas.pydata.org
- License: BSD 3-Clause License
- Usage: optional ML training and evaluation workflows

### scikit-learn

- Upstream: scikit-learn developers
- Website: https://scikit-learn.org
- License: BSD 3-Clause License
- Usage: optional ML training and evaluation workflows
- Note: some binary distributions may include additional runtime files under
  their own license terms

### PyTorch

- Upstream: PyTorch contributors
- Website: https://pytorch.org
- License: BSD 3-Clause License
- Usage: optional environment checks and optional advanced ML workflows
- Packaging note: current light build configuration excludes `torch`,
  `torchaudio`, and `torchvision` from the default bundled app build

## Build Tools

### PyInstaller

- Upstream: PyInstaller Development Team
- Website: https://pyinstaller.org
- License: GPL v2 or later, with PyInstaller bootloader exception
- Usage: build-time packaging tool for Windows executables
- Note: the PyInstaller exception allows use for non-open-source and
  commercial/non-commercial distributions, but PyInstaller itself remains under
  its own license

## Bundled Runtime Components

### FFmpeg

- Upstream: FFmpeg project and bundled codec/library contributors
- Website: https://ffmpeg.org
- Windows binary source currently referenced by build script:
  https://www.gyan.dev/ffmpeg/builds/
- Usage: audio/media processing for bundled Windows releases
- Important: if FFmpeg binaries are redistributed with Auto OTO, the release
  package must also include the corresponding FFmpeg license notices and any
  other required attribution or source-reference information for that specific
  FFmpeg build
- Important: the exact obligations depend on the specific FFmpeg build that is
  bundled

## Build Configuration Notes

At the time of writing, the project build script:

- installs `PyInstaller` at build time
- installs dependencies from `requirements.txt` and `requirements-ml.txt`
- excludes `torch`, `torchaudio`, `torchvision`, and `ml` from the default app
  bundle
- copies FFmpeg binaries into the Windows distributable bundle

Release maintainers should re-check this notice whenever the build pipeline or
dependency list changes.

## Redistribution Guidance

When distributing Auto OTO binaries or packaged source releases, it is
recommended to include:

- this notice file
- the original license text for each bundled third-party dependency
- separate FFmpeg notices if FFmpeg is bundled
- any additional notices required by binary wheels or bundled runtime libraries

## No Warranty

Third-party software is provided under the terms of its respective license.
Please review each upstream license for warranty disclaimers and conditions.
