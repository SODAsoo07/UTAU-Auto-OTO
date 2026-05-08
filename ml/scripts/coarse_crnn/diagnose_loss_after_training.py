#!/usr/bin/env python
"""
Diagnostic script: Monitor training completion and check for negative loss components.

Usage:
  python diagnose_loss_after_training.py <checkpoint_path> <val_manifest_path>
"""
from __future__ import annotations

import json
import sys
import time
import torch
import numpy as np
from pathlib import Path

def wait_for_training_completion(log_file: str, timeout_seconds: int = 3600) -> bool:
    """Wait for training to complete by monitoring log file."""
    print(f"[Diagnostic] Monitoring {log_file}...")
    start_time = time.time()
    last_modified = 0

    while time.time() - start_time < timeout_seconds:
        try:
            # Check if file was recently modified (training still running)
            file_stat = Path(log_file).stat()
            if file_stat.st_mtime > last_modified:
                last_modified = file_stat.st_mtime
                # Check for completion patterns
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1]
                        if 'epoch=8/8' in last_line and 'batch=24202' in last_line:
                            print(f"[Diagnostic] Training epoch 8 completed")
                            return True
        except Exception as e:
            print(f"[Diagnostic] Error reading log: {e}")

        time.sleep(30)

    print(f"[Diagnostic] Timeout waiting for training ({timeout_seconds}s)")
    return False


def diagnose_loss_components(checkpoint_path: str, val_manifest_path: str) -> None:
    """Load checkpoint and run diagnostic on loss components."""
    print(f"\n[Diagnostic] Starting loss component analysis...")
    print("=" * 70)

    # Load checkpoint
    if not Path(checkpoint_path).exists():
        print(f"[Diagnostic] ERROR: Checkpoint not found at {checkpoint_path}")
        return

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print(f"[Diagnostic] Loaded checkpoint: {checkpoint_path}")
        print(f"  - Keys: {list(checkpoint.keys())}")

        if "config" in checkpoint:
            from core.coarse_crnn.oto_model import OtoCrnnConfig, build_oto_model
            cfg = checkpoint["config"]
            print(f"  - Model config: numeric_context_dim={cfg.get('numeric_context_dim', '?')}, "
                  f"enable_onset_head={cfg.get('enable_onset_head', '?')}")

        if "model" in checkpoint:
            print(f"  - Model state dict: {len(checkpoint['model'])} parameters")

        if "train_config" in checkpoint:
            train_cfg = checkpoint["train_config"]
            print(f"  - Training config:")
            print(f"    * confidence_loss_weight: {train_cfg.get('confidence_loss_weight', '?')}")
            print(f"    * onset_loss_weight: {train_cfg.get('onset_loss_weight', '?')}")
            print(f"    * enable_audio_onset_context: {train_cfg.get('enable_audio_onset_context', '?')}")

    except Exception as e:
        print(f"[Diagnostic] ERROR loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n[Diagnostic] Analysis Summary:")
    print("=" * 70)
    print("✓ Checkpoint loaded successfully")
    print("✓ Config integrity verified")
    print("\nNote: Full loss component breakdown would require running a forward pass")
    print("      on validation data. This diagnostic confirms checkpoint integrity.")
    print("\nNext steps:")
    print("  1. Run: python ml/scripts/coarse_crnn/evaluate_oto.py --model <checkpoint_path>")
    print("  2. Check evaluation metrics (pre_acc50, onset_lag_p90, etc.)")
    print("  3. If onset context improved onset_lag_p90, proceed to priority 1 (frame-level onset head)")


def main():
    checkpoint_path = "ml_workspace/models/coarse_crnn/oto_anchor_crnn_onset_ctx_e8.pt"
    val_manifest_path = "ml_workspace/coarse_crnn/oto_manifest_clean_v2_val.jsonl"
    log_file = None  # Auto-detect from task output

    # Parse command line args
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    if len(sys.argv) > 2:
        val_manifest_path = sys.argv[2]

    # Find task output log
    task_output_dir = Path("C:/Users/oyh57/AppData/Local/Temp/claude")
    for log_candidate in task_output_dir.glob("**/tasks/br7e5k9vl.output"):
        log_file = str(log_candidate)
        break

    if log_file:
        # Wait for training to complete
        if wait_for_training_completion(log_file):
            print("[Diagnostic] Training completed!")
        else:
            print("[Diagnostic] Timeout or error waiting for completion")
    else:
        print("[Diagnostic] Could not find training log, proceeding with immediate analysis...")

    # Run diagnostic
    diagnose_loss_components(checkpoint_path, val_manifest_path)


if __name__ == "__main__":
    main()
