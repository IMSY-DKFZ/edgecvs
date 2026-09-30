# helper functions
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf


def seed_torch(seed=42):
    """
    Seed various random number generators to ensure reproducibility.

    Args:
        seed (int): Seed value to set for random number generators.

    Returns:
        None
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_experiment_config(CFG):
    """
    Merge the saved experiment config over the launch-time CFG for inference.

    Loads `output_dir/runs/{exp}/config.yaml` (the config snapshotted at train
    time), merges it on top of the Hydra CFG, then re-applies a whitelist of
    runtime keys so values like paths/device/batch size follow the current
    invocation rather than the training run. Inference flags are forced on.

    Returns the merged config, or the original CFG (with inference forced) when
    no saved config is found.
    """
    base_output = CFG.output_dir if CFG.output_dir else os.path.join(CFG.parent_path, "output")
    exp_config_path = os.path.join(base_output, "runs", CFG.exp, "config.yaml")

    if not os.path.exists(exp_config_path):
        print(f"[Warning] Config file not found at {exp_config_path}.")
        OmegaConf.set_struct(CFG, False)
        CFG.inference = True
        return CFG

    print(f"[Config] Loading saved experiment config from: {exp_config_path}")
    saved_cfg = OmegaConf.load(exp_config_path)

    # Unlock both configs so the merge accepts keys absent from the other schema.
    OmegaConf.set_struct(CFG, False)
    OmegaConf.set_struct(saved_cfg, False)

    merged_cfg = OmegaConf.merge(CFG, saved_cfg)

    # Runtime overrides: these follow the current invocation, not the saved run.
    runtime_keys = [
        "parent_path", "output_dir", "device",
        "num_workers", "batch_size", "valid_batch_size",
        "debug", "inference", "run_inference",
        "tta", "n_tta", "seg_head",
    ]
    for key in runtime_keys:
        if hasattr(CFG, key):
            OmegaConf.update(merged_cfg, key, getattr(CFG, key))

    # Force inference flags on.
    merged_cfg.inference = True
    merged_cfg.run_inference = True

    return merged_cfg
