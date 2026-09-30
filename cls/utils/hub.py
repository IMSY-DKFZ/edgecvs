import os
import torch
from omegaconf import OmegaConf


def release_weights_name(exp, fold):
    """Released weights file of one fold, e.g. edgecvs-5m-fold0.safetensors."""
    return f"{exp}-fold{fold}.safetensors"


def download_run(CFG):
    """
    Fetch a released model from the Hugging Face Hub. The Hub repo holds one
    subfolder per model (<exp>/), mirrored into the local run folder:
      <output_dir>/runs/<exp>/config.yaml
      <output_dir>/runs/<exp>/<exp>-fold<k>.safetensors  (for each trn_fold)
    Files already present locally are not downloaded again.
    """
    from huggingface_hub import hf_hub_download

    runs_dir = os.path.join(CFG.output_dir, "runs")
    print(f"[Hub] Fetching {CFG.hf_repo}/{CFG.exp} -> {os.path.join(runs_dir, CFG.exp)}")

    config_path = hf_hub_download(CFG.hf_repo, "config.yaml", subfolder=CFG.exp, local_dir=runs_dir)
    for fold in OmegaConf.load(config_path).trn_fold:
        hf_hub_download(CFG.hf_repo, release_weights_name(CFG.exp, fold), subfolder=CFG.exp, local_dir=runs_dir)


def load_checkpoint(CFG, fold, device):
    """
    Load a fold's weights: the released `<exp>-fold<k>.safetensors` if present,
    otherwise a training checkpoint `fold<k>/checkpoints/last.pth`. Returns the state_dict, or None.
    """
    run_dir = os.path.join(CFG.output_dir, "runs", CFG.exp)
    safetensors_path = os.path.join(run_dir, release_weights_name(CFG.exp, fold))
    ckpt_path = os.path.join(run_dir, f"fold{fold}", "checkpoints", "last.pth")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file

        print(f"Loading weights from: {safetensors_path}")
        return load_file(safetensors_path, device=str(device))

    if os.path.exists(ckpt_path):
        print(f"Loading weights from: {ckpt_path}")
        return torch.load(ckpt_path, map_location=device, weights_only=False)["model"]

    return None
