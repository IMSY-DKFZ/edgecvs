import os
import hydra
import torch
import pandas as pd
import numpy as np
import warnings
from hydra.utils import to_absolute_path

from engine.helper import get_inference_loader, inference_fn
from models.classifier import build_model
from utils.hub import download_run, load_checkpoint
from utils.utils import seed_torch, load_experiment_config

# Suppress warnings
warnings.filterwarnings("ignore")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def list_images(root):
    """Recursively list images under `root`, as sorted paths relative to it."""
    image_paths = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for filename in filenames:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                image_paths.append(os.path.relpath(os.path.join(dirpath, filename), root))
    return sorted(image_paths)


def run(CFG):
    """
    Predict C1/C2/C3 probabilities for every image in a folder (recursively).
    """
    print(f"\n[Inference] Starting folder inference for Experiment: {CFG.exp}")

    # 1. Reproducibility
    seed_torch(seed=CFG.seed)

    # 2. Setup Device
    device = torch.device(CFG.device)

    # 3. Build the dataframe of images
    input_dir = to_absolute_path(CFG.input_dir)
    image_paths = list_images(input_dir)
    if not image_paths:
        print(f"[Error] No images found under {input_dir}")
        return
    print(f"Found {len(image_paths)} images under {input_dir}")

    # TrainDataset reads parent_path/train_path/dataset/image_path
    CFG.parent_path = input_dir
    CFG.train_path = ""

    folds = pd.DataFrame({"image_path": image_paths})
    # TrainDataset reads target_size label columns starting at col0; unlabeled here
    for i in range(CFG.target_size):
        folds[f"C{i + 1}"] = 0.0

    # Container for Ensemble Averaging
    ensemble_preds_list = []

    # 4. Loop over trained folds
    for fold in CFG.trn_fold:
        print(f"\nProcessing Model/Fold {fold}...")

        # Treat the whole dataframe as the inference set for this model
        folds["fold"] = fold

        # --- A. Locate Weights ---
        state_dict = load_checkpoint(CFG, fold, device)

        if state_dict is None:
            print(f"[Warning] No checkpoint found for {CFG.exp} fold {fold}. Skipping Fold {fold}.")
            continue

        # --- B. Build & Run ---
        model = build_model(CFG)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        _, inference_loader = get_inference_loader(CFG, fold, folds)
        preds = inference_fn(CFG, inference_loader, model, device)
        ensemble_preds_list.append(preds)

        # Clean up memory
        del model, state_dict
        torch.cuda.empty_cache()

    if not ensemble_preds_list:
        print("[Inference] No predictions were generated.")
        return

    # 5. Average logits across models, then map to probabilities
    avg_preds = np.mean(np.array(ensemble_preds_list), axis=0)
    probs = 1.0 / (1.0 + np.exp(-avg_preds))

    predictions = pd.DataFrame({"image_path": image_paths})
    for i in range(CFG.target_size):
        predictions[f"C{i + 1}"] = probs[:, i]

    output_csv = to_absolute_path(CFG.output_csv)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    predictions.to_csv(output_csv, index=False)
    print(f"\n[Inference] Saved {len(predictions)} predictions to {output_csv}")


@hydra.main(config_path="../configs/cls", config_name="config", version_base=None)
def main(CFG):
    # Fetch the released weights from the Hugging Face Hub (skipped if hf_repo is null)
    if CFG.hf_repo:
        download_run(CFG)

    # Load the saved experiment config, merged with runtime overrides.
    final_cfg = load_experiment_config(CFG)

    # Run the inference loop
    run(final_cfg)


if __name__ == "__main__":
    main()
