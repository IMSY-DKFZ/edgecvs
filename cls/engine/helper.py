import numpy as np
import gc
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


from data.augmentation import get_transforms
from data.dataset import TrainDataset

def inference_fn(
    CFG,
    valid_loader,
    model,
    device,
):
    model.eval()  # Set the model to evaluation mode

    preds = []  # Initialize a list to store predictions

    for step, data in tqdm(enumerate(valid_loader), desc="Processing batches", total=len(valid_loader)):

        images = data["image"].to(CFG.device)

        with torch.no_grad():
            if CFG.tta:
                tta_preds = []
                for _ in range(CFG.n_tta):
    
                    model_output = model(images)
                    
                    tta_preds.append(model_output["logits"].detach().cpu().numpy())  # Detach and move to CPU
                    del y_preds  # Immediately delete to free memory
                    gc.collect()  # Force garbage collection

                y_preds = np.mean(tta_preds, axis=0)  # Compute mean on CPU
                del tta_preds  # Delete the TTA predictions list
            else:
             
                model_output = model(images)

                y_preds = model_output["logits"].detach().cpu().numpy()  # Detach and move to CPU

            preds.append(y_preds)  # Append predictions

        del images, model_output  # Delete variables
        gc.collect()  # Force garbage collection

    predictions = np.concatenate(preds, axis=0)  # Concatenate all predictions
    del preds  # Delete the predictions list
    gc.collect()  # Final garbage collection

    return predictions


def get_inference_loader(CFG, fold, folds):

   
    inference_folds = folds
    
    inference_dataset = TrainDataset(
        inference_folds,
        fold,
        transform=get_transforms(CFG=CFG, data="valid"),
        inference=True,
        CFG=CFG,
    )

    # Pytorch dataloader
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.nworkers,
        pin_memory=False,
        drop_last=False,
    )
    print("returned inference folds", inference_folds.shape)
    return inference_folds, inference_loader
