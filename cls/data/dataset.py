import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# Fix number of threads used by opencv
cv2.setNumThreads(1)



def seg_mask_path_for(CFG, dataset, file_name):
    """Resolve the on-disk PNG mask path for an image.

    Single source of truth shared by ``TrainDataset._load_mask_from_disk`` and
    the offline seg-stats script so the sampler measures exactly what the model
    trains on.
    """
    return os.path.join(
        CFG.parent_path,
        CFG.train_path,
        CFG.seg_mask_path,
        dataset,
        file_name.replace(".jpg", ".png"),
    )


class TrainDataset(Dataset):
    def __init__(self, df, fold, CFG, transform=None, inference=False):
        self.df = df
        self.CFG = CFG
        self.fold = fold
        self.file_names = df["image_path"].values
        self.transform = transform
        self.inference = inference
        self.dataset = df["dataset"].values if "dataset" in df.columns else [""] * len(df)
        
        # Pre-load labels
        index_no = df.columns.get_loc(CFG.col0)
        self.labels = torch.FloatTensor(
            df.iloc[:, index_no : index_no + CFG.target_size].values.astype(np.float16)
        )
        
        # Cache the strategy type for speed
        self.strategy = CFG.strategy

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
            file_name = self.file_names[index]
            dataset = self.dataset[index]
            image = self._load_image(file_name, dataset)

            # 1. Initialize the output dictionary
            batch = {}

            # 2. Strategies that load masks from disk
            if self.strategy in ("aux_seg_disk", "aux_seg_disk_attn", "seg") and not self.inference:
                mask = self._load_mask_from_disk(file_name, dataset)
                if self.transform:
                    aug = self.transform(image=image, mask=mask)
                    batch["image"] = aug["image"]
                    batch["mask"] = aug["mask"]
            else:
                # 3. Standard processing
                if self.transform:
                    batch["image"] = self.transform(image=image)["image"]
                else:
                    batch["image"] = image

            # 4. Attach classification targets if available
            if not self.inference and self.CFG.target_size > 0 and self.strategy != "seg":
                batch["target"] = self.labels[index]
                
            return batch
    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _load_image(self, file_name, dataset):
        file_path = os.path.join(
            self.CFG.parent_path, self.CFG.train_path, dataset, file_name
        )
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_mask_from_disk(self, file_name, dataset):
        """
        Only called for disk-mask segmentation strategies.
        No need to check config flags here anymore.
        """
        # Determine Path
        mask_path = seg_mask_path_for(self.CFG, dataset, file_name)

        # Case B: Standard PNG Masks
        mask = cv2.imread(mask_path)
        # mask = cv2.resize(mask, (self.CFG.height, self.CFG.width))
        mask = mask[..., 0] # RGB to Class Index (channel 0)

        # Class Merging/Removal Logic
        mask = self._process_mask_classes(mask)
        return mask
    
    def _process_mask_classes(self, mask):
        """Clean up buggy pixels and handle class mapping"""
        if not hasattr(self.CFG, 'seg'): 
            return mask
        
        # FIX: Clip values to [0, num_classes - 1]
        # This prevents out-of-bounds errors during one-hot encoding
        num_classes = self.CFG.seg_num_classes 
        mask = np.clip(mask, 0, num_classes - 1)
        
        return mask
