from typing import Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transform_from_config(transform_config: Dict[str, Any]) -> A.BasicTransform:
    """
    Create an Albumentations transform from config dictionary.

    Args:
        transform_config: Dictionary containing transform name and parameters

    Returns:
        A.BasicTransform: Instantiated transform
    """
    name = transform_config["name"]
    params = transform_config.get("params", {})

    # Handle special case for ToTensorV2
    if name == "ToTensorV2":
        return ToTensorV2(**params)

    # Get the transform class from albumentations
    transform_cls = getattr(A, name)
    return transform_cls(**params)


def get_transforms(*, data: str, CFG) -> A.Compose:
    """
    Get image augmentation transforms based on configuration.

    Args:
        data: Either "train" or "valid" to specify the dataset type
        CFG: Configuration object containing augmentation parameters

    Returns:
        A.Compose: Composed transformation pipeline
    """

    if data not in ["train", "valid"]:
        raise ValueError(
            f"Invalid data type: {data}. Must be either 'train' or 'valid'"
        )

    # Get transforms for current mode
    transforms = [
        get_transform_from_config(t_cfg) for t_cfg in getattr(CFG, data).transforms
    ]

    # Handle TTA for validation
    if data == "valid":
        tta_enabled = getattr(CFG.valid, "tta", False)
        if tta_enabled:
            print("TTA (Test Time Augmentation) activated!")
            # Add TTA-specific transforms if defined
            if hasattr(CFG.valid, "tta_transforms"):
                tta_transforms = [
                    get_transform_from_config(t_cfg)
                    for t_cfg in CFG.valid.tta_transforms
                ]
                transforms.extend(tta_transforms)

    return A.Compose(transforms, additional_targets={"image2": "image"})
