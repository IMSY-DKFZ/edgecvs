import torch
from typing import Dict, Any

from models.registry import ModelRegistry  # autodiscovers all models on import


def build_model(CFG: Dict[str, Any], fold: int = None) -> torch.nn.Module:
    # 1. Determine model type using the registry's logic
    model_type = ModelRegistry.determine_model_from_cfg(CFG)

    # 2. Get the class
    model_class = ModelRegistry.get_model_class(model_type)

    print(f"Model used is {model_type}")

    # 3. Instantiate
    if model_type == "CLSModel":
        model = model_class(CFG, model_name=CFG.model_name, pretrained=CFG.pretrained)
    else:
        model = model_class(CFG)

    return model.to(CFG.device)
