import timm


def create_backbone(model_name, img_size, **kwargs):
    """
    Create a timm backbone, passing img_size only when the model supports it.
    ViT-based models (EVA02, ViT, DeiT …) accept img_size for position-embedding
    initialisation; CNN-based models (EdgeNeXt, ConvNeXt …) do not.
    """
    try:
        return timm.create_model(model_name, img_size=img_size, **kwargs)
    except TypeError:
        return timm.create_model(model_name, **kwargs)
