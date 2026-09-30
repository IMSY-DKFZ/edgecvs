import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import ModelRegistry
from models.shared.auxiliary_aspp_head import create_backbone


def _cfg_get(node, key, default=None):
    """OmegaConf/dict/object safe getter."""
    if node is None:
        return default
    try:
        if hasattr(node, "get"):
            return node.get(key, default)
    except Exception:
        pass
    try:
        return getattr(node, key)
    except Exception:
        return default


def _make_norm(num_channels):
    for groups in (32, 16, 8, 4, 2, 1):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


def _resolve_out_indices(CFG):
    segformer_cfg = _cfg_get(CFG, "segformer")
    aspp_cfg = _cfg_get(CFG, "aspp")
    out_indices = _cfg_get(
        segformer_cfg,
        "encoder_out_indices",
        _cfg_get(aspp_cfg, "encoder_out_indices", (0, 1, 2, 3)),
    )
    return tuple(int(i) for i in out_indices)


def _feature_channels(feature_info, out_indices):
    try:
        channels = list(feature_info.channels())
        if len(channels) == len(out_indices):
            return channels
    except Exception:
        pass

    channels = []
    for pos, idx in enumerate(out_indices):
        try:
            channels.append(feature_info[idx]["num_chs"])
        except Exception:
            channels.append(feature_info[pos]["num_chs"])
    return channels


def _ensure_nchw(feature, expected_channels, stage_idx):
    if feature.ndim != 4:
        raise RuntimeError(
            f"SegFormerAuxSeg expects 4D feature maps. "
            f"Stage {stage_idx} has shape {tuple(feature.shape)}."
        )

    if feature.shape[1] == expected_channels:
        return feature
    if feature.shape[-1] == expected_channels:
        return feature.permute(0, 3, 1, 2).contiguous()

    raise RuntimeError(
        f"Could not infer feature layout at stage {stage_idx}: "
        f"shape={tuple(feature.shape)}, expected_channels={expected_channels}."
    )


class SegFormerProjection(nn.Module):
    """1x1 projection used as the SegFormer MLP-equivalent for 2D maps."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            _make_norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.proj(x)


class SegFormerFusionHead(nn.Module):
    """
    SegFormer-style decode head:
    project each stage to a shared channel dimension, resize to the highest
    spatial stage, concatenate, fuse, lightly refine, then classify pixels.
    """

    def __init__(self, feature_channels, decoder_dim, num_classes, dropout=0.1):
        super().__init__()
        self.projections = nn.ModuleList(
            SegFormerProjection(ch, decoder_dim) for ch in feature_channels
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * len(feature_channels), decoder_dim, kernel_size=1, bias=False),
            _make_norm(decoder_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(
                decoder_dim,
                decoder_dim,
                kernel_size=3,
                padding=1,
                groups=decoder_dim,
                bias=False,
            ),
            _make_norm(decoder_dim),
            nn.GELU(),
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=1, bias=False),
            _make_norm(decoder_dim),
            nn.GELU(),
        )

        self.classifier = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, features, output_size=None):
        ref_shape = features[0].shape[-2:]
        projected = []

        for feature, projection in zip(features, self.projections):
            x = projection(feature)
            if x.shape[-2:] != ref_shape:
                x = F.interpolate(x, size=ref_shape, mode="bilinear", align_corners=False)
            projected.append(x)

        x = torch.cat(projected, dim=1)
        x = self.fuse(x)
        x = self.refine(x)
        logits = self.classifier(x)

        if output_size is not None and logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

        return logits


class LightweightAuxSegHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)


@ModelRegistry.register(
    name="SegFormerAuxSeg",
    matcher=lambda cfg: cfg.get("vmodel") == "SegFormerAuxSeg",
    build_msg=lambda cfg: f"Aux. SegFormer-style Model {cfg.vmodel}",
)
class SegFormerAuxSeg(nn.Module):
    """
    Multi-task classifier + SegFormer-style auxiliary segmentation model.

    This keeps the ASPPAuxSeg contract:
      {"logits": (B, target_size), "seg_output": (B, C, H, W)}
    and optionally returns "aux_segs" for DeepSupervisionFocalDiceLoss.
    """

    def __init__(self, CFG, override_model_name=None):
        super().__init__()
        self.CFG = CFG

        segformer_cfg = _cfg_get(CFG, "segformer")
        aspp_cfg = _cfg_get(CFG, "aspp")
        current_model_name = override_model_name if override_model_name else CFG.model_name
        out_indices = _resolve_out_indices(CFG)

        seg_size = _cfg_get(CFG.seg, "seg_input_size")
        self.seg_h = _cfg_get(seg_size, "height", CFG.height)
        self.seg_w = _cfg_get(seg_size, "width", CFG.width)

        pretrained = bool(_cfg_get(segformer_cfg, "pretrained", _cfg_get(aspp_cfg, "pretrained", True)))

        self.backbone = create_backbone(
            current_model_name,
            img_size=(self.seg_h, self.seg_w),
            pretrained=pretrained,
            num_classes=0,
            features_only=True,
            out_indices=out_indices,
            in_chans=3,
        )

        self.feature_info = self.backbone.feature_info
        self.feature_channels = _feature_channels(self.feature_info, out_indices)
        self.backbone_channels = self.feature_channels[-1]
        self.embedding_size = self.backbone_channels

        decoder_dim = int(
            _cfg_get(segformer_cfg, "decoder_dim", _cfg_get(aspp_cfg, "aspp_out_channels", 256))
        )
        decoder_dropout = float(_cfg_get(segformer_cfg, "decoder_dropout", 0.1))
        cls_dropout = float(_cfg_get(segformer_cfg, "cls_dropout", _cfg_get(aspp_cfg, "cls_dropout", 0.3)))

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=cls_dropout),
            nn.Linear(self.embedding_size, CFG.target_size),
        )

        num_seg_classes = len(CFG.seg.classes)
        self.segmentation_head = SegFormerFusionHead(
            feature_channels=self.feature_channels,
            decoder_dim=decoder_dim,
            num_classes=num_seg_classes,
            dropout=decoder_dropout,
        )

        default_aux = float(_cfg_get(CFG.seg, "aux_weight", 0.0)) > 0.0
        self.aux_supervision = bool(_cfg_get(segformer_cfg, "aux_supervision", default_aux))
        if self.aux_supervision:
            aux_hidden = int(_cfg_get(segformer_cfg, "aux_hidden_channels", max(32, decoder_dim // 2)))
            self.aux_seg_heads = nn.ModuleList(
                LightweightAuxSegHead(ch, aux_hidden, num_seg_classes)
                for ch in self.feature_channels[:-1]
            )
        else:
            self.aux_seg_heads = nn.ModuleList()

        # Heads are always built so checkpoints load strictly; seg_head=false
        # only skips running them (classification-only fast path).
        self.run_seg_head = bool(_cfg_get(CFG, "seg_head", True))

    def forward(self, x):
        if x.shape[-2:] != (self.seg_h, self.seg_w):
            x = F.interpolate(x, size=(self.seg_h, self.seg_w), mode="bilinear", align_corners=False)

        raw_features = self.backbone(x)
        features = [
            _ensure_nchw(feature, channels, idx)
            for idx, (feature, channels) in enumerate(zip(raw_features, self.feature_channels))
        ]

        cls_output = self.classifier(features[-1])
        if not self.run_seg_head:
            return {"logits": cls_output}

        seg_output = self.segmentation_head(features, output_size=(self.seg_h, self.seg_w))

        output = {"logits": cls_output, "seg_output": seg_output}
        if len(self.aux_seg_heads) > 0:
            output["aux_segs"] = [
                head(feature) for head, feature in zip(self.aux_seg_heads, features[:-1])
            ]

        return output
