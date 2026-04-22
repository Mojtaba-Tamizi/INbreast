from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as e:
    raise ImportError(
        "MBSSwinUNet requires timm. Install it with: pip install timm"
    ) from e


# ============================================================
# Small helpers
# ============================================================


def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported activation type: {name}")


def make_norm(name: str, num_channels: int) -> nn.Module:
    name = name.lower()
    if name == "batchnorm":
        return nn.BatchNorm2d(num_channels)
    if name == "instancenorm":
        return nn.InstanceNorm2d(num_channels, affine=True)
    if name == "groupnorm":
        num_groups = min(32, num_channels)
        while num_groups > 1 and num_channels % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups, num_channels)
    if name == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm type: {name}")


# ============================================================
# Basic building blocks
# ============================================================


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = False,
        norm: str = "groupnorm",
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.norm = make_norm(norm, out_channels)
        self.act = make_activation(activation)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class ResidualRefinementBlock(nn.Module):
    """
    Lightweight decoder/fusion refinement block:
      1) 1x1 projection for alignment
      2) 3x3 conv
      3) 3x3 conv
      4) residual addition
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = "groupnorm",
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.proj = ConvNormAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            norm=norm,
            activation=activation,
            dropout=0.0,
        )
        self.conv1 = ConvNormAct(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            norm=norm,
            activation=activation,
            dropout=dropout,
        )
        self.conv2 = ConvNormAct(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            norm=norm,
            activation="none",
            dropout=0.0,
        )
        self.act = make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        x = self.act(x)
        return x


class UpsampleBlock(nn.Module):
    """
    Decoder stage:
      1) bilinear upsample to skip size
      2) channel projection
      3) concatenate with skip
      4) refinement
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm: str = "groupnorm",
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.pre = ConvNormAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            norm=norm,
            activation=activation,
            dropout=0.0,
        )
        self.refine = ResidualRefinementBlock(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            norm=norm,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.pre(x)
        x = torch.cat([x, skip], dim=1)
        x = self.refine(x)
        return x


# ============================================================
# Backbone wrapper
# ============================================================


class SwinEncoder(nn.Module):
    """
    timm-based Swin encoder returning 4 hierarchical feature maps in NCHW.

    For patch size 4, the typical spatial pyramid is:
      H/4, H/8, H/16, H/32

    This wrapper is tolerant to image sizes like 256 and 512.
    """

    def __init__(
        self,
        backbone_name: str,
        backbone_pretrained: bool,
        in_channels: int,
        image_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        create_kwargs = {
            "pretrained": backbone_pretrained,
            "features_only": True,
            "in_chans": in_channels,
            "out_indices": (0, 1, 2, 3),
        }
        create_kwargs["output_fmt"] = "NCHW"
        if image_size is not None:
            create_kwargs["img_size"] = image_size

        try:
            self.encoder = timm.create_model(backbone_name, **create_kwargs)
        except TypeError:
            # Some backbones may not accept img_size explicitly in all timm versions.
            create_kwargs.pop("img_size", None)
            self.encoder = timm.create_model(backbone_name, **create_kwargs)

        self.channels = tuple(self.encoder.feature_info.channels())

        if len(self.channels) != 4:
            raise RuntimeError(
                f"Expected 4 encoder feature maps, got {len(self.channels)}."
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.encoder(x)

        if len(features) != 4:
            raise RuntimeError(
                f"Expected 4 encoder feature maps, got {len(features)}."
            )

        out: List[torch.Tensor] = []

        for feat, ch in zip(features, self.channels):
            if feat.ndim != 4:
                raise RuntimeError(
                    f"Expected 4D encoder feature map, got shape {tuple(feat.shape)}"
                )

            # Case 1: already NCHW
            if feat.shape[1] == ch:
                out.append(feat.contiguous())
                continue

            # Case 2: NHWC -> convert to NCHW
            if feat.shape[-1] == ch:
                feat = feat.permute(0, 3, 1, 2).contiguous()
                out.append(feat)
                continue

            raise RuntimeError(
                f"Could not infer feature layout for shape {tuple(feat.shape)} "
                f"with expected channel size {ch}."
            )

        return out


# ============================================================
# Multi-scale fusion
# ============================================================


class MultiScaleFusion(nn.Module):
    """
    Fuses selected decoder features into a shared final feature map.
    Typical starting choice from the design text:
      fusion_levels = ("d3", "d2", "df")
    """

    def __init__(
        self,
        level_to_channels: Dict[str, int],
        fusion_levels: Sequence[str],
        fusion_channels: int = 64,
        norm: str = "groupnorm",
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        allowed = {"d3", "d2", "d1", "df"}
        fusion_levels = tuple(fusion_levels)

        if len(fusion_levels) == 0:
            raise ValueError("fusion_levels cannot be empty when fusion is enabled.")
        invalid = set(fusion_levels) - allowed
        if invalid:
            raise ValueError(f"Invalid fusion levels: {sorted(invalid)}")

        self.fusion_levels = fusion_levels
        self.align_layers = nn.ModuleDict()

        for level in self.fusion_levels:
            self.align_layers[level] = ConvNormAct(
                in_channels=level_to_channels[level],
                out_channels=fusion_channels,
                kernel_size=1,
                norm=norm,
                activation=activation,
                dropout=0.0,
            )

        self.fuse = ResidualRefinementBlock(
            in_channels=fusion_channels * len(self.fusion_levels),
            out_channels=fusion_channels,
            norm=norm,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, feature_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        missing = [level for level in self.fusion_levels if level not in feature_dict]
        if missing:
            raise KeyError(f"Missing fusion levels in feature_dict: {missing}")

        target_size = feature_dict["df"].shape[-2:]
        aligned: List[torch.Tensor] = []

        for level in self.fusion_levels:
            x = feature_dict[level]
            x = self.align_layers[level](x)
            if x.shape[-2:] != target_size:
                x = F.interpolate(
                    x,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            aligned.append(x)

        x = torch.cat(aligned, dim=1)
        x = self.fuse(x)
        return x


# ============================================================
# Output heads
# ============================================================


class BoundaryHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        norm: str = "groupnorm",
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                norm=norm,
                activation=activation,
                dropout=dropout,
            ),
            ConvNormAct(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                norm=norm,
                activation=activation,
                dropout=0.0,
            ),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        hidden_channels: int = 64,
        norm: str = "groupnorm",
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                norm=norm,
                activation=activation,
                dropout=dropout,
            ),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ExpertHead(nn.Module):
    """
    Lightweight expert head closer to the text:
      3x3 Conv + Act/Norm
      1x1 Conv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        hidden_channels: int = 64,
        norm: str = "groupnorm",
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            ConvNormAct(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                norm=norm,
                activation=activation,
                dropout=dropout,
            ),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class GatingNetwork(nn.Module):
    """
    Lightweight image-level gate:
      GAP -> FC -> Act -> FC -> Softmax
    """

    def __init__(
        self,
        in_channels: int,
        num_experts: int,
        hidden_dim: int = 32,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_experts)
        self.act = make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x).flatten(1)
        gate = self.fc1(pooled)
        gate = self.act(gate)
        gate = self.fc2(gate)
        gate = torch.softmax(gate, dim=1)
        return gate


class MoESegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        num_experts: int = 2,
        expert_hidden_channels: int = 64,
        gate_hidden_dim: int = 32,
        norm: str = "groupnorm",
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if num_experts < 2:
            raise ValueError("MoE head requires at least 2 experts.")

        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [
                ExpertHead(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    hidden_channels=expert_hidden_channels,
                    norm=norm,
                    activation=activation,
                    dropout=dropout,
                )
                for _ in range(num_experts)
            ]
        )
        self.gate = GatingNetwork(
            in_channels=in_channels,
            num_experts=num_experts,
            hidden_dim=gate_hidden_dim,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        expert_logits = [expert(x) for expert in self.experts]
        stacked = torch.stack(expert_logits, dim=1)  # [B, E, C, H, W]

        gate_weights = self.gate(x)  # [B, E]
        weights = gate_weights[:, :, None, None, None]
        fused_logits = (stacked * weights).sum(dim=1)

        return {
            "logits": fused_logits,
            "expert_logits": stacked,
            "gate_weights": gate_weights,
        }


# ============================================================
# Main architecture
# ============================================================


class MBSSwinUNet(nn.Module):
    """
    MBS-SwinUNet
    Modular Boundary-aware Multi-scale Swin-UNet

    Repo-facing output convention:
      - returns logits only
      - public keys are:
          "mask"
          "boundary"   (if enabled)

    Optional debug/analysis tensors are returned only when:
      return_aux=True
    """

    def __init__(
        self,
        image_size: Optional[int] = 256,
        in_channels: int = 3,
        num_classes: int = 1,
        backbone_name: str = "swin_tiny_patch4_window7_224",
        backbone_pretrained: bool = True,
        decoder_channels: Sequence[int] = (384, 192, 96, 64),
        refine_dropout: float = 0.0,
        use_multiscale_fusion: bool = True,
        fusion_levels: Sequence[str] = ("d3", "d2", "df"),
        fusion_channels: int = 64,
        use_boundary_head: bool = True,
        use_moe_head: bool = False,
        num_experts: int = 2,
        gate_hidden_dim: int = 32,
        seg_head_hidden_channels: int = 64,
        boundary_head_hidden_channels: int = 64,
        expert_hidden_channels: int = 64,
        activation: str = "gelu",
        norm: str = "groupnorm",
        return_aux_by_default: bool = False,
    ) -> None:
        super().__init__()

        decoder_channels = tuple(decoder_channels)
        fusion_levels = tuple(fusion_levels)

        if len(decoder_channels) != 4:
            raise ValueError("decoder_channels must contain 4 values.")
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")

        self.image_size = image_size
        self.use_multiscale_fusion = use_multiscale_fusion
        self.use_boundary_head = use_boundary_head
        self.use_moe_head = use_moe_head
        self.return_aux_by_default = return_aux_by_default

        # --------------------
        # Encoder
        # --------------------
        self.encoder = SwinEncoder(
            backbone_name=backbone_name,
            backbone_pretrained=backbone_pretrained,
            in_channels=in_channels,
            image_size=image_size,
        )
        enc_ch = self.encoder.channels  # usually (96, 192, 384, 768)

        # --------------------
        # Decoder
        # --------------------
        c_d3, c_d2, c_d1, c_df = decoder_channels

        self.dec3 = UpsampleBlock(
            in_channels=enc_ch[3],
            skip_channels=enc_ch[2],
            out_channels=c_d3,
            norm=norm,
            activation=activation,
            dropout=refine_dropout,
        )
        self.dec2 = UpsampleBlock(
            in_channels=c_d3,
            skip_channels=enc_ch[1],
            out_channels=c_d2,
            norm=norm,
            activation=activation,
            dropout=refine_dropout,
        )
        self.dec1 = UpsampleBlock(
            in_channels=c_d2,
            skip_channels=enc_ch[0],
            out_channels=c_d1,
            norm=norm,
            activation=activation,
            dropout=refine_dropout,
        )

        self.final_up_proj = ConvNormAct(
            in_channels=c_d1,
            out_channels=c_df,
            kernel_size=3,
            norm=norm,
            activation=activation,
            dropout=0.0,
        )
        self.final_up_refine = ResidualRefinementBlock(
            in_channels=c_df,
            out_channels=c_df,
            norm=norm,
            activation=activation,
            dropout=refine_dropout,
        )

        # --------------------
        # Multi-scale fusion
        # --------------------
        level_to_channels = {
            "d3": c_d3,
            "d2": c_d2,
            "d1": c_d1,
            "df": c_df,
        }

        if self.use_multiscale_fusion:
            self.fusion = MultiScaleFusion(
                level_to_channels=level_to_channels,
                fusion_levels=fusion_levels,
                fusion_channels=fusion_channels,
                norm=norm,
                activation=activation,
                dropout=refine_dropout,
            )
            shared_channels = fusion_channels
        else:
            self.fusion = None
            self.post_no_fusion = ResidualRefinementBlock(
                in_channels=c_df,
                out_channels=c_df,
                norm=norm,
                activation=activation,
                dropout=refine_dropout,
            )
            shared_channels = c_df

        # --------------------
        # Boundary head
        # --------------------
        if self.use_boundary_head:
            self.boundary_head = BoundaryHead(
                in_channels=shared_channels,
                hidden_channels=boundary_head_hidden_channels,
                norm=norm,
                activation=activation,
                dropout=refine_dropout,
            )
        else:
            self.boundary_head = None

        # --------------------
        # Segmentation head / MoE head
        # --------------------
        if self.use_moe_head:
            self.seg_head = MoESegmentationHead(
                in_channels=shared_channels,
                out_channels=num_classes,
                num_experts=num_experts,
                expert_hidden_channels=expert_hidden_channels,
                gate_hidden_dim=gate_hidden_dim,
                norm=norm,
                activation=activation,
                dropout=refine_dropout,
            )
        else:
            self.seg_head = SegmentationHead(
                in_channels=shared_channels,
                out_channels=num_classes,
                hidden_channels=seg_head_hidden_channels,
                norm=norm,
                activation=activation,
                dropout=refine_dropout,
            )

    def _decode(
        self,
        features: List[torch.Tensor],
        input_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e1, e2, e3, b = features

        d3 = self.dec3(b, e3)   # H/16
        d2 = self.dec2(d3, e2)  # H/8
        d1 = self.dec1(d2, e1)  # H/4

        df = F.interpolate(
            d1,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        df = self.final_up_proj(df)
        df = self.final_up_refine(df)

        return d3, d2, d1, df

    def _make_shared_feature(
        self,
        d3: torch.Tensor,
        d2: torch.Tensor,
        d1: torch.Tensor,
        df: torch.Tensor,
    ) -> torch.Tensor:
        feature_dict = {
            "d3": d3,
            "d2": d2,
            "d1": d1,
            "df": df,
        }

        if self.use_multiscale_fusion:
            shared = self.fusion(feature_dict)
        else:
            shared = self.post_no_fusion(df)

        return shared

    def forward(
        self,
        x: torch.Tensor,
        return_aux: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        if return_aux is None:
            return_aux = self.return_aux_by_default

        features = self.encoder(x)
        d3, d2, d1, df = self._decode(features, input_size=input_size)
        shared = self._make_shared_feature(d3, d2, d1, df)

        outputs: Dict[str, torch.Tensor] = {}

        # Segmentation branch
        seg_aux: Dict[str, torch.Tensor] = {}
        if self.use_moe_head:
            seg_out = self.seg_head(shared)
            mask_logits = seg_out["logits"]
            seg_aux["expert_logits"] = seg_out["expert_logits"]
            seg_aux["gate_weights"] = seg_out["gate_weights"]
        else:
            mask_logits = self.seg_head(shared)

        if mask_logits.shape[-2:] != input_size:
            mask_logits = F.interpolate(
                mask_logits,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )

        outputs["mask"] = mask_logits

        # Boundary branch
        if self.use_boundary_head and self.boundary_head is not None:
            boundary_logits = self.boundary_head(shared)
            if boundary_logits.shape[-2:] != input_size:
                boundary_logits = F.interpolate(
                    boundary_logits,
                    size=input_size,
                    mode="bilinear",
                    align_corners=False,
                )
            outputs["boundary"] = boundary_logits

        # Optional auxiliary/debug outputs
        if return_aux:
            outputs["shared_feature"] = shared
            outputs["decoder_d3"] = d3
            outputs["decoder_d2"] = d2
            outputs["decoder_d1"] = d1
            outputs["decoder_df"] = df
            outputs.update(seg_aux)

        return outputs