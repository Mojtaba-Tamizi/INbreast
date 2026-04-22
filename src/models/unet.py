from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True) -> None:
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        layers.extend(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            ]
        )
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, use_batchnorm=use_batchnorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        bilinear: bool = True,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            up_out_channels = in_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            up_out_channels = in_channels // 2

        self.conv = DoubleConv(
            up_out_channels + skip_channels,
            out_channels,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        x = F.pad(
            x,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        encoder_channels: list[int] | tuple[int, ...] = (64, 128, 256, 512),
        bottleneck_channels: int = 1024,
        use_batchnorm: bool = True,
        bilinear: bool = True,
        aux_heads: dict | None = None,
    ) -> None:
        super().__init__()

        if len(encoder_channels) != 4:
            raise ValueError("For this baseline UNet, encoder_channels must have length 4.")

        aux_heads = aux_heads or {}
        self.use_boundary_head = bool(aux_heads.get("boundary", False))

        c1, c2, c3, c4 = encoder_channels

        self.stem = DoubleConv(in_channels, c1, use_batchnorm=use_batchnorm)
        self.down1 = DownBlock(c1, c2, use_batchnorm=use_batchnorm)
        self.down2 = DownBlock(c2, c3, use_batchnorm=use_batchnorm)
        self.down3 = DownBlock(c3, c4, use_batchnorm=use_batchnorm)
        self.down4 = DownBlock(c4, bottleneck_channels, use_batchnorm=use_batchnorm)

        self.up1 = UpBlock(
            in_channels=bottleneck_channels,
            skip_channels=c4,
            out_channels=c4,
            bilinear=bilinear,
            use_batchnorm=use_batchnorm,
        )
        self.up2 = UpBlock(
            in_channels=c4,
            skip_channels=c3,
            out_channels=c3,
            bilinear=bilinear,
            use_batchnorm=use_batchnorm,
        )
        self.up3 = UpBlock(
            in_channels=c3,
            skip_channels=c2,
            out_channels=c2,
            bilinear=bilinear,
            use_batchnorm=use_batchnorm,
        )
        self.up4 = UpBlock(
            in_channels=c2,
            skip_channels=c1,
            out_channels=c1,
            bilinear=bilinear,
            use_batchnorm=use_batchnorm,
        )

        self.mask_head = nn.Conv2d(c1, num_classes, kernel_size=1)

        if self.use_boundary_head:
            self.boundary_head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x1 = self.stem(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        d1 = self.up1(x5, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)

        outputs = {
            "mask": self.mask_head(d4),
        }

        if self.use_boundary_head:
            outputs["boundary"] = self.boundary_head(d4)

        return outputs