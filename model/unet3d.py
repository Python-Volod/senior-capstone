"""Small two-level 3D U-Net for learning and CPU-sized patch experiments."""
import torch
from torch import nn
from torch.nn import functional as F


def _block(inputs: int, outputs: int) -> nn.Sequential:
    """Two convolutions learn features while preserving spatial dimensions."""
    # A small negative slope keeps gradients flowing in this very narrow network.
    return nn.Sequential(
        nn.Conv3d(inputs, outputs, 3, padding=1), nn.LeakyReLU(.01),
        nn.Conv3d(outputs, outputs, 3, padding=1), nn.LeakyReLU(.01),
    )


class UNet3D(nn.Module):
    """Map [B,1,D,H,W] to same-size voxel logits, without thresholding."""

    def __init__(self, base_channels: int = 4) -> None:
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be positive.")
        channels = base_channels
        # Encoder: learn details, then reduce each spatial dimension by half.
        self.encoder1 = _block(1, channels)
        self.encoder2 = _block(channels, channels * 2)
        self.pool = nn.MaxPool3d(2)
        # Bottleneck: combine features at the smallest spatial scale.
        self.bottleneck = _block(channels * 2, channels * 4)
        # Decoder: combine upsampled features with earlier encoder details.
        self.decoder2 = _block(channels * 6, channels * 2)
        self.decoder1 = _block(channels * 3, channels)
        self.output = nn.Conv3d(channels, 1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5 or image.shape[1] != 1 or min(image.shape[2:]) < 4:
            raise ValueError("Expected [B,1,D,H,W] with spatial dimensions at least 4.")
        detail1 = self.encoder1(image)
        detail2 = self.encoder2(self.pool(detail1))
        deepest = self.bottleneck(self.pool(detail2))
        # Matching explicit sizes also handles odd input dimensions.
        up2 = F.interpolate(deepest, size=detail2.shape[2:], mode="trilinear", align_corners=False)
        decoded2 = self.decoder2(torch.cat([up2, detail2], dim=1))
        up1 = F.interpolate(decoded2, size=detail1.shape[2:], mode="trilinear", align_corners=False)
        decoded1 = self.decoder1(torch.cat([up1, detail1], dim=1))
        # Raw logits work with BCEWithLogitsLoss; sigmoid is applied only for display.
        return self.output(decoded1)
