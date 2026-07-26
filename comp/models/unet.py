"""
U-Net Encoder-Decoder baseline for DAPI -> Marker image translation.

Architecture:
  - ResNet-style blocks with GroupNorm
  - 5-level encoder-decoder with skip connections
  - Configurable base channels and depth
  - Optional attention gates in decoder
  - Supports multi-marker output via separate decoder heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block: Conv -> GN -> ReLU -> Conv -> GN + residual."""

    def __init__(self, ch: int, gn_groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(gn_groups, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(gn_groups, ch)

    def forward(self, x):
        r = x
        x = F.relu(self.norm1(self.conv1(x)), inplace=True)
        x = self.norm2(self.conv2(x))
        return F.relu(r + x, inplace=True)


class EncoderBlock(nn.Module):
    """Encoder level: ResBlock(s) + Downsample."""

    def __init__(self, in_ch: int, out_ch: int, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            *[ResBlock(out_ch) for _ in range(num_blocks)],
        )
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1, bias=False)

    def forward(self, x):
        feats = self.blocks(x)
        return self.down(feats), feats


class DecoderBlock(nn.Module):
    """Decoder level: Upsample + Concat(skip) + ResBlock(s)."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, num_blocks: int = 2):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        )
        self.blocks = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            *[ResBlock(out_ch) for _ in range(num_blocks)],
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if sizes mismatch
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.blocks(x)


class AttentionGate(nn.Module):
    """
    Attention gate: filters skip connections based on decoder signal.
    g: decoder signal (gate), x: encoder skip
    """

    def __init__(self, g_ch: int, x_ch: int, inter_ch: int = None):
        super().__init__()
        if inter_ch is None:
            inter_ch = x_ch // 2
        self.W_g = nn.Conv2d(g_ch, inter_ch, 1, bias=False)
        self.W_x = nn.Conv2d(x_ch, inter_ch, 1, bias=False)
        self.psi = nn.Conv2d(inter_ch, 1, 1, bias=False)

    def forward(self, x, g):
        # Resize gate to match skip spatial dims
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=True)
        attn = F.relu(self.W_g(g) + self.W_x(x), inplace=True)
        attn = torch.sigmoid(self.psi(attn))
        return x * attn


class UNet(nn.Module):
    """
    U-Net for grayscale image-to-image translation.

    Args:
        in_channels:  input channels (1 for grayscale DAPI)
        out_channels: output channels (1 for single marker, N for multi)
        base_channels: channels at first encoder level
        num_blocks: ResBlocks per level
        use_attention: add attention gates in decoder skip connections
        num_markers: if >1, use separate decoder heads for each marker
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        num_blocks: int = 2,
        use_attention: bool = False,
        num_markers: int = 1,
    ):
        super().__init__()
        self.num_markers = num_markers
        self.use_attention = use_attention

        ch = base_channels
        chs = [ch, ch * 2, ch * 4, ch * 8, ch * 8]

        # Shared encoder
        self.enc1 = EncoderBlock(in_channels, chs[0], num_blocks)
        self.enc2 = EncoderBlock(chs[0], chs[1], num_blocks)
        self.enc3 = EncoderBlock(chs[1], chs[2], num_blocks)
        self.enc4 = EncoderBlock(chs[2], chs[3], num_blocks)
        self.enc5 = EncoderBlock(chs[3], chs[4], num_blocks)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(chs[4]),
            ResBlock(chs[4]),
        )

        # Decoder — single head (num_markers=1) or multi-head
        self.decoders = nn.ModuleList()
        for _ in range(num_markers):
            dec = self._make_decoder(chs)
            self.decoders.append(dec)

    def _make_decoder(self, chs):
        dec = nn.ModuleDict({
            "dec4": DecoderBlock(chs[4], chs[3], chs[3]),
            "dec3": DecoderBlock(chs[3], chs[2], chs[2]),
            "dec2": DecoderBlock(chs[2], chs[1], chs[1]),
            "dec1": DecoderBlock(chs[1], chs[0], chs[0]),
            "final": nn.Sequential(
                nn.Conv2d(chs[0], chs[0], 3, padding=1, bias=False),
                nn.GroupNorm(8, chs[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(chs[0], 1, 1),
                nn.Tanh(),
            ),
        })
        if self.use_attention:
            # g_ch = prev decoder output channels, x_ch = skip channels
            dec["attn4"] = AttentionGate(chs[4], chs[3])   # g:512, x:512
            dec["attn3"] = AttentionGate(chs[3], chs[2])   # g:512, x:256
            dec["attn2"] = AttentionGate(chs[2], chs[1])   # g:256, x:128
            dec["attn1"] = AttentionGate(chs[1], chs[0])   # g:128, x:64
        return dec

    def _encode(self, x):
        """Shared encoder forward. Returns list of skip features."""
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)
        x, s5 = self.enc5(x)
        x = self.bottleneck(x)
        return x, [s5, s4, s3, s2, s1]

    def _decode(self, x, skips, decoder):
        """Single decoder forward."""
        s5, s4, s3, s2, s1 = skips

        x = decoder["dec4"](x, s4 if not self.use_attention else decoder["attn4"](s4, x))
        x = decoder["dec3"](x, s3 if not self.use_attention else decoder["attn3"](s3, x))
        x = decoder["dec2"](x, s2 if not self.use_attention else decoder["attn2"](s2, x))
        x = decoder["dec1"](x, s1 if not self.use_attention else decoder["attn1"](s1, x))
        return decoder["final"](x)

    def forward(self, x, marker_idx: int = 0):
        """
        Args:
            x: (B, 1, H, W) DAPI image [-1, 1]
            marker_idx: which decoder head to use (for multi-marker)

        Returns:
            (B, 1, H, W) predicted marker image [-1, 1]
        """
        x, skips = self._encode(x)
        return self._decode(x, skips, self.decoders[marker_idx])

    def forward_all(self, x):
        """Forward all markers at once. Returns list of (B, 1, H, W)."""
        x, skips = self._encode(x)
        return [self._decode(x, skips, dec) for dec in self.decoders]
