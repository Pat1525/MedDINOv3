"""
Primus_Multiscale_3D: slice-wise DINOv3 + 3D transposed-conv decoder.

Architecture:
    Input [B, 1, D, H, W]
        --> fold D into batch
    [B*D, 1, H, W] --> repeat to 3 channels --> [B*D, 3, H, W]
        --> frozen DINOv3 ViT-B/16, multi-scale features
    list of 4 tensors [B*D, embed_dim, H/16, W/16]
        --> concat along channel
    [B*D, embed_dim*4, H/16, W/16]
        --> unfold D from batch
    [B, embed_dim*4, D, H/16, W/16]
        --> 3D transposed-conv decoder (upsamples H, W by 16; D unchanged)
    [B, num_classes, D, H, W]

Notes:
    - DINOv3 is frozen (parameters.requires_grad = False, eval mode).
    - Single input channel only (DCE-MRI multi-phase deferred).
    - Deep supervision returns a single output (matches MedDINOv3's 2D baseline).
"""

from typing import List, Tuple, Union

import torch
import torch.nn as nn


class LayerNormNd(nn.Module):
    """LayerNorm over channel dim for tensors of shape [B, C, ...]."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W] -> [B, D, H, W, C] -> norm -> back
        dims = list(range(x.dim()))
        # move channel from 1 to last
        x = x.permute(0, *dims[2:], 1)
        x = self.norm(x)
        # move channel back to position 1
        x = x.permute(0, -1, *range(1, x.dim() - 1))
        return x


class InitWeights_He:
    """He initialization, matching nnU-Net / MedDINOv3 convention."""

    def __init__(self, neg_slope: float = 1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module: nn.Module):
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, a=self.neg_slope, nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class PatchDecode3D(nn.Module):
    """
    3D transposed-conv decoder that upsamples H and W by `patch_embed_size`,
    while leaving D unchanged. Mirrors the 2D PatchDecode design.

    patch_embed_size is a tuple (pD, pH, pW). For our slice-wise setup pD=1
    and pH=pW=16, requiring 4 stages of 2x upsampling in H and W.
    """

    def __init__(
        self,
        patch_embed_size: Tuple[int, int, int],
        in_channels: int,
        num_classes: int,
        norm=LayerNormNd,
        activation=nn.GELU,
    ):
        super().__init__()
        pD, pH, pW = patch_embed_size
        assert pD == 1, f"Expected pD=1 for slice-wise model, got {pD}"
        assert pH == pW, f"Expected square in-plane patch, got pH={pH}, pW={pW}"

        # number of 2x upsampling stages needed in H and W
        import math
        n_stages = int(math.log2(pH))
        assert 2 ** n_stages == pH, f"patch_embed_size in-plane must be a power of 2, got {pH}"

        # progressive channel reduction: in_channels -> in/2 -> in/4 -> ... -> base
        # at each stage, halve the channels until we hit num_classes territory
        layers = []
        c_in = in_channels
        for i in range(n_stages):
            c_out = max(c_in // 2, num_classes * 2)
            # final stage outputs to num_classes
            if i == n_stages - 1:
                c_out = num_classes

            layers.append(
                nn.ConvTranspose3d(
                    c_in,
                    c_out,
                    kernel_size=(1, 2, 2),
                    stride=(1, 2, 2),
                )
            )
            # don't norm/activate the final classification output
            if i < n_stages - 1:
                layers.append(norm(c_out))
                layers.append(activation())
            c_in = c_out

        self.up = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H/16, W/16] -> [B, num_classes, D, H, W]
        return self.up(x)


class Primus_Multiscale_3D(nn.Module):
    """
    3D-aware port of MedDINOv3's Primus_Multiscale.

    Processes 3D volumes by running a frozen 2D DINOv3 ViT slice-wise,
    concatenating multi-scale features, then decoding with a 3D
    transposed-conv head.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: Union[int, Tuple[int, int, int]],
        num_classes: int,
        num_input_channels: int = 1,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder: nn.Module = None,
        interaction_indices: List[int] = (2, 5, 8, 11),
    ):
        super().__init__()

        # Normalize patch_embed_size to a tuple
        if isinstance(patch_embed_size, int):
            patch_embed_size = (1, patch_embed_size, patch_embed_size)
        elif len(patch_embed_size) == 2:
            patch_embed_size = (1, *patch_embed_size)

        assert num_input_channels == 1, (
            f"Baseline supports num_input_channels=1 only; got {num_input_channels}. "
            f"Multi-phase DCE-MRI support is deferred to a follow-up."
        )

        self.num_input_channels = num_input_channels
        self.interaction_indices = list(interaction_indices)
        self.dino_encoder = dino_encoder

        # Freeze DINOv3 entirely
        for p in self.dino_encoder.parameters():
            p.requires_grad = False
        self.dino_encoder.eval()

        self.up_projection = PatchDecode3D(
            patch_embed_size=patch_embed_size,
            in_channels=embed_dim * len(self.interaction_indices),
            num_classes=num_classes,
            norm=decoder_norm,
            activation=decoder_act,
        )
        self.up_projection.apply(InitWeights_He(1e-2))

    def train(self, mode: bool = True):
        """Override train() to keep DINOv3 in eval mode always (frozen BN/dropout)."""
        super().train(mode)
        self.dino_encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W]
        assert x.dim() == 5, f"Expected 5D input [B, C, D, H, W], got shape {x.shape}"
        B, C, D, H, W = x.shape
        assert C == 1, f"Expected single-channel input, got C={C}"
        assert H % 16 == 0 and W % 16 == 0, (
            f"H and W must be divisible by 16 (DINOv3 patch size); got H={H}, W={W}"
        )

        # Fold depth into batch -> slice-wise 2D
        # [B, C, D, H, W] -> [B, D, C, H, W] -> [B*D, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

        # DINOv3 expects 3-channel RGB input
        x = x.repeat(1, 3, 1, 1)  # [B*D, 3, H, W]

        # Multi-scale features from frozen DINOv3
        # With reshape=True, each element is [B*D, embed_dim, H/16, W/16]
        with torch.no_grad():
            hier = self.dino_encoder.get_intermediate_layers(
                x, n=self.interaction_indices, reshape=True
            )
        hier = torch.cat(hier, dim=1)  # [B*D, embed_dim * n_layers, H/16, W/16]

        # Unfold depth back: [B*D, C', H', W'] -> [B, D, C', H', W'] -> [B, C', D, H', W']
        C_feat = hier.shape[1]
        Hp, Wp = hier.shape[2], hier.shape[3]
        hier = hier.reshape(B, D, C_feat, Hp, Wp).permute(0, 2, 1, 3, 4).contiguous()

        # 3D transposed-conv decoder
        out = self.up_projection(hier)  # [B, num_classes, D, H, W]
        return out
