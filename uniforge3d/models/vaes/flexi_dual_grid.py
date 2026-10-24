from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules import sparse as sp
from .sparse_unet import (
    SparseResBlock3d,
    SparseConvNeXtBlock3d,

    SparseResBlockDownsample3d,
    SparseResBlockUpsample3d,
    SparseResBlockS2C3d,
    SparseResBlockC2S3d,
)
from .sparse_unet import (
    SparseUnetVaeDecoder,
)
from ...representations import Mesh
from o_voxel.convert import flexible_dual_grid_to_mesh


class FlexiDualGridVaeDecoder(SparseUnetVaeDecoder):
    def __init__(
        self,
        resolution: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List[Dict[str, Any]],
        voxel_margin: float = 0.5,
        use_fp16: bool = False,
    ):
        self.resolution = resolution
        self.voxel_margin = voxel_margin

        super().__init__(
            7,
            model_channels,
            latent_channels,
            num_blocks,
            block_type,
            up_block_type,
            block_args,
            use_fp16,
        )

    def set_resolution(self, resolution: int) -> None:
        self.resolution = resolution

    def forward(self, x: sp.SparseTensor, **kwargs):
        decoded = super().forward(x, **kwargs)
        out_list = list(decoded) if isinstance(decoded, tuple) else [decoded]
        h = out_list[0]
        vertices = h.replace((1 + 2 * self.voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - self.voxel_margin)
        intersected = h.replace(h.feats[..., 3:6] > 0)
        #quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
        quad_lerp = h.replace(F.softplus(h.feats[..., 6:7] * 1.15) * 1.1) #sharper
        print(f"[FlexiDualGridVaeDecoder] Decoded sparse tensor @ grid_size {self.resolution}")
        mesh = [Mesh(*flexible_dual_grid_to_mesh(
            v.coords[:, 1:], v.feats, i.feats, q.feats,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            grid_size=self.resolution,
            train=True
        )) for v, i, q in zip(vertices, intersected, quad_lerp)]
        out_list[0] = mesh
        return out_list[0] if len(out_list) == 1 else tuple(out_list)


    def decode_from_stage(self, h: sp.SparseTensor, start_stage: int):
        """Continue from intermediate hidden features and construct the mesh."""
        assert self.pred_subdiv is True

        if not 0 <= start_stage < len(self.blocks):
            raise ValueError(
                f"Invalid start_stage={start_stage}; "
                f"decoder has {len(self.blocks)} stages"
            )

        # Continue the remaining decoder stages without applying from_latent again.
        subs = []
        for i in range(start_stage, len(self.blocks)):
            res = self.blocks[i]
            for j, block in enumerate(res):
                if i < len(self.blocks) - 1 and j == len(res) - 1:
                    h, sub = block(h)
                    subs.append(sub)
                else:
                    h = block(h)


        # Match the output layer's dtype before applying its linear projection.
        output_dtype = self.output_layer.weight.dtype
        normalized = F.layer_norm(
            h.feats.float(),
            h.feats.shape[-1:],
        )
        h = h.replace(normalized.to(dtype=output_dtype).contiguous())
        h = self.output_layer(h)


        # Reuse the same seven-channel-to-mesh conversion as forward().
        vertices = h.replace(
            (1 + 2 * self.voxel_margin) * F.sigmoid(h.feats[..., 0:3])
            - self.voxel_margin
        )
        intersected = h.replace(h.feats[..., 3:6] > 0)
        quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))

        meshes = [
            Mesh(*flexible_dual_grid_to_mesh(
                v.coords[:, 1:],
                v.feats,
                i.feats,
                q.feats,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                grid_size=self.resolution,
                train=False,
            ))
            for v, i, q in zip(vertices, intersected, quad_lerp)
        ]

        return meshes, subs
