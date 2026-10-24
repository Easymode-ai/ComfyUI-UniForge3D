"""PBR sampling, decoding and baking; model placement belongs to the pipeline."""

from __future__ import annotations

from typing import Any

import torch
import trimesh

from ..utils.device import release_cuda

from ..modules.sparse import SparseTensor
from ..utils.preprocessing import InputBatch
from ..utils.texture_baking import to_glb
from .samplers import FlowEulerSampler, model_predictor
from .types import NormalizedShapeLatent


def denormalize_shape_latent(
    config: dict[str, Any],
    normalized_latent: NormalizedShapeLatent,
    device: torch.device,
) -> SparseTensor:
    """Convert normalized Shape1024 flow output to the shape VAE's latent space."""
    coords, feats = normalized_latent.coords, normalized_latent.feats
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"Expected Shape1024 coords [N,4], got {tuple(coords.shape)}")
    if feats.ndim != 2 or feats.shape[0] != coords.shape[0]:
        raise ValueError(
            f"Shape1024 feature/coordinate mismatch: feats={tuple(feats.shape)}, "
            f"coords={tuple(coords.shape)}"
        )
    spatial = coords[:, 1:]
    if spatial.numel() == 0 or int(spatial.min()) < 0 or int(spatial.max()) >= 64:
        raise ValueError("Shape1024 coordinates are outside [0, 64)")
    if not torch.isfinite(feats).all():
        raise ValueError("Shape1024 features contain NaN or Inf")
        
    normalization = config["shape_slat_normalization"]
    
    # Create mean and std directly on the same device as feats to prevent device mismatch errors
    mean = torch.tensor(normalization["mean"], dtype=torch.float32, device=feats.device)[None]
    std = torch.tensor(normalization["std"], dtype=torch.float32, device=feats.device)[None]
    
    # Denormalize in full float32 and clamp to prevent numerical overflow NaNs
    feats = feats.float() * std + mean
    feats = torch.clamp(feats, min=-1e4, max=1e4)
    
    return SparseTensor(feats=feats, coords=coords.int()).to(device)

@torch.no_grad()
def sample_texture(
    model: torch.nn.Module,
    config: dict[str, Any],
    conditioner,
    shape_vae_latent: SparseTensor,
    batch: InputBatch,
    dino_features: torch.Tensor,
    view_assignment: dict[str, torch.Tensor],
    device: torch.device,
) -> SparseTensor:
    shape_norm = config["shape_slat_normalization"]
    mean = torch.tensor(shape_norm["mean"], device=device)[None]
    std = torch.tensor(shape_norm["std"], device=device)[None]
    

    # Preserve the original VAE -> normalized flow conversion and its rounding.
    shape_condition = shape_vae_latent.replace((shape_vae_latent.feats - mean) / std)
    
    coords = shape_condition.coords[:, 1:].to(dtype=torch.int32)


    
    if coords.numel() == 0 or int(coords.min()) < 0 or int(coords.max()) >= 64:
        raise RuntimeError("Shape1024 latent does not contain valid coords64")

    condition = conditioner.encode(
        model.adapter,
        dino_features=dino_features.to(device),
        vggt_images=batch.vggt_images.to(device),
        structure_conditioned_coords=shape_condition.coords,
        input_masks=batch.input_masks.to(device),
        view_assignment=view_assignment,
    )
    noise_channels = model.in_channels - shape_vae_latent.feats.shape[1]
    noise = shape_condition.replace(
        torch.randn(len(coords), noise_channels, device="cpu").to(coords.device)
    )
    condition["cond"] = model.prepare_condition(noise, condition["cond"])
    condition["neg_cond"] = model.prepare_condition(noise, condition["neg_cond"])
    sampling = config["sampler"]
    sampler = FlowEulerSampler(**sampling["args"])
    predict = model_predictor(
        model,
        condition["cond"],
        condition["neg_cond"],
        concat_cond=shape_condition,
    )
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        texture_slat = sampler.sample(
            noise,
            predict,
            **sampling["params"],
            verbose=True,
            tqdm_desc="PBR texture sampling",
        ).samples
    texture_norm = config["texture_slat_normalization"]
    mean = torch.tensor(texture_norm["mean"], device=device)[None]
    std = torch.tensor(texture_norm["std"], device=device)[None]
    print("Finished PBR texture sampling.")
    return texture_slat.replace(texture_slat.feats * std + mean)


@torch.no_grad()
def decode_shape_and_texture(
    shape_decoder: torch.nn.Module,
    texture_decoder: torch.nn.Module,
    shape_vae_latent: SparseTensor,
    texture_vae_latent: SparseTensor,
):
    """Decode with shared subdivisions and fill holes in the reference mesh."""
    meshes, subdivisions = shape_decoder(shape_vae_latent, return_subs=True)

    del shape_vae_latent
    print("[decode_shape_and_texture] Shape decoding complete.")
    release_cuda()
    voxels = texture_decoder(texture_vae_latent, guide_subs=subdivisions) * 0.5 + 0.5
    del texture_vae_latent
    release_cuda()
    print("[decode_shape_and_texture] Texture decoding complete.")
    meshes[0].fill_holes()
    print("[decode_shape_and_texture] Filled holes in the reference mesh.")
    return meshes[0], voxels[0]


@torch.no_grad()
def remesh_and_bake_texture(
    texture_config: dict[str, Any],
    mesh_config: dict[str, Any],
    mesh,
    pbr_voxel: SparseTensor | None = None,
    decimation_target: int = 10000,
    texture_size: int = 1024,
    uv_mode: str = "comfy_gpu",
) -> trimesh.Trimesh:
    """Remesh and optionally bake texture. If pbr_voxel is None, export without texture."""
    print("[remesh_and_bake_texture] Simplifying mesh...")
    mesh.simplify(int(mesh_config["pre_simplify_target"]))

    if pbr_voxel is None:
        # No texture: convert to trimesh and return
        return trimesh.Trimesh(
            vertices=mesh.vertices.detach().cpu().numpy(),
            faces=mesh.faces.detach().cpu().numpy(),
        )

    return to_glb(
        mesh.vertices,
        mesh.faces,
        attr_volume=pbr_voxel.feats,
        coords=pbr_voxel.coords[:, 1:],
        attr_layout={
            "base_color": slice(0, 3),
            "metallic": slice(3, 4),
            "roughness": slice(4, 5),
            "alpha": slice(5, 6),
        },
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        grid_size=int(mesh_config["grid_size"]),
        texture_size=int(texture_size),
        decimation_target=decimation_target,
        remesh=True,
        remesh_band=float(mesh_config["remesh_band"]),
        remesh_project=float(mesh_config["remesh_project"]),
        verbose=True,
        uv_mode=uv_mode,
    )
