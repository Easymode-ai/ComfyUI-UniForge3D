"""Sparse shape latent sampling and mesh decoding."""

from __future__ import annotations

from typing import Any

import torch

from ..models.vaes.flexi_dual_grid import FlexiDualGridVaeDecoder
from ..modules import sparse as sp
from .samplers import FlowEulerSampler, model_predictor

from tqdm import tqdm

@torch.no_grad()
def sample_shape_latent(
    model: torch.nn.Module,
    latent_coords: torch.Tensor,
    condition: dict[str, Any],
    sampling: dict[str, Any],
    *,
    description: str = "Shape sampling",
) -> sp.SparseTensor:
    """Sample latent features at fixed coordinates using an encoded condition."""
    # Keep noise on the CPU RNG stream used by the original inference path.
    features = torch.randn(len(latent_coords), int(model.in_channels), device="cuda:0")
    features = features.to(latent_coords.device)
    noise = sp.SparseTensor(feats=features, coords=latent_coords)
    condition["cond"] = model.prepare_condition(noise, condition["cond"])
    condition["neg_cond"] = model.prepare_condition(noise, condition["neg_cond"])

    sampler = FlowEulerSampler(**sampling["args"])
    predict = model_predictor(model, condition["cond"], condition["neg_cond"])

    with torch.autocast(device_type=latent_coords.device.type, dtype=torch.bfloat16):
        return sampler.sample(
            noise,
            predict,
            **sampling["params"],
            verbose=True,
            tqdm_desc=description,
        ).samples


@torch.no_grad()
def decode_shape_latent(
    decoder: FlexiDualGridVaeDecoder | None,
    normalized_latent,
    normalization: dict[str, Any] | None,
    *,
    batch_size: int = 1024*8,
):
    print("Decoding shape latent with batch size:", batch_size)
    """Denormalize and batch-decode latent features returning a sparse latent tensor."""
    value = normalized_latent
    if normalization is not None:
        mean = torch.tensor(normalization["mean"], dtype=torch.float32, device=value.device)
        std = torch.tensor(normalization["std"], dtype=torch.float32, device=value.device)
        mean = mean.view(1, -1)
        std = std.view(1, -1)
        value = value.replace(value.feats * std + mean)

    if decoder is None:
        return value

    # Match feature dtype to the decoder's first layer weights to prevent Half/Float mismatches
    if hasattr(decoder, "from_latent") and hasattr(decoder.from_latent, "weight"):
        target_dtype = decoder.from_latent.weight.dtype
        value = value.replace(value.feats.to(target_dtype))

    num_tokens = len(value.coords)
    if num_tokens <= batch_size:
        return decoder(value)

    decoded_feats_list = []
    decoded_coords_list = []
    num_batches = (num_tokens + batch_size - 1) // batch_size

    for i in tqdm(range(0, num_tokens, batch_size), desc="Decoding latent", total=num_batches):
        chunk_coords = value.coords[i : i + batch_size]
        chunk_feats = value.feats[i : i + batch_size]
        chunk_tensor = sp.SparseTensor(feats=chunk_feats, coords=chunk_coords)
        
        chunk_out = decoder(chunk_tensor)
        if hasattr(chunk_out, "feats") and hasattr(chunk_out, "coords"):
            decoded_feats_list.append(chunk_out.feats)
            decoded_coords_list.append(chunk_out.coords)
        else:
            decoded_feats_list.append(chunk_out)

    if decoded_coords_list:
        all_feats = torch.cat(decoded_feats_list, dim=0)
        all_coords = torch.cat(decoded_coords_list, dim=0)
        return sp.SparseTensor(feats=all_feats, coords=all_coords)

    return value