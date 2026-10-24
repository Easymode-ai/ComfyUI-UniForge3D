"""Initialize models and load original training weights over their pretrained bases."""

import json
import re
from pathlib import Path
from typing import Any

import torch
from easydict import EasyDict as edict
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file, load_model

from . import __getattr__ as get_model_class
from .conditioning import StructuredLatentConditioner
from .dino import DinoV3Encoder
from .lora import attach_stage1_lora
from .structured_latent_flow import VGGTConditionedSLatFlowModel
from .vaes.flexi_dual_grid import FlexiDualGridVaeDecoder
from .vaes.sparse_structure import SparseStructureDecoder
from .vaes.sparse_unet import SparseUnetVaeDecoder


def resolve_checkpoint(
    checkpoint: str | Path | dict[str, str],
    root: Path,
    *,
    decoder: bool = False,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> str:
    """Resolve a local checkpoint or an explicit Hub reference."""
    if isinstance(checkpoint, (str, Path)):
        return str(root / Path(checkpoint).expanduser())
    filename = checkpoint["filename"]
    options = {
        "repo_id": checkpoint["repo_id"],
        "revision": checkpoint.get("revision"),
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }
    if decoder:
        folder = snapshot_download(
            **options,
            allow_patterns=[f"{filename}.json", f"{filename}.safetensors"],
        )
        return str(Path(folder) / filename)
    return hf_hub_download(filename=filename, **options)


def _decoder_args(checkpoint: str | Path, expected_name: str) -> dict[str, Any]:
    """Read and validate the constructor arguments beside a decoder checkpoint."""
    config_path = Path(f"{checkpoint}.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["name"] != expected_name:
        raise ValueError(
            f"Expected {expected_name} in {config_path}, found {config['name']}"
        )
    return config["args"]

def _load_decoder_weights(model: torch.nn.Module, checkpoint: str | Path) -> None:
    """Strictly load the Safetensors file beside a decoder configuration."""
    weights_path = Path(f"{checkpoint}.safetensors")
    load_model(model, weights_path, strict=True, device="cpu")
    model.eval().requires_grad_(False)

def load_shape_decoder(
    checkpoint: str | Path,
    resolution: int,
) -> FlexiDualGridVaeDecoder:
    """Construct the shape decoder and load its frozen inference weights."""
    model = FlexiDualGridVaeDecoder(
        **_decoder_args(checkpoint, "FlexiDualGridVaeDecoder")
    )
    _load_decoder_weights(model, checkpoint)
    model.set_resolution(int(resolution))
    return model

def load_sparse_structure_decoder(checkpoint: str | Path) -> SparseStructureDecoder:
    model = SparseStructureDecoder(**_decoder_args(checkpoint, "SparseStructureDecoder"))
    _load_decoder_weights(model, checkpoint)
    return model

def load_texture_decoder(checkpoint: str | Path) -> SparseUnetVaeDecoder:
    """Construct the PBR decoder and load its frozen inference weights."""
    model = SparseUnetVaeDecoder(
        **_decoder_args(checkpoint, "SparseUnetVaeDecoder")
    )
    _load_decoder_weights(model, checkpoint)
    return model

def convert_checkpoint_to_safetensors(input_path: str, output_path: str):

    from pathlib import Path
    from safetensors.torch import save_file

    if not Path(input_path).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {input_path}")

    state = torch.load(input_path, map_location="cpu", weights_only=True)
    
    if "model" in state:
        state_dict = state["model"]
    elif "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state
        
    cleaned_state_dict = {
        k: t.contiguous().cpu() for k, t in state_dict.items() if isinstance(t, torch.Tensor)
    }
            
    save_file(cleaned_state_dict, output_path)
    print(f"Converted {input_path} -> {output_path}")

def _read_weights(checkpoint: str) -> dict[str, torch.Tensor]:
    checkpoint_str = str(checkpoint)
    
    # 1. If it's explicitly a safetensors file
    if checkpoint_str.endswith(".safetensors"):
        return load_file(checkpoint_str, device="cuda:0")
    
    # 2. Try standard PyTorch load first
    try:
        state = torch.load(checkpoint_str, map_location="cuda:0", mmap=False, weights_only=False)
    except Exception:
        # 3. Fallback: .pt file is actually a safetensors file under the hood
        try:
            state = load_file(checkpoint_str, device="cuda:0")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load weights from {checkpoint_str}. "
                f"Ensure it's a valid PyTorch or Safetensors file, or check if it's an incomplete Git LFS pointer. Error: {e}"
            )

    return state["model"] if isinstance(state, dict) and "model" in state else state

def _load_trainable_weights(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """A compact checkpoint must cover every trainable parameter over its loaded base."""
    print(f"Loading trainable weights for model: {model.__class__.__name__}")
    required = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing = required - state.keys()
    if missing:
        raise RuntimeError(f"Missing training weights: {sorted(missing)[:20]}")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected training weights: {result.unexpected_keys[:20]}")


def _load_structure_weights(model: torch.nn.Module, weights: dict) -> None:
    # Bases are loaded into FP32 modules, just as in the training constructor.
    dit = _read_weights(weights["dit_base"])
    # RoPE phases are deterministic buffers, absent from the official base file.
    dit.setdefault("rope_phases", model.dit.rope_phases)
    model.dit.load_state_dict(dit, strict=True)
    vggt = _read_weights(weights["vggt_base"])
    vggt = {name.removeprefix("model."): value for name, value in vggt.items()}
    # The published model omits the unused camera head and mask token.
    vggt = {
        name: value for name, value in vggt.items()
        if not name.startswith("camera_head.") and name != "aggregator.patch_embed.mask_token"
    }
    model.vggt.load_state_dict(vggt, strict=True)
    attach_stage1_lora(model.dit, **weights["dit_lora"])
    attach_stage1_lora(model.vggt, **weights["vggt_lora"])

    state = _read_weights(weights["checkpoint"])
    state = {
        re.sub(r"^fusion_groups\.(\d+)\.f_from_z\.", r"dit_to_vggt.\1.", name): value
        for name, value in state.items()
        if not re.match(r"^fusion_groups\.\d+\.z_from_f\.", name)
    }
    _load_trainable_weights(model, state)
    model.eval().requires_grad_(False)


def _load_flow_model(spec: edict, weights: dict):
    if spec.name != "VGGTConditionedSLatFlowModel":
        raise ValueError(f"Unsupported flow model: {spec.name}")
    model = VGGTConditionedSLatFlowModel(**spec.args)
    from peft import LoraConfig, get_peft_model

    if "base_checkpoint" in weights:
        model.denoiser.load_state_dict(_read_weights(weights["base_checkpoint"]), strict=True)
    model.denoiser.requires_grad_(False)
    model.denoiser = get_peft_model(model.denoiser, LoraConfig(**weights["lora"]))
    state = _read_weights(weights["checkpoint"])
    if "base_checkpoint" in weights:
        _load_trainable_weights(model.denoiser, state)
    else:
        model.denoiser.load_state_dict(state, strict=True)
    model.adapter.load_state_dict(_read_weights(weights["adapter_checkpoint"]), strict=True)
    model.eval().requires_grad_(False)
    return model


def load_models(
    config: dict[str, Any], *, cache_dir: str | Path | None = None,
    local_files_only: bool = False, models_to_load: list[str] | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Load only specified models. If models_to_load is None, load all.
    If device is specified, load directly to that device (no CPU)."""

    # If no filter specified, load all
    if models_to_load is None:
        models_to_load = [
            "dino", "conditioner",
            "structure_model", "structure_decoder",
            "shape512_model", "shape512_decoder",
            "shape1024_model", "shape1024_decoder",
            "texture_model", "texture_decoder",
        ]

    result = {}

    if "dino" in models_to_load:
        print("Loading DINO model...")
        dino = DinoV3Encoder(
            config["external_models"]["dino"],
            cache_dir=cache_dir, local_files_only=local_files_only,
        ).eval().requires_grad_(False)
        if device is not None:
            dino.to(device)
        result["dino"] = dino

    if "conditioner" in models_to_load:
        conditioner = StructuredLatentConditioner(**config["conditioner"])
        if device is not None:
            conditioner.to(device)
        result["conditioner"] = conditioner

    if "structure_model" in models_to_load or "structure_decoder" in models_to_load:
        structure = edict(config["sparse_structure"])
        if "structure_model" in models_to_load:
            spec = structure.model
            structure_model = get_model_class(spec.name)(**spec.args)
            _load_structure_weights(structure_model, structure.weights)
            if device is not None:
                structure_model.to(device)
                # Verify model is on device
                for param in structure_model.parameters():
                    assert str(param.device).startswith(str(device)), f"Model not on {device}: {param.device}"
            result["structure_model"] = structure_model
        if "structure_decoder" in models_to_load:
            structure_decoder = load_sparse_structure_decoder(structure.decoder.checkpoint)
            if device is not None:
                structure_decoder.to(device)
                # Verify decoder is on device
                for param in structure_decoder.parameters():
                    assert str(param.device).startswith(str(device)), f"Decoder not on {device}: {param.device}"
            result["structure_decoder"] = structure_decoder

    if "shape512_model" in models_to_load or "shape512_decoder" in models_to_load:
        shape512 = edict(config["structured_latent_512"])
        if "shape512_model" in models_to_load:
            shape512_model = _load_flow_model(shape512.model, shape512.weights)
            if device is not None:
                shape512_model.to(device)
            result["shape512_model"] = shape512_model
        if "shape512_decoder" in models_to_load:
            shape512_decoder = load_shape_decoder(
                shape512.decoder.checkpoint, int(shape512.decoder.resolution),
            )
            if device is not None:
                shape512_decoder.to(device)
            result["shape512_decoder"] = shape512_decoder

    if "shape1024_model" in models_to_load or "shape1024_decoder" in models_to_load or "texture_model" in models_to_load or "texture_decoder" in models_to_load:
        shape1024 = edict(config["structured_latent_1024"])
        texture = edict(config["texture"])
        if "shape1024_model" in models_to_load:
            shape1024_model = _load_flow_model(shape1024.model, shape1024.weights)
            if device is not None:
                shape1024_model.to(device)
            result["shape1024_model"] = shape1024_model
        if "texture_model" in models_to_load:
            texture_model = _load_flow_model(texture.model, texture.weights)
            if device is not None:
                texture_model.to(device)
            result["texture_model"] = texture_model
        if "shape1024_decoder" in models_to_load:
            shape1024_decoder = load_shape_decoder(
                texture.shape_decoder_checkpoint, 1024
            )
            if device is not None:
                shape1024_decoder.to(device)
            result["shape1024_decoder"] = shape1024_decoder
        if "texture_decoder" in models_to_load:
            texture_decoder = load_texture_decoder(texture.texture_decoder_checkpoint)
            if device is not None:
                texture_decoder.to(device)
            result["texture_decoder"] = texture_decoder

    return result
