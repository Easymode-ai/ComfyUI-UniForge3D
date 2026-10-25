from random import seed
import sys
import os
from pathlib import Path

import numpy as np
from PIL import Image
import tempfile

import torch
import trimesh

from .uniforge3d.pipelines.samplers.flow_euler import DEFAULT_SAMPLER_METHOD
from .uniforge3d.utils import mesh


from .uniforge3d import UniForge3DImageTo3DPipeline

from .uniforge3d.utils.device import release_cuda
from .uniforge3d.pipelines.texture import denormalize_shape_latent
from .uniforge3d.pipelines.sparse_structure import denoise_sparse_coords
from .uniforge3d.pipelines.types import SparseStructure


class UniForge3DModelLoader:
    """Load and cache UniForge3D model pipeline."""

    def __init__(self):
        self.cached_pipeline = None
        self.cached_config = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "xibushu/UniForge3D"}),
                "device_id": ("INT", {"default": 0, "min": 0, "max": 8}),
                "low_vram": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1})
            }
        }

    RETURN_TYPES = ("UNIFORGE3D_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "3D/UniForge3D"

    def load_model(self, model_name, device_id, low_vram, seed):
        current_config = (model_name, device_id, low_vram)

        if self.cached_pipeline is None or self.cached_config != current_config:
            checkpoints_dir = Path(__file__).parent / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(checkpoints_dir)
            os.environ["HF_HUB_CACHE"] = str(checkpoints_dir)
            os.environ["TMPDIR"] = str(checkpoints_dir / "tmp")
            os.environ["TEMP"] = str(checkpoints_dir / "tmp")
            (checkpoints_dir / "tmp").mkdir(parents=True, exist_ok=True)

            self.cached_pipeline = UniForge3DImageTo3DPipeline.from_pretrained(
                model_name,
                device=device_id,
                low_vram=low_vram,
                cache_dir=checkpoints_dir,
            )
            self.cached_config = current_config

            torch.cuda.set_device(device_id)
            torch.manual_seed(seed)

        return (self.cached_pipeline,)


class UniForge3DPreprocessImages:
    """Stage 0: Preprocesses input images into tensor batches."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("UNIFORGE3D_MODEL",),
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("UNIFORGE3D_BATCH",)
    RETURN_NAMES = ("batch",)
    FUNCTION = "preprocess"
    CATEGORY = "3D/UniForge3D/Stages"

    def preprocess(self, model, images):
        checkpoints_dir = Path(__file__).parent / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()

        pil_images = []
        for i in range(images.shape[0]):
            img_np = images[i]
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

            if img_np.shape[2] == 4:
                pil_img = Image.fromarray(img_np, 'RGBA')
            elif img_np.shape[2] == 3:
                pil_img = Image.fromarray(img_np, 'RGB')
            else:
                pil_img = Image.fromarray(img_np[:, :, 0], 'L')
            pil_images.append(pil_img)

        temp_img_dir = checkpoints_dir / "temp_images"
        temp_img_dir.mkdir(parents=True, exist_ok=True)
        image_paths = []
        for i, pil_img in enumerate(pil_images):
            temp_img_path = temp_img_dir / f"input_{i}.png"
            pil_img.save(temp_img_path)
            image_paths.append(str(temp_img_path))

        batch = model._preprocess_images(image_paths)
        return (batch,)


class UniForge3DEncodeDino512:
    """Stage 1: Encodes DINO 512 features."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("UNIFORGE3D_MODEL",),
                "batch": ("UNIFORGE3D_BATCH",),
            }
        }

    RETURN_TYPES = ("DINO_FEATURES_512",)
    RETURN_NAMES = ("dino_features_512",)
    FUNCTION = "encode"
    CATEGORY = "3D/UniForge3D/Stages"

    @torch.no_grad()
    def encode(self, model, batch):
        torch.cuda.set_device(model.device)
        print("[UniForge3D] Stage 1/5: Encoding DINO features (512)...")
        dino_features_512 = model._encode_dino_512(batch)
        release_cuda()
        return (dino_features_512,)

class UniForge3DEncodeDino1024:
    """Stage 1: Encodes DINO 1024 features."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("UNIFORGE3D_MODEL",),
                "batch": ("UNIFORGE3D_BATCH",),
            }
        }

    RETURN_TYPES = ("DINO_FEATURES_1024",)
    RETURN_NAMES = ("dino_features_1024",)
    FUNCTION = "encode"
    CATEGORY = "3D/UniForge3D/Stages"

    @torch.no_grad()
    def encode(self, model, batch):
        torch.cuda.set_device(model.device)
        print("[UniForge3D] Stage 1/5: Encoding DINO features (1024)...")
        dino_features_1024 = model._encode_dino_1024(batch)
        release_cuda()
        return (dino_features_1024,)

class UniForge3DSparseStructure:
    """Stage 2: Generates and optionally denoises the sparse structure with full configuration parameters."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("UNIFORGE3D_MODEL",),
                "batch": ("UNIFORGE3D_BATCH",),
                "dino_features_512": ("DINO_FEATURES_512",),
                "denoise_sparse": ("BOOLEAN", {"default": True}),
                "denoise_iterations": ("INT", {"default": 1, "min": 1, "max": 10}),
                "structure_steps": ("INT", {"default": 12, "min": 1, "max": 200}),
                "structure_guidance_strength": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "structure_guidance_interval_start": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "structure_guidance_interval_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "structure_guidance_rescale": ("FLOAT", {"default": 0.7, "min": -1.0, "max": 1.0, "step": 0.01}),
                "structure_rescale_t": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "structure_sigma_min": ("FLOAT", {"default": 1e-05, "min": 1e-6, "max": 0.1, "step": 1e-6}),
                "sampler_method": (["euler", "rk4"], {"default": DEFAULT_SAMPLER_METHOD}),

            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR",)
    RETURN_NAMES = ("sparse_structure_32", "sparse_structure_64")
    FUNCTION = "generate_sparse"
    CATEGORY = "3D/UniForge3D/Stages"

    @torch.no_grad()
    def generate_sparse(
        self, model, batch, dino_features_512, denoise_sparse,
        denoise_iterations, structure_steps, structure_guidance_strength,
        structure_guidance_interval_start, structure_guidance_interval_end,
        structure_guidance_rescale, structure_rescale_t, structure_sigma_min, sampler_method, 
    ):
        torch.cuda.set_device(model.device)
        verbose = True
        # Configure sampling params dynamically matching the full config structure
        params = model.config["sparse_structure"]["sampler"]["params"]
        params["steps"] = int(structure_steps)
        params["guidance_strength"] = float(structure_guidance_strength)
        params["guidance_interval"] = [float(structure_guidance_interval_start), float(structure_guidance_interval_end)]
        params["guidance_rescale"] = float(structure_guidance_rescale)
        params["rescale_t"] = float(structure_rescale_t)
        
        model.config["sparse_structure"]["sampler"]["args"]["sigma_min"] = float(structure_sigma_min)
      
        print("[UniForge3D] Stage 2/5: Generating sparse structure...")
        sparse_structure = model._generate_structure(batch, dino_features_512, sampler_method=sampler_method)
        release_cache = getattr(globals(), "release_cuda", None)
        if release_cache:
            release_cache()

        if denoise_sparse:
            print("[UniForge3D] Stage 2/5: Denoising sparse structure...")

            sparse_structure = SparseStructure(
                coords32=denoise_sparse_coords(sparse_structure.coords32, iterations=denoise_iterations),
                coords64=denoise_sparse_coords(sparse_structure.coords64, iterations=denoise_iterations),
            )

        sparse_structure_32 = sparse_structure.coords32
        sparse_structure_64 = sparse_structure.coords64
        

        return (sparse_structure_32, sparse_structure_64)

from scipy.ndimage import (
    binary_closing,
    binary_erosion,
    binary_fill_holes,
    label,
)

class UniForge3DSparsePostProcess:
    """
    Post-process a sparse voxel tensor without converting it to coordinates.
    Supports batched 3D grids and optional channel dimension.
    """

    ALGORITHMS = [
        "remove_small_holes",
        "morphological_closing",
        "flood_fill",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "coords": ("TENSOR",),
                "fill_holes": ("BOOLEAN", {"default": True}),
                "hole_fill_algorithm": (
                    cls.ALGORITHMS,
                    {"default": "remove_small_holes"},
                ),
                "hole_structure": (
                    "INT",
                    {"default": 1, "min": 1, "max": 15, "step": 1},
                ),
                "hole_iterations": (
                    "INT",
                    {"default": 1, "min": 1, "max": 10, "step": 1},
                ),
                "keep_only_shell": ("BOOLEAN", {"default": False}),
                "verbose": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("coords",)
    FUNCTION = "process"
    CATEGORY = "3D/Sparse"

    def process(
        self,
        coords,
        fill_holes=True,
        hole_fill_algorithm="remove_small_holes",
        hole_structure=1,
        hole_iterations=1,
        keep_only_shell=False,
        verbose=False,
    ):
        if not isinstance(coords, torch.Tensor):
            raise TypeError("coords must be a torch.Tensor")

        # Sparse coordinate input: [N,4] = [batch,z,y,x].
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError(
                f"Expected sparse coordinates [N,4], got {tuple(coords.shape)}"
            )

        if coords.numel() == 0:
            return (coords,)

        original_device = coords.device
        original_dtype = coords.dtype

        # Coordinates are integer voxel indices, not a dense mask.
        points = coords.detach().to(device="cpu", dtype=torch.int64).numpy()

        if np.any(points < 0):
            raise ValueError("Voxel coordinates must be non-negative.")

        if not fill_holes and not keep_only_shell:
            return (coords,)

        batch_ids = np.unique(points[:, 0])
        output_points = []

        structure_size = max(1, int(hole_structure))
        iterations = max(1, int(hole_iterations))
        structure = np.ones(
            (structure_size,) * 3,
            dtype=bool,
        )

        for batch_id in batch_ids:
            batch_points = points[points[:, 0] == batch_id, 1:4]

            # Preserve the coordinate system. Use the existing grid extent.
            # Padding allows closing/filling to operate at the boundaries.
            maximum = batch_points.max(axis=0)
            shape = tuple((maximum + 2).tolist())

            volume = np.zeros(shape, dtype=bool)
            volume[
                batch_points[:, 0],
                batch_points[:, 1],
                batch_points[:, 2],
            ] = True

            if fill_holes:
                if hole_fill_algorithm == "morphological_closing":
                    volume = binary_closing(
                        volume,
                        structure=structure,
                        iterations=iterations,
                    )

                elif hole_fill_algorithm == "flood_fill":
                    volume = binary_closing(
                        volume,
                        structure=structure,
                        iterations=iterations,
                    )
                    volume = binary_fill_holes(volume)

                    labels, count = label(volume)
                    if count:
                        sizes = np.bincount(labels.ravel())
                        sizes[0] = 0
                        volume = labels == sizes.argmax()

                elif hole_fill_algorithm == "remove_small_holes":
                    from skimage.morphology import remove_small_holes

                    for z in range(volume.shape[0]):
                        volume[z] = remove_small_holes(
                            volume[z],
                            max_size=max(2, structure_size ** 2),
                        )

                else:
                    raise ValueError(
                        f"Unknown algorithm: {hole_fill_algorithm}"
                    )

            if keep_only_shell:
                eroded = binary_erosion(
                    volume,
                    structure=np.ones((3, 3, 3), dtype=bool),
                    iterations=2,
                    border_value=0,
                )
                volume &= ~eroded

            # Convert the processed dense volume back into sparse coordinates.
            voxel_indices = np.argwhere(volume)

            if len(voxel_indices):
                batch_column = np.full(
                    (len(voxel_indices), 1),
                    batch_id,
                    dtype=np.int64,
                )
                output_points.append(
                    np.concatenate([batch_column, voxel_indices], axis=1)
                )

            if verbose:
                print(
                    f"[SparsePostProcess] Batch {batch_id}: "
                    f"{len(batch_points)} input voxels -> "
                    f"{len(voxel_indices)} output voxels"
                )

        if output_points:
            result_np = np.concatenate(output_points, axis=0)
            result = torch.as_tensor(
                result_np,
                device=original_device,
                dtype=original_dtype,
            )
        else:
            result = torch.empty(
                (0, 4),
                device=original_device,
                dtype=original_dtype,
            )



        if verbose:
            print(
                f"[SparsePostProcess] {tuple(coords.shape)} -> "
                f"{tuple(result.shape)}, "
                f"algorithm={hole_fill_algorithm}, "
                f"device={result.device}"
            )

        return (result,)

class ComfyMeshDict(dict):
    """Dictionary subclass supporting both key and attribute access for ComfyUI mesh compatibility."""
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise AttributeError(f"'ComfyMeshDict' object has no attribute '{attr}'")
    def __setattr__(self, attr, value):
        self[attr] = value


class UniForge3DShape512Mesh:
    """Stage 3: Generates shape 512 latents and returns a mesh dictionary with PyTorch tensor vertices/faces for ComfyUI-Mesh-Quad-Reconstruct."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("UNIFORGE3D_MODEL",),
                "batch": ("UNIFORGE3D_BATCH",),
                "sparse_structure_32": ("TENSOR",),
                "sparse_structure_64": ("TENSOR",),
                "dino_features_512": ("DINO_FEATURES_512",),
                "sparse_structure_resolution": ("INT", {"default": 64}),
                "decode_resolution": ("INT", {"default": 1536}),
                "shape512_steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "shape512_guidance_strength": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "shape512_guidance_rescale": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "shape512_rescale_t": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "shape512_sigma_min": ("FLOAT", {"default": 1e-05, "min": 1e-6, "max": 0.1, "step": 1e-6}),
                "shape512_guidance_interval_start": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shape512_guidance_interval_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sampler_method": (["euler", "rk4"], {"default": DEFAULT_SAMPLER_METHOD}),
            }
        }

    RETURN_TYPES = ("MESH","TENSOR")
    RETURN_NAMES = ("mesh512","coords")
    FUNCTION = "generate_shape512"
    CATEGORY = "3D/UniForge3D/Stages"

    @torch.no_grad()
    def generate_shape512(
        self, model, batch, sparse_structure_32, sparse_structure_64, dino_features_512,
        sparse_structure_resolution, decode_resolution, shape512_steps, shape512_guidance_strength,shape512_guidance_rescale,
        shape512_rescale_t, shape512_sigma_min, shape512_guidance_interval_start,
        shape512_guidance_interval_end, sampler_method
    ):
        torch.cuda.set_device(model.device)
        
        params = model.config["structured_latent_512"]["sampler"]["params"]
        params["steps"] = int(shape512_steps)
        params["guidance_strength"] = float(shape512_guidance_strength)
        params["guidance_rescale"] = float(shape512_guidance_rescale)
        params["guidance_interval"] = [float(shape512_guidance_interval_start), float(shape512_guidance_interval_end)]
        params["rescale_t"] = float(shape512_rescale_t)
        
        model.config["structured_latent_512"]["sampler"]["args"]["sigma_min"] = float(shape512_sigma_min)

        print("[UniForge3D] Stage 3/5: Generating shape 512 latents (lightweight)...")
        shape512_vae_latent, mesh512 = model._generate_shape512(
            batch, sparse_structure_32, sparse_structure_64, dino_features_512,
            sparse_resolution=sparse_structure_resolution,
            lr_resolution=decode_resolution,
            sampler_method=sampler_method,
        )

     
        #structure_conditioned_coords_original = model._shape512_to_coords64(mesh512, sparse_structure_resolution)
        print("[UniForge3D] ✓ Shape512 latents generated")
        release_cuda()

        colors = mesh512.colors if hasattr(mesh512, "colors") else None
        if colors is not None:
            if hasattr(colors, "detach"):
                colors = colors.detach().cpu()
            elif hasattr(colors, "numpy"):
                colors = colors.numpy()
            colors = torch.as_tensor(colors, dtype=torch.float32)
        # Ensure vertices and faces are PyTorch tensors so downstream nodes can call .unsqueeze(0)
        vertices = mesh512.vertices
        if hasattr(vertices, "detach"):
            vertices = vertices.detach().cpu()
        elif hasattr(vertices, "numpy"):
            vertices = vertices.numpy()
        vertices = torch.as_tensor(vertices, dtype=torch.float32)

        faces = mesh512.faces
        if hasattr(faces, "detach"):
            faces = faces.detach().cpu()
        elif hasattr(faces, "numpy"):
            faces = faces.numpy()
        faces = torch.as_tensor(faces, dtype=torch.long)

        mesh_dict = ComfyMeshDict({
            "vertices": vertices,
            "faces": faces,
            "colors": colors,
            "face_counts": mesh512.face_counts if hasattr(mesh512, "face_counts") else None
        })


        coords = model._shape512_to_coords64(
                mesh512,
                sparse_structure_resolution
            )

        return (mesh_dict, coords)
    
class UniForge3DCoordinateUpsample:
    """Stage 4a: Takes in a MESH dictionary, voxelizes coordinates, and performs VRAM-safe upsampling."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("UNIFORGE3D_MODEL",),
                "mesh": ("MESH",),
                "decoder_upsample_times": ("INT", {"default": 2, "min": 1, "max": 10}),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "max_resolution": ("INT", {"default": 1536, "min": 64, "max": 4096}),
                "voxel_resolution": ("INT", {"default": 64, "min": 8, "max": 512}),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("upsampled_coords",)
    FUNCTION = "upsample_coords"
    CATEGORY = "3D/UniForge3D/Stages"

    @torch.no_grad()
    def upsample_coords(self, model, mesh, decoder_upsample_times, resolution, max_resolution, voxel_resolution):
        # Handle MESH dictionary input directly (tensor vertices & faces)
        class MeshTensorWrapper:
            def __init__(self, mesh_obj):
                if hasattr(mesh_obj, "vertices"):
                    v = mesh_obj.vertices
                elif isinstance(mesh_obj, dict) or hasattr(mesh_obj, "__getitem__"):
                    v = mesh_obj["vertices"]
                else:
                    raise TypeError("Provided MESH object has no 'vertices' attribute or key.")

                if hasattr(mesh_obj, "faces"):
                    f = mesh_obj.faces
                elif isinstance(mesh_obj, dict) or hasattr(mesh_obj, "__getitem__"):
                    f = mesh_obj["faces"]
                else:
                    raise TypeError("Provided MESH object has no 'faces' attribute or key.")

                # Safely detach and strip batch dimension if present ([1, N, 3] -> [N, 3])
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu()
                    if v.ndim == 3 and v.shape[0] == 1:
                        v = v.squeeze(0)
                elif isinstance(v, np.ndarray):
                    if v.ndim == 3 and v.shape[0] == 1:
                        v = v.squeeze(0)

                if isinstance(f, torch.Tensor):
                    f = f.detach().cpu()
                    if f.ndim == 3 and f.shape[0] == 1:
                        f = f.squeeze(0)
                elif isinstance(f, np.ndarray):
                    if f.ndim == 3 and f.shape[0] == 1:
                        f = f.squeeze(0)

                if v is None or len(v) == 0:
                    raise ValueError("[UniForge3D] Mesh vertices are empty [0, 3]. Check Stage 3 execution.")

                self.vertices = v
                self.faces = f

        mesh_wrapped = MeshTensorWrapper(mesh)
 
        structure_conditioned_coords_original = model._shape512_to_coords64(mesh_wrapped, surface_voxel_resolution=voxel_resolution)

        def interpolate_sparse_coords_densely(coords, scale_factor=2, resolution=64, max_val=1536):
            device = coords.device
            b = coords[:, 0:1]
            xyz = coords[:, 1:4].long()
            
            xyz = torch.clamp(xyz, 0, resolution - 1)
            base_xyz = xyz * scale_factor
            
            offsets = torch.stack(
                torch.meshgrid([torch.arange(scale_factor, device=device)] * 3, indexing='ij'),
                dim=-1
            ).reshape(-1, 3)
            
            expanded_xyz = base_xyz.unsqueeze(1) + offsets.unsqueeze(0)
            expanded_xyz = expanded_xyz.reshape(-1, 3)
            
            expanded_b = b.repeat_interleave(scale_factor ** 3, dim=0)
            
            new_coords = torch.cat([expanded_b, expanded_xyz], dim=1)
            new_coords[:, 1:] = torch.clamp(new_coords[:, 1:], 0, max_val - 1)
            
            return new_coords.unique(dim=0).contiguous()
        
        coords = structure_conditioned_coords_original.clone()
        coords = interpolate_sparse_coords_densely(coords, scale_factor=decoder_upsample_times, resolution=resolution, max_val=max_resolution)
 
        return (coords,)


class UniForge3DUpsampleSparse:
    """Stage 4b: Takes in a sparse tensor, extracts coordinates, and performs VRAM-safe upsampling."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("UNIFORGE3D_MODEL",),
                "sparse_64": ("TENSOR",),
                "decoder_upsample_times": ("INT", {"default": 2, "min": 1, "max": 10}),
                "resolution": ("INT", {"default": 64, "min": 32, "max": 512}),
                "max_resolution": ("INT", {"default": 1536, "min": 32, "max": 4096}),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("upsample_sparses",)
    FUNCTION = "upsample_sparses"
    CATEGORY = "3D/UniForge3D/Stages"

    @torch.no_grad()
    def upsample_sparses(self, model, sparse_64, decoder_upsample_times, resolution, max_resolution):
        structure_conditioned_coords_original = sparse_64

        def interpolate_sparse_coords_densely(coords, scale_factor=2, resolution=64, max_val=1536):
            device = coords.device
            b = coords[:, 0:1]
            xyz = coords[:, 1:4].long()
            
            xyz = torch.clamp(xyz, 0, resolution - 1)
            base_xyz = xyz * scale_factor
            
            offsets = torch.stack(
                torch.meshgrid([torch.arange(scale_factor, device=device)] * 3, indexing='ij'),
                dim=-1
            ).reshape(-1, 3)
            
            expanded_xyz = base_xyz.unsqueeze(1) + offsets.unsqueeze(0)
            expanded_xyz = expanded_xyz.reshape(-1, 3)
            
            expanded_b = b.repeat_interleave(scale_factor ** 3, dim=0)
            
            new_coords = torch.cat([expanded_b, expanded_xyz], dim=1)
            new_coords[:, 1:] = torch.clamp(new_coords[:, 1:], 0, max_val - 1)
            
            return new_coords.unique(dim=0).contiguous()

        coords = structure_conditioned_coords_original.clone()
        coords = interpolate_sparse_coords_densely(coords, scale_factor=decoder_upsample_times, resolution=resolution, max_val=max_resolution)
        return (coords,)

class UniForge3DShape1024:
    """Stage 4b: Shape 1024 Latent sampling and final mesh decoding with complete parameter configuration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("UNIFORGE3D_MODEL",),
                "batch": ("UNIFORGE3D_BATCH",),
                "sparse_structure": ("TENSOR",),
                "structure_conditioned_coords_original": ("TENSOR",),
                "dino_features_1024": ("DINO_FEATURES_1024",),
                "shape1024_steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "shape1024_guidance_strength": ("FLOAT", {"default": 6.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "shape1024_guidance_rescale": ("FLOAT", {"default": 0.3, "min": -1.0, "max": 1.0, "step": 0.01}),
                "shape1024_rescale_t": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "shape1024_sigma_min": ("FLOAT", {"default": 1e-05, "min": 1e-6, "max": 0.1, "step": 1e-6}),
                "shape1024_guidance_interval_start": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shape1024_guidance_interval_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sampler_method": (["euler", "rk4"], {"default": DEFAULT_SAMPLER_METHOD}),
                                        
            }
        }

    RETURN_TYPES = ("TENSOR","VIEW_ASSIGNMENT")
    RETURN_NAMES = ("shape1024_vae_latent","view_assignment")
    FUNCTION = "decode_1024"
    CATEGORY = "3D/UniForge3D/Stages"

    @torch.no_grad()
    def decode_1024(
        self, model, batch, sparse_structure, structure_conditioned_coords_original,
        dino_features_1024, shape1024_steps, shape1024_guidance_strength,
        shape1024_guidance_rescale, shape1024_rescale_t, shape1024_sigma_min,
        shape1024_guidance_interval_start, shape1024_guidance_interval_end,
        sampler_method
    ):
        torch.cuda.set_device(model.device)
        
        params = model.config["structured_latent_1024"]["sampler"]["params"]
        params["steps"] = int(shape1024_steps)
        params["guidance_strength"] = float(shape1024_guidance_strength)
        params["guidance_rescale"] = float(shape1024_guidance_rescale)
        params["rescale_t"] = float(shape1024_rescale_t)
        params["guidance_interval"] = [float(shape1024_guidance_interval_start), float(shape1024_guidance_interval_end)]
        
        model.config["structured_latent_1024"]["sampler"]["args"]["sigma_min"] = float(shape1024_sigma_min)

        print("[UniForge3D] Sampling 1024 shape using structure guidance...")
        shape1024_latent, view_assignment = model._generate_shape1024(
            batch,
            latent_coords=structure_conditioned_coords_original,
            structure_conditioned_coords=sparse_structure,
            dino_features=dino_features_1024,
            sampler_method=sampler_method,
        )

        #del dino_features_1024
        release_cuda()

        shape1024_vae_latent = denormalize_shape_latent(
            model.config["texture"],
            shape1024_latent,
            model.device,
        )
        #del shape1024_latent
        release_cuda()
        return (shape1024_vae_latent, view_assignment)


class UniForge3DVaeDecoder:
    """Stage 4b: Shape Decoder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("UNIFORGE3D_MODEL",),
                "shape1024_vae_latent": ("TENSOR",),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 18096, "step": 64}),
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "vae_decoder"
    CATEGORY = "3D/UniForge3D/Stages"

    @torch.no_grad()
    def vae_decoder(self, model, shape1024_vae_latent, resolution):
        #torch.cuda.set_device(model.device)
        shape1024_vae_latent.to(device=model.device)

        mesh = model._decode_shape_latent_1024(
            model.shape1024_decoder,
            shape1024_vae_latent,
            resolution=resolution,
        )

        print("[UniForge3D] ✓ Shape1024 mesh decoded")
        release_cuda()

        # Ensure vertices and faces are PyTorch tensors inside a ComfyMeshDict
        colors = mesh.colors if hasattr(mesh, "colors") else None
        if colors is not None:
            if hasattr(colors, "detach"):
                colors = colors.detach().cpu()
            elif hasattr(colors, "numpy"):
                colors = colors.numpy()
            colors = torch.as_tensor(colors, dtype=torch.float32)
        vertices = mesh.vertices
        if hasattr(vertices, "detach"):
            vertices = vertices.detach().cpu()
        elif hasattr(vertices, "numpy"):
            vertices = vertices.numpy()
        vertices = torch.as_tensor(vertices, dtype=torch.float32)

        faces = mesh.faces
        if hasattr(faces, "detach"):
            faces = faces.detach().cpu()
        elif hasattr(faces, "numpy"):
            faces = faces.numpy()
        faces = torch.as_tensor(faces, dtype=torch.long)

        mesh_dict = ComfyMeshDict({
            "vertices": vertices,
            "faces": faces,
            "colors": colors,
            "face_counts": mesh.face_counts if hasattr(mesh, "face_counts") else None
        })

        return (mesh_dict,)

import torch

class Uniforge3DGenerateTexture:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("UNIFORGE3D_MODEL",),
                "batch": ("UNIFORGE3D_BATCH",),
                "shape1024_vae_latent": ("TENSOR",),
                "view_assignment": ("VIEW_ASSIGNMENT",),
                "dino_features": ("DINO_FEATURES_1024",),
                "decimate_target_faces": ("INT",{"default": 100_000,"min": 1000, "max": 99999999}),
                "texture_size": ("INT",{"default": 1024,"min": 1, "max": 8192}),
                "mesh_resolution": ("INT",{"default": 1024,"min": 1, "max": 8192}),
                "uv_mode": ("STRING",{"default": "comfy_gpu", "options": ["comfy_gpu", "xatlas", "keep_existing"]})
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("textured_mesh",)
    FUNCTION = "generate_texture"
    CATEGORY = "Uniforge3D/Texture"

    def generate_texture(self, model, batch, shape1024_vae_latent, view_assignment, dino_features, decimate_target_faces, texture_size, mesh_resolution, uv_mode):
        # Replace or wire up your underlying generation method here
        textured_mesh = model._generate_texture(
            batch,
            shape1024_vae_latent,
            view_assignment,
            dino_features,
            decimate_target_faces=decimate_target_faces,
            texture_size=texture_size,
            mesh_resolution=mesh_resolution,
            uv_mode=uv_mode
        )
        return (textured_mesh,)




NODE_CLASS_MAPPINGS = {
    "UniForge3DModelLoader": UniForge3DModelLoader,
    "UniForge3DPreprocessImages": UniForge3DPreprocessImages,
    "UniForge3DEncodeDino512": UniForge3DEncodeDino512,
    "UniForge3DEncodeDino1024": UniForge3DEncodeDino1024,
    "UniForge3DSparseStructure": UniForge3DSparseStructure,
    "UniForge3DShape512Mesh": UniForge3DShape512Mesh,
    "UniForge3DCoordinateUpsample": UniForge3DCoordinateUpsample,
    "UniForge3DShape1024": UniForge3DShape1024,
    "UniForge3DUpsampleSparse": UniForge3DUpsampleSparse,
    "UniForge3DVaeDecoder": UniForge3DVaeDecoder,
    "UniForge3DSparsePostProcess": UniForge3DSparsePostProcess,
    "Uniforge3DGenerateTexture": Uniforge3DGenerateTexture

}

DISPLAY_NAME_MAPPINGS = {
    "UniForge3DModelLoader": "UniForge3D Model Loader",
    "UniForge3DPreprocessImages": "UniForge3D Preprocess Images",
    "UniForge3DEncodeDino512": "UniForge3D Encode DINO 512",
    "UniForge3DEncodeDino1024": "UniForge3D Encode DINO 1024",
    "UniForge3DSparseStructure": "UniForge3D Sparse Structure",
    "UniForge3DShape512Mesh": "UniForge3D Shape 512 & Mesh",
    "UniForge3DCoordinateUpsample": "UniForge3D Coordinate Upsample",
    "UniForge3DShape1024": "UniForge3D Shape 1024",
    "UniForge3DUpsampleSparse": "UniForge3D Upsample Sparse", 
    "UniForge3DVaeDecoder": "UniForge3D VAE Decoder",
    "UniForge3DSparsePostProcess": "UniForge3D Sparse Post Process",
    "Uniforge3DGenerateTexture": "Generate Texture (Uniforge3D)",
}