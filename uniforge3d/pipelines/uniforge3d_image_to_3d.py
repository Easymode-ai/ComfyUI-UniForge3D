

from __future__ import annotations

import json
from contextlib import ExitStack, contextmanager
from json import decoder
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
import trimesh
from huggingface_hub import snapshot_download

from .samplers.flow_euler import DEFAULT_SAMPLER_METHOD

from ..models.loaders import load_models, resolve_checkpoint
from ..utils.device import release_cuda
from ..utils.mesh import (
    display_to_model,
    dual_contour_from_occupancy,
    export_glb,
    normalize_to_cube,
    official_shape_only_postprocess,
    surface_voxelize,
)
from ..utils.preprocessing import BackgroundRemover, InputBatch, build_input_batch
from .shape import decode_shape_latent, sample_shape_latent
from .sparse_structure import decode_sparse_structure, decoded_coordinates, sample_sparse_structure
from .texture import (
    decode_shape_and_texture,
    denormalize_shape_latent,
    remesh_and_bake_texture,
    sample_texture,
)
from ..modules import sparse as sp
from ..modules.sparse.basic import SparseTensor

from .types import NormalizedShapeLatent, SparseStructure


class UniForge3DImageTo3DPipeline:


    def __init__(
        self,
        config: dict,
        *,
        device: int = 0,
        low_vram: bool = True,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.config = config
        self.device = torch.device("cuda", int(device))
        self.low_vram = low_vram
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self._currently_loaded = {}  # Currently on VRAM only

    @property
    def dino(self):
        if "dino" in self._currently_loaded:
            return self._currently_loaded["dino"]
        return "dino"

    @property
    def conditioner(self):
        if "conditioner" in self._currently_loaded:
            return self._currently_loaded["conditioner"]
        return "conditioner"

    @property
    def structure_model(self):
        if "structure_model" in self._currently_loaded:
            return self._currently_loaded["structure_model"]
        return "structure_model"

    @property
    def structure_decoder(self):
        if "structure_decoder" in self._currently_loaded:
            return self._currently_loaded["structure_decoder"]
        return "structure_decoder"

    @property
    def shape512_model(self):
        if "shape512_model" in self._currently_loaded:
            return self._currently_loaded["shape512_model"]
        return "shape512_model"

    @property
    def shape512_decoder(self):
        if "shape512_decoder" in self._currently_loaded:
            return self._currently_loaded["shape512_decoder"]
        return "shape512_decoder"

    @property
    def shape1024_model(self):
        if "shape1024_model" in self._currently_loaded:
            return self._currently_loaded["shape1024_model"]
        return "shape1024_model"

    @property
    def shape1024_decoder(self):
        if "shape1024_decoder" in self._currently_loaded:
            return self._currently_loaded["shape1024_decoder"]
        return "shape1024_decoder"

    @property
    def texture_model(self):
        if "texture_model" in self._currently_loaded:
            return self._currently_loaded["texture_model"]
        return "texture_model"

    @property
    def texture_decoder(self):
        if "texture_decoder" in self._currently_loaded:
            return self._currently_loaded["texture_decoder"]
        return "texture_decoder"

    @property
    def remove_background(self):
        return BackgroundRemover(
            self.config["image_preprocess"]["rembg_model"], self.device,
            cache_dir=self.cache_dir, local_files_only=self.local_files_only,
        )

    @classmethod
    def from_pretrained(
        cls, path: str | Path, *, device: int = 0, low_vram: bool = True,
        revision: str | None = None, cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> "UniForge3DImageTo3DPipeline":
        """Load a local model directory or a Hugging Face repository."""
        root = Path(path).expanduser()
        if not root.is_dir():
            root = Path(snapshot_download(
                str(path), revision=revision, cache_dir=cache_dir,
                local_files_only=local_files_only,
                allow_patterns=["pipeline.json", "uniforge3d/**/*.pt"],
            ))
        return cls.from_config(
            root / "pipeline.json", weights_root=root, device=device,
            low_vram=low_vram, cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

    @classmethod
    def from_config(
        cls, config_path: str | Path, *, device: int = 0, low_vram: bool = True,
        weights_root: str | Path | None = None,
        cache_dir: str | Path | None = None, local_files_only: bool = False,
    ) -> "UniForge3DImageTo3DPipeline":
        """Load a JSON config; local weights default to the repository's weights/."""
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        print(f"[UniForge3D] Loading configuration from {config_path}")
        root = (Path(weights_root).expanduser() if weights_root is not None
                else Path(__file__).resolve().parents[2] / "weights")
        options = {"cache_dir": cache_dir, "local_files_only": local_files_only}
        for stage in ("sparse_structure", "structured_latent_512", "structured_latent_1024", "texture"):
            weights = config[stage]["weights"]
            for key in ("checkpoint", "adapter_checkpoint", "dit_base", "vggt_base", "base_checkpoint"):
                if key in weights:
                    weights[key] = resolve_checkpoint(weights[key], root, **options)
            if "decoder" in config[stage]:
                decoder = config[stage]["decoder"]
                decoder["checkpoint"] = resolve_checkpoint(
                    decoder["checkpoint"], root, decoder=True, **options,
                )
        for key in ("shape_decoder_checkpoint", "texture_decoder_checkpoint"):
            config["texture"][key] = resolve_checkpoint(
                config["texture"][key], root, decoder=True, **options,
            )
        vggt = config["conditioner"]["vggt_feature_extractor"]
        vggt["checkpoint_path"] = resolve_checkpoint(vggt["checkpoint_path"], root, **options)
        return cls(config, device=device, low_vram=low_vram, **options)

    @contextmanager
    def _use_models(self, *model_names):
        """Load only needed models directly to VRAM from disk. No CPU caching."""
        string_names = []
        for item in model_names:
            if isinstance(item, str):
                string_names.append(item)

        if string_names:
            # Load ONLY the requested models directly to VRAM (no CPU intermediate)
            print(f"[UniForge3D] Loading {', '.join(string_names)} directly to {self.device}...")
            all_models = load_models(
                self.config, cache_dir=self.cache_dir, local_files_only=self.local_files_only,
                models_to_load=string_names,  # Load only these models
                device=self.device,  # Load directly to VRAM, no CPU
            )
            # Store loaded models
            for name in string_names:
                self._currently_loaded[name] = all_models[name]
            # Delete the dict to free memory
            del all_models

        try:
            yield
        finally:
            if string_names:
                print(f"[UniForge3D] Unloading {', '.join(string_names)} from VRAM...")
                for name in string_names:
                    self._currently_loaded.pop(name, None)
                if self.low_vram:
                    release_cuda()

    def _preprocess_images(self, images: Sequence[str | Path]) -> InputBatch:

        with ExitStack() as stack:
            background_loaded = False

            def remove_background(image):
                nonlocal background_loaded
                if not background_loaded:
                    stack.enter_context(self._use_models(self.remove_background.model))
                    background_loaded = True
                return self.remove_background(image)

            return build_input_batch(
                images,
                self.config["image_preprocess"],
                remove_background,
            )

    @torch.no_grad()
    def _encode_dino_512(self, batch: InputBatch) -> torch.Tensor:
        """Encode 512 resolution DINO features - returned on CPU only."""
        with self._use_models(self.dino):
            images512 = batch.dino_images512.to(self.device, dtype=torch.bfloat16)  # (1, 3, H, W)
            print(f"[UniForge3D] Encoding 512 resolution, shape: {images512.shape}")

            # Ensure DINO model is in bfloat16
            dino_model = self.dino
            dino_model.to(dtype=torch.bfloat16)


            # Encode
            features = dino_model(images512)#.cpu()  # Keep on CPU, not VRAM


            del images512
            #assert features.device.type == 'cpu', "Features must be on CPU"
            print(f"[UniForge3D] → Finished encoding 512 features, shape: {features.shape}, device: {features.device}")
            release_cuda()
            return features

    @torch.no_grad()
    def _encode_dino_1024(self, batch: InputBatch) -> torch.Tensor:
        """Encode 1024 resolution DINO features - returned on CPU only."""
        with self._use_models(self.dino):
            images1024 = batch.dino_images1024.to(self.device, dtype=torch.bfloat16)  # (1, 3, H, W)
            print(f"[UniForge3D] Encoding 1024 resolution, shape: {images1024.shape}")

            # Ensure DINO model is in bfloat16
            dino_model = self.dino
            dino_model.to(dtype=torch.bfloat16)

     
            # Encode
            features = dino_model(images1024)#.cpu()  # Keep on CPU, not VRAM
  

            del images1024
            #assert features.device.type == 'cpu', "Features must be on CPU"
            print(f"[UniForge3D] → Finished encoding 1024 features, shape: {features.shape}, device: {features.device}")
            release_cuda()
            return features

    def _generate_structure(
        self,
        batch: InputBatch,
        dino_features: torch.Tensor,
        sampler_method: str,

    ) -> SparseStructure:
        print("[UniForge3D] → Loading Structure model...")
        config = self.config["sparse_structure"]
        sampling = config["sampler"]
        with self._use_models(self.structure_model):
            dit = self.structure_model.dit
            resolution = dit.resolution
            noise = torch.randn(
                (1, dit.in_channels, resolution, resolution, resolution),
                device="cpu",
            ).to(self.device)

        
            # Create noise with actual model resolution and in_channels
            model = self.structure_model
            noise = torch.randn(
                (1, model.dit.in_channels, model.dit.resolution, model.dit.resolution, model.dit.resolution),
                device=self.device,
            )
            vggt_images = batch.vggt_images.to(self.device)
        
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            latent = sample_sparse_structure(
                model,
                noise,
                vggt_images,
                dino_features.to(self.device),
                **sampling["args"],
                **sampling["params"],
                sampler_method=sampler_method,
            )
        with self._use_models(self.structure_decoder):
            occupancy_logits = decode_sparse_structure(
                self.structure_decoder,
                latent,
                config["decoder"].get("normalization"),
            )

        coords32, coords64 = decoded_coordinates(occupancy_logits)
        return SparseStructure(coords32.cpu(), coords64.cpu())

        
    def _sample_shape(
        self,
        model_name: str,
        config: dict,
        batch: InputBatch,
        *,
        latent_coords: torch.Tensor,
        structure_conditioned_coords: torch.Tensor,
        dino_features: torch.Tensor,
        return_view_assignment: bool,
        grid_resolution: int,
        sampler_method: str,
    ):
        latent_coords = latent_coords.int().contiguous()
        structure_conditioned_coords = (
            structure_conditioned_coords.int().contiguous()
        )

        with self._use_models(model_name, "conditioner"):
            conditioning = config["conditioning"]

            latent_coords = latent_coords.to(
                device=self.device,
                dtype=torch.int32,
            )
            structure_conditioned_coords = structure_conditioned_coords.to(
                device=self.device,
                dtype=torch.int32,
            )

            if model_name == "shape512_model":
                model = self.shape512_model
            elif model_name == "shape1024_model":
                model = self.shape1024_model
            else:
                raise ValueError(f"Unknown model: {model_name}")

            condition = self.conditioner.encode(
                model.adapter,
                dino_features=dino_features.to(
                    device=self.device,
                    dtype=torch.float32,
                ),
                vggt_images=batch.vggt_images.to(
                    device=self.device,
                    dtype=torch.float32,
                ),
                structure_conditioned_coords=structure_conditioned_coords,
                input_masks=batch.input_masks.to(
                    device=self.device,
                    dtype=torch.bool,
                ),
                latent_coords=latent_coords,
                router_grid=int(conditioning["grid_resolution"]),
            )

            # Shape1024 generates its own latent features.
            # The upsampled 512 coordinates guide conditioning instead.
            latent = sample_shape_latent(
                model,
                latent_coords,
                condition,
                config["sampler"],
                description=(
                    f"Shape-{config['decoder']['resolution']} sampling"
                ),
                sampler_method=sampler_method
            )

            view_assignment = (
                {"selected_view": condition["cond"]["selected_view"].cpu()}
                if return_view_assignment
                else None
            )

        if not torch.isfinite(latent.feats).all():
            raise RuntimeError(
                "Shape sampling produced non-finite latent features"
            )

        return latent, view_assignment
    
    def _decode_shape_latent_1024(
        self,
        decoder,
        latent,
        resolution=1024,
    ) -> trimesh.Trimesh:
        
        with self._use_models(self.shape1024_decoder):
            self.shape1024_decoder.resolution = resolution
            mesh, _ = self.shape1024_decoder(
                    latent,
                    return_subs=True,
                )
        return mesh[0]


    
    def _generate_shape512(
        self,
        batch: InputBatch,
        structure_32: SparseStructure,
        structure_64: SparseStructure,
        dino_features: torch.Tensor,
        sparse_resolution: int = 32,
        lr_resolution: int = 512,
        decoder_upsample_times: int = 4,
        sampler_method: str = DEFAULT_SAMPLER_METHOD,
    ) -> trimesh.Trimesh:
        # Sparse structure is decoded at 64^3, downscale to 32^3 for shape512
        latent_coords = structure_32
        structure_conditioned_coords = structure_64

        print("[UniForge3D] → Loading Shape512 model...")
        latent, _ = self._sample_shape(
            "shape512_model",
            self.config["structured_latent_512"],
            batch,
            latent_coords=latent_coords,
            structure_conditioned_coords=structure_conditioned_coords,
            dino_features=dino_features,
            return_view_assignment=False,
            grid_resolution=32,
            sampler_method=sampler_method
            #initial_sparse=None, Initialized CuMesh with vertices and faces
        )

        print("[UniForge3D] → Loading Shape512 Decoder...")

        with self._use_models(self.shape512_decoder):
            self.shape512_decoder.resolution = lr_resolution
            mesh = decode_shape_latent(
                self.shape512_decoder,
                latent,
                self.config["structured_latent_512"]["decoder"].get("normalization"),
            )[0]

            

        print("[UniForge3D] → Shape512 decoding completed.")

        #return mesh#mesh.vertices, mesh.faces
    
        return latent, mesh

    def _generate_shape1024(
        self,
        batch: InputBatch,
        latent_coords: torch.Tensor,
        structure_conditioned_coords: torch.Tensor,
        dino_features: torch.Tensor,
        sampler_method: str,
    ) -> tuple[NormalizedShapeLatent, dict[str, torch.Tensor]]:
        print("[UniForge3D] → Loading Shape1024 model...")
        latent, view_assignment = self._sample_shape(
            "shape1024_model",
            self.config["structured_latent_1024"],
            batch,
            latent_coords=latent_coords,
            structure_conditioned_coords=structure_conditioned_coords,
            dino_features=dino_features,
            return_view_assignment=True,
            grid_resolution=64,
            sampler_method=sampler_method
            #initial_sparse=initial_sparse,
        )

        if view_assignment is None:
            raise RuntimeError("Shape1024 must provide its texture routing result")
        shape = NormalizedShapeLatent(coords=latent.coords.cpu(), feats=latent.feats.cpu())
        return shape, view_assignment

    def _shape512_to_coords64(self, mesh, surface_voxel_resolution: int) -> torch.Tensor:
            """Convert the already-remeshed Shape-512 surface to coords64."""
            config = self.config["structured_latent_handoff"]
            
            # Ensure vertices and faces are safely moved to CPU numpy arrays
            vertices = mesh.vertices
            if hasattr(vertices, "detach"):
                vertices = vertices.detach().cpu().numpy()
            faces = mesh.faces
            if hasattr(faces, "detach"):
                faces = faces.detach().cpu().numpy()

            
            model_vertices = display_to_model(vertices)
            print("[_shape512_to_coords64] Converted vertices to model coordinates.")
            normalized_vertices, _ = normalize_to_cube(model_vertices, config["target_max_side"])
            print("[_shape512_to_coords64] Normalized vertices to cube.")
            xyz = torch.from_numpy(
                surface_voxelize(normalized_vertices, faces, surface_voxel_resolution)
            )
            print(f"[_shape512_to_coords64] Voxelized surface to resolution {surface_voxel_resolution}.")
            return torch.cat((torch.zeros((len(xyz), 1), dtype=torch.int32), xyz.int()), dim=1)

    def _generate_texture(
        self,
        batch: InputBatch,
        normalized_shape_latent: NormalizedShapeLatent,
        view_assignment: dict[str, torch.Tensor],
        dino_features: torch.Tensor,
        decimate_target_faces: int,
        texture_size: int = 1024,
        mesh_resolution: int = 1024,
        uv_mode: str = "comfy_gpu",
    ) -> trimesh.Trimesh:
        config = self.config["texture"]
        with self._use_models(
            self.texture_model,
            self.shape1024_decoder,
            self.texture_decoder,
            self.conditioner,
        ):

            print("[_generate_texture] Denormalizing shape latent.")
            shape_vae_latent = denormalize_shape_latent(
                config,
                normalized_shape_latent,
                self.device,
            )
            print("[_generate_texture] Sampling texture latent.")
            texture_vae_latent = sample_texture(
                self.texture_model,
                config,
                self.conditioner,
                shape_vae_latent,
                batch,
                dino_features,
                view_assignment,
                self.device,
            )
            release_cuda()
            self.shape1024_decoder.resolution = mesh_resolution
            print(f"[_generate_texture] Loading and Decoding shape and texture @ mesh_resolution={mesh_resolution}")
            reference_mesh, pbr_voxel = decode_shape_and_texture(
                
                self.shape1024_decoder, self.texture_decoder, shape_vae_latent, texture_vae_latent
            )
            del shape_vae_latent,texture_vae_latent
        print(f"[_generate_texture] Remeshing and baking texture...")
        release_cuda()
        return remesh_and_bake_texture(
            config,
            self.config["mesh_postprocess"],
            reference_mesh,
            pbr_voxel,
            decimation_target=decimate_target_faces,
            texture_size=texture_size,
            uv_mode=uv_mode
        )

    @torch.no_grad()
    def run(
        self,
        images: Sequence[str | Path],
        output: str | Path,
        *,
        seed: int | None = None,
        overwrite: bool = False,
        skip_texture: bool = False,
        sparse_structure_resolution: int = 64,
        denoise_sparse: bool = True,
        decode_stage: bool = False,
        hr_resolution: int = 1536,
        lr_resolution: int = 512,
        max_num_tokens: int = 90000,
        decoder_upsample_times: int = 4,
        denoise_iterations: int = 1,
    ) -> tuple[trimesh.Trimesh, torch.Tensor]:

        output_path = Path(output).expanduser().resolve()
        if output_path.exists() and (not overwrite or not output_path.is_file()):
            raise FileExistsError(
                f"Output already exists: {output_path}; use --overwrite to recompute"
            )

        print("[UniForge3D] Starting pipeline execution...")
        batch = self._preprocess_images(images)
        print(f"[UniForge3D] ✓ Images preprocessed")
        release_cuda()

        torch.cuda.set_device(self.device)
        torch.manual_seed(seed)
        try:
            # Encode DINO 512 - only what we need now
            print("[UniForge3D] Stage 1/5: Encoding DINO features (512)...")
            dino_features_512 = self._encode_dino_512(batch)
            print("[UniForge3D] ✓ DINO 512 encoding complete")
            # Free GPU memory
            release_cuda()

            print("[UniForge3D] Stage 2/5: Generating sparse structure...")
            sparse_structure = self._generate_structure(batch, dino_features_512)
            print("[UniForge3D] ✓ Sparse structure generated")
            release_cuda()

            if denoise_sparse:
                print("[UniForge3D] Denoising sparse structure...")
                from .sparse_structure import denoise_sparse_coords
                from .types import SparseStructure

                sparse_structure = SparseStructure(
                    coords32=denoise_sparse_coords(sparse_structure.coords32, iterations=denoise_iterations),
                    coords64=denoise_sparse_coords(sparse_structure.coords64, iterations=denoise_iterations),
                )
                print("[UniForge3D] ✓ Sparse structure denoised")
            print("num coords32 tokens:", sparse_structure.coords32.shape[0])
            print("num coords64 tokens:", sparse_structure.coords64.shape[0])

            # Instead of running full 512 VAE mesh generation, obtain lightweight latent tokens directly from DiT
            print("[UniForge3D] Stage 3/5: Generating shape 512 latents (lightweight)...")
            shape512_vae_latent, mesh512 = self._generate_shape512(
                batch, sparse_structure, dino_features_512,
                sparse_resolution=sparse_structure_resolution,
                lr_resolution=hr_resolution,
            )

            print("[UniForge3D] ✓ Shape512 mesh generated. Begin")
            structure_conditioned_coords_original = self._shape512_to_coords64(mesh512, 64)

            print("[UniForge3D] ✓ Shape512 latents generated")
            
            del dino_features_512
            release_cuda()

            def interpolate_sparse_coords_densely(coords, scale_factor=2, resolution=512, max_val=1024):
                """
                VRAM-safe, pure-GPU vectorized nearest-neighbor coordinate upsampling.
                Mathematically equivalent to F.interpolate(..., mode='nearest') on dense grids,
                but computed entirely via sparse coordinate expansion without OOM.
                """
                device = coords.device
                
                b = coords[:, 0:1]       # [N, 1]
                xyz = coords[:, 1:4].long() # [N, 3]
                
                # Clamp base coordinates to valid input range
                xyz = torch.clamp(xyz, 0, resolution - 1)
                base_xyz = xyz * scale_factor
                
                # Generate local block offsets for nearest-neighbor expansion (scale_factor^3)
                offsets = torch.stack(
                    torch.meshgrid([torch.arange(scale_factor, device=device)] * 3, indexing='ij'),
                    dim=-1
                ).reshape(-1, 3) # [scale_factor^3, 3]
                
                # Expand coordinates: [N, 1, 3] + [1, K, 3] -> [N, K, 3]
                expanded_xyz = base_xyz.unsqueeze(1) + offsets.unsqueeze(0)
                expanded_xyz = expanded_xyz.reshape(-1, 3) # [N * K, 3]
                
                # Repeat batch indices for each expanded point
                expanded_b = b.repeat_interleave(scale_factor ** 3, dim=0) # [N * K, 1]
                
                # Combine and clamp to target resolution bounds
                new_coords = torch.cat([expanded_b, expanded_xyz], dim=1)
                new_coords[:, 1:] = torch.clamp(new_coords[:, 1:], 0, max_val - 1)
                
                # Remove duplicate coordinates
                return new_coords.unique(dim=0).contiguous()
            # Derive structure-conditioned coordinates directly from the sparse latent coordinates instead of a full mesh
            #structure_conditioned_coords = shape512_vae_latent.coords.clone()
            # Scale up coordinates to match the higher target resolution context safely
            '''
            structure_conditioned_coords = structure_conditioned_coords_original.clone()
            print("structure_conditioned_coords before interpolation:", structure_conditioned_coords.shape)
            structure_conditioned_coords = interpolate_sparse_coords_densely(structure_conditioned_coords, scale_factor=decoder_upsample_times, resolution=512)
            print("structure_conditioned_coords after interpolation:", structure_conditioned_coords.shape)
            
            quant_coords = torch.cat(
                [
                    structure_conditioned_coords[:, :1],
                    torch.clamp(
                        (structure_conditioned_coords[:, 1:].float() ).round().int(),
                        min=0,
                        max=512
                    )
                ],
                dim=1,
            )
            
            coords = quant_coords.unique(dim=0)
            target_coords = coords.contiguous()

            print("num tokens after quantization:", target_coords.shape[0])
            
            
            release_cuda()
            import gc
            del shape512_vae_latent
            
            gc.collect()
            torch.cuda.empty_cache()
            release_cuda()
            # ============================================================
            # STAGE 5: SAMPLE 1024 FROM THE UPSAMPLED 512 STRUCTURE
            # ============================================================
            print("[UniForge3D] Encoding DINO features (1024)...")
            '''
            dino_features_1024 = self._encode_dino_1024(batch)
            
            print("[UniForge3D] Sampling 1024 shape using the upsampled 512 structure...")
            shape1024_latent, view_assignment = self._generate_shape1024(
                batch,
                latent_coords=structure_conditioned_coords_original,               # Target Shape1024 latent coordinates
                structure_conditioned_coords=sparse_structure.coords64, # 512 mesh-derived structural guidance
                dino_features=dino_features_1024,
            )

            del dino_features_1024
            release_cuda()
            
            # ============================================================
            # FINAL DECODING
            # ============================================================
            shape1024_vae_latent = denormalize_shape_latent(
                self.config["texture"],
                shape1024_latent,
                self.device,
            )

            del shape1024_latent
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            release_cuda()  
            
            with self._use_models(self.shape1024_decoder):
                self.shape1024_decoder.resolution = hr_resolution

                #meshes, _ = self.shape1024_decoder.decode_from_stage(
                #    shape1024_vae_latent,
                 #   start_stage=4,
                #)
   

                meshes, _ = self.shape1024_decoder(
                    shape1024_vae_latent,
                    return_subs=True,
                )

                #mesh = decode_shape_latent(
                #    self.shape1024_decoder,
                #    shape1024_vae_latent,
                #    self.config["structured_latent_1024"]["decoder"].get("normalization"),
                #)[0]
                    

                del shape1024_vae_latent
                gc.collect()
                torch.cuda.empty_cache()
                release_cuda()

                output_mesh = meshes[0]

                trimesh_obj = trimesh.Trimesh(
                    vertices=output_mesh.vertices.detach().cpu().numpy(),
                    faces=output_mesh.faces.detach().cpu().numpy(),
                    process=False,
                )

            print("[UniForge3D] ✓ Shape1024 mesh decoded")
             
            release_cuda()

            print("[UniForge3D] Returning Shape1024 mesh without texture")
            return output_mesh
       

            # Texture reuses the Shape1024 latent and its per-token view routing.
            print("[UniForge3D] Stage 5/5: Generating texture...")
            textured_mesh = self._generate_texture(
                batch,
                shape1024_latent,
                view_assignment,
                dino_features["shape1024"],
            )
            print("[UniForge3D] ✓ Texture generated")
            release_cuda()
            print("[UniForge3D] Exporting final mesh...")
            export_glb(textured_mesh, output_path, overwrite=overwrite)
            print(f"[UniForge3D] ✓ Pipeline complete! Mesh saved to: {output_path}")
            # Return raw 1024 mesh (same as skip_texture path for consistency)
            return textured_mesh
        finally:
            print("[UniForge3D] Cleaning up resources...")
            release_cuda()
