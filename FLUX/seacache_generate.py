#!/usr/bin/env python3
from typing import Any, Dict, Optional, Union
import argparse
import os
import re
from datetime import datetime

import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from util_seacache import rel_l1, apply_sea_with_scheduler

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def seacache_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    """
    Drop-in replacement for FluxTransformer2DModel.forward with SeaCache gating.
    Logic mirrors the denoise_cache gating (accumulated rescaled relative L1).
    """

    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective.")

    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids is not None and txt_ids.ndim == 3:
        logger.warning("`txt_ids` passed as 3D Tensor; dropping batch dim for rotary embedding cache.")
        txt_ids = txt_ids[0]
    if img_ids is not None and img_ids.ndim == 3:
        logger.warning("`img_ids` passed as 3D Tensor; dropping batch dim for rotary embedding cache.")
        img_ids = img_ids[0]

    if txt_ids is not None and img_ids is not None:
        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)
    else:
        # Fallback: some pipelines may pass precomputed rotary or leave ids None in edge cases.
        image_rotary_emb = None

    # ---- SeaCache gating ----
    should_calc = True
    if getattr(self, "enable_seacache", False):
        inp = hidden_states
        temb_ = temb
        modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.transformer_blocks[0].norm1(inp, emb=temb_)
        # Track change based on the first block's norm1 modulation
        if self.cnt == 0 or self.cnt == self.num_steps - 1 or self.previous_modulated_input is None:
            should_calc = True
            self.accumulated_rel_l1_distance = 0.0
        else:
            # Apply SEA filtering before computing distance
            modulated_inp = modulated_inp.reshape(
                modulated_inp.shape[0],
                int(img_ids[:, 1].max().item() + 1),
                int(img_ids[:, 2].max().item() + 1),
                modulated_inp.shape[-1],
            )
            modulated_inp = apply_sea_with_scheduler(
                modulated_inp,
                self.scheduler,
                getattr(self, "cnt", 0),
                power_exp=2.0,
                dims=(-2, -3),
                norm_mode="mean",
            )
            modulated_inp = modulated_inp.reshape(modulated_inp.shape[0], -1, modulated_inp.shape[-1])
            self.accumulated_rel_l1_distance += rel_l1(modulated_inp, self.previous_modulated_input)

            if self.accumulated_rel_l1_distance < float(self.seacache_thresh):
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0.0

        self.previous_modulated_input = modulated_inp
        self.cnt += 1
        if self.cnt == self.num_steps:
            # Reset step counter at the end of a full trajectory
            self.cnt = 0

    # ---- Main block compute / skip ----
    if getattr(self, "enable_seacache", False) and not should_calc and (self.previous_residual is not None):
        # Reuse residual from the previous timestep
        hidden_states = hidden_states + self.previous_residual
    else:
        ori_hidden_states = hidden_states
        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb):
                        return module(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = int(np.ceil(len(self.transformer_blocks) / len(controlnet_block_samples)))
                if controlnet_blocks_repeat:
                    hidden_states = hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb):
                        return module(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            if controlnet_single_block_samples is not None:
                interval_control = int(np.ceil(len(self.single_transformer_blocks) / len(controlnet_single_block_samples)))
                hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

        if getattr(self, "enable_seacache", False):
            # Cache residual from the current step
            self.previous_residual = hidden_states - ori_hidden_states

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


# Replace Diffusers model forward
FluxTransformer2DModel.forward = seacache_forward


# ----------------------------
# Utils
# ----------------------------
def read_prompts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def safe_filename(name: str) -> str:
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"[^0-9A-Za-z._-]", "", name)
    return name or "img"


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    parser = argparse.ArgumentParser(
        description="Diffusers + SeaCache (FLUX) — image generation from prompts (aligned with the second script)."
    )
    # Prompt input options
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Path to a text file containing one prompt per line.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photo of an astronaut riding a horse",
        help="Single prompt string. Used if --prompt_file is not provided.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the generated images.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width (multiple of 16, same default as the second script).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height (multiple of 16, same default as the second script).",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of sampling steps. Defaults: flux-dev=50, flux-schnell=4.",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=3.5,
        help="Guidance scale (default 3.5, same as the second script).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed (global sample index is added to this).",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate per prompt.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="flux-dev",
        choices=["flux-dev", "flux-schnell"],
        help="Model name (same choices as the second script).",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Optional explicit HF model id. If set, overrides model_name.",
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Use CPU offload (`enable_model_cpu_offload`).",
    )

    # SeaCache threshold: conceptually equivalent to specache_thresh in the other script
    parser.add_argument(
        "--seacache_thresh",
        type=float,
        default=0.3,
        help="SeaCache threshold (equivalent to specache_thresh in the other script). 0.3: 2x, 0.6: 3x",
    )

    # dtype selection (matches the second script's default bfloat16)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Computation dtype (default bf16, as in the second script).",
    )

    args = parser.parse_args()

    # Decide where prompts come from
    if args.prompt_file:
        prompts = read_prompts(args.prompt_file)
        prompt_source = args.prompt_file
    elif args.prompt:
        prompts = [args.prompt]
        prompt_source = "<inline-prompt>"
    else:
        parser.error("You must provide either --prompt_file or --prompt.")

    os.makedirs(args.output_dir, exist_ok=True)

    # ====== Device / dtype configuration ======
    device = "cuda" if torch.cuda.is_available() else "cpu"
    want_bf16 = args.dtype == "bf16"
    torch_dtype = torch.bfloat16 if (want_bf16 and device == "cuda") else torch.float16

    # ====== Model mapping (same naming as the second script) ======
    if args.model_id is not None:
        model_id = args.model_id
        model_name = "flux-dev" if "dev" in model_id else "flux-schnell"
    else:
        if args.model_name == "flux-dev":
            model_id = "black-forest-labs/FLUX.1-dev"
        else:
            model_id = "black-forest-labs/FLUX.1-schnell"
        model_name = args.model_name

    # ====== Default steps ======
    if args.num_inference_steps is None:
        num_steps = 4 if model_name == "flux-schnell" else 50
    else:
        num_steps = args.num_inference_steps

    # ====== Load pipeline ======
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    # SeaCache state variables stored on the model instance
    tr = pipe.transformer
    tr.scheduler = pipe.scheduler
    tr.enable_seacache = True
    tr.seacache_thresh = float(args.seacache_thresh)

    # Random sequence: same pattern as the second script (global sample index)
    base_seed = int(args.seed)
    global_sample_idx = 0  # Index over all generated images

    print(
        f"[{now_str()}] Start | model_id={model_id} ({model_name}) | steps={num_steps} | "
        f"guidance={args.guidance} | seacache_thresh={args.seacache_thresh} | "
        f"prompts={len(prompts)} | seed={base_seed} | dtype={torch_dtype}"
    )

    for p_idx, prompt in enumerate(prompts):
        n = int(args.num_images_per_prompt)
        for i in range(n):
            # Per-sample seed
            seed_this = base_seed + global_sample_idx

            # Reset SeaCache state 
            tr.cnt = 0
            tr.num_steps = int(num_steps)
            tr.accumulated_rel_l1_distance = 0.0
            tr.previous_modulated_input = None
            tr.previous_residual = None

            generator = torch.Generator(device=device).manual_seed(seed_this)

            out = pipe(
                prompt=prompt,
                num_inference_steps=num_steps,
                height=(args.height // 16) * 16,
                width=(args.width // 16) * 16,
                guidance_scale=gscale
                if (gscale := (0.0 if model_name == "flux-schnell" else float(args.guidance)))
                is not None
                else 0.0,
                max_sequence_length=256 if model_name == "flux-schnell" else 512,
                num_images_per_prompt=1,
                generator=generator,
            )

            # Save image
            img = out.images[0]
            base = safe_filename(prompt)[:80]
            fn = f"SeaCache_{global_sample_idx:05d}-{base}.png"
            img.save(os.path.join(args.output_dir, fn))

            truncated_prompt = prompt[:60] + ("…" if len(prompt) > 60 else "")
            print(
                f"  - [{p_idx + 1}/{len(prompts)}] '{truncated_prompt}' "
                f"# {i + 1}/{n} => 1 image"
            )

            global_sample_idx += 1
        break  # NOTE: kept from your original script (only first prompt is processed).

    # Summary (no timing)
    if global_sample_idx > 0:
        total_images = global_sample_idx
        summary_lines = [
            "",
            "[Generation Summary]",
            f"  Output dir:                 {args.output_dir}",
            f"  Model id:                   {model_id}",
            f"  Model name:                 {model_name}",
            f"  Steps per image:            {num_steps}",
            f"  Guidance:                   {args.guidance}",
            f"  SeaCache threshold:         {args.seacache_thresh}",
            f"  Prompt source:              {prompt_source}",
            f"  Seed (base):                {base_seed}",
            f"  Total images:               {total_images}",
            f"  Size (WxH):                 {args.width}x{args.height}",
        ]
        print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
