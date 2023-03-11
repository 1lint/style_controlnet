# modified starting from HuggingFace diffusers train_dreambooth.py example
# https://github.com/huggingface/diffusers/blob/024c4376fb19caa85275c038f071b6e1446a5cad/examples/dreambooth/train_dreambooth.py

import logging
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available

from diffusers import AutoencoderKL, StableDiffusionPipeline

from torchvision.utils import make_grid
import numpy as np

from .modified_diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
)
from .data import PNGDataModule

logger = get_logger(__name__)


def main(args):
    logging_dir = Path(args.output_dir)

    accelerator_project_config = ProjectConfiguration(
        logging_dir=logging_dir,
        total_limit=5,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    vae = AutoencoderKL.from_pretrained("lint/anime_vae")

    if os.path.isfile(args.pretrained_model_name_or_path):
        pipe = StableDiffusionPipeline.from_ckpt(args.pretrained_model_name_or_path)
        pipe.safety_checker = None
        pipe.feature_extractor = None
        pipe.vae = vae
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )

    #unet = UNet2DConditionModel.from_config(pipe.unet.config)
    #unet.load_state_dict(pipe.unet.state_dict(), strict=False)
    #pipe.unet = unet

    unet = pipe.unet
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    noise_scheduler = pipe.scheduler

    if os.path.isfile(args.controlnet_weights_path):
        controlnet = ControlNetModel.from_config(
            ControlNetModel.load_config("configs/controlnet_config.json")
        )
        controlnet.load_weights_from_sd_ckpt(args.controlnet_weights_path)
    else:
        controlnet = ControlNetModel.from_pretrained(
            args.controlnet_weights_path,
            subfolder=args.controlnet_subfolder,
            low_cpu_mem_usage=False,
            device_map=None,
            controlnet_conditioning_embedding_type="null",
            controlnet_conditioning_channels=0,
        )

    control_pipe = StableDiffusionControlNetPipeline(
        **pipe.components,
        controlnet=controlnet,
        requires_safety_checker=False,
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    if args.train_whole_controlnet:
        controlnet.requires_grad_(True)
    else:
        controlnet.requires_grad_(False)
        controlnet.controlnet_down_blocks.requires_grad_(True)
        controlnet.controlnet_mid_block.requires_grad_(True)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    args.learning_rate = (
        args.learning_rate
        * args.gradient_accumulation_steps
        * args.batch_size
        * accelerator.num_processes
    )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        import bitsandbytes as bnb

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    if args.train_whole_controlnet:
        params_to_optimize = list(controlnet.parameters())
    else:
        # optimize only the zero convolution weights
        params_to_optimize = list(
            controlnet.controlnet_down_blocks.parameters()
        ) + list(controlnet.controlnet_mid_block.parameters())

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
    )

    def collate_fn(examples):
        input_ids = []
        for example in examples:
            input_ids.append(
                tokenizer(
                    example["instance_tags"],
                    padding="do_not_pad",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                ).input_ids
            )

        pixel_values = [example["instance_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            return_tensors="pt",
            max_length=tokenizer.model_max_length,
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch

    datamodule = PNGDataModule(
        train_dir=args.train_data_dir,
        val_dir=args.valid_data_dir,
        from_hf_hub=args.from_hf_hub,
        resolution=[args.resolution, args.resolution],
        target_key="instance_images",
        cond_key="instance_tags",
        persistent_workers=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        batch_size=args.batch_size,
    )

    datamodule.setup(stage="fit")

    train_dataloader = datamodule.train_dataloader()

    if args.valid_data_dir:
        valid_dataloader = accelerator.prepare(datamodule.val_dataloader())

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    (
        unet,
        controlnet,
        optimizer,
        train_dataloader,
    ) = accelerator.prepare(
        unet, controlnet, optimizer, train_dataloader
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    controlnet.to(accelerator.device, dtype=torch.float32)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("tb_logs", config=vars(args))

    # Train!
    total_batch_size = (
        args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )

    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    def compute_loss(batch):
        latents = vae.encode(
            batch["pixel_values"].to(dtype=weight_dtype)
        ).latent_dist.sample()
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(batch["input_ids"])[0]

        down_block_res_samples, mid_block_res_sample = controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=None,
            return_dict=False,
        )

        noise_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {noise_scheduler.config.prediction_type}"
            )

        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

        return loss, encoder_hidden_states

    def log_images(batch, encoder_hidden_states, cond_scales=[0.0, 0.5, 1.0]):
        input_tensors = batch["pixel_values"]
        input_tensors = (input_tensors / 2 + 0.5).clamp(0, 1).cpu()

        [height, width] = input_tensors.shape[-2:]

        output_tensors = []
        with torch.autocast("cuda"), torch.no_grad():
            for cond_scale in cond_scales:
                np_image = control_pipe(
                    prompt_embeds=encoder_hidden_states,
                    controlnet_conditioning_scale=cond_scale,
                    height=height,
                    width=width,
                    output_type="numpy",
                )[0]

                pred_tensor = torch.tensor(np_image).permute(0, 3, 1, 2)
                output_tensors.append(pred_tensor)

            output_tensors = torch.cat(output_tensors)

        image_tensors = torch.cat([input_tensors, output_tensors])
        grid = make_grid(
            image_tensors.detach().cpu(), normalize=False, nrow=args.batch_size
        )
        grid = grid.permute(1, 2, 0).squeeze(-1) * 255
        grid = grid.numpy().astype(np.uint8)

        image_grid = Image.fromarray(grid)
        image_grid.save(
            Path(accelerator.trackers[0].logging_dir) / f"{global_step}.png"
        )

    for epoch in range(first_epoch, args.num_train_epochs):
        # run training loop
        controlnet.train()
        for step, batch in enumerate(train_dataloader):
            loss, encoder_hidden_states = compute_loss(batch)

            loss /= args.gradient_accumulation_steps
            accelerator.backward(loss)
            if global_step % args.gradient_accumulation_steps == 1:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        if args.save_whole_pipeline:
                            control_pipe.controlnet = accelerator.unwrap_model(
                                controlnet
                            )
                            control_pipe.save_pretrained(
                                save_path, safe_serialization=True
                            )
                        else:
                            accelerator.unwrap_model(controlnet).save_pretrained(
                                save_path, safe_serialization=True
                            )

                    if (
                        args.image_logging_steps
                        and global_step % args.image_logging_steps == 1
                    ):
                        log_images(batch, encoder_hidden_states)

            logs = {"training_loss": loss.detach().item()}
            accelerator.log(logs, step=step)
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

        # run validation loop
        if args.valid_data_dir:
            controlnet.eval()
            for step, batch in enumerate(valid_dataloader):
                with torch.no_grad():
                    loss, encoder_hidden_states = compute_loss(batch)

                logs = {"validation_loss": loss.detach().item()}
                accelerator.log(logs, step=step)
                progress_bar.set_postfix(**logs)

            accelerator.wait_for_everyone()

    accelerator.end_training()

