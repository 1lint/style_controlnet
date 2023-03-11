import gradio as gr
import torch
import random
from PIL import Image
import os
import argparse
import shutil
import gc
import importlib
import json

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipelineLegacy,
)

from modified_diffusers.pipeline_stable_diffusion_controlnet import StableDiffusionControlNetPipeline
from modified_diffusers.controlnet import ControlNetModel

from .textual_inversion import main as run_textual_inversion
from .shared import default_scheduler, scheduler_dict, model_ids, controlnet_ids


_xformers_available = importlib.util.find_spec("xformers") is not None
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'
dtype = torch.float16 if device == "cuda" else torch.float32
low_vram_mode = False


tab_to_pipeline = {
    1: StableDiffusionControlNetPipeline,
    2: StableDiffusionImg2ImgPipeline,
    3: StableDiffusionInpaintPipelineLegacy,
}


def load_pipe(model_id, controlnet_id, scheduler_name, tab_index=1, pipe_kwargs="{}"):
    global pipe, loaded_model_id

    scheduler = scheduler_dict[scheduler_name]

    pipe_class = tab_to_pipeline[tab_index]

    if model_id == 'xxlmaes/VinteProtogenMix':
        # author didn't put sd weights on HF so I converted them and put them under my repo
        # I substitute the repo name here for clarity and attribution (https://civitai.com/models/5657/vinteprotogenmix-v10)
        model_id = 'lint/simpathizer'

    # load new weights from disk only when changing model_id
    if model_id != loaded_model_id:
        pipe = pipe_class.from_pretrained(
            model_id,
            controlnet=ControlNetModel.from_pretrained(controlnet_id, subfolder='controlnet'),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            scheduler=scheduler.from_pretrained(model_id, subfolder="scheduler"),
            **json.loads(pipe_kwargs),
        )
        loaded_model_id = model_id

    # if same model_id, instantiate new pipeline with same underlying pytorch objects to avoid reloading weights from disk
    elif pipe_class != pipe.__class__ or not isinstance(pipe.scheduler, scheduler):
        pipe.components["scheduler"] = scheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        pipe = pipe_class(**pipe.components)

    if device == "cuda":
        for component in pipe.components.values():
            if hasattr(component, 'device'):
                component.to('cuda', torch.float16)

        if _xformers_available:
            pipe.enable_xformers_memory_efficient_attention()
        if low_vram_mode:
            pipe.enable_attention_slicing()
            print("using attention slicing to lower VRAM")

    return pipe


pipe = None
loaded_model_id = ""
pipe = load_pipe(model_ids[0], controlnet_ids[0], default_scheduler)


def pad_image(image):
    w, h = image.size
    if w == h:
        return image
    elif w > h:
        new_image = Image.new(image.mode, (w, w), (0, 0, 0))
        new_image.paste(image, (0, (w - h) // 2))
        return new_image
    else:
        new_image = Image.new(image.mode, (h, h), (0, 0, 0))
        new_image.paste(image, ((h - w) // 2, 0))
        return new_image

@torch.no_grad()
def generate(
    model_name,
    controlnet_name,
    scheduler_name,
    prompt,
    guidance,
    steps,
    n_images=1,
    width=512,
    height=512,
    seed=0,
    #image=None,
    #strength=0.5,
    #inpaint_image=None,
    #inpaint_strength=0.5,
    #inpaint_radio="",
    neg_prompt="",
    controlnet_prompt="",
    controlnet_cond_scale=1.0,
    controlnet_mix_mode=None,
    tab_index=1,
    pipe_kwargs="{}",
    progress=gr.Progress(track_tqdm=True),
):

    if controlnet_mix_mode == "default":
        controlnet_mix_mode = None
    if seed == -1:
        seed = random.randint(0, 2147483647)

    generator = torch.Generator(device).manual_seed(seed)

    pipe = load_pipe(
        model_id=model_name,
        controlnet_id=controlnet_name,
        scheduler_name=scheduler_name,
        tab_index=tab_index,
        pipe_kwargs=pipe_kwargs,
    )

    status_message = f"Prompt: '{prompt}' | Seed: {seed} | Guidance: {guidance} | Scheduler: {scheduler_name} | Steps: {steps}"

    if controlnet_prompt == "":
        controlnet_prompt = None # pass None so pipeline uses base prompt as controlnet_prompt


    if tab_index == 1:
        status_message = "Text to Image " + status_message

        result = pipe(
            prompt,
            negative_prompt=neg_prompt,
            num_images_per_prompt=n_images,
            num_inference_steps=int(steps),
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator,
            controlnet_prompt=controlnet_prompt,
            controlnet_conditioning_scale=controlnet_cond_scale,
            controlnet_mode=controlnet_mix_mode,
        )

    '''
    elif tab_index == 2:

        status_message = "Image to Image " + status_message
        print(image.size)
        image = image.resize((width, height))
        print(image.size)

        result = pipe(
            prompt,
            negative_prompt=neg_prompt,
            num_images_per_prompt=n_images,
            image=image,
            num_inference_steps=int(steps),
            strength=strength,
            guidance_scale=guidance,
            generator=generator,
        )

    elif tab_index == 3:
        status_message = "Inpainting " + status_message

        init_image = inpaint_image["image"].resize((width, height))
        mask = inpaint_image["mask"].resize((width, height))

        result = pipe(
            prompt,
            negative_prompt=neg_prompt,
            num_images_per_prompt=n_images,
            image=init_image,
            mask_image=mask,
            num_inference_steps=int(steps),
            strength=inpaint_strength,
            preserve_unmasked_image=(
                inpaint_radio == "preserve non-masked portions of image"
            ),
            guidance_scale=guidance,
            generator=generator,
        )

    else:
        return None, f"Unhandled tab index: {tab_index}"
    '''
    return result.images, status_message
    

# based on lvkaokao/textual-inversion-training
def train_textual_inversion(
    model_name,
    scheduler_name,
    type_of_thing,
    files,
    concept_word,
    init_word,
    text_train_steps,
    text_train_bsz,
    text_learning_rate,
    progress=gr.Progress(track_tqdm=True),
):

    if device == "cpu":
        raise gr.Error("Textual inversion training not supported on CPU")

    pipe = load_pipe(
        model_id=model_name,
        scheduler_name=scheduler_name,
        tab_index=1,
    )

    pipe.disable_xformers_memory_efficient_attention()  # xformers handled by textual inversion script

    concept_dir = "concept_images"
    output_dir = "output_model"
    training_resolution = 512

    if os.path.exists(output_dir):
        shutil.rmtree("output_model")
    if os.path.exists(concept_dir):
        shutil.rmtree("concept_images")

    os.makedirs(concept_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    gc.collect()
    torch.cuda.empty_cache()

    if concept_word == "" or concept_word == None:
        raise gr.Error("You forgot to define your concept prompt")

    for j, file_temp in enumerate(files):
        file = Image.open(file_temp.name)
        image = pad_image(file)
        image = image.resize((training_resolution, training_resolution))
        extension = file_temp.name.split(".")[1]
        image = image.convert("RGB")
        image.save(f"{concept_dir}/{j+1}.{extension}", quality=100)

    args_general = argparse.Namespace(
        train_data_dir=concept_dir,
        learnable_property=type_of_thing,
        placeholder_token=concept_word,
        initializer_token=init_word,
        resolution=training_resolution,
        train_batch_size=text_train_bsz,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        mixed_precision="fp16",
        use_bf16=False,
        max_train_steps=int(text_train_steps),
        learning_rate=text_learning_rate,
        scale_lr=True,
        lr_scheduler="constant",
        lr_warmup_steps=0,
        output_dir=output_dir,
    )

    try:
        final_result = run_textual_inversion(pipe, args_general)
    except Exception as e:
        raise gr.Error(e)

    pipe.text_encoder = pipe.text_encoder.eval().to(device, dtype=dtype)
    pipe.unet = pipe.unet.eval().to(device, dtype=dtype)

    gc.collect()
    torch.cuda.empty_cache()

    return (
        f"Finished training! Check the {output_dir} directory for saved model weights"
    )
