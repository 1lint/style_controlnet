
from argparse import Namespace
from multiprocessing import cpu_count
from pathlib import Path

from src.train_controlstyle import main
from src.parse_args import parse_args

models_path = "/mnt/d/repos/stable-diffusion-webui/models/Stable-diffusion/"
#base_ckpt = "dreamshaper_331BakedVae.safetensors"
base_ckpt = "realdosmix_.safetensors"

controlnet_ckpt = "anything-v4.0.ckpt"

args = Namespace(
    
    pretrained_model_name_or_path=str(Path(models_path)/base_ckpt),
    #pretrained_model_name_or_path='lint/simpathizer',

    #controlnet_weights_path=str(Path(models_path)/controlnet_ckpt),
    #controlnet_subfolder='unet',

    controlnet_weights_path='models/realdosmix__animestyler2/checkpoint-6169',
    controlnet_subfolder=None,

    # dataset args
    train_data_dir="/mnt/d/data/anybooru/train",
    valid_data_dir="/mnt/d/data/anybooru/valid",
    resolution=512,
    from_hf_hub=False,

    # training args
    train_whole_controlnet=True, # whether to train whole controlnet or just zero convolution weights 
    learning_rate=5e-6,
    num_train_epochs=1000,
    max_train_steps=None,
    seed=3434554,
    max_grad_norm=1.0,
    gradient_accumulation_steps=4,

    # VRAM args
    batch_size=3,
    mixed_precision="fp16", # set to "fp16" for mixed-precision training.
    gradient_checkpointing=True, # set this to True to lower the memory usage.
    use_8bit_adam=True, # use 8bit optimizer from bitsandbytes
    enable_xformers_memory_efficient_attention=True,
    allow_tf32=True,
    dataloader_num_workers=cpu_count(),

    # logging args
    output_dir=f"./models/{base_ckpt.rsplit('.',1)[0]}_animestyler2",
    report_to='tensorboard',
    image_logging_steps=0, # disabled when 0. costs additional VRAM to log images
    save_whole_pipeline=False,
    checkpointing_steps=10000, 
)

if __name__ == '__main__':
    #args = parse_args()
    main(args)