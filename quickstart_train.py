
from argparse import Namespace
from multiprocessing import cpu_count

from src.train_controlstyle import main
from src.parse_args import parse_args

args = Namespace(
    
    # start training from preexisting models
    pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5',
    controlnet_weights_path='andite/anything-v4.0', 
    controlnet_subfolder='unet',

    # uncomment to resume training from pretrained weights
    #pretrained_model_name_or_path='lint/simpathizer',
    #controlnet_weights_path='lint/simpathizer', 
    #controlnet_subfolder='controlnet',

    # dataset args
    train_data_dir="lint/anybooru",
    valid_data_dir=None,
    resolution=512,
    from_hf_hub=True,

    # training args
    train_whole_controlnet=False, # whether to train whole controlnet or just zero convolution weights 
    learning_rate=5e-6,
    num_train_epochs=1000,
    max_train_steps=None,
    seed=3434554,
    max_grad_norm=1.0,
    gradient_accumulation_steps=4,

    # VRAM args
    batch_size=1,
    mixed_precision="fp16", # set to "fp16" for mixed-precision training.
    gradient_checkpointing=True, # set this to True to lower the memory usage.
    use_8bit_adam=True, # use 8bit optimizer from bitsandbytes
    enable_xformers_memory_efficient_attention=True,
    allow_tf32=True,
    dataloader_num_workers=cpu_count(),

    # logging args
    output_dir=f"./models/controlstyle",
    report_to='tensorboard',
    image_logging_steps=400, # disabled when 0. costs additional VRAM to log images
    save_whole_pipeline=True,
    checkpointing_steps=4000, 
)

if __name__ == '__main__':
    #args = parse_args()
    main(args)