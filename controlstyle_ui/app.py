import gradio as gr
from multiprocessing import cpu_count
from utils.functions import generate, train_textual_inversion
from utils.shared import model_ids, scheduler_names, default_scheduler, controlnet_ids

default_img_size = 512

with open("html/header.html") as fp:
    header = fp.read()

with open("html/footer.html") as fp:
    footer = fp.read()

controlnet_mix_modes = ["default", "+w", "-w", "+h", "-h", "+c", "-c"]

with gr.Blocks(css="html/style.css") as demo:

    pipe_state = gr.State(lambda: 1)

    gr.HTML(header)

    with gr.Row():

        with gr.Column(scale=70):

            # with gr.Row():
            prompt = gr.Textbox(
                label="Prompt", placeholder="<Shift+Enter> to generate", lines=2
            )
            neg_prompt = gr.Textbox(label="Negative Prompt", placeholder="", lines=2)

            controlnet_prompt = gr.Textbox(label="Controlnet Prompt", placeholder="", lines=2)

        with gr.Column(scale=30):
            model_name = gr.Dropdown(
                label="Model", choices=model_ids, value=model_ids[0]
            )
            controlnet_name = gr.Dropdown(
                label="Controlnet", choices=controlnet_ids, value=controlnet_ids[0]
            )
            scheduler_name = gr.Dropdown(
                label="Scheduler", choices=scheduler_names, value=default_scheduler
            )
            generate_button = gr.Button(value="Generate", elem_id="generate-button")

    with gr.Row():

        with gr.Column():

            with gr.Tab("Text to Image") as tab:
                tab.select(lambda: 1, [], pipe_state)

            '''
            with gr.Tab("Image to image") as tab:
                tab.select(lambda: 2, [], pipe_state)

                image = gr.Image(
                    label="Image to Image",
                    source="upload",
                    tool="editor",
                    type="pil",
                    elem_id="image_upload",
                ).style(height=default_img_size)
                strength = gr.Slider(
                    label="Denoising strength",
                    minimum=0,
                    maximum=1,
                    step=0.02,
                    value=0.8,
                )

            with gr.Tab("Inpainting") as tab:
                tab.select(lambda: 3, [], pipe_state)

                inpaint_image = gr.Image(
                    label="Inpainting",
                    source="upload",
                    tool="sketch",
                    type="pil",
                    elem_id="image_upload",
                ).style(height=default_img_size)
                inpaint_strength = gr.Slider(
                    label="Denoising strength",
                    minimum=0,
                    maximum=1,
                    step=0.02,
                    value=0.8,
                )
                inpaint_options = [
                    "preserve non-masked portions of image",
                    "output entire inpainted image",
                ]
                inpaint_radio = gr.Radio(
                    inpaint_options,
                    value=inpaint_options[0],
                    show_label=False,
                    interactive=True,
                )

            with gr.Tab("Textual Inversion") as tab:
                tab.select(lambda: 4, [], pipe_state)

                type_of_thing = gr.Dropdown(
                    label="What would you like to train?",
                    choices=["object", "person", "style"],
                    value="object",
                    interactive=True,
                )

                text_train_bsz = gr.Slider(
                    label="Training Batch Size",
                    minimum=1,
                    maximum=8,
                    step=1,
                    value=1,
                )

                files = gr.File(
                    label=f"""Upload the images for your concept""",
                    file_count="multiple",
                    interactive=True,
                    visible=True,
                )

                text_train_steps = gr.Number(label="How many steps", value=1000)

                text_learning_rate = gr.Number(label="Learning Rate", value=5.0e-4)

                concept_word = gr.Textbox(
                    label=f"""concept word - use a unique, made up word to avoid collisions"""
                )
                init_word = gr.Textbox(
                    label=f"""initial word - to init the concept embedding"""
                )

                textual_inversion_button = gr.Button(value="Train Textual Inversion")

                training_status = gr.Text(label="Training Status")
            '''

            with gr.Row():
                controlnet_cond_scale = gr.Slider(
                    label="Controlnet Weight", value=0.5, minimum=0.0, maximum=1.0, step=0.1
                )
                controlnet_mix_mode = gr.Dropdown(
                    label="Controlnet Mix Mode", choices=controlnet_mix_modes, value=controlnet_mix_modes[0]
                )


            with gr.Row():
                batch_size = gr.Slider(
                    label="Batch Size", value=1, minimum=1, maximum=8, step=1
                )
                seed = gr.Slider(-1, 2147483647, label="Seed", value=-1, step=1)

            with gr.Row():
                guidance = gr.Slider(
                    label="Guidance scale", value=7.5, minimum=0, maximum=20
                )
                steps = gr.Slider(
                    label="Steps", value=20, minimum=1, maximum=100, step=1
                )

            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    value=default_img_size,
                    minimum=64,
                    maximum=1024,
                    step=32,
                )
                height = gr.Slider(
                    label="Height",
                    value=default_img_size,
                    minimum=64,
                    maximum=1024,
                    step=32,
                )

        with gr.Column():
            gallery = gr.Gallery(
                label="Generated images", show_label=False, elem_id="gallery"
            ).style(height=default_img_size, grid=2)

            generation_details = gr.Markdown()

            pipe_kwargs = gr.Textbox(label="Pipe kwargs", value="{\n\t\n}")

            # if torch.cuda.is_available():
            #  giga = 2**30
            #  vram_guage = gr.Slider(0, torch.cuda.memory_reserved(0)/giga, label='VRAM Allocated to Reserved (GB)', value=0, step=1)
            #  demo.load(lambda : torch.cuda.memory_allocated(0)/giga, inputs=[], outputs=vram_guage, every=0.5, show_progress=False)

    gr.HTML(footer)

    inputs = [
        model_name,
        controlnet_name,
        scheduler_name,
        prompt,
        guidance,
        steps,
        batch_size,
        width,
        height,
        seed,
        #image,
        #strength,
        #inpaint_image,
        #inpaint_strength,
        #inpaint_radio,
        neg_prompt,
        controlnet_prompt,
        controlnet_cond_scale,
        controlnet_mix_mode,
        pipe_state,
        pipe_kwargs,
    ]
    outputs = [gallery, generation_details]

    prompt.submit(generate, inputs=inputs, outputs=outputs)
    generate_button.click(generate, inputs=inputs, outputs=outputs)

    '''
    textual_inversion_inputs = [
        model_name,
        scheduler_name,
        type_of_thing,
        files,
        concept_word,
        init_word,
        text_train_steps,
        text_train_bsz,
        text_learning_rate,
    ]

    textual_inversion_button.click(
        train_textual_inversion,
        inputs=textual_inversion_inputs,
        outputs=[training_status],
    )
    '''

demo.queue(concurrency_count=cpu_count())

demo.launch()
