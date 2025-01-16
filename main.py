import gradio as gr
import models
import utils
import time
from IPython.display import clear_output

clear_output()
print("Setup Complete")

all_images = []
config = models.PromptConfig()

def main():
    
    # gradio interface
    with gr.Blocks() as token_interface:
        gr.Markdown("## Hugging Face token")
        token_input = gr.Textbox(
            label="input your Hugging Face token (opsional)",
            placeholder="Enter your token here (opsional)...",
            type="password"
        )
        save_button = gr.Button("Submit", variant="primary")
        output_label = gr.Label(label= "your token :")
        save_button.click(fn=utils.save_token, inputs=token_input, outputs=output_label)
    
    token_interface.launch(share=True)

    while not utils.token_set:
        time.sleep(1)

    with gr.Blocks(theme='JohnSmith9982/small_and_pretty', js=utils.checkThemeMode) as ui:
        with gr.Row():
            gr.Markdown(
                """
                # **Basic Stable Diffusion**

                ***support SDXL model***

                **by Xshini/KZR**
                """
            )

        with gr.Row(show_progress=True, variant="panel" ):
            model_id_input = gr.Dropdown(choices=config.options_models,label="Model", value=config.model_id, allow_custom_value=True)
            lora_id_input = gr.Dropdown(choices=config.options_loras,label="LoRA", value=config.lora_id, allow_custom_value=True)
            with gr.Column():
                load_model_btn = gr.Button("Load Model", variant="primary", size='lg')
                toggle_dark = gr.Button(value="", size='lg')
                toggle_dark.click(
                    None,
                    js="""
                    () => {
                        const button = document.querySelector('button:nth-of-type(2)');
                        document.body.classList.toggle('dark');
                        button.innerText = document.body.classList.contains('dark') ? 'Light Mode' : 'Dark Mode';
                    }
                    """,
                )


        with gr.Row():
            with gr.Column(variant ='panel'):
                copyright_tags_input = gr.Textbox(label="Copyright Tags", value=config.copyright_tags, lines=2)
                character_tags_input = gr.Textbox(label="Character Tags", value=config.character_tags, lines=2)
                general_tags_input = gr.Textbox(label="General Tags", value=config.general_tags, lines=2)
                rating_input = gr.Radio(utils.ratings, label="Rating", value=config.rating)
                aspect_ratio_tags_input = gr.Radio(config.options_aspect_ratio, label="Aspect Ratio", value=config.aspect_ratio_tags)
                length_prompt_input = gr.Radio(config.options_length_prompt, label="Length Prompt", value=config.length_prompt)
            with gr.Column(variant ='panel'):
                generated_imgs_tags_btn = gr.Button("Generate Prompt", variant="primary")
                with gr.Group():
                    prompt_output = gr.Textbox(label="Generate Prompt", info="this is an optional feature, you can directly generate images in ''Advanced Prompt Images''", lines=3, value="", interactive=False)
                    clipboard_btn = gr.Button(value="Copy to clipboard", interactive=False,)
                generated_imgs_with_tags_btn = gr.Button(value="Generate image with this prompt!",variant='primary', interactive=False)

                with gr.Accordion(label="Advanced Prompt Images", open=False):
                    model_id_lower = model_id_input.value.lower()
                    prompt_input = gr.Textbox(label="Prompt", value=config.prompt, lines=5)
                    negative_prompt_input = gr.Textbox(label="Negative Prompt", value=config.negative_prompt, lines=3)
                    with gr.Row():
                        scheduler_input = gr.Dropdown(
                            choices=list(config.schedulers.keys()),
                            label="Scheduler",
                            value="Euler",
                            interactive=True,
                        )

                        type_prediction_input = gr.Dropdown(
                            choices=["epsilon", "v_prediction"],
                            label="Type Prediction",
                            value="v_prediction",
                            interactive=True,
                        )

                    width_input = gr.Slider(minimum=256, maximum=2048, step=64, label="Width", value=config.width)
                    height_input = gr.Slider(minimum=256, maximum=2048, step=64, label="Height", value=config.height)
                    steps_input = gr.Slider(minimum=1, maximum=50, step=1, label="Steps", value=config.steps)
                    scale_input = gr.Slider(minimum=1, maximum=20, step=0.5, label="Scale", value=config.scale)
                    clip_skip_input = gr.Slider(minimum=1, maximum=12, step=1, label="Clip Skip", value=config.clip_skip, visible=True)
                    if "sd-xl" in model_id_lower or "sdxl" in model_id_lower or "xl" in model_id_lower:
                        clip_skip_input.visible = False
                    num_images_input = gr.Slider(minimum=1, maximum=5, step=1, label="Number of Images", value=config.num_images)
                    generated_imgs_btn = gr.Button("Generate Images", variant="primary", interactive=False)
                image_output = gr.Gallery(label="Generated Image",show_label=False,columns=[2], rows=[2], object_fit="contain", height="auto")
                
        btn_check = gr.State()
        pipe = gr.State()
        model_id_input.change(
            utils.clip_skip_visibility, 
            inputs= model_id_input, 
            outputs= clip_skip_input
        )
        load_model_btn.click(
            utils.load_model, 
            inputs=[
                model_id_input, 
                lora_id_input, 
                btn_check,
                pipe
            ], 
            outputs=[
                pipe, 
                model_id_input, 
                lora_id_input, 
                generated_imgs_btn, 
                generated_imgs_with_tags_btn
            ]
        )
        generated_imgs_tags_btn.click(
            utils.generated_imgs_tags, 
            inputs=[
                copyright_tags_input, 
                character_tags_input, 
                general_tags_input, 
                rating_input, 
                aspect_ratio_tags_input, 
                length_prompt_input, 
                pipe
            ], 
            outputs=[
                prompt_output,
                clipboard_btn, 
                generated_imgs_with_tags_btn, 
                btn_check
            ]
        )
        clipboard_btn.click(
            utils.gradio_copy_text, 
            inputs= prompt_output, 
            js=utils.COPY_ACTION_JS
        )
        generated_imgs_with_tags_btn.click(
            utils.generated_imgs, 
            inputs=[
                model_id_input, 
                prompt_output, 
                negative_prompt_input, 
                scheduler_input, 
                type_prediction_input, 
                width_input, 
                height_input, 
                steps_input, 
                scale_input, 
                clip_skip_input, 
                num_images_input,
                pipe
            ], 
            outputs= image_output
        )
        generated_imgs_btn.click(
            utils.generated_imgs, 
            inputs=[
                model_id_input, 
                prompt_input, 
                negative_prompt_input, 
                scheduler_input, 
                type_prediction_input, 
                width_input, 
                height_input, 
                steps_input, 
                scale_input, 
                clip_skip_input, 
                num_images_input,
                pipe
            ], 
            outputs= image_output
        )
        clear_output()
        ui.queue() 
        ui.launch(share=True, inline=False, debug=True)

if __name__ == "__main__": 
    main()