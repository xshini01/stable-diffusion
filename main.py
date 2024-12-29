import gradio as gr
import time
import gc
import torch
from IPython.display import clear_output
from diffusers import StableDiffusionPipeline, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, HeunDiscreteScheduler, LMSDiscreteScheduler, DPMSolverSinglestepScheduler, StableDiffusionXLPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from compel import Compel, ReturnedEmbeddingsType

clear_output()
print("Setup Complete")

#function

hf_token = None
token_set = False

def save_token(token):
    global hf_token, token_set
    hf_token = token
    token_set = True  
    masked_token = token[:4] + "*" * (len(token) - 4)
    if hf_token:
      return f"Your token: {masked_token}"
    else:
      return "continue without token"

def update_clip_skip_visibility(model_id):
    model_id_lower = model_id.lower()
    if "sd-xl" in model_id_lower or "sdxl" in model_id_lower or "xl" in model_id_lower:
        return gr.update(visible=False)
    else:
        return gr.update(visible=True)

def load_model(model_id, lora_id, btn_check, pipe, progress=gr.Progress(track_tqdm=True)):

    model_id_lower = model_id.lower()
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    if "sd-xl" in model_id_lower or "sdxl" in model_id_lower or "xl" in model_id_lower:
        gr.Info("wait a minute the model is loading!")
        progress(0.2, desc="Starting model loading")
        time.sleep(1)
        pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16, token=hf_token if hf_token else None)
    else:
        gr.Info("wait a minute the model is loading!")
        progress(0.2, desc="Starting model loading")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None, torch_dtype=torch.float16, token=hf_token if hf_token else None)
        
    pipe.enable_xformers_memory_efficient_attention()

    if lora_id:
        try:
            gr.Info("wait a minute the Lora is loading!")
            progress(0.5, desc="Load LoRA weight")
            pipe.load_lora_weights(lora_id, adapter_name=lora_id)
            pipe.fuse_lora(lora_scale=0.7)
            gr.Info(f"Load LoRA {lora_id} Success")
        except Exception as e:
            gr.Info(f"LoRA {lora_id} not compatible with model {model_id}")
            gr.Info(f"Use another Lora, if sdxl model use Lora xl")
            gr.Info(f"Load Model without LoRA")
    else:
        print(f"without lora")

    pipe = pipe.to("cuda")
    gr.Info(f"Load Model {model_id} and {lora_id} Success")
    progress(1, desc="Model loaded successfully")
    generate_imgs = gr.Button(interactive=True)
    generated_imgs_with_tags = gr.Button()
    clear_output()
    if btn_check:
        generated_imgs_with_tags = gr.Button(interactive=True)
    return pipe, model_id, lora_id, generate_imgs, generated_imgs_with_tags

def create_scheduler(scheduler_type, config, scheduler_args):

    SchedulerClass = schedulers.get(scheduler_type, DPMSolverMultistepScheduler)
    if "Karras" in scheduler_type :
        return SchedulerClass.from_config(config, use_karras_sigmas=True, **scheduler_args)
    else:
        return SchedulerClass.from_config(config, **scheduler_args)

def generated_imgs_tags(copyright_tags, character_tags, general_tags, rating, aspect_ratio_tags, Length_prompt, pipe):
    MODEL_NAME = "p1atdev/dart-v2-moe-sft"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    prompt_template = (
    f"<|bos|>"
    f"<copyright>{copyright_tags}</copyright>"
    f"<character>{character_tags}</character>"
    f"<|rating:{rating}|><|aspect_ratio:{aspect_ratio_tags}|><|length:{Length_prompt}|>"
    f"<general>{general_tags}<|identity:none|><|input_end|>"
    )
    inputs = tokenizer(prompt_template, return_tensors="pt").input_ids

    with torch.no_grad():
        outputs = model.generate(
        inputs,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=100,
        max_new_tokens=128,
        num_beams=1,
        )

    generated_text = ", ".join([tag for tag in tokenizer.batch_decode(outputs[0], skip_special_tokens=True) if tag.strip() != ""])
    copy = gr.Button(interactive=True)
    tags_imgs = gr.Button()
    btn_check = generated_text
    if pipe:
      tags_imgs = gr.Button(interactive=True)
    return generated_text, copy, tags_imgs, btn_check

def gradio_copy_text(_text: None):
    gr.Info("Copied!")

COPY_ACTION_JS = """\
(inputs, _outputs) => {
if (inputs.trim() !== "") {
    navigator.clipboard.writeText(inputs);
}
}"""

checkThemeMode="""
() => {
    const button = document.querySelector('button:nth-of-type(2)');
    button.innerText = document.body.classList.contains('dark') ? 'Light Mode' : 'Dark Mode';
}"""

all_images = []
def generated_imgs(model_id, prompt, negative_prompt, scheduler_name, type_prediction, width, height, steps, scale, clip_skip, num_images,pipe, progress=gr.Progress(track_tqdm=True)):
    all_images = []
    model_id_lower = model_id.lower()

    scheduler_args = {
        "prediction_type": type_prediction,
        **({"rescale_betas_zero_snr": True} if type_prediction == "v_prediction" else {})
    }

    pipe.scheduler = create_scheduler(scheduler_name, pipe.scheduler.config, scheduler_args)

    for _ in range(num_images):
        if "sd-xl" in model_id_lower or "sdxl" in model_id_lower or "xl" in model_id_lower:
            compel = Compel(
                tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True]
            )
            conditioning, pooled = compel(prompt)
            image = pipe(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, height=height, num_inference_steps=steps, width=width,
                         negative_prompt=negative_prompt, guidance_scale=scale).images[0]
        else:
            compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
            embeds = compel(prompt)
            image = pipe(prompt_embeds=embeds, height=height, num_inference_steps=steps, width=width,
                         negative_prompt=negative_prompt, guidance_scale=scale, clip_skip=clip_skip).images[0]
        image_path = f"output_image_{len(all_images) + 1}.png"
        image.save(image_path)
        all_images.append(image_path)
    return all_images

# default value
model_id = "Laxhar/noobai-XL-Vpred-0.9r"
lora_id = "xshini/Nakano_Miku_xl"

copyright_tags= "Go-Toubun no Hanayome"
character_tags = "nakano miku"
general_tags = "masterpiece, best quality, newest, absurdres, highres"
rating = "general"
aspect_ratio_tags = "square"
Length_prompt= "short"

prompt = "masterpiece, best quality, newest, absurdres, highres, 1girl, nakano miku, solo, green skirt, headphones around neck, looking at viewer, blush, closed mouth, white shirt, long sleeves, blue cardigan, pleated skirt, black pantyhose"
negative_prompt = "worst quality, old, early, nsfw, low quality, lowres, signature, username, logo, bad hands, mutated hands, mammal, anthro, ambiguous form, feral, semi-anthro"
width = 1024
height = 1024
steps = 30
scale = 5
clip_skip= 2
num_images = 1

# Choice input
choices_Models = ["stablediffusionapi/abyssorangemix3a1b","Ojimi/anime-kawai-diffusion","Linaqruf/anything-v3-1","circulus/canvers-anime-v3.8.1",
                 "redstonehero/cetusmix_v4","DGSpitzer/Cyberpunk-Anime-Diffusion","dreamlike-art/dreamlike-anime-1.0","Lykon/dreamshaper-8",
                 "emilianJR/majicMIX_realistic_v6","Meina/MeinaMix_V11","Meina/MeinaPastel_V7","jzli/RealCartoon3D-v11","Meina/MeinaUnreal_V5",
                 "redstonehero/xxmix_9realistic_v40","stablediffusionapi/yesmix-v35","Lykon/AAM_AnyLora_AnimeMix","Lykon/AnyLoRA","xshini/pooribumix_V1",
                 "GraydientPlatformAPI/perfectpony-xl","cagliostrolab/animagine-xl-3.1","John6666/anima-pencil-xl-v5-sdxl", "Laxhar/noobai-XL-Vpred-0.9r"]

choices_Loras = ["xshini/KizunaAi","xshini/NakanoMiku","xshini/HiguchiKaede","xshini/tokisaki-Kurumi-XL","xshini/Nakano_Miku_xl"]

schedulers = {
    "DPM++ 2M": DPMSolverMultistepScheduler,
    "DPM++ 2M Karras": DPMSolverMultistepScheduler,
    "DPM++ SDE": DPMSolverSinglestepScheduler,
    "DPM++ SDE Karras": DPMSolverSinglestepScheduler,
    "DPM2": KDPM2DiscreteScheduler,
    "DPM2 Karras": KDPM2DiscreteScheduler,
    "DPM2 a": KDPM2AncestralDiscreteScheduler,
    "DPM2 a Karras": KDPM2AncestralDiscreteScheduler,
    "Euler": EulerDiscreteScheduler,
    "Euler a": EulerAncestralDiscreteScheduler	,
    "Heun": HeunDiscreteScheduler,
    "LMS": LMSDiscreteScheduler,
    "LMS Karras": LMSDiscreteScheduler,
}
choices_Ratings =["sfw","general","sensitive"]
choices_AspectRasio =["ultra_wide","wide","square", "tall", "ultra_tall"]
choices_LongPrompt= ["very_short","short","medium", "long", "very_long"]

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
    save_button.click(fn=save_token, inputs=token_input, outputs=output_label)

with gr.Blocks(theme='JohnSmith9982/small_and_pretty', js=checkThemeMode) as ui:
    with gr.Row():
        gr.Markdown(
            """
            # **Basic Stable Diffusion**

            ***support SDXL model***

            **by Xshini/KZR**
            """
        )

    with gr.Row(show_progress=True, variant="panel" ):
        model_id_input = gr.Dropdown(choices=choices_Models,label="Model", value=model_id, allow_custom_value=True)
        lora_id_input = gr.Dropdown(choices=choices_Loras,label="LoRA", value=lora_id, allow_custom_value=True)
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
            copyright_tags_input = gr.Textbox(label="Copyright Tags", value=copyright_tags, lines=2)
            character_tags_input = gr.Textbox(label="Character Tags", value=character_tags, lines=2)
            general_tags_input = gr.Textbox(label="General Tags", value=general_tags, lines=2)
            rating_input = gr.Radio(choices_Ratings, label="Rating", value=rating)
            aspect_ratio_tags_input = gr.Radio(choices_AspectRasio, label="Aspect Ratio", value=aspect_ratio_tags)
            Length_prompt_input = gr.Radio(choices_LongPrompt, label="Length Prompt", value=Length_prompt)
        with gr.Column(variant ='panel'):
            generated_imgs_tags_btn = gr.Button("Generate Prompt", variant="primary")
            with gr.Group():
                prompt_output = gr.Textbox(label="Generate Prompt", info="this is an optional feature, you can directly generate images in ''Advanced Prompt Images''", lines=3, value="", interactive=False)
                clipboard_btn = gr.Button(value="Copy to clipboard", interactive=False,)
            generated_imgs_with_tags_btn = gr.Button(value="Generate image with this prompt!",variant='primary', interactive=False)

            with gr.Accordion(label="Advanced Prompt Images", open=False):
                model_id_lower = model_id_input.value.lower()
                prompt_input = gr.Textbox(label="Prompt", value=prompt, lines=5)
                negative_prompt_input = gr.Textbox(label="Negative Prompt", value=negative_prompt, lines=3)
                with gr.Row():
                    scheduler_input = gr.Dropdown(
                        choices=list(schedulers.keys()),
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

                width_input = gr.Slider(minimum=256, maximum=2048, step=64, label="Width", value=width)
                height_input = gr.Slider(minimum=256, maximum=2048, step=64, label="Height", value=height)
                steps_input = gr.Slider(minimum=1, maximum=50, step=1, label="Steps", value=steps)
                scale_input = gr.Slider(minimum=1, maximum=20, step=0.5, label="Scale", value=scale)
                clip_skip_input = gr.Slider(minimum=1, maximum=12, step=1, label="Clip Skip", value=clip_skip, visible=True)
                if "sd-xl" in model_id_lower or "sdxl" in model_id_lower or "xl" in model_id_lower:
                    clip_skip_input.visible = False
                num_images_input = gr.Slider(minimum=1, maximum=5, step=1, label="Number of Images", value=num_images)
                generated_imgs_btn = gr.Button("Generate Images", variant="primary", interactive=False)
            image_output = gr.Gallery(label="Generated Image",show_label=False,columns=[2], rows=[2], object_fit="contain", height="auto")

    btn_check = gr.State()
    pipe = gr.State()
    model_id_input.change(
        update_clip_skip_visibility, 
        inputs= model_id_input, 
        outputs= clip_skip_input
    )
    load_model_btn.click(
        load_model, 
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
        generated_imgs_tags, 
        inputs=[
            copyright_tags_input, 
            character_tags_input, 
            general_tags_input, 
            rating_input, 
            aspect_ratio_tags_input, 
            Length_prompt_input, 
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
        gradio_copy_text, 
        inputs= prompt_output, 
        js=COPY_ACTION_JS
    )
    generated_imgs_with_tags_btn.click(
        generated_imgs, 
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
        generated_imgs, 
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

def main():
    clear_output()
    token_interface.launch(share=True)
    while not token_set:
        pass
    time.sleep(2)
    clear_output()
    ui.queue()
    ui.launch(share=True, inline=False, debug=True)

main()
