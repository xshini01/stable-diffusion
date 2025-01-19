import gc
import os
import torch
import gradio as gr
import time
import models
import bcrypt
from google.colab import userdata
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from transformers import AutoTokenizer, AutoModelForCausalLM
from compel import Compel, ReturnedEmbeddingsType
from IPython.display import clear_output

# Token set 
hf_token = None
token_set = False
ratings = None
    
def save_token(token):
    global hf_token, token_set, ratings
    hf_token = token
    token_set = True  
    masked_token = token[:4] + "*" * (len(token) - 4)
    ratings = set_ratings()
    return f"Your token: {masked_token}" if hf_token else "Continue without token"

# auto remove clip_skip if sdxl model
def clip_skip_visibility(model_id):
    model_id_lower = model_id.lower()
    if "sd-xl" in model_id_lower or "sdxl" in model_id_lower or "xl" in model_id_lower:
        return gr.update(visible=False)
    else:
        return gr.update(visible=True)


# load Model and Lora
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
        pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None if verify_token() else True, torch_dtype=torch.float16, token=hf_token if hf_token else None)
        
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

def verify_token(): 
    my_token = userdata.get('my_token')
    stored_hash = b'$2b$12$o.DA9bq6AOg.jL4848kIvu5oy2K/2Qs35dWENbi/p8yDQQH2epmZy'
    if my_token is None:
        print("my_token is not set in the environment variables.")
        return False
    if bcrypt.checkpw(my_token.encode('utf-8'), stored_hash): 
        return True  
    return False
        
    
# set scheduler 
def set_scheduler(scheduler_type, config, scheduler_args):

    SchedulerClass = models.PromptConfig().schedulers.get(scheduler_type, DPMSolverMultistepScheduler)
    if "Karras" in scheduler_type :
        return SchedulerClass.from_config(config, use_karras_sigmas=True, **scheduler_args)
    else:
        return SchedulerClass.from_config(config, **scheduler_args)

def set_ratings():
    config = models.PromptConfig()
    return config.options_ratings if verify_token() else config.options_ratings[0:3]

# generated imgs tags / genreated imgs prompts
def generated_imgs_tags(copyright_tags, character_tags, general_tags, rating, aspect_ratio_tags, Length_prompt, pipe, progress=gr.Progress(track_tqdm=True)):
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

# gradio notif copy form generated imgs tags
def gradio_copy_text(_text: None):
    gr.Info("Copied!")


# copy function
COPY_ACTION_JS = """\
(inputs, _outputs) => {
if (inputs.trim() !== "") {
    navigator.clipboard.writeText(inputs);
}
}"""

# current theme mode checking function
checkThemeMode="""
() => {
    const button = document.querySelector('button:nth-of-type(2)');
    button.innerText = document.body.classList.contains('dark') ? 'Light Mode' : 'Dark Mode';
}"""

# generated imgs
def generated_imgs(model_id, prompt, negative_prompt, scheduler_name, type_prediction, width, height, steps, scale, clip_skip, num_images,pipe, progress=gr.Progress(track_tqdm=True)):
    all_images = []
    gr.Progress(track_tqdm=True)
    model_id_lower = model_id.lower()

    scheduler_args = {
        "prediction_type": type_prediction,
        **({"rescale_betas_zero_snr": True} if type_prediction == "v_prediction" else {})
    }
 
    pipe.scheduler = set_scheduler(scheduler_name, pipe.scheduler.config, scheduler_args)

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
