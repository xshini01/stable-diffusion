import gc
import os
import torch
import gradio as gr
import time
import requests
import models
from tqdm import tqdm
import bcrypt
from google.colab import userdata
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from transformers import AutoTokenizer, AutoModelForCausalLM
from compel import Compel, ReturnedEmbeddingsType
from IPython.display import clear_output

hf_token = None
token_set = False
ratings = None
civitai_token = None
model_name= None
    
def save_token(token):
    global hf_token, token_set, ratings, civitai_token
    if token.startswith("hf_"):
        hf_token = token
        token_type = "Hugging Face"
    else:
        civitai_token = token
        token_type = "Civitai"
    token_set = True  
    masked_token = token[:4] + "*" * (len(token) - 4)
    ratings = set_ratings()
    if hf_token or civitai_token:
        return f"Your {token_type} token: {masked_token}"
    else:
        return "Continue without token"

def is_sdxl(model_path):
    return any(keyword in model_path for keyword in ["sd-xl", "sdxl", "xl", "illustrious"])

# auto remove clip_skip if sdxl model
def clip_skip_visibility(model_id):
    if is_sdxl(model_id.lower()):
        return gr.update(visible=False)
    if model_name is not None and is_sdxl(model_name.lower()):
        return gr.update(visible=False)
    return gr.update(visible=True)
    
def download_file(url):
    save_dir="/content/stable-diffusion/models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    headers = {}
    if civitai_token:
        headers["Authorization"] = f"Bearer {civitai_token}"
    response = requests.get(url, stream=True, headers=headers if civitai_token else "")
    total_size = int(response.headers.get("content-length", 0))
    progress = tqdm(total=total_size, unit="B", unit_scale=True)

    # Cek jika respons sukses
    if response.status_code == 200:
        # Ambil nama file dari header Content-Disposition (jika ada)
        if "content-disposition" in response.headers:
            content_disposition = response.headers["content-disposition"]
            filename = content_disposition.split("filename=")[-1].strip('"')
        else:
            # Jika tidak ada, buat nama file berdasarkan nomor model dan format
            model_id = url.split("/")[-1].split("?")[0]  # Ambil nomor model
            filename = f"model_{model_id}.safetensors"
        
        # Simpan file dengan nama asli
        file_path = os.path.join(save_dir, filename)
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                progress.update(len(chunk))
        progress.close()
        return file_path
    else:
        if response.status_code == 401 :
            gr.Error("Failed to download file. Please input civitai tokens")
        else :
            gr.Error(f"Failed to download file. status code status: {response.status_code}")
        return None

# load Model and Lora
def load_model(model_id, lora_id, btn_check, pipe, progress=gr.Progress(track_tqdm=True)):
    global model_name

    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    try:
        # Handle URL model
        if model_id.startswith(("http://", "https://")):
            gr.Info("Downloading model...")
            progress(0.1, desc="Downloading model")
            
            download_path = download_file(model_id)
            model_name = os.path.basename(download_path)
            
            pipeline_class = StableDiffusionXLPipeline if is_sdxl(model_name.lower()) else StableDiffusionPipeline
            pipe = pipeline_class.from_single_file(
                download_path,
                torch_dtype=torch.float16,
                safety_checker=None if verify_token() else True,
                token=hf_token or None
            )

        # Handle huggingFace SDXL model
        elif is_sdxl(model_id.lower()):
            gr.Info("Loading SDXL model...")
            progress(0.2, desc="Loading SDXL model")
            
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                cache_dir="/content/stable-diffusion/models",
                torch_dtype=torch.float16,
                token=hf_token or None,
                safety_checker=None if verify_token() else True
            )

        # Handle huggingFace standard Stable Diffusion model
        else:
            gr.Info("Loading standard model...")
            progress(0.2, desc="Loading standard model")
            
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                cache_dir="/content/stable-diffusion/models",
                torch_dtype=torch.float16,
                token=hf_token or None,
                safety_checker=None if verify_token() else True
            )
    
        if lora_id:
            try:
                gr.Info("Loading LoRA...")
                progress(0.5, desc="Loading LoRA")
                
                pipe.load_lora_weights(lora_id, adapter_name=lora_id)
                pipe.fuse_lora(lora_scale=0.7)
                gr.Info(f"LoRA {lora_id} loaded successfully")
            except Exception as e:
                gr.Warning(f"LoRA Error: {str(e)}")
                gr.Info("Proceeding without LoRA")

        # Move to GPU
        pipe = pipe.enable_xformers_memory_efficient_attention()
        pipe = pipe.to("cuda")
        gr.Info(f"Load Model {model_id} and {lora_id} Success")
        progress(1, desc="Model loaded successfully")
        
    except Exception as e :
        gr.Warning(f"Loading Failed: {str(e)}")
    generate_imgs = gr.Button(interactive=True)
    generated_imgs_with_tags = gr.Button()
    clear_output()
    if btn_check:
        generated_imgs_with_tags = gr.Button(interactive=True)
    return pipe, model_id, lora_id, generate_imgs, generated_imgs_with_tags

def verify_token(): 
    stored_hash = b'$2b$12$o.DA9bq6AOg.jL4848kIvu5oy2K/2Qs35dWENbi/p8yDQQH2epmZy'
    try:
        my_token = userdata.get('my_token')
    except userdata.SecretNotFoundError or userdata.NotebookAccessError:
        return False
    if my_token is None:
        return False      
    return bcrypt.checkpw(my_token.encode('utf-8'), stored_hash)
        
    
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

    scheduler_args = {
        "prediction_type": type_prediction,
        **({"rescale_betas_zero_snr": True} if type_prediction == "v_prediction" else {})
    }
 
    pipe.scheduler = set_scheduler(scheduler_name, pipe.scheduler.config, scheduler_args)

    for _ in range(num_images):
        if is_sdxl(model_id.lower()) or (model_name is not None and is_sdxl(model_name.lower())):
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
