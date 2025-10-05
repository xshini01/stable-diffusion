import gc
import os
import torch
import gradio as gr
import requests
import models
import re
from huggingface_hub import snapshot_download
from tqdm import tqdm
import bcrypt
from google.colab import userdata
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from transformers import AutoTokenizer, AutoModelForCausalLM
from compel import CompelForSD, CompelForSDXL
from IPython.display import clear_output

hf_token = None
token_set = False
ratings = None
civitai_token = None

# save token
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

# check SDXL model
def is_sdxl(model_path):
    return any(keyword in model_path for keyword in ["sd-xl", "sdxl", "xl", "illustrious"])


def is_file_exists(path, file_target):

    os.makedirs(path, exist_ok=True)

    files = os.listdir(path)

    if not file_target.endswith(".safetensors"):
        normalized_file_target = re.sub(r'[-/]', '', file_target)
        
        for name_file in files:
            normalized_name_file = re.sub(r'[-/]', '', name_file)
            lora_file = os.path.splitext(name_file)[0]
            if normalized_file_target == normalized_name_file or lora_file == file_target:
                return True
    else:
        for name_file in files:
            if name_file == file_target:
                return True
                
    return False


# auto remove clip_skip if sdxl model
def clip_skip_visibility(model_id):
    if is_sdxl(model_id.lower()):
        return gr.update(visible=False)
    else :
        return gr.update(visible=True)

# download model/lora
def download_file(url, save_dir, progress=gr.Progress(track_tqdm=True)):
    
    os.makedirs(save_dir, exist_ok=True)
    headers = {}
    if civitai_token:
        headers["Authorization"] = f"Bearer {civitai_token}"
    response = requests.get(url, stream=True, headers=headers if civitai_token else "")
    total_size = int(response.headers.get("content-length", 0))
    progress = tqdm(total=total_size, unit="B", unit_scale=True, desc= "Downloading")
    
    if response.status_code == 200:
        # get file name from header Content-Disposition
        if "content-disposition" in response.headers:
            content_disposition = response.headers["content-disposition"]
            filename = content_disposition.split("filename=")[-1].strip('"')
        else:
            # or get file name from url
            model_id = url.split("/")[-1].split("?")[0]
            filename = f"{model_id}.safetensors"
        
        # save file
        clip_skip_visibility(filename)
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
    model_name = None
    if pipe and not is_file_exists("/content/stable-diffusion/models", model_id) :
        del pipe
        torch.cuda.empty_cache()
        gc.collect()
    try:
        
        if is_file_exists("/content/stable-diffusion/models", model_id):
            pass
        # Handle URL model
        elif model_id.startswith(("http://", "https://")):
            gr.Info("Downloading model...")
            progress(0.1, desc="Downloading model")
            
            model_path = download_file(model_id, "/content/stable-diffusion/models")
            model_name = os.path.basename(model_path)
            
            pipeline_class = StableDiffusionXLPipeline if is_sdxl(model_name.lower()) else StableDiffusionPipeline
            pipe = pipeline_class.from_single_file(
                model_path,
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
                use_safetensors=True,
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
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e :
        gr.Warning(f"Loading Failed: {str(e)}")
    
    # Load Lora
    lora_name = None
    if lora_id:
        try:
            gr.Info("Loading LoRA...")
            # Handle URL lora
            if is_file_exists("/content/stable-diffusion/loras", lora_id):
                pass
            elif lora_id.startswith(("http://", "https://")):
                lora_path = download_file(lora_id, "/content/stable-diffusion/loras")
                lora_name = os.path.splitext(os.path.basename(lora_path))[0]
                pipe.load_lora_weights(lora_path, adapter_name=lora_name)
                pipe.fuse_lora(lora_scale=0.7)
            # Handle huggingface repo
            else:
                lora_path = snapshot_download(repo_id=lora_id, cache_dir="/content/stable-diffusion/loras")
                pipe.load_lora_weights(lora_path, adapter_name=lora_id)
                pipe.fuse_lora(lora_scale=0.7)
            gr.Info(f"LoRA {lora_id} loaded successfully")    
        except Exception as e:
            gr.Warning(f"LoRA Error: {str(e)}")
            gr.Info("Proceeding without LoRA")

    # Move to GPU
    pipe = pipe.to("cuda")
    gr.Info(f"Load Model {model_id} and {lora_id} Success")
    progress(1, desc="Model loaded successfully")
    generate_imgs = gr.Button(interactive=True)
    generated_imgs_with_tags = gr.Button()
    clear_output()
    if btn_check:
        generated_imgs_with_tags = gr.Button(interactive=True)
    return pipe, gr.update(value= model_name or model_id), gr.update(value=lora_name or lora_id), generate_imgs, generated_imgs_with_tags

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

# generated imgs tags / generated imgs prompts
def generated_imgs_tags(copyright_tags, character_tags, general_tags, rating, aspect_ratio_tags, length_prompt, pipe, progress=gr.Progress(track_tqdm=True)):
    MODEL_NAME = "p1atdev/dart-v2-moe-sft"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    prompt_template = (
    f"<|bos|>"
    f"<copyright>{copyright_tags}</copyright>"
    f"<character>{character_tags}</character>"
    f"<|rating:{rating}|><|aspect_ratio:{aspect_ratio_tags}|><|length:{length_prompt}|>"
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
    output_dir = "/content/stable-diffusion/images"
    os.makedirs(output_dir, exist_ok=True)
    gr.Progress(track_tqdm=True)

    scheduler_args = {
        "prediction_type": type_prediction,
        **({"rescale_betas_zero_snr": True} if type_prediction == "v_prediction" else {})
    }
 
    pipe.scheduler = set_scheduler(scheduler_name, pipe.scheduler.config, scheduler_args)

    for _ in range(num_images):
        # Handle prompt SDXL model
        if is_sdxl(model_id.lower()):
            compel = CompelForSDXL(pipe)
            conditioning = compel(prompt, negative_prompt=negative_prompt)
           

            image = pipe(prompt_embeds=conditioning.embeds, 
                pooled_prompt_embeds=conditioning.pooled_embeds,
                negative_prompt_embeds=conditioning.negative_embeds,
                negative_pooled_prompt_embeds=conditioning.negative_pooled_embeds,
                height=height, num_inference_steps=steps, width=width,guidance_scale=scale).images[0]

        else:
            compel = CompelForSD(pipe)
            conditioning = compel(prompt, negative_prompt=negative_prompt)

            image = pipe(prompt_embeds=conditioning.embeds, 
                negative_prompt_embed=conditioning.negative_embeds,
                height=height, num_inference_steps=steps, width=width,guidance_scale=scale,clip_skip=clip_skip).images[0]

        image_path = os.path.join(output_dir, f"output_image_{len(all_images) + 1}.png")
        image.save(image_path)
        all_images.append(image_path)
    return all_images
