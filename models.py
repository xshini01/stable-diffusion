from diffusers import (
    DPMSolverMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    LMSDiscreteScheduler,
)

class PromptConfig:
    def __init__(self):
        self.model_id = "Laxhar/noobai-XL-Vpred-1.0"
        self.lora_id = "xshini/Nakano_Miku_xl"
        self.copyright_tags = "Go-Toubun no Hanayome"
        self.character_tags = "nakano miku"
        self.general_tags = "masterpiece, 1 girl, solo"
        self.rating = "general"
        self.aspect_ratio_tags = "square"
        self.length_prompt = "medium"
        self.prompt = (
            "masterpiece, best quality, newest, absurdres, highres, "
            "1girl, nakano miku, solo, green skirt, headphones around neck, "
            "looking at viewer, blush, closed mouth, white shirt, long sleeves, "
            "blue cardigan, pleated skirt, black pantyhose"
        )
        self.negative_prompt = (
            "worst quality, old, early, nsfw, low quality, lowres, "
            "signature, username, logo, bad hands, mutated hands, mammal, "
            "anthro, ambiguous form, feral, semi-anthro"
        )
        self.width = 1024
        self.height = 1024
        self.steps = 30
        self.scale = 5
        self.clip_skip = 2
        self.num_images = 1
        self.options_models = [
            "stablediffusionapi/abyssorangemix3a1b", "Ojimi/anime-kawai-diffusion",
            "Linaqruf/anything-v3-1", "circulus/canvers-anime-v3.8.1",
            "redstonehero/cetusmix_v4", "DGSpitzer/Cyberpunk-Anime-Diffusion",
            "dreamlike-art/dreamlike-anime-1.0", "Lykon/dreamshaper-8",
            "emilianJR/majicMIX_realistic_v6", "Meina/MeinaMix_V11",
            "Meina/MeinaPastel_V7", "jzli/RealCartoon3D-v11", "Meina/MeinaUnreal_V5",
            "redstonehero/xxmix_9realistic_v40", "stablediffusionapi/yesmix-v35",
            "Lykon/AAM_AnyLora_AnimeMix", "Lykon/AnyLoRA", "xshini/pooribumix_V1",
            "GraydientPlatformAPI/perfectpony-xl", "cagliostrolab/animagine-xl-3.1",
            "John6666/anima-pencil-xl-v5-sdxl", "Laxhar/noobai-XL-Vpred-1.0"
        ]
        self.options_loras = [
            "xshini/KizunaAi", "xshini/NakanoMiku", "xshini/HiguchiKaede",
            "xshini/tokisaki-Kurumi-XL", "xshini/Nakano_Miku_xl"
        ]
        self.schedulers = {
            "DPM++ 2M": DPMSolverMultistepScheduler,
            "DPM++ 2M Karras": DPMSolverMultistepScheduler,
            "DPM++ SDE": DPMSolverSinglestepScheduler,
            "DPM++ SDE Karras": DPMSolverSinglestepScheduler,
            "DPM2": KDPM2DiscreteScheduler,
            "DPM2 Karras": KDPM2DiscreteScheduler,
            "DPM2 a": KDPM2AncestralDiscreteScheduler,
            "DPM2 a Karras": KDPM2AncestralDiscreteScheduler,
            "Euler": EulerDiscreteScheduler,
            "Euler a": EulerAncestralDiscreteScheduler,
            "Heun": HeunDiscreteScheduler,
            "LMS": LMSDiscreteScheduler,
            "LMS Karras": LMSDiscreteScheduler,
        }
        self.options_ratings = ["sfw", "general", "sensitive", "nsfw", "questionable", "explicit"]
        self.options_aspect_ratio = ["ultra_wide", "wide", "square", "tall", "ultra_tall"]
        self.options_length_prompt = ["very_short", "short", "medium", "long", "very_long"]