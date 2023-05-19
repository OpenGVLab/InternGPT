import torch
from diffusers import StableUnCLIPImg2ImgPipeline
from .utils import gen_new_name, prompts

from . import imagebind as ib


class Anything2Image:
    def __init__(self, device):
        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
        )
        self.device = device
        self.pipe = pipe.to(device)
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_slicing()

        self.model = ib.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(device)


class Audio2Image:
    template_model = True

    def __init__(self, Anything2Image):
        self.pipe = Anything2Image.pipe
        self.model = Anything2Image.model
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,' \
            ' fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image from Audio",
             description="useful when you want to generate a real image from audio. "
                         "like: generate a real image from audio, "
                         "or generate a new image based on the given audio. "
                         "The input to this tool should be a string, representing the audio_path")
    @torch.no_grad()
    def inference(self, inputs):
        audio_paths = [inputs]
        embeddings = self.model.forward({
            ib.ModalityType.AUDIO: ib.load_and_transform_audio_data(audio_paths, self.device),
        })
        embeddings = embeddings[ib.ModalityType.AUDIO]
        images = self.pipe(image_embeds=embeddings.half(), width=512, height=512).images
        new_img_name = gen_new_name(audio_paths[0], 'Audio2Image')
        images[0].save(new_img_name)
        return new_img_name


class Thermal2Image:
    template_model = True

    def __init__(self, Anything2Image):
        self.pipe = Anything2Image.pipe
        self.model = Anything2Image.model
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,' \
            ' fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image from Thermal Image",
             description="useful when you want to generate a real image from a thermal image. "
                         "like: generate a real image from thermal image, "
                         "or generate a new image based on the given thermal image. "
                         "The input to this tool should be a string, representing the thermal_path")
    @torch.no_grad()
    def inference(self, inputs):
        thermal_paths = [inputs]
        embeddings = self.model.forward({
            ib.ModalityType.THERMAL: ib.load_and_transform_thermal_data(thermal_paths, self.device),
        })
        embeddings = embeddings[ib.ModalityType.THERMAL]
        images = self.pipe(image_embeds=embeddings.half(), width=512, height=512).images
        new_img_name = gen_new_name(thermal_paths[0], 'Thermal2Image')
        images[0].save(new_img_name)
        return new_img_name


class AudioImage2Image:
    template_model = True

    def __init__(self, Anything2Image):
        self.pipe = Anything2Image.pipe
        self.model = Anything2Image.model
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,' \
            ' fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image from Audio and Image",
             description="useful when you want to generate a real image from audio and image. "
                         "like: generate a real image from audio and image, "
                         "or generate a new image based on the given audio and image. "
                         "The input to this tool should be two string, representing the audio_path and image path")
    @torch.no_grad()
    def inference(self, inputs):
        audio_path, image_path = inputs.split(',')
        audio_path, image_path = audio_path.strip(), image_path.strip()
        embeddings = self.model.forward({
            ib.ModalityType.VISION: ib.load_and_transform_vision_data([image_path], self.device),
        }, normalize=False)
        img_embeddings = embeddings[ib.ModalityType.VISION]
        embeddings = self.model.forward({
            ib.ModalityType.AUDIO: ib.load_and_transform_audio_data([audio_path], self.device),
        })
        audio_embeddings = embeddings[ib.ModalityType.AUDIO]
        embeddings = (img_embeddings + audio_embeddings) / 2
        images = self.pipe(image_embeds=embeddings.half(), width=512, height=512).images
        new_img_name = gen_new_name(audio_path, 'AudioImage2Image')
        images[0].save(new_img_name)
        return new_img_name


class AudioText2Image:
    template_model = True

    def __init__(self, Anything2Image):
        self.pipe = Anything2Image.pipe
        self.model = Anything2Image.model
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,' \
            ' fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image from Audio",
             description="useful when you want to generate a real image from audio and text prompt. "
                         "like: generate a real image from audio and text prompt, "
                         "or generate a new image based on the given audio and text prompt. "
                         "The input to this tool should be two string, representing the text prompt and audio_path")
    @torch.no_grad()
    def inference(self, inputs):
        prompt, audio_path = inputs.split(',')
        prompt, audio_path = prompt.strip(), audio_path.strip()
        audio_paths = [audio_path]
        embeddings = self.model.forward({
            ib.ModalityType.AUDIO: ib.load_and_transform_audio_data(audio_paths, self.device),
        })
        embeddings = embeddings[ib.ModalityType.AUDIO]
        images = self.pipe(prompt=prompt, image_embeds=embeddings.half(), width=512, height=512).images
        new_img_name = gen_new_name(audio_paths[0], 'AudioText2Image')
        images[0].save(new_img_name)
        return new_img_name
