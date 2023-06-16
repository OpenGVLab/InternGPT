import torch
from diffusers import StableUnCLIPImg2ImgPipeline
from .utils import gen_new_name, prompts

from . import imagebind as ib


class Anything2Image:
    def __init__(self, device,e_mode):
        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
        )
        self.device = device
        self.e_mode = e_mode
        self.pipe = pipe
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_slicing()
        self.model = ib.imagebind_huge(pretrained=True)
        self.model.eval()
        if self.e_mode is not True:
            self.pipe.to(device)
            self.model.to(device)


class Audio2Image:
    template_model = True

    def __init__(self, Anything2Image):
        self.pipe = Anything2Image.pipe
        self.model = Anything2Image.model
        self.device = Anything2Image.device
        self.e_mode = Anything2Image.e_mode
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
        if self.e_mode:
            self.pipe.to(self.device)
            self.model.to(self.device)

        audio_paths = [inputs]
        embeddings = self.model.forward({
            ib.ModalityType.AUDIO: ib.load_and_transform_audio_data(audio_paths, self.device),
        })
        embeddings = embeddings[ib.ModalityType.AUDIO]
        images = self.pipe(image_embeds=embeddings.half(), width=512, height=512).images
        new_img_name = gen_new_name(audio_paths[0], 'Audio2Image')
        images[0].save(new_img_name)
        if self.e_mode:
            self.pipe.to("cpu")
            self.model.to("cpu")

        return new_img_name


class Thermal2Image:
    template_model = True

    def __init__(self, Anything2Image):
        self.pipe = Anything2Image.pipe
        self.model = Anything2Image.model
        self.device = Anything2Image.device
        self.e_mode = Anything2Image.e_mode
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,' \
            ' fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image from Thermal Image",
             description="useful when you want to generate a real image from a thermal image. "
                         "like: generate a real image from thermal image, "
                         "or generate a new image based on the given thermal image. "
                         "The input to this tool should be a string, representing the image_path")
    @torch.no_grad()
    def inference(self, inputs):
        if self.e_mode:
            self.pipe.to(self.device)
            self.model.to(self.device)

        thermal_paths = [inputs]
        embeddings = self.model.forward({
            ib.ModalityType.THERMAL: ib.load_and_transform_thermal_data(thermal_paths, self.device),
        })
        embeddings = embeddings[ib.ModalityType.THERMAL]
        images = self.pipe(image_embeds=embeddings.half(), width=512, height=512).images
        new_img_name = gen_new_name(thermal_paths[0], 'Thermal2Image')
        images[0].save(new_img_name)
        if self.e_mode:
            self.pipe.to("cpu")
            self.model.to("cpu")

        return new_img_name


class AudioImage2Image:
    template_model = True

    def __init__(self, Anything2Image):
        self.pipe = Anything2Image.pipe
        self.model = Anything2Image.model
        self.device = Anything2Image.device
        self.e_mode = Anything2Image.e_mode
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,' \
            ' fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image from Image and Audio",
             description="useful when you want to generate a real image from image and audio. "
                         "like: generate a real image from image and audio, "
                         "or generate a new image based on the given image and audio. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and audio_path")
    @torch.no_grad()
    def inference(self, inputs):
        if self.e_mode:
            self.pipe.to(self.device)
            self.model.to(self.device)
        print(f'AudioImage2Image: {inputs}')
        image_path, audio_path = inputs.split(',')
        image_path, audio_path = image_path.strip(), audio_path.strip()
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
        if self.e_mode:
            self.pipe.to(self.device)
            self.model.to(self.device)
        return new_img_name


class AudioText2Image:
    template_model = True

    def __init__(self, Anything2Image):
        self.pipe = Anything2Image.pipe
        self.model = Anything2Image.model
        self.device = Anything2Image.device
        self.e_mode = Anything2Image.e_mode
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,' \
            ' fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image from Audio And Text",
             description="useful when you want to generate a real image from audio and text prompt. "
                         "like: generate a real image from audio with user's prompt, "
                         "or generate a new image based on the given audio with user's description. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing audio_path and prompt")
    @torch.no_grad()
    def inference(self, inputs):
        if self.e_mode:
            self.pipe.to(self.device)
            self.model.to(self.device)

        audio_path  = inputs.split(',')[0]
        prompt = ','.join(inputs.split(',')[1:])
        audio_path = audio_path.strip()
        prompt = prompt.strip()
        audio_paths = [audio_path]
        embeddings = self.model.forward({
            ib.ModalityType.TEXT: ib.load_and_transform_text([prompt], self.device),
        }, normalize=False)
        text_embeddings = embeddings[ib.ModalityType.TEXT]

        embeddings = self.model.forward({
            ib.ModalityType.AUDIO: ib.load_and_transform_audio_data(audio_paths, self.device),
        })
        audio_embeddings = embeddings[ib.ModalityType.AUDIO]
        # embeddings = (text_embeddings + audio_embeddings) / 2
        embeddings = text_embeddings * 0.5 + audio_embeddings * 0.5
        # images = self.pipe(prompt=prompt, image_embeds=embeddings.half(), width=512, height=512).images
        images = self.pipe(image_embeds=embeddings.half(), width=512, height=512).images
        new_img_name = gen_new_name(audio_paths[0], 'AudioText2Image')
        images[0].save(new_img_name)
        if self.e_mode:
            self.pipe.to(self.device)
            self.model.to(self.device)

        return new_img_name
