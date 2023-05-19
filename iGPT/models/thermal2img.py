import torch
from diffusers import StableUnCLIPImg2ImgPipeline
from .utils import gen_new_name, prompts

from . import imagebind as ib


class Thermal2Image:
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
        }, normalize=True)
        embeddings = embeddings[ib.ModalityType.THERMAL]
        images = self.pipe(image_embeds=embeddings.half(), width=512, height=512).images
        new_img_name = gen_new_name(thermal_paths[0], 'Thermal2Image')
        images[0].save(new_img_name)
        return new_img_name
