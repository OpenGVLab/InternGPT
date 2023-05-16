import torch
from diffusers import StableUnCLIPImg2ImgPipeline
from .utils import gen_new_name, prompts

from . import imagebind

class Audio2Image:
    def __init__(self, device):
        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
        )
        self.pipe = pipe.to(device)
        
        self.model = imagebind.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(device)


    @prompts(name="Generate Image from Audio",
             description="Generate Image from Audio")
    @torch.no_grad()
    def inference(self, inputs):
        audio_paths=[inputs]
        embeddings = self.model.forward({
            imagebind.ModalityType.AUDIO: imagebind.load_and_transform_audio_data(audio_paths, device),
        })
        embeddings = embeddings[imagebind.ModalityType.AUDIO]
        images = self.pipe(image_embeds=embeddings.half()).images
        new_img_name = gen_new_name(audio_paths[0], 'LDMInpainter')
        images[0].save(new_img_name)
        return new_img_name