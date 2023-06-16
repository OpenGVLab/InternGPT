import numpy as np
import torch
from PIL import Image
import random
import os
import uuid

from .drag_gan_src import drag_gan, stylegan2, get_path
from .utils import gen_new_name, prompts, to_image


CKPT_SIZE = {
    'stylegan2-ffhq-config-f.pt': 1024,
    # 'stylegan2-cat-config-f.pt': 256,
    # 'stylegan2-church-config-f.pt': 256,
    # 'stylegan2-horse-config-f.pt': 256,
}

class StyleGAN:
    def __init__(self, device,e_mode):
        self.e_mode = e_mode
        self.device = device
        #self.g_ema = stylegan2().to(device)
        self.g_ema = stylegan2()
        if self.e_mode is not True:
            self.g_ema.to(device=self.device)
        self.image_size = 1024

    # @prompts(name="Generate Image with StyleGAN",
    #          description="useful when you want to generate a real image with StyleGAn. "
    #                      "like: generate a real image with StyleGAn, "
    #                      "or generate a new image based on with StyleGAn. "
    #                      "This tool does not need input")
    @torch.no_grad()
    def gen_image(self, inputs):
        if self.e_mode:
            self.g_ema.to(device=self.device)
        sample_z = torch.randn([1, 512], device=self.device)
        latent, noise = self.g_ema.prepare([sample_z])
        sample, F = self.g_ema.generate(latent, noise)
        image = Image.fromarray(to_image(sample))
        state = {
            'latent': latent,
            'noise': noise,
            'F': F
        }
        new_img_name = gen_new_name('tmp', 'DragGAN')
        image.save(new_img_name)
        if self.e_mode:
            self.g_ema.to(device="cpu")
        return new_img_name
    
    def change_ckpt(self, ckpt=None):
        if ckpt is None:
            ckpt = random.sample(CKPT_SIZE.keys(), 1)[0]
        assert ckpt in CKPT_SIZE.keys()
        checkpoint = torch.load(get_path(ckpt), map_location=f'{self.device}')
        #checkpoint = torch.load(get_path(ckpt), map_location="cpu")
        self.g_ema.load_state_dict(checkpoint["g_ema"], strict=False)
        self.g_ema.requires_grad_(False)
        self.g_ema.eval()


class DragGAN:
    template_model = True

    def __init__(self, StyleGAN):
        self.g_ema = StyleGAN
        self.device = self.g_ema.device
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,' \
            ' fewer digits, cropped, worst quality, low quality'

    @prompts(name="Edit Image with DragGAN",
             description="useful when you want to edit a StyleGAN generated image with DragGAN. "
                         "like: drag this image."
                         "The input to this tool should be ...")
    @torch.no_grad()
    def inference(self, inputs):
        
        state, points = inputs
        max_iters = 20
        latent = state['latent']
        noise = state['noise']
        F = state['F']

        handle_points = [torch.tensor(p).float() for p in points['handle']]
        target_points = [torch.tensor(p).float() for p in points['target']]

        mask = Image.fromarray(mask['mask']).convert('L')
        mask = np.array(mask) == 255

        mask = torch.from_numpy(mask).float().to(self.device)
        mask = mask.unsqueeze(0).unsqueeze(0)

        sample2, latent, F, handle_points = drag_gan(self.g_ema, latent, noise, F,
                                                     handle_points, target_points, mask,
                                                     self.device, max_iters=max_iters)
        image = Image.fromarray(to_image(sample2))
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:6]}.png")
        new_img_name = gen_new_name(image_filename, 'DragGAN')
        image.save(new_img_name)

        return new_img_name
