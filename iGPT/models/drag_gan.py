import numpy as np
import torch
from PIL import Image

from .drag_gan_src import drag_gan, stylegan2
from .utils import gen_new_name, prompts


def to_image(tensor):
    tensor = tensor.squeeze(0).permute(1, 2, 0)
    arr = tensor.detach().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = arr * 255
    return arr.astype('uint8')


class StyleGAN:
    def __init__(self, device):
        self.g_ema = stylegan2().to(device)
        self.device = device

    @prompts(name="Generate Image with StyleGAN",
             description="useful when you want to generate a real image with StyleGAn. "
                         "like: generate a real image with StyleGAn, "
                         "or generate a new image based on with StyleGAn. "
                         "This tool does not need input")
    @torch.no_grad()
    def inference(self, inputs):
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
        return new_img_name


class DragGAN:
    template_model = True

    def __init__(self, StyleGAN):
        self.g_ema = StyleGAN
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
                                                     max_iters=max_iters)
        image = Image.fromarray(to_image(sample2))
        new_img_name = gen_new_name('tmp', 'DragGAN')
        image.save(new_img_name)

        return new_img_name
