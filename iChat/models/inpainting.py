import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from .utils import gen_new_name, prompts
import torch
from omegaconf import OmegaConf
import numpy as np
from .inpainting_src.ldm_inpainting.ldm.models.diffusion.ddim import DDIMSampler
from .inpainting_src.ldm_inpainting.ldm.util import instantiate_from_config
from .utils import cal_dilate_factor, dilate_mask


def make_batch(image, mask, device):
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)

    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)
        
    masked_image = (1 - mask) * image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k] * 2.0 - 1.0
    return batch


class LDMInpainting:
    def __init__(self, device):
        ckpt_path = 'model_zoo/ldm_inpainting_big.ckpt'
        config = './iChat/models/inpainting_src/ldm_inpainting/config.yaml'
        self.ddim_steps = 50
        self.device = device
        config = OmegaConf.load(config)
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load(ckpt_path)["state_dict"], strict=False)
        self.model = model.to(device=device)
        self.sampler = DDIMSampler(model)
    
    @prompts(name="Remove the Masked Object",
             description="useful when you want to remove an object by masking the region in the image. "
                         "like: remove masked object or inpaint the masked region.. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and mask_path")
    @torch.no_grad()
    def inference(self, inputs):
        print(f'inputs: {inputs}')
        # image, mask, device
        img_path, mask_path = inputs.split(',')[0], inputs.split(',')[1]
        img_path = img_path.strip()
        mask_path = mask_path.strip()
        image = Image.open(img_path)
        mask = Image.open(mask_path).convert('L')
        w, h = image.size
        image = image.resize((512, 512))
        mask = mask.resize((512, 512))
        image = np.array(image)
        mask = np.array(mask)
        dilate_factor = cal_dilate_factor(mask.astype(np.uint8))
        mask = dilate_mask(mask, dilate_factor)
        
        with self.model.ema_scope():
            batch = make_batch(image, mask, device=self.device)
            # encode masked image and concat downsampled mask
            c = self.model.cond_stage_model.encode(batch["masked_image"])
            cc = torch.nn.functional.interpolate(batch["mask"],
                                                 size=c.shape[-2:])
            c = torch.cat((c, cc), dim=1)

            shape = (c.shape[1] - 1,) + c.shape[2:]
            samples_ddim, _ = self.sampler.sample(S=self.ddim_steps,
                                                    conditioning=c,
                                                    batch_size=c.shape[0],
                                                    shape=shape,
                                                    verbose=False)
            x_samples_ddim = self.model.decode_first_stage(samples_ddim)

            image = torch.clamp((batch["image"] + 1.0) / 2.0,
                                min=0.0, max=1.0)
            mask = torch.clamp((batch["mask"] + 1.0) / 2.0,
                               min=0.0, max=1.0)
            predicted_image = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                          min=0.0, max=1.0)

            inpainted = (1 - mask) * image + mask * predicted_image
            inpainted = inpainted.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
        
        # print(type(inpainted))
        inpainted = inpainted.astype(np.uint8)
        new_img_name = gen_new_name(img_path, 'LDMInpainter')
        new_img = Image.fromarray(inpainted)
        new_img = new_img.resize((w, h))
        new_img.save(new_img_name)
        print(
            f"\nProcessed LDMInpainting, Inputs: {inputs}, "
            f"Output Image: {new_img_name}")
        return new_img_name
        # return inpainted

'''
if __name__ == '__main__':
    painting = LDMInpainting('cuda:0')
    res = painting.inference(f'image/82e612_fe54ca_raw.png,image/04a785_fe54ca_mask.png.')
    print(res)
'''

    
