import os
import torch
from PIL import Image, ImageOps
import math
import time
import numpy as np
import uuid
import cv2

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering

from .utils import seed_everything, gen_new_name, prompts, GLOBAL_SEED
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, StableDiffusionInstructPix2PixPipeline
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector, MLSDdetector, HEDdetector

from segment_anything.utils.amg import remove_small_regions
from segment_anything import build_sam, sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
import yaml

import matplotlib.pyplot as plt
# Please DO NOT MOVE THE IMPORT ORDER FOR easyocr.
import easyocr

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo

# from .utils import prompts


# GLOBAL_SEED=19120623


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


class MaskFormer:
    def __init__(self, device):
        print(f"Initializing MaskFormer to {device}")
        self.device = device
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

    def inference(self, image_path, text):
        threshold = 0.5
        min_area = 0.02
        padding = 20
        original_image = Image.open(image_path)
        image = original_image.resize((512, 512))
        inputs = self.processor(text=text, images=image, padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        mask = torch.sigmoid(outputs[0]).squeeze().cpu().numpy() > threshold
        area_ratio = len(np.argwhere(mask)) / (mask.shape[0] * mask.shape[1])
        if area_ratio < min_area:
            return None
        true_indices = np.argwhere(mask)
        mask_array = np.zeros_like(mask, dtype=bool)
        for idx in true_indices:
            padded_slice = tuple(slice(max(0, i - padding), i + padding + 1) for i in idx)
            mask_array[padded_slice] = True
        visual_mask = (mask_array * 255).astype(np.uint8)
        image_mask = Image.fromarray(visual_mask)
        return image_mask.resize(original_image.size)


class ImageEditing:
    def __init__(self, device):
        print(f"Initializing ImageEditing to {device}")
        self.device = device
        self.mask_former = MaskFormer(device=self.device)
        self.revision = 'fp16' if 'cuda' in device else None
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", revision=self.revision, torch_dtype=self.torch_dtype).to(device)

    @prompts(name="Remove Something From The Photo",
             description="useful when you want to remove and object or something from the photo "
                         "from its description or location. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the object need to be removed. ")
    def inference_remove(self, inputs):
        image_path, to_be_removed_txt = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        return self.inference_replace(f"{image_path},{to_be_removed_txt},background")

    @prompts(name="Replace Something From The Photo",
             description="useful when you want to replace an object from the object description or "
                         "location with another object from its description. "
                         "The input to this tool should be a comma separated string of three, "
                         "representing the image_path, the object to be replaced, the object to be replaced with ")
    def inference_replace(self, inputs):
        image_path, to_be_replaced_txt, replace_with_txt = inputs.split(",")
        original_image = Image.open(image_path)
        original_size = original_image.size
        mask_image = self.mask_former.inference(image_path, to_be_replaced_txt)
        updated_image = self.inpaint(prompt=replace_with_txt, image=original_image.resize((512, 512)),
                                     mask_image=mask_image.resize((512, 512))).images[0]
        # updated_image_path = get_new_image_name(image_path, func_name="replace-something")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        updated_image = updated_image.resize(original_size)
        updated_image.save(updated_image_path)
        print(
            f"\nProcessed ImageEditing, Input Image: {image_path}, Replace {to_be_replaced_txt} to {replace_with_txt}, "
            f"Output Image: {updated_image_path}")
        return updated_image_path


class InstructPix2Pix:
    def __init__(self, device):
        print(f"Initializing InstructPix2Pix to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix",
                                                                           safety_checker=None,
                                                                           torch_dtype=self.torch_dtype).to(device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

    @prompts(name="Instruct Image Using Text",
             description="useful when you want to the style of the image to be like the text. "
                         "like: make it look like a painting. or make it like a robot. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the text. ")
    def inference(self, inputs):
        """Change style of image."""
        print("===>Starting InstructPix2Pix Inference")
        image_path, text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        original_image = Image.open(image_path)
        image = self.pipe(text, image=original_image, num_inference_steps=40, image_guidance_scale=1.2).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="pix2pix")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        image.save(updated_image_path)
        print(f"\nProcessed InstructPix2Pix, Input Image: {image_path}, Instruct Text: {text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path


class Text2Image:
    def __init__(self, device):
        print(f"Initializing Text2Image to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                            torch_dtype=self.torch_dtype)
        self.pipe.to(device)
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                        'fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image From User Input Text",
             description="useful when you want to generate an image from a user input text and save it to a file. "
                         "like: generate an image of an object or something, or generate an image that includes some objects. "
                         "The input to this tool should be a string, representing the text used to generate image. ")
    def inference(self, text):
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        prompt = text + ', ' + self.a_prompt
        image = self.pipe(prompt, negative_prompt=self.n_prompt).images[0]
        image.save(image_filename)
        print(
            f"\nProcessed Text2Image, Input Text: {text}, Output Image: {image_filename}")
        return image_filename


class ImageCaptioning:
    def __init__(self, device):
        print(f"Initializing ImageCaptioning to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=self.torch_dtype).to(self.device)

    @prompts(name="Get Photo Description",
             description="useful when you want to know what is inside the photo. receives image_path as input. "
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, image_path):
        inputs = self.processor(Image.open(image_path), return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs)
        captions = self.processor.decode(out[0], skip_special_tokens=True)
        print(f"\nProcessed ImageCaptioning, Input Image: {image_path}, Output Text: {captions}")
        return captions


class Image2Canny:
    def __init__(self, device):
        print("Initializing Image2Canny")
        self.low_threshold = 100
        self.high_threshold = 200

    @prompts(name="Edge Detection On Image",
             description="useful when you want to detect the edge of the image. "
                         "like: detect the edges of this image, or canny detection on image, "
                         "or perform edge detection on this image, or detect the canny image of this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        image = Image.open(inputs)
        image = np.array(image)
        canny = cv2.Canny(image, self.low_threshold, self.high_threshold)
        canny = canny[:, :, None]
        canny = np.concatenate([canny, canny, canny], axis=2)
        canny = Image.fromarray(canny)
        # updated_image_path = get_new_image_name(inputs, func_name="edge")
        updated_image_path = gen_new_name(inputs, f'{type(self).__name__}')
        canny.save(updated_image_path)
        print(f"\nProcessed Image2Canny, Input Image: {inputs}, Output Text: {updated_image_path}")
        return updated_image_path


class CannyText2Image:
    def __init__(self, device):
        print(f"Initializing CannyText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-canny",
                                                          torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                            'fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image Condition On Canny Image",
             description="useful when you want to generate a new real image from both the user description and a canny image."
                         " like: generate a real image of a object or something from this canny image,"
                         " or generate a new real image of a object or something from this edge image. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description. ")
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="canny2image")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        image.save(updated_image_path)
        print(f"\nProcessed CannyText2Image, Input Canny: {image_path}, Input Text: {instruct_text}, "
              f"Output Text: {updated_image_path}")
        return updated_image_path


class Image2Line:
    def __init__(self, device):
        print("Initializing Image2Line")
        self.detector = MLSDdetector.from_pretrained('lllyasviel/ControlNet')

    @prompts(name="Line Detection On Image",
             description="useful when you want to detect the straight line of the image. "
                         "like: detect the straight lines of this image, or straight line detection on image, "
                         "or perform straight line detection on this image, or detect the straight line image of this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        image = Image.open(inputs)
        mlsd = self.detector(image)
        # updated_image_path = get_new_image_name(inputs, func_name="line-of")
        updated_image_path = gen_new_name(inputs, f'{type(self).__name__}')
        mlsd.save(updated_image_path)
        print(f"\nProcessed Image2Line, Input Image: {inputs}, Output Line: {updated_image_path}")
        return updated_image_path


class LineText2Image:
    def __init__(self, device):
        print(f"Initializing LineText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-mlsd",
                                                          torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                            'fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image Condition On Line Image",
             description="useful when you want to generate a new real image from both the user description "
                         "and a straight line image. "
                         "like: generate a real image of a object or something from this straight line image, "
                         "or generate a new real image of a object or something from this straight lines. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description. ")
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        # updated_image_path = get_new_image_name(image_path, func_name="line2image")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        image.save(updated_image_path)
        print(f"\nProcessed LineText2Image, Input Line: {image_path}, Input Text: {instruct_text}, "
              f"Output Text: {updated_image_path}")
        return updated_image_path


class Image2Hed:
    def __init__(self, device):
        print("Initializing Image2Hed")
        self.detector = HEDdetector.from_pretrained('lllyasviel/ControlNet')

    @prompts(name="Hed Detection On Image",
             description="useful when you want to detect the soft hed boundary of the image. "
                         "like: detect the soft hed boundary of this image, or hed boundary detection on image, "
                         "or perform hed boundary detection on this image, or detect soft hed boundary image of this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        image = Image.open(inputs)
        hed = self.detector(image)
        # updated_image_path = get_new_image_name(inputs, func_name="hed-boundary")
        updated_image_path = gen_new_name(inputs, f'{type(self).__name__}')
        hed.save(updated_image_path)
        print(f"\nProcessed Image2Hed, Input Image: {inputs}, Output Hed: {updated_image_path}")
        return updated_image_path


class HedText2Image:
    def __init__(self, device):
        print(f"Initializing HedText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-hed",
                                                          torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                            'fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image Condition On Soft Hed Boundary Image",
             description="useful when you want to generate a new real image from both the user description "
                         "and a soft hed boundary image. "
                         "like: generate a real image of a object or something from this soft hed boundary image, "
                         "or generate a new real image of a object or something from this hed boundary. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description")
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="hed2image")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        image.save(updated_image_path)
        print(f"\nProcessed HedText2Image, Input Hed: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path


class Image2Scribble:
    def __init__(self, device):
        print("Initializing Image2Scribble")
        self.detector = HEDdetector.from_pretrained('lllyasviel/ControlNet')

    @prompts(name="Sketch Detection On Image",
             description="useful when you want to generate a scribble of the image. "
                         "like: generate a scribble of this image, or generate a sketch from this image, "
                         "detect the sketch from this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        image = Image.open(inputs)
        scribble = self.detector(image, scribble=True)
        # updated_image_path = get_new_image_name(inputs, func_name="scribble")
        updated_image_path = gen_new_name(inputs, f'{type(self).__name__}')
        scribble.save(updated_image_path)
        print(f"\nProcessed Image2Scribble, Input Image: {inputs}, Output Scribble: {updated_image_path}")
        return updated_image_path


class ScribbleText2Image:
    def __init__(self, device):
        print(f"Initializing ScribbleText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-scribble",
                                                          torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                            'fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image Condition On Sketch Image",
             description="useful when you want to generate a new real image from both the user description and "
                         "a scribble image or a sketch image. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description")
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        # updated_image_path = get_new_image_name(image_path, func_name="scribble2image")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        image.save(updated_image_path)
        print(f"\nProcessed ScribbleText2Image, Input Scribble: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path


class Image2Pose:
    def __init__(self, device):
        print("Initializing Image2Pose")
        self.detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

    @prompts(name="Pose Detection On Image",
             description="useful when you want to detect the human pose of the image. "
                         "like: generate human poses of this image, or generate a pose image from this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        image = Image.open(inputs)
        pose = self.detector(image)
        # updated_image_path = get_new_image_name(inputs, func_name="human-pose")
        updated_image_path = gen_new_name(inputs, f'{type(self).__name__}')
        pose.save(updated_image_path)
        print(f"\nProcessed Image2Pose, Input Image: {inputs}, Output Pose: {updated_image_path}")
        return updated_image_path


class PoseText2Image:
    def __init__(self, device):
        print(f"Initializing PoseText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-openpose",
                                                          torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.num_inference_steps = 20
        self.seed = -1
        self.unconditional_guidance_scale = 9.0
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,' \
                            ' fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image Condition On Pose Image",
             description="useful when you want to generate a new real image from both the user description "
                         "and a human pose image. "
                         "like: generate a real image of a human from this human pose image, "
                         "or generate a new real image of a human from this pose. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description")
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        # updated_image_path = get_new_image_name(image_path, func_name="pose2image")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        image.save(updated_image_path)
        print(f"\nProcessed PoseText2Image, Input Pose: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path


class SegText2Image:
    def __init__(self, device):
        print(f"Initializing SegText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-seg",
                                                          torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,' \
                            ' fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image Condition On Segmentations",
             description="useful when you want to generate a new real image from both the user description and segmentations. "
                         "like: generate a real image of a object or something from this segmentation image, "
                         "or generate a new real image of a object or something from these segmentations. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description")
    def a_inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        seed_everything(GLOBAL_SEED)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        # updated_image_path = get_new_image_name(image_path, func_name="segment2image")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        image.save(updated_image_path)
        print(f"\nProcessed SegText2Image, Input Seg: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path
    

'''
class ConditionalText2Image:
    template_model = True
    def __init__(self, models):
        self.models = models
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,' \
                            ' fewer digits, cropped, worst quality, low quality'

    @prompts(name="Beautify the Image with other Conditions",
             description="useful when you want to generate a new real image from both the user description and condition. "
                         "Condition can be one of segmentation, canny, scribble"
                         "like: Beautify a image from its segmentation and description "
                         "or generate a image from its canny and description. "
                         "or generate a real image from scribble and description. "
                         "The input to this tool should be a comma separated string of three, "
                         "representing the image_path, condition, instruction")
    def inference(self, inputs):
        image_path, condition, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        # image = Image.open(image_path)
        if 'segmentation' in condition.lower():
            seg_path = self.models['SegmentAnything'].inference(image_path)
            updated_image_path = self.models['SegText2Image'].a_inference(seg_path+","+instruct_text)
        else:
            raise NotImplementedError(condition)
        
        print(f"\nProcessed SegText2Image, Input Seg: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path
'''

class Image2Depth:
    def __init__(self, device):
        print("Initializing Image2Depth")
        self.depth_estimator = pipeline('depth-estimation')

    @prompts(name="Predict Depth On Image",
             description="useful when you want to detect depth of the image. like: generate the depth from this image, "
                         "or detect the depth map on this image, or predict the depth for this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        image = Image.open(inputs)
        depth = self.depth_estimator(image)['depth']
        depth = np.array(depth)
        depth = depth[:, :, None]
        depth = np.concatenate([depth, depth, depth], axis=2)
        depth = Image.fromarray(depth)
        # updated_image_path = get_new_image_name(inputs, func_name="depth")
        updated_image_path = gen_new_name(inputs, f'{type(self).__name__}')
        depth.save(updated_image_path)
        print(f"\nProcessed Image2Depth, Input Image: {inputs}, Output Depth: {updated_image_path}")
        return updated_image_path


class DepthText2Image:
    def __init__(self, device):
        print(f"Initializing DepthText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-depth", torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,' \
                            ' fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image Condition On Depth",
             description="useful when you want to generate a new real image from both the user description and depth image. "
                         "like: generate a real image of a object or something from this depth image, "
                         "or generate a new real image of a object or something from the depth map. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description")
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        # updated_image_path = get_new_image_name(image_path, func_name="depth2image")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        image.save(updated_image_path)
        print(f"\nProcessed DepthText2Image, Input Depth: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path


class Image2Normal:
    def __init__(self, device):
        print("Initializing Image2Normal")
        self.depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
        self.bg_threhold = 0.4

    @prompts(name="Predict Normal Map On Image",
             description="useful when you want to detect norm map of the image. "
                         "like: generate normal map from this image, or predict normal map of this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        image = Image.open(inputs)
        original_size = image.size
        image = self.depth_estimator(image)['predicted_depth'][0]
        image = image.numpy()
        image_depth = image.copy()
        image_depth -= np.min(image_depth)
        image_depth /= np.max(image_depth)
        x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        x[image_depth < self.bg_threhold] = 0
        y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        y[image_depth < self.bg_threhold] = 0
        z = np.ones_like(x) * np.pi * 2.0
        image = np.stack([x, y, z], axis=2)
        image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
        image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
        image = image.resize(original_size)
        updated_image_path = get_new_image_name(inputs, func_name="normal-map")
        updated_image_path = gen_new_name(inputs, f'{type(self).__name__}')
        image.save(updated_image_path)
        print(f"\nProcessed Image2Normal, Input Image: {inputs}, Output Depth: {updated_image_path}")
        return updated_image_path


class NormalText2Image:
    def __init__(self, device):
        print(f"Initializing NormalText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-normal", torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,' \
                            ' fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image Condition On Normal Map",
             description="useful when you want to generate a new real image from both the user description and normal map. "
                         "like: generate a real image of a object or something from this normal map, "
                         "or generate a new real image of a object or something from the normal map. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description")
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="normal2image")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        image.save(updated_image_path)
        print(f"\nProcessed NormalText2Image, Input Normal: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path


class VisualQuestionAnswering:
    def __init__(self, device):
        print(f"Initializing VisualQuestionAnswering to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base", torch_dtype=self.torch_dtype).to(self.device)

    @prompts(name="Answer Question About The Image",
             description="useful when you need an answer for a question based on an image. "
                         "like: what is the background color of the last image, how many cats in this figure, what is in this figure. "
                         "The input to this tool should be a comma separated string of two, representing the image_path and the question")
    def inference(self, inputs):
        image_path, question = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        raw_image = Image.open(image_path).convert('RGB')
        inputs = self.processor(raw_image, question, return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs)
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        print(f"\nProcessed VisualQuestionAnswering, Input Image: {image_path}, Input Question: {question}, "
              f"Output Answer: {answer}")
        return answer


class InfinityOutPainting:
    template_model = True # Add this line to show this is a template model.
    def __init__(self, ImageCaptioning, ImageEditing, VisualQuestionAnswering):
        self.llm = OpenAI(temperature=0)
        self.ImageCaption = ImageCaptioning
        self.ImageEditing = ImageEditing
        self.ImageVQA = VisualQuestionAnswering
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                        'fewer digits, cropped, worst quality, low quality'

    def get_BLIP_vqa(self, image, question):
        inputs = self.ImageVQA.processor(image, question, return_tensors="pt").to(self.ImageVQA.device,
                                                                                  self.ImageVQA.torch_dtype)
        out = self.ImageVQA.model.generate(**inputs)
        answer = self.ImageVQA.processor.decode(out[0], skip_special_tokens=True)
        print(f"\nProcessed VisualQuestionAnswering, Input Question: {question}, Output Answer: {answer}")
        return answer

    def get_BLIP_caption(self, image):
        inputs = self.ImageCaption.processor(image, return_tensors="pt").to(self.ImageCaption.device,
                                                                                self.ImageCaption.torch_dtype)
        out = self.ImageCaption.model.generate(**inputs)
        BLIP_caption = self.ImageCaption.processor.decode(out[0], skip_special_tokens=True)
        return BLIP_caption

    def check_prompt(self, prompt):
        check = f"Here is a paragraph with adjectives. " \
                f"{prompt} " \
                f"Please change all plural forms in the adjectives to singular forms. "
        return self.llm(check)

    def get_imagine_caption(self, image, imagine):
        BLIP_caption = self.get_BLIP_caption(image)
        background_color = self.get_BLIP_vqa(image, 'what is the background color of this image')
        style = self.get_BLIP_vqa(image, 'what is the style of this image')
        imagine_prompt = f"let's pretend you are an excellent painter and now " \
                         f"there is an incomplete painting with {BLIP_caption} in the center, " \
                         f"please imagine the complete painting and describe it" \
                         f"you should consider the background color is {background_color}, the style is {style}" \
                         f"You should make the painting as vivid and realistic as possible" \
                         f"You can not use words like painting or picture" \
                         f"and you should use no more than 50 words to describe it"
        caption = self.llm(imagine_prompt) if imagine else BLIP_caption
        caption = self.check_prompt(caption)
        print(f'BLIP observation: {BLIP_caption}, ChatGPT imagine to {caption}') if imagine else print(
            f'Prompt: {caption}')
        return caption

    def resize_image(self, image, max_size=1000000, multiple=8):
        aspect_ratio = image.size[0] / image.size[1]
        new_width = int(math.sqrt(max_size * aspect_ratio))
        new_height = int(new_width / aspect_ratio)
        new_width, new_height = new_width - (new_width % multiple), new_height - (new_height % multiple)
        return image.resize((new_width, new_height))

    def dowhile(self, original_img, tosize, expand_ratio, imagine, usr_prompt):
        old_img = original_img
        while (old_img.size != tosize):
            prompt = self.check_prompt(usr_prompt) if usr_prompt else self.get_imagine_caption(old_img, imagine)
            crop_w = 15 if old_img.size[0] != tosize[0] else 0
            crop_h = 15 if old_img.size[1] != tosize[1] else 0
            old_img = ImageOps.crop(old_img, (crop_w, crop_h, crop_w, crop_h))
            temp_canvas_size = (expand_ratio * old_img.width if expand_ratio * old_img.width < tosize[0] else tosize[0],
                                expand_ratio * old_img.height if expand_ratio * old_img.height < tosize[1] else tosize[
                                    1])
            temp_canvas, temp_mask = Image.new("RGB", temp_canvas_size, color="white"), Image.new("L", temp_canvas_size,
                                                                                                  color="white")
            x, y = (temp_canvas.width - old_img.width) // 2, (temp_canvas.height - old_img.height) // 2
            temp_canvas.paste(old_img, (x, y))
            temp_mask.paste(0, (x, y, x + old_img.width, y + old_img.height))
            resized_temp_canvas, resized_temp_mask = self.resize_image(temp_canvas), self.resize_image(temp_mask)
            image = self.ImageEditing.inpaint(prompt=prompt, image=resized_temp_canvas, mask_image=resized_temp_mask,
                                              height=resized_temp_canvas.height, width=resized_temp_canvas.width,
                                              num_inference_steps=50).images[0].resize(
                (temp_canvas.width, temp_canvas.height), Image.ANTIALIAS)
            image = blend_gt2pt(old_img, image)
            old_img = image
        return old_img

    @prompts(name="Extend An Image",
             description="useful when you need to extend an image into a larger image."
                         "like: extend the image into a resolution of 2048x1024, extend the image into 2048x1024. "
                         "The input to this tool should be a comma separated string of two, representing the image_path and the resolution of widthxheight")
    def inference(self, inputs):
        image_path, resolution = inputs.split(',')
        width, height = resolution.split('x')
        tosize = (int(width), int(height))
        image = Image.open(image_path)
        image = ImageOps.crop(image, (10, 10, 10, 10))
        out_painted_image = self.dowhile(image, tosize, 4, True, False)
        # updated_image_path = get_new_image_name(image_path, func_name="outpainting")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        out_painted_image.save(updated_image_path)
        print(f"\nProcessed InfinityOutPainting, Input Image: {image_path}, Input Resolution: {resolution}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path


##################### New Models #####################
class SegmentAnything:
    def __init__(self, device):
        print(f"Initializing SegmentAnything to {device}")

        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.predictor = SamPredictor(self.sam)
        self.sam.to(device=device)
        self.clicked_region = None
        self.img_path = None

    @prompts(name="Segment Anything on Image",
             description="useful when you want to segment anything in the image. "
                         "like: segment anything from this image, "
                         "The input to this tool should be a string, representing the image_path")             
    def inference(self, inputs):
        print("Inputs: ", inputs)
        img_path = inputs.strip()
        self.img_path = img_path
        annos = self.segment_anything(img_path)
        full_img, _ = self.show_annos(annos)
        # full_img = Image.fromarray(full_img)
        # res = Image.fromarray(res)
        # print(os.path.splitext(img_path))
        seg_all_image_path = gen_new_name(img_path, 'SegmentAnything')
        full_img.save(seg_all_image_path, "PNG")

        print(f"\nProcessed SegmentAnything, Input Image: {inputs}, Output Depth: {seg_all_image_path}")
        return seg_all_image_path
    
        
    @prompts(name="Segment the Clicked Region",
             description="useful when you want to segment the masked region or block in the image. "
                         "like: segment the masked region in this image, "
                         "or segment the clicked region in this image, "
                         "The input to this tool should be None.")        
    def inference_by_mask(self, inputs=None):
        # mask = np.array(Image.open(mask_path).convert('L'))
        res_mask = self.segment_by_mask(self.clicked_region)
        filaname = gen_new_name(self.img_path, 'SegmentAnything')
        mask_img = Image.fromarray(res_mask.astype(np.uint8)*255)
        mask_img.save(filaname, "PNG")
        return filaname
    
    def segment_by_mask(self, mask=None):
        import random
        random.seed(GLOBAL_SEED)
        if mask is None:
            mask = self.clicked_region 
        idxs = np.nonzero(mask)
        # print(idxs)
        num_points = min(max(1, int(len(idxs[0]) * 0.01)), 16)
        sampled_idx = random.sample(range(0, len(idxs[0])), num_points)
        new_mask = []
        for i in range(len(idxs)):
            new_mask.append(idxs[i][sampled_idx])
        points = np.array(new_mask).reshape(2, -1).transpose(1, 0)[:, ::-1]
        labels = np.array([1] * num_points)

        res_masks, scores, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )

        return res_masks[np.argmax(scores), :, :]


    def segment_anything(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask_generator = SamAutomaticMaskGenerator(self.sam)
        annos = mask_generator.generate(img)
        return annos
    
    def get_detection_map(self, img_path):
        annos = self.segment_anything(img_path)
        _, detection_map = self.show_anns(annos)

        return detection_map

    def preprocess(self, img, img_path):
        self.predictor.set_image(img)
        self.img_path = img_path

    def reset(self):
        self.predictor.reset_image()
        self.clicked_region = None
        self.img_path = None
    
    def show_annos(self, anns):
        # From https://github.com/sail-sg/EditAnything/blob/main/sam2image.py#L91
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        full_img = None

        # for ann in sorted_anns:
        for i in range(len(sorted_anns)):
            ann = anns[i]
            m = ann['segmentation']
            if full_img is None:
                full_img = np.zeros((m.shape[0], m.shape[1], 3))
                map = np.zeros((m.shape[0], m.shape[1]), dtype=np.uint16)
            map[m != 0] = i + 1
            color_mask = np.random.random((1, 3)).tolist()[0]
            full_img[m != 0] = color_mask
        full_img = full_img * 255
        # anno encoding from https://github.com/LUSSeg/ImageNet-S
        res = np.zeros((map.shape[0], map.shape[1], 3))
        res[:, :, 0] = map % 256
        res[:, :, 1] = map // 256
        res.astype(np.float32)
        full_img = Image.fromarray(np.uint8(full_img))
        return full_img, res

    def segment_by_points(self, img, points, lables):
        # TODO
        # masks, _, _ = self.predictor.predict(
        #     point_coords=np.array(points[-1]),
        #     point_labels=np.array(lables[-1]),
        #     # mask_input=mask_input[-1],
        #     multimask_output=True, # SAM outputs 3 masks, we choose the one with highest score
        # )
        # # return masks_[np.argmax(scores_), :, :]
        # return masks
        pass


class RemoveClickedAnything:
    """
    prepare:
    ```
    curl -L $(yadisk-direct https://disk.yandex.ru/d/ouP6l8VJ0HpMZg) -o big-lama.zip
    unzip big-lama.zip
    ```
    """
    def __init__(self, device):
        print(f"Initializing RemoveClickedAnything to {device}")
        self.device=device
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.click_point_state = None

    @prompts(name="Remove the clicked object",
             description="useful when you want to remove an object by clicking in the image. "
                         "like: remove object by clicking, "
                         "The input to this tool should be a string, representing the image_path")
    def inference_replace(self, inputs):
        if self.click_point_state is None:
            print(f"The current point state is None, return the input path {inputs}")
            return inputs

        print("Inputs: ", inputs)
        image_path = inputs
        img = np.array(Image.open(image_path))

        predictor = SamPredictor(self.sam)
        predictor.set_image(img)
        point_coords = [self.click_point_state]
        point_labels = [1]
        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)

        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
         )
        # '''
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(img)
            show_mask(mask, plt.gca())
            show_points(point_coords, point_labels, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.savefig("show_mask{}.jpg".format(i))
        # '''

        masks = masks.astype(np.uint8) * 255
        masks = [self.dilate_mask(mask) for mask in masks]
        
        imgs = []
        for idx, mask in enumerate(masks):
            img_inpainted = self.inpaint_img_with_lama(
                img, mask, "./lama/configs/prediction/default.yaml", "big-lama")
            img_inpainted = img_inpainted.astype(np.uint8)
            each = resize_image(img_inpainted, 512)
            imgs.append(torch.DoubleTensor(each))
        image_list = torch.stack(imgs).permute(0, 3, 1, 2)
        
        updated_image_path = gen_new_name(inputs, f'{type(self).__name__}')
        save_image(image_list, updated_image_path, nrow=3,  # requires data type: torch.float64
                   normalize=True, value_range=(0, 255))
        print(f"\nProcessed SegmentAnything, Input Image: {inputs}, Output Depth: {updated_image_path}")
        return updated_image_path


    def inpaint_img_with_lama(self, img, mask, config_p,
        ckpt_p: str="./lama/configs/prediction/default.yaml",
        mod=8):
        assert len(mask.shape) == 2
        if np.max(mask) == 1:
            mask = mask * 255
        img = torch.from_numpy(img).float().div(255.)
        mask = torch.from_numpy(mask).float()
        predict_config = OmegaConf.load(config_p)
        predict_config.model.path = ckpt_p
        device = torch.device(predict_config.device)

        train_config_path = os.path.join(
            predict_config.model.path, 'config.yaml')

        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = os.path.join(
            predict_config.model.path, 'models',
            predict_config.model.checkpoint
        )
        model = load_checkpoint(
            train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)

        batch = {}
        batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
        batch['mask'] = mask[None, None]
        unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
        batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
        batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
        batch = move_to_device(batch, device)
        batch['mask'] = (batch['mask'] > 0) * 1

        batch = model(batch)
        cur_res = batch[predict_config.out_key][0].permute(1, 2, 0)
        cur_res = cur_res.detach().cpu().numpy()

        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        return cur_res

    def dilate_mask(self, mask, dilate_factor=15):
        # dilate mask
        mask = mask.astype(np.uint8)
        mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)
        
        return mask


class RemovesMaskedAnything:
    """
    prepare:
    ```
    curl -L $(yadisk-direct https://disk.yandex.ru/d/ouP6l8VJ0HpMZg) -o big-lama.zip
    unzip big-lama.zip
    ```
    """
    template_model=True # Add this line to show this is a template model.
    def __init__(self, SegmentAnything):
        self.SegmentAnything = SegmentAnything
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                        'fewer digits, cropped, worst quality, low quality'

    @prompts(name="Remove the masked object",
             description="useful when you want to remove an object by masking the region in the image. "
                         "like: remove object by the masked region"
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and seg_path")
    def inference_replace(self, inputs):
        print("Inputs: ", inputs)
        image_path, seg_mask_path = inputs.split(',')
        image_path = image_path.strip()
        seg_mask_path = seg_mask_path.strip()
        img = np.array(Image.open(image_path))
        seg_mask = Image.open(seg_mask_path).convert('L')
        seg_mask = np.array(seg_mask)

        seg_mask = self.dilate_mask(seg_mask)
        inpainted_img = self.inpaint_img_with_lama(
                img, seg_mask, "./lama/configs/prediction/default.yaml", "big-lama")

        inpainted_img_path = gen_new_name(image_path, "RemovesMaskedAnything")
        Image.fromarray(inpainted_img).save(inpainted_img_path, 'PNG')
    
        print(f"\nProcessed SegmentAnything, Input Image: {inputs}, Output Image: {inpainted_img_path}")
        return inpainted_img_path


    def preprocess_img(self, img):
        self.SegmentAnything.set_image(img)

    def inpaint_img_with_lama(self, img, mask, config_p,
        ckpt_p: str="./lama/configs/prediction/default.yaml",
        mod=8):
        assert len(mask.shape) == 2
        if np.max(mask) == 1:
            mask = mask * 255
        img = torch.from_numpy(img).float().div(255.)
        mask = torch.from_numpy(mask).float()
        predict_config = OmegaConf.load(config_p)
        predict_config.model.path = ckpt_p
        device = torch.device(predict_config.device)

        train_config_path = os.path.join(
            predict_config.model.path, 'config.yaml')

        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = os.path.join(
            predict_config.model.path, 'models',
            predict_config.model.checkpoint
        )
        model = load_checkpoint(
            train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)

        batch = {}
        batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
        batch['mask'] = mask[None, None]
        unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
        batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
        batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
        batch = move_to_device(batch, device)
        batch['mask'] = (batch['mask'] > 0) * 1

        batch = model(batch)
        cur_res = batch[predict_config.out_key][0].permute(1, 2, 0)
        cur_res = cur_res.detach().cpu().numpy()

        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        return cur_res

    def dilate_mask(self, mask, dilate_factor=15):
        # dilate mask
        mask = mask.astype(np.uint8)
        mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)
        
        return mask
    

class InpaintMaskedAnything:
    """
    prepare:
    ```
    curl -L $(yadisk-direct https://disk.yandex.ru/d/ouP6l8VJ0HpMZg) -o big-lama.zip
    unzip big-lama.zip
    ```
    """
    template_model=True # Add this line to show this is a template model.
    def __init__(self, SegmentAnything):
        self.SegmentAnything = SegmentAnything
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                        'fewer digits, cropped, worst quality, low quality'

    @prompts(name="Inpaint the Masked Object",
             description="useful when you want to remove an object by masking the region in the image. "
                         "like: inpaint the masked region. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and seg_path")
    def inference_replace(self, inputs):
        print("Inputs: ", inputs)
        image_path, seg_mask_path = inputs.split(',')
        image_path = image_path.strip()
        seg_mask_path = seg_mask_path.strip()
        img = np.array(Image.open(image_path))
        seg_mask = Image.open(seg_mask_path).convert('L')
        seg_mask = np.array(seg_mask)

        seg_mask = self.dilate_mask(seg_mask)
        inpainted_img = self.inpaint_img_with_lama(
                img, seg_mask, "./lama/configs/prediction/default.yaml", "big-lama")

        time_stamp = int(time.time())
        inpainted_img_path = gen_new_name(image_path, "InpaintMaskedAnything")
        Image.fromarray(inpainted_img).save(inpainted_img_path, 'PNG')
    
        print(f"\nProcessed InpaintMaskedAnything, Input Image: {inputs}, Output Image: {inpainted_img_path}")
        return inpainted_img_path


    def preprocess_img(self, img):
        self.SegmentAnything.set_image(img)

    def inpaint_img_with_lama(self, img, mask, config_p,
        ckpt_p: str="./lama/configs/prediction/default.yaml",
        mod=8):
        assert len(mask.shape) == 2
        if np.max(mask) == 1:
            mask = mask * 255
        img = torch.from_numpy(img).float().div(255.)
        mask = torch.from_numpy(mask).float()
        predict_config = OmegaConf.load(config_p)
        predict_config.model.path = ckpt_p
        device = torch.device(predict_config.device)

        train_config_path = os.path.join(
            predict_config.model.path, 'config.yaml')

        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = os.path.join(
            predict_config.model.path, 'models',
            predict_config.model.checkpoint
        )
        model = load_checkpoint(
            train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)

        batch = {}
        batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
        batch['mask'] = mask[None, None]
        unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
        batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
        batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
        batch = move_to_device(batch, device)
        batch['mask'] = (batch['mask'] > 0) * 1

        batch = model(batch)
        cur_res = batch[predict_config.out_key][0].permute(1, 2, 0)
        cur_res = cur_res.detach().cpu().numpy()

        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        return cur_res

    def dilate_mask(self, mask, dilate_factor=15):
        # dilate mask
        mask = mask.astype(np.uint8)
        mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)
        
        return mask
    

class ExtractMaskedAnything:
    template_model=True # Add this line to show this is a template model.
    def __init__(self, SegmentAnything):
        self.SegmentAnything = SegmentAnything
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                        'fewer digits, cropped, worst quality, low quality'

    # @prompts(name="Extract the masked anything",
    #          description="useful when you want to extract the masked region in the image. "
    #                      "like: extract the masked region or keep the masked region in the image. "
    #                      "The input to this tool should be a comma separated string of two, "
    #                      "representing the image_path and mask_path")
    @prompts(name="Extract the Masked anything",
             description="useful when you want to extract the masked region in the image. "
                         "like: extract the masked region in the image or "
                         "keep the masked region in the image. "
                         "The input to this tool should be a string, "
                         "representing the image_path")
    def inference(self, inputs):
        print("Inputs: ", inputs)
        image_path = inputs.strip()
        seg_mask_path = self.SegmentAnything.inference_by_mask()
        # if len(inputs.split(',')) == 1:
        #     seg_mask_path = self.SegmentAnything.inference_by_mask()
        #     image_path = inputs.strip()
        # else:
        #     image_path, seg_mask_path = inputs.split(',')
        #     image_path = image_path.strip()
        seg_mask_path = seg_mask_path.strip()
        img = np.array(Image.open(image_path).convert("RGB"))
        seg_mask = Image.open(seg_mask_path).convert('RGB')
        seg_mask = np.array(seg_mask, dtype=np.uint8)
        new_img = img * (seg_mask // 255)
        print(new_img.shape)
        print(seg_mask.shape)
        rgba_img = np.concatenate((new_img, seg_mask[:, :, :1]), axis=-1).astype(np.uint8)
        rgba_img = Image.fromarray(rgba_img).convert("RGBA")
        new_name = gen_new_name(image_path, "ExtractMaskedAnything")
        rgba_img.save(new_name, 'PNG')
    
        print(f"\nProcessed ExtractMaskedAnything, Input Image: {inputs}, Output Image: {new_name}")
        return new_name


class ReplaceClickedAnything:
    def __init__(self, device):
        print(f"Initializing ReplaceClickedAnything to {device}")
        self.device=device
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.click_point_state = None
        self.revision = 'fp16' if 'cuda' in device else None
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", revision=self.revision, torch_dtype=self.torch_dtype).to(device)
    

    @prompts(name="Replace the clicked object",
             description="useful when you want to replace an object by clicking in the image. "
                         "like: replace object by clicking, "
                         "The input to this tool should be a string, representing the image_path")
    def inference_replace(self, inputs):
        if self.click_point_state is None:
            print(f"The current point state is None, return the input path {inputs}")
            return inputs

        print("Inputs: ", inputs)
        image_path = inputs
        img = np.array(Image.open(image_path))

        predictor = SamPredictor(self.sam)
        predictor.set_image(img)
        point_coords = [self.click_point_state]
        point_labels = [1]
        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)

        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
         )
        # '''
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(img)
            show_mask(mask, plt.gca())
            show_points(point_coords, point_labels, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.savefig("show_mask{}.jpg".format(i))
        # '''

        imgs = []
        for mask in masks:
            mask_pil = Image.fromarray(mask)
            image_pil = Image.fromarray(img)

            image_pil = image_pil.resize((512, 512))
            mask_pil = mask_pil.resize((512, 512))
            each = self.inpaint(prompt='cute dog', image=image_pil, mask_image=mask_pil).images[0]
            each = resize_image(np.array(each), 512)
            imgs.append(torch.DoubleTensor(each))
        image_list = torch.stack(imgs).permute(0, 3, 1, 2)

        # updated_image_path = get_new_image_name(inputs, func_name="clickreplace")
        updated_image_path = gen_new_name(inputs, f'{type(self).__name__}')
        save_image(image_list, updated_image_path, nrow=3,  # requires data type: torch.float64
                   normalize=True, value_range=(0, 255))
        print(f"\nProcessed ReplaceClickedAnything, Input Image: {inputs}, Output Depth: {updated_image_path}")
        return updated_image_path


class ReplaceMaskedAnything:
    def __init__(self, device):
        print(f"Initializing ReplaceClickedAnything to {device}")
        self.device=device
        self.revision = 'fp16' if 'cuda' in device else None
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", revision=self.revision, torch_dtype=self.torch_dtype).to(device)
    

    @prompts(name="Replace the masked object",
             description="useful when you want to replace an object by clicking in the image. "
                         "like: replace the masked object"
                         "The input to this tool should be a comma separated string of Three, "
                         "representing the image_path and the seg_path and the prompt")
    def inference_replace(self, inputs):
        print("Inputs: ", inputs)
        image_path, seg_path = inputs.split(',')[:2]
        image_path = image_path.strip()
        seg_path = seg_path.strip()
        prompt = ','.join(inputs.split(',')[2:]).strip()
        img = Image.open(image_path)
        original_shape = img.size
        img = img.resize((512, 512))
        seg_img = Image.open(seg_path).convert("L").resize((512, 512))

        gen_img = self.inpaint(prompt=prompt, image=img, mask_image=seg_img).images[0]
        # gen_img = resize_image(np.array(gen_img), 512)
        gen_img = gen_img.resize(original_shape)
        gen_img_path = gen_new_name(image_path, 'ReplaceMaskedAnything')
        gen_img.save(gen_img_path, 'PNG')
        print(f"\nProcessed ReplaceMaskedAnything, Input Image: {inputs}, Output Depth: {gen_img_path}.")
        return gen_img_path


class ImageOCRRecognition:
    def __init__(self, device):
        print(f"Initializing ImageOCRRecognition to {device}")
        self.device = device
        self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=device) # this needs to run only once to load the model into memory
        self.result = None
        self.image_path=None
        self.clicked_region = None
    

    @prompts(name="recognize the optical characters in the image",
             description="useful when you want to recognize the characters or words in the clicked region of image. "
                         "like: recognize the characters or words in the clicked region."
                         "The input to this tool should be a comma separated string of two, "
                         "The input to this tool should be None.")
    def inference_by_mask(self, inputs=None):
        mask = self.clicked_region
        inds =np.where(mask != 0)
        coord = []
        for ind_per_dim in inds:
            coord.append(int(ind_per_dim.mean()))
        
        if self.image_path is None or len(inds[0]) == 0:
            # self.result = self.reader.readtext(image_path)
            return 'No characters in the image'

        # stat = [100, 595] # todo

        # reader = easyocr.Reader(['ch_sim', 'en', 'fr', 'it', 'ja', 'ko', 'ru', 'de', 'pt']) # this needs to run only once to load the model into memory
        orc_text = self.search((coord[1], coord[0]))
        if orc_text is None or len(orc_text) == 0:
            orc_text = 'No characters in the image'

        print(
            f"\nProcessed ImageOCRRecognition, Input Image: {self.image_path}, "
            f"Output Text: {orc_text}.")
        return orc_text
    
    @prompts(name="recognize all optical characters in the image",
             description="useful when you want to recognize all characters or words in the image. "
                         "like: recognize all characters and words in the image."
                         "The input to this tool should be a string, "
                         "representing the image_path.")
    def inference(self, inputs):
        image_path = inputs.strip()
        if self.image_path != image_path:
            self.result = self.reader.readtext(image_path)
            self.image_path = image_path
        # print(self.result)
        res_text = []
        for item in self.result:
            # ([[x, y], [x, y], [x, y], [x, y]], text, confidence)
            res_text.append(item[1])
        print(
            f"\nProcessed ImageOCRRecognition, Input Image: {self.image_path}, "
            f"Output Text: {res_text}")
        return res_text
    
    def preprocess(self, img, img_path):
        self.image_path = img_path
        self.result = self.reader.readtext(self.image_path)

    def search(self, coord):
        for item in self.result:
            left_top = item[0][0]
            right_bottom=item[0][-2]
            if (coord[0] >= left_top[0] and coord[1] >= left_top[1]) and \
                (coord[0] <= right_bottom[0] and coord[1] <= right_bottom[1]):
                return item[1]

        return None

    def reset(self):
        self.image_path = None
        self.result = None
        self.mask = None
