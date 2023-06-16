import os
import torch
from PIL import Image
import random
import time
import numpy as np
import uuid
import cv2
import wget

from transformers import pipeline

from .utils import (cal_dilate_factor, dilate_mask, gen_new_name,
                    seed_everything, prompts, resize_800,
                    gen_new_seed, GLOBAL_SEED)
# from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, StableDiffusionInstructPix2PixPipeline
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector, MLSDdetector, HEDdetector


from segment_anything.utils.amg import remove_small_regions
from segment_anything import build_sam, sam_model_registry, SamAutomaticMaskGenerator
from .sam_preditor import SamPredictor

# Please DO NOT MOVE THE IMPORT ORDER FOR easyocr.
import easyocr


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
        # updated_image_path = get_new_image_name(image_path, func_name="pix2pix")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        image.save(updated_image_path)
        print(f"\nProcessed InstructPix2Pix, Input Image: {image_path}, Instruct Text: {text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path


class Text2Image:
    def __init__(self, device,e_mode):
        print(f"Initializing Text2Image to {device}")
        self.device = device
        self.e_mode = e_mode
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                            torch_dtype=self.torch_dtype)
        if self.e_mode is not True:
            self.pipe.to(self.device)
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                        'fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image From User Input Text",
             description="useful when you want to generate an image from a user input text and save it to a file. "
                         "like: generate an image of an object or something, or generate an image that includes some objects. "
                         "The input to this tool should be a string, representing the text used to generate image. ")
    def inference(self, text):
        if self.e_mode:
            self.pipe.to(self.device)
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:6]}.png")
        image_filename = gen_new_name(image_filename)
        prompt = text + ', ' + self.a_prompt
        image = self.pipe(prompt, negative_prompt=self.n_prompt).images[0]
        image.save(image_filename)
        print(
            f"\nProcessed Text2Image, Input Text: {text}, Output Image: {image_filename}")
        if self.e_mode:
            self.pipe.to("cpu")
        return image_filename


class Image2Canny:
    def __init__(self, device,e_mode):
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
    def __init__(self, device,e_mode):
        print(f"Initializing CannyText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-canny",
                                                          torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.device = device
        self.e_mode = e_mode
        if self.e_mode is not True:
            self.pipe.to(device)
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
        # to device
        if self.e_mode:
            self.pipe.to(self.device)
        image = Image.open(image_path)
        w, h = image.size
        image = resize_800(image)
        seed = gen_new_seed()
        seed_everything(seed)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        # updated_image_path = get_new_image_name(image_path, func_name="canny2image")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        image = image.resize((w, h))
        image.save(updated_image_path)
        print(f"\nProcessed CannyText2Image, Input Canny: {image_path}, Input Text: {instruct_text}, "
              f"Output Text: {updated_image_path}")
        if self.e_mode:
            self.pipe.to("cpu")
            print("GPU memory: ", torch.cuda.memory_allocated())
        return updated_image_path


class Image2Line:
    def __init__(self, device,e_mode):
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
        #self.pipe.to(device)
        self.device = device 
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
        self.pipe.to(self.device)
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        w, h = image.size
        image = resize_800(image)
        seed = gen_new_seed()
        seed_everything(seed)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        # updated_image_path = get_new_image_name(image_path, func_name="line2image")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        image = image.resize((w, h))
        image.save(updated_image_path)
        print(f"\nProcessed LineText2Image, Input Line: {image_path}, Input Text: {instruct_text}, "
              f"Output Text: {updated_image_path}")
        self.pipe.to("cpu")
        return updated_image_path


class Image2Hed:
    def __init__(self, device, e_mode):
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
        self.device =device
        # self.pipe.to(device)
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
        self.pipe.to(self.device)
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        w, h = image.size
        image = resize_800(image)
        seed = gen_new_seed()
        seed_everything(seed)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        # updated_image_path = get_new_image_name(image_path, func_name="hed2image")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        image = image.resize((w, h))
        image.save(updated_image_path)
        print(f"\nProcessed HedText2Image, Input Hed: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        self.pipe.to("cpu")
        return updated_image_path


class Image2Scribble:
    def __init__(self, device, e_mode):
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
    def __init__(self, device,e_mode):
        print(f"Initializing ScribbleText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-scribble",
                                                          torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.e_mode = e_mode
        if self.e_mode is not True:
            self.pipe.to(device)
        self.device = device 
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
        if self.e_mode:
            self.pipe.to(self.device)
        image = Image.open(image_path)
        w, h = image.size
        image = resize_800(image)
        seed = gen_new_seed()
        seed_everything(seed)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        # updated_image_path = get_new_image_name(image_path, func_name="scribble2image")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        image = image.resize((w, h))
        image.save(updated_image_path)
        print(f"\nProcessed ScribbleText2Image, Input Scribble: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        if self.e_mode:
            self.pipe.to("cpu")
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
        # self.pipe.to(device)
        self.device = device
        self.num_inference_steps = 20
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
        self.pipe.to(self.device)
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        w, h = image.size
        image = resize_800(image)
        seed = gen_new_seed()
        seed_everything(seed)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        # updated_image_path = get_new_image_name(image_path, func_name="pose2image")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        image = image.resize((w, h))
        image.save(updated_image_path)
        print(f"\nProcessed PoseText2Image, Input Pose: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        self.pipe.to("cpu")
        return updated_image_path


class SegText2Image:
    def __init__(self, device,e_mode):
        print(f"Initializing SegText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-seg",
                                                          torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        # self.pipe.to(device)
        self.device = device 
        self.e_mode =e_mode
        if self.e_mode is not True:
            self.pipe.to(device)
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,' \
                            ' fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image Condition On Segmentations",
             description="useful when you want to generate a new real image from both the user description and segmentations. "
                         "like: generate a real image of a object or something from this segmentation image, "
                         "or generate a new real image of a object or something from these segmentations. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description")
    def inference(self, inputs):
        if self.e_mode:
            self.pipe.to(self.device)
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        w, h = image.size
        image = resize_800(image)
        seed = gen_new_seed()
        seed_everything(seed)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        # updated_image_path = get_new_image_name(image_path, func_name="segment2image")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        image = image.resize((w, h))
        image.save(updated_image_path)
        print(f"\nProcessed SegText2Image, Input Seg: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        if self.e_mode:
            self.pipe.to("cpu")
        return updated_image_path


class ImageText2Image:
    template_model=True
    def __init__(self, SegText2Image, SegmentAnything):
        # print(f"Initializing SegText2Image to {device}")
        # self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.SegText2Image = SegText2Image
        self.SegmentAnything = SegmentAnything
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,' \
                            ' fewer digits, cropped, worst quality, low quality'

    @prompts(name="Beautify The Image",
             description="useful when you want to beatify or create a new real image from both the user description and segmentations. "
                         "like: generate a real image from its segmentation image, "
                         "beautify this image with it's segmentation, "
                         "or beautify this image by user description. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description")
    def inference(self, inputs):
        img_path, prompt = inputs.split(',')[0], inputs.split(',')[1]
        seg_path = self.SegmentAnything.inference(img_path)
        res_path = self.SegText2Image.inference(f'{seg_path},{prompt}')
        print(f"\nProcessed SegText2Image, Input Seg: {img_path}, Input Text: {res_path}, "
              f"Output Image: {res_path}")
        return res_path


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
        self.device = device 
        # self.pipe.to(device)
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
        self.pipe.to(self.device)
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        w, h = image.size
        image = resize_800(image)
        seed = gen_new_seed()
        seed_everything(seed)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        # updated_image_path = get_new_image_name(image_path, func_name="depth2image")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        image = image.resize((w, h))
        image.save(updated_image_path)
        print(f"\nProcessed DepthText2Image, Input Depth: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        self.pipe.to("cpu")
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
        # self.pipe.to(device)
        self.device = device 
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
        self.pipe.to(self.device)
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        image.save(updated_image_path)
        print(f"\nProcessed NormalText2Image, Input Normal: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        self.pipe.to("cpu")
        return updated_image_path


class SegmentAnything:
    def __init__(self, device,e_mode):
        print(f"Initializing SegmentAnything to {device}")

        self.device = device
        self.e_mode = e_mode 
        self.model_checkpoint_path = "model_zoo/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        self.download_parameters()
        self.sam = sam_model_registry[model_type](checkpoint=self.model_checkpoint_path)
        if self.e_mode is not True:
            self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        
    def download_parameters(self):
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        if not os.path.exists(self.model_checkpoint_path):
            wget.download(url, out=self.model_checkpoint_path)

    @prompts(name="Segment Anything On Image",
             description="useful when you want to segment anything in the image. "
                         "like: segment anything from this image, "
                         "The input to this tool should be a string, "
                         "representing the image_path.")             
    def inference(self, inputs):
        print("Inputs: ", inputs)

        img_path = inputs.strip()
        img = np.array(Image.open(img_path))
        annos = self.segment_anything(img)
        full_img, _ = self.show_annos(annos)
        seg_all_image_path = gen_new_name(img_path, 'seg')
        full_img.save(seg_all_image_path, "PNG")

        print(f"\nProcessed SegmentAnything, Input Image: {inputs}, Output Depth: {seg_all_image_path}")
        return seg_all_image_path
        
    @prompts(name="Segment The Clicked Region In The Image",
             description="useful when you want to segment the masked region or block in the image. "
                         "like: segment the masked region in this image, "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the mask_path")        
    def inference_by_mask(self, inputs):
        img_path, mask_path = inputs.split(',')[0], inputs.split(',')[1]
        img_path = img_path.strip()
        mask_path = mask_path.strip()
        clicked_mask = Image.open(mask_path).convert('L')
        clicked_mask = np.array(clicked_mask, dtype=np.uint8)
        # mask = np.array(Image.open(mask_path).convert('L'))
        res_mask = self.segment_by_mask(clicked_mask)
        
        res_mask = res_mask.astype(np.uint8)*255
        filaname = gen_new_name(self.img_path, 'mask')
        mask_img = Image.fromarray(res_mask)
        mask_img.save(filaname, "PNG")
        return filaname
    
    def segment_by_mask(self, mask, features):

        # to device 
        if self.e_mode:
            self.sam.to(device=self.device)
        random.seed(GLOBAL_SEED)
        idxs = np.nonzero(mask)
        num_points = min(max(1, int(len(idxs[0]) * 0.01)), 16)
        sampled_idx = random.sample(range(0, len(idxs[0])), num_points)
        new_mask = []
        for i in range(len(idxs)):
            new_mask.append(idxs[i][sampled_idx])
        points = np.array(new_mask).reshape(2, -1).transpose(1, 0)[:, ::-1]
        labels = np.array([1] * num_points)

        res_masks, scores, _ = self.predictor.predict(
            features=features,
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        # to cpu 
        if self.e_mode:
            self.sam.to(device="cpu") 
            print("Current allocated memory:", torch.cuda.memory_allocated())
        return res_masks[np.argmax(scores), :, :]

    def segment_anything(self, img):
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # to device
        if self.e_mode:
            self.sam.to(device=self.device)
        mask_generator = SamAutomaticMaskGenerator(self.sam)
        annos = mask_generator.generate(img)
        # to cpu 
        if self.e_mode:
            self.sam.to(device="cpu")
            print("Current allocated memory:", torch.cuda.memory_allocated())
        return annos
    
    def get_detection_map(self, img_path):
        annos = self.segment_anything(img_path)
        _, detection_map = self.show_anns(annos)

        return detection_map

    def get_image_embedding(self, img):
        # to device 
        if self.e_mode:
            self.sam.to(device=self.device)
        embedding = self.predictor.set_image(img)
        # to cpu 
        if self.e_mode:
            self.sam.to(device="cpu")
            print("Current allocated memory:", torch.cuda.memory_allocated())
        return embedding

        
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


class ExtractMaskedAnything:
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

    @prompts(name="Extract The Masked Anything",
             description="useful when you want to extract or save the masked region in the image. "
                         "like: extract the masked region, keep the clicked region in the image "
                         "or save the masked region in the image. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and mask_path")
    def inference(self, inputs):
        print("Inputs: ", inputs)
        image_path, seg_mask_path = inputs.split(',')
        image_path = image_path.strip()
        seg_mask_path = seg_mask_path.strip()
        img = np.array(Image.open(image_path).convert("RGB"))
        seg_mask = Image.open(seg_mask_path).convert('RGB')
        seg_mask = np.array(seg_mask, dtype=np.uint8)
        new_img = img * (seg_mask // 255)
        rgba_img = np.concatenate((new_img, seg_mask[:, :, :1]), axis=-1).astype(np.uint8)
        rgba_img = Image.fromarray(rgba_img).convert("RGBA")
        new_name = gen_new_name(image_path, "ExtractMaskedAnything")
        rgba_img.save(new_name, 'PNG')
    
        print(f"\nProcessed ExtractMaskedAnything, Input Image: {inputs}, Output Image: {new_name}")
        return new_name


class ReplaceMaskedAnything:
    def __init__(self, device,e_mode):
        print(f"Initializing ReplaceMaskedAnything to {device}")
        self.device=device
        self.e_mode = e_mode
        self.revision = 'fp16' if 'cuda' in device else None
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        # self.inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        #     "runwayml/stable-diffusion-inpainting", revision=self.revision, torch_dtype=self.torch_dtype).to(device)
        self.inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", revision=self.revision, torch_dtype=self.torch_dtype)
        if self.e_mode is not True:
            self.inpaint.to(device)

    @prompts(name="Replace The Masked Object",
             description="useful when you want to replace an object by clicking in the image "
                         "with other object or something. "
                         "like: replace the masked object with a new object or something. "
                         "The input to this tool should be a comma separated string of three, "
                         "representing the image_path and the mask_path and the prompt")
    def inference(self, inputs):
        if self.e_mode is True:
            self.inpaint.to(self.device)
        print("Inputs: ", inputs)
        image_path, mask_path = inputs.split(',')[:2]
        image_path = image_path.strip()
        mask_path = mask_path.strip()
        prompt = ','.join(inputs.split(',')[2:]).strip()
        img = Image.open(image_path)
        original_shape = img.size
        img = img.resize((512, 512))
        mask_img = Image.open(mask_path).convert("L").resize((512, 512))
        mask = np.array(mask_img, dtype=np.uint8)
        dilate_factor = cal_dilate_factor(mask)
        mask = dilate_mask(mask, dilate_factor)

        gen_img = self.inpaint(prompt=prompt, image=img, mask_image=mask_img).images[0]
        # gen_img = resize_image(np.array(gen_img), 512)
        gen_img = gen_img.resize(original_shape)
        gen_img_path = gen_new_name(image_path, 'ReplaceMaskedAnything')
        gen_img.save(gen_img_path, 'PNG')
        print(f"\nProcessed ReplaceMaskedAnything, Input Image: {inputs}, Output Depth: {gen_img_path}.")
        if self.e_mode is True:
            self.inpaint.to("cpu")
        return gen_img_path


class ImageOCRRecognition:
    def __init__(self, device,e_mode):
        print(f"Initializing ImageOCRRecognition to {device}")
        self.device = device
        self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=device) # this needs to run only once to load the model into memory

    @prompts(name="Recognize The Optical Characters By Clicking",
             description="useful when you want to recognize the characters or words in the clicked region of image. "
                         "like: recognize the characters or words in the clicked region."
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the mask_path")
    def inference_by_mask(self, inputs=None):
        image_path, mask_path = inputs.split(',')[0], inputs.split(',')[1]
        image_path = image_path.strip()
        mask_path = mask_path.strip()
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask, dtype=np.uint8)
        ocr_res = self.readtext(image_path)
        seleted_ocr_text = self.get_ocr_by_mask(mask, ocr_res)
        print(
            f"\nProcessed ImageOCRRecognition, Input Image: {inputs}, "
            f"Output Text: {seleted_ocr_text}.")
        return seleted_ocr_text

    def get_ocr_by_mask(self, mask, ocr_res):
        inds =np.where(mask != 0)
        inds = (inds[0][::8], inds[1][::8])
        # self.result = self.reader.readtext(self.image_path)
        if len(inds[0]) == 0:
            # self.result = self.reader.readtext(image_path)
            return 'No characters in the image'

        # reader = easyocr.Reader(['ch_sim', 'en', 'fr', 'it', 'ja', 'ko', 'ru', 'de', 'pt']) # this needs to run only once to load the model into memory
        ocr_text_list = []
        for i in range(len(inds[0])):
            res = self.search((inds[1][i], inds[0][i]), ocr_res)
            if res is not None and len(res) > 0:
                ocr_text_list.append(res)
        ocr_text_list = list(dict.fromkeys(ocr_text_list))
        ocr_text = '\n'.join(ocr_text_list)
        if ocr_text is None or len(ocr_text.strip()) == 0:
            ocr_text = 'No characters in the image'
        else:
            ocr_text = '\n' + ocr_text
        
        return ocr_text

    @prompts(name="Recognize All Optical Characters",
             description="useful when you want to recognize all characters or words in the image. "
                         "like: recognize all characters and words in the image."
                         "The input to this tool should be a string, "
                         "representing the image_path.")
    def inference(self, inputs):
        image_path = inputs.strip()
        result = self.reader.readtext(image_path)
        # print(self.result)
        res_text = self.parse_result(result)
        print(
            f"\nProcessed ImageOCRRecognition, Input Image: {inputs}, "
            f"Output Text: {res_text}")
        return res_text
    
    def parse_result(self, result):
        res_text = []
        for item in result:
            # ([[x, y], [x, y], [x, y], [x, y]], text, confidence)
            res_text.append(item[1])
        return res_text
    
    def readtext(self, img_path):
        return self.reader.readtext(img_path)

    def search(self, coord, orc_res):
        for item in orc_res:
            left_top = item[0][0]
            right_bottom=item[0][-2]
            if (coord[0] >= left_top[0] and coord[1] >= left_top[1]) and \
                (coord[0] <= right_bottom[0] and coord[1] <= right_bottom[1]):
                return item[1]

        return ''


class ImageCaptioning:
    def __init__(self, device):
        print(f"Initializing ImageCaptioning to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        # self.model = BlipForConditionalGeneration.from_pretrained(
        #     "Salesforce/blip-image-captioning-base", torch_dtype=self.torch_dtype).to(self.device)
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=self.torch_dtype)     

    @prompts(name="Get Photo Description",
             description="useful when you want to know what is inside the photo. receives image_path as input. "
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, image_path):
        self.model.to(self.device)
        inputs = self.processor(Image.open(image_path), return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs)
        captions = self.processor.decode(out[0], skip_special_tokens=True)
        print(f"\nProcessed ImageCaptioning, Input Image: {image_path}, Output Text: {captions}")
        self.model.to("cpu")
        return captions
    

class VisualQuestionAnswering:
    def __init__(self, device):
        print(f"Initializing VisualQuestionAnswering to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        # self.model = BlipForQuestionAnswering.from_pretrained(
        #     "Salesforce/blip-vqa-base", torch_dtype=self.torch_dtype).to(self.device)
        self.model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base", torch_dtype=self.torch_dtype)

    @prompts(name="Answer Question About The Image",
             description="useful when you need an answer for a question based on an image. "
                         "like: what is the background color of the last image, how many cats in this figure, what is in this figure. "
                         "The input to this tool should be a comma separated string of two, representing the image_path and the question")
    def inference(self, inputs):
        self.model.to(self.device)
        image_path, question = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        raw_image = Image.open(image_path).convert('RGB')
        inputs = self.processor(raw_image, question, return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs)
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        print(f"\nProcessed VisualQuestionAnswering, Input Image: {image_path}, Input Question: {question}, "
              f"Output Answer: {answer}")
        self.model.to("cpu")
        return answer
    

class MaskedVisualQuestionAnswering:
    template_model=True
    def __init__(self, VisualQuestionAnswering, SegmentAnything):
        self.VisualQuestionAnswering = VisualQuestionAnswering
        self.SegmentAnything = SegmentAnything
        print(f"Initializing MaskedVisualQuestionAnswering")

    @prompts(name="Answer Question About The Masked Image",
             description="useful when you need an answer for a question based on a masked image. "
                         "like: what is the background color in the masked region, how many cats in this masked figure, what is in this masked figure. "
                         "The input to this tool should be a comma separated string of two, representing the image_path and the question")
    def inference_by_mask(self, inputs):
        image_path, question = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        mask_path = self.SegmentAnything.inference_by_mask(image_path)
        raw_image = Image.open(image_path).convert('RGB')
        # mask_image = Image.open(mask_path).convert('L')
        mask_image = Image.open(mask_path).convert('RGB')
        new_image_arr = np.array(raw_image, dtype=np.uint8) // (np.array(mask_image) // 255)
        new_image = Image.fromarray(new_image_arr)
        new_image_path = gen_new_name(image_path, '')
        new_image.save(new_image_path, 'PNG')

        answer = self.VisualQuestionAnswering.inference(f'{new_image_path},{question}')
        print(f"\nProcessed VisualQuestionAnswering, Input Image: {image_path}, Input Question: {question}, "
              f"Output Answer: {answer}")
        return answer
