# coding: utf-8
import os
os.environ['CURL_CA_BUNDLE'] = ''

try:
    import detectron2
except:
    os.system('pip install git+https://github.com/facebookresearch/detectron2.git')

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "third-party" / "lama"))

import random
import torch
import cv2
import re
import uuid
from PIL import Image, ImageOps
import math
import numpy as np
import argparse
import inspect
from functools import partial
import shutil
import whisper

import gradio as gr
import gradio.themes.base as ThemeBase
from gradio.themes.utils import colors, fonts, sizes

from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering

from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, StableDiffusionInstructPix2PixPipeline
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector, MLSDdetector, HEDdetector

from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI

from iGPT.models import VideoCaption, ActionRecognition, DenseCaption, GenerateTikTokVideo
from iGPT.models import HuskyVQA, LDMInpainting
from iGPT.models.utils import (cal_dilate_factor, dilate_mask, gen_new_name,
                                seed_everything, prompts, blend_gt2pt)

# from segment_anything.utils.amg import remove_small_regions
from segment_anything import build_sam, sam_model_registry, SamAutomaticMaskGenerator
from iGPT.models.sam_preditor import SamPredictor
from bark import SAMPLE_RATE, generate_audio

import matplotlib.pyplot as plt
# Please DO NOT MOVE THE IMPORT ORDER FOR easyocr.
import easyocr

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo
import openai

# openai.api_base = 'https://closeai.deno.dev/v1'

GLOBAL_SEED=1912

INTERN_CHAT_PREFIX = """InternGPT is designed to be able to assist with a wide range of text and visual related tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. InternGPT is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

InternGPT is able to process and understand large amounts of text and images. As a language model, InternGPT can not directly read images, but it has a list of tools to finish different visual tasks. Each image will have a file name formed as "image/xxx.png", and InternGPT can invoke different tools to indirectly understand pictures. When talking about images, InternGPT is very strict to the file name and will never fabricate nonexistent files. When using tools to generate new image files, InternGPT is also known that the image may not be the same as the user's demand, and will use other visual question answering tools or description tools to observe the real image. InternGPT is able to use tools in a sequence, and is loyal to the tool observation outputs rather than faking the image content and image file name. It will remember to provide the file name from the last tool observation, if a new image is generated.

Human may provide new figures to InternGPT with a description. The description helps InternGPT to understand this image, but InternGPT should use tools to finish following tasks, rather than directly imagine from the description.

Overall, InternGPT is a powerful visual dialogue assistant tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 


TOOLS:
------

InternGPT  has access to the following tools:"""

INTERN_CHAT_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, you can find all input paths in the history but can not feed the tool's description into the tool.
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

INTERN_CHAT_SUFFIX = """You are very strict to the filename correctness and will never fake a file name if it does not exist.
You will remember to provide the image file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Since InternGPT is a text language model, InternGPT must use tools to observe images rather than imagination.
The thoughts and observations are only visible for InternGPT, InternGPT should remember to repeat important information in the final response for Human. 
Thought: Do I need to use a tool? {agent_scratchpad} Let's think step by step.
"""

INTERN_CHAT_PREFIX_CN = """InternGPT 旨在能够协助完成范围广泛的文本和视觉相关任务，从回答简单的问题到提供对广泛主题的深入解释和讨论。 InternGPT 能够根据收到的输入生成类似人类的文本，使其能够进行听起来自然的对话，并提供连贯且与手头主题相关的响应。

InternGPT 能够处理和理解大量文本和图像。作为一种语言模型，InternGPT 不能直接读取图像，但它有一系列工具来完成不同的视觉任务。每张图片都会有一个文件名，格式为“image/xxx.png”，InternGPT可以调用不同的工具来间接理解图片。在谈论图片时，InternGPT 对文件名的要求非常严格，绝不会伪造不存在的文件。在使用工具生成新的图像文件时，InternGPT也知道图像可能与用户需求不一样，会使用其他视觉问答工具或描述工具来观察真实图像。 InternGPT 能够按顺序使用工具，并且忠于工具观察输出，而不是伪造图像内容和图像文件名。如果生成新图像，它将记得提供上次工具观察的文件名。

Human 可能会向 InternGPT 提供带有描述的新图形。描述帮助 InternGPT 理解这个图像，但 InternGPT 应该使用工具来完成以下任务，而不是直接从描述中想象。有些工具将会返回英文描述，但你对用户的聊天应当采用中文。

总的来说，InternGPT 是一个强大的可视化对话辅助工具，可以帮助处理范围广泛的任务，并提供关于范围广泛的主题的有价值的见解和信息。

工具列表:
------

InternGPT 可以使用这些工具:"""

INTERN_CHAT_FORMAT_INSTRUCTIONS_CN = """用户使用中文和你进行聊天，但是工具的参数应当使用英文。如果要调用工具，你必须遵循如下格式:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

当你不再需要继续调用工具，而是对观察结果进行总结回复时，你必须使用如下格式：


```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

INTERN_CHAT_SUFFIX_CN = """你对文件名的正确性非常严格，而且永远不会伪造不存在的文件。

开始!

因为InternGPT是一个文本语言模型，必须使用工具去观察图片而不是依靠想象。
推理想法和观察结果只对InternGPT可见，需要记得在最终回复时把重要的信息重复给用户，你只能给用户返回中文句子。我们一步一步思考。在你使用工具时，工具的参数只能是英文。

聊天历史:
{chat_history}

新输入: {input}
Thought: Do I need to use a tool? {agent_scratchpad}
"""

os.makedirs('image', exist_ok=True)


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
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:6]}.png")
        image_filename = gen_new_name(image_filename)
        prompt = text + ', ' + self.a_prompt
        image = self.pipe(prompt, negative_prompt=self.n_prompt).images[0]
        image.save(image_filename)
        print(
            f"\nProcessed Text2Image, Input Text: {text}, Output Image: {image_filename}")
        return image_filename


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
        w, h = image.size
        image = resize_800(image)
        seed_everything(GLOBAL_SEED)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        # updated_image_path = get_new_image_name(image_path, func_name="canny2image")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        image = image.resize((w, h))
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
        w, h = image.size
        image = resize_800(image)
        seed_everything(GLOBAL_SEED)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        # updated_image_path = get_new_image_name(image_path, func_name="line2image")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        image = image.resize((w, h))
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
        w, h = image.size
        image = resize_800(image)
        seed_everything(GLOBAL_SEED)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        # updated_image_path = get_new_image_name(image_path, func_name="hed2image")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        image = image.resize((w, h))
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
        w, h = image.size
        image = resize_800(image)
        seed_everything(GLOBAL_SEED)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        # updated_image_path = get_new_image_name(image_path, func_name="scribble2image")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        image = image.resize((w, h))
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
        w, h = image.size
        image = resize_800(image)
        seed_everything(GLOBAL_SEED)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        # updated_image_path = get_new_image_name(image_path, func_name="pose2image")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        image = image.resize((w, h))
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
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        w, h = image.size
        image = resize_800(image)
        seed_everything(GLOBAL_SEED)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        # updated_image_path = get_new_image_name(image_path, func_name="segment2image")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        image = image.resize((w, h))
        image.save(updated_image_path)
        print(f"\nProcessed SegText2Image, Input Seg: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path


# '''
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
                         "beautify this image with it's segmentations, "
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
# '''


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
        w, h = image.size
        image = resize_800(image)
        seed_everything(GLOBAL_SEED)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        # updated_image_path = get_new_image_name(image_path, func_name="depth2image")
        updated_image_path = gen_new_name(image_path, f'{type(self).__name__}')
        image = image.resize((w, h))
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

        self.device = device
        sam_checkpoint = "model_zoo/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.predictor = SamPredictor(self.sam)
        self.sam.to(device=device)
        # self.clicked_region = None
        # self.img_path = None
        # self.history_mask_res = None

    @prompts(name="Segment Anything on Image",
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
        
    @prompts(name="Segment the Clicked Region in the Image",
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

        return res_masks[np.argmax(scores), :, :]


    def segment_anything(self, img):
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask_generator = SamAutomaticMaskGenerator(self.sam)
        annos = mask_generator.generate(img)
        return annos
    
    def get_detection_map(self, img_path):
        annos = self.segment_anything(img_path)
        _, detection_map = self.show_anns(annos)

        return detection_map

    def get_image_embedding(self, img):
        return self.predictor.set_image(img)

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

    @prompts(name="Extract the masked anything",
             description="useful when you want to extract the masked region in the image. "
                         "like: extract the masked region or keep the masked region in the image"
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
    def __init__(self, device):
        print(f"Initializing ReplaceMaskedAnything to {device}")
        self.device=device
        self.revision = 'fp16' if 'cuda' in device else None
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", revision=self.revision, torch_dtype=self.torch_dtype).to(device)
    

    @prompts(name="Replace the Masked Object",
             description="useful when you want to replace an object by clicking in the image "
                         "with other object or something. "
                         "like: replace the masked object with a new object or something. "
                         "The input to this tool should be a comma separated string of three, "
                         "representing the image_path and the mask_path and the prompt")
    def inference(self, inputs):
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
        return gen_img_path


class ImageOCRRecognition:
    def __init__(self, device):
        print(f"Initializing ImageOCRRecognition to {device}")
        self.device = device
        self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=device) # this needs to run only once to load the model into memory

    @prompts(name="recognize the optical characters in the image",
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

    @prompts(name="recognize all optical characters in the image",
             description="useful when you want to recognize all characters or words in the image. "
                         "like: recognize all characters and words in the image."
                         "The input to this tool should be a string, "
                         "representing the image_path.")
    def inference(self, inputs):
        image_path = inputs.strip()
        result = self.reader.readtext(image_path)
        # print(self.result)
        res_text = []
        for item in result:
            # ([[x, y], [x, y], [x, y], [x, y]], text, confidence)
            res_text.append(item[1])
        print(
            f"\nProcessed ImageOCRRecognition, Input Image: {inputs}, "
            f"Output Text: {res_text}")
        return res_text
    
    # def preprocess(self, img, img_path):
        # self.image_path = img_path
        # self.result = self.reader.readtext(self.image_path)
    
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



class ConversationBot:
    def __init__(self, load_dict):
        print(f"Initializing InternGPT, load_dict={load_dict}")
        if 'HuskyVQA' not in load_dict:
            raise ValueError("You have to load ImageCaptioning as a basic function for iGPT")
        if 'SegmentAnything' not in load_dict:
            raise ValueError("You have to load SegmentAnything as a basic function for iGPT")
        if 'ImageOCRRecognition' not in load_dict:
            raise ValueError("You have to load ImageOCRRecognition as a basic function for iGPT")

        self.models = {}
        self.audio_model = whisper.load_model("small").to('cuda:0')
        # Load Basic Foundation Models
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)

        # Load Template Foundation Models
        for class_name, module in globals().items():
            if getattr(module, 'template_model', False):
                template_required_names = {k for k in inspect.signature(module.__init__).parameters.keys() if k!='self'}
                loaded_names = set([type(e).__name__ for e in self.models.values()])
                if template_required_names.issubset(loaded_names):
                    self.models[class_name] = globals()[class_name](
                        **{name: self.models[name] for name in template_required_names})
        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith('inference'):
                    func = getattr(instance, e)
                    self.tools.append(Tool(name=func.name, description=func.description, func=func))

    
    def find_latest_image(self, file_list):
        res = None
        prev_mtime = None
        for file_item in file_list:
            file_path = os.path.basename(file_item[0])
            if not os.path.exists(file_item[0]):
                continue

            if res is None:
                res = file_item[0]
                ms = int(file_path.split('_')[0][3:]) * 0.001
                prev_mtime = int(os.path.getmtime(file_item[0])) + ms
            else:
                
                ms = int(file_path.split('_')[0][3:]) * 0.001
                cur_mtime = int(os.path.getmtime(file_item[0])) + ms
                # cur_mtime = cur_mtime + ms
                if cur_mtime > prev_mtime:
                    prev_mtime = cur_mtime
                    res = file_item[0]
        return res

    def run_task(self, use_voice, text, audio_path, state, user_state):
        if use_voice:
            state, _, user_state = self.run_audio(audio_path, state, user_state)
        else:
            state, _, user_state = self.run_text(text, state, user_state)
        return state, state, user_state

    def find_param(self, msg, keyword, excluded=False):
        p1 = re.compile(f'(image/[-\\w]*.(png|mp4))')
        p2 = re.compile(f'(image/[-\\w]*_{keyword}.(png|mp4))')
        if keyword == None or len(keyword) == 0:
            out_filenames = p1.findall(msg)
        elif not excluded:
            out_filenames = p2.findall(msg)
        elif excluded:
            all_files = p1.findall(msg)
            excluded_files = p2.findall(msg)
            out_filenames = set(all_files) - set(excluded_files)

        res = self.find_latest_image(out_filenames)
        return res

    def rectify_action(self, inputs, history_msg, user_state):
        print('Rectify the action.')
        print(inputs)
        func = None
        func_name = None
        func_inputs = None
        if 'remove' in inputs.lower() or 'erase' in inputs.lower():
            # func = self.models['RemoveMaskedAnything']
            # cls = self.models.get('RemoveMaskedAnything', None)
            cls = self.models.get('LDMInpainting', None)
            if cls is not None:
                func = cls.inference
            mask_path = self.find_param(history_msg+inputs, 'mask')
            img_path = self.find_param(history_msg+inputs, 'mask', excluded=True)
            func_inputs = f'{img_path},{mask_path}'
            func_name = 'RemoveMaskedAnything'
        elif 'replace' in inputs.lower():
            cls = self.models.get('ReplaceMaskedAnything', None)
            if cls is not None:
                func = cls.inference
            mask_path = self.find_param(history_msg+inputs, 'mask')
            # img_path = self.find_param(history_msg, 'raw')
            img_path = self.find_param(history_msg+inputs, 'mask', excluded=True)
            prompt = inputs.strip()
            func_inputs = f'{img_path},{mask_path},{prompt}'
            func_name = 'ReplaceMaskedAnything'
        elif 'generate' in inputs.lower() or 'beautify' in inputs.lower():
            # print('*' * 40)
            cls = self.models.get('ImageText2Image', None)
            if cls is not None:
                func = cls.inference
            img_path = self.find_param(history_msg+inputs, '')
            # img_path = self.find_param(history_msg, 'raw')
            prompt = inputs.strip()
            func_inputs = f'{img_path},{prompt}'
            func_name = 'ImageText2Image'
        elif 'describe' in inputs.lower() or 'introduce' in inputs.lower():
            cls = self.models.get('HuskyVQA', None)
            func_name = 'HuskyVQA'
            if cls is not None and 'mask' in inputs.lower():
                prompt = inputs.strip()
                func = cls.inference_by_mask
                img_path = self.find_param(history_msg+inputs, 'mask', excluded=True)
                mask_path = self.find_param(history_msg+inputs, 'mask')
                func_inputs = f'{img_path},{mask_path},{prompt}'
            elif cls is not None: 
                prompt = inputs.strip()
                func = cls.inference
                img_path = self.find_param(history_msg+inputs, 'mask', excluded=True)
                func_inputs = f'{img_path}'

        elif 'image' in inputs.lower() or 'figure' in inputs.lower() or 'picture' in inputs.lower():
            cls = self.models.get('HuskyVQA', None)
            func_name = 'HuskyVQA'
            if cls is not None:
                func = cls.inference
            img_path = self.find_param(history_msg+inputs, 'mask', excluded=True)
            # img_path = self.find_param(history_msg, 'raw')
            prompt = inputs.strip()
            func_inputs = f'{img_path},{prompt}'
        else:
            # raise NotImplementedError('Can not find the matched function.')
            res = user_state[0]['agent'](f"You can use history message to sanswer this question without using any tools. {inputs}")
            res = res['output'].replace("\\", "/")

        print(f'{func_name}: {func_inputs}')
        return_res = None
        if func is None:
            res = f"I have tried to use the tool: \"{func_name}\" to acquire the results, but it is not sucessfully loaded."
        else:
            return_res = func(func_inputs)
            if os.path.exists(return_res):
                res = f"I have used the tool: \"{func_name}\" to obtain the results. The output image is named {return_res}."
            else:
                res = f"I have used the tool: \"{func_name}\" to obtain the results. {return_res}"
        print(f"I have used the tool: \"{func_name}\" to obtain the results. The Inputs: {func_inputs}. Result: {return_res}.")
        return res
    
    def check_illegal_files(self, file_list):
        illegal_files = []
        for file_item in file_list:
            if not os.path.exists(file_item[0]):
                illegal_files.append(file_item[0])

        return illegal_files
        
    def run_text(self, text, state, user_state):
        if text is None or len(text) == 0:
            state += [(None, 'Please input text.')]
            return state, state, user_state
        user_state[0]['agent'].memory.buffer = cut_dialogue_history(user_state[0]['agent'].memory.buffer, keep_last_n_words=500)
        pattern = re.compile('(image/[-\\w]*.(png|mp4))')
        try:
            response = user_state[0]['agent']({"input": text.strip()})['output']
            response = response.replace("\\", "/")
            out_filenames = pattern.findall(response)
            illegal_files = self.check_illegal_files(out_filenames)
            if len(illegal_files) > 0:
                raise FileNotFoundError(f'{illegal_files} do (does) not exist.')
            res = self.find_latest_image(out_filenames)
        except Exception as err1:
            # state += [(text, 'Sorry, I failed to understand your instruction. You can try it again or turn to more powerful language model.')]
            print(f'Error: {err1}')
            try:
                response = self.rectify_action(text, user_state[0]['agent'].memory.buffer[:], user_state)
                # print('response = ', response)
                out_filenames = pattern.findall(response)
                res = self.find_latest_image(out_filenames)
                # print(out_filenames)
                user_state[0]['agent'].memory.buffer += f'\nHuman: {text.strip()}\n' + f'AI:{response})'

            except Exception as err2:
                print(f'Error: {err2}')
                state += [(text, 'Sorry, I failed to understand your instruction. You can try it again or turn to more powerful language model.')]
                return state, state, user_state

        if res is not None and user_state[0]['agent'].memory.buffer.count(res) <= 1:
            state = state + [(text, response + f' `{res}` is as follows: ')]
            state = state + [(None, (res, ))]
        else:
            state = state + [(text, response)]
            
        print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
              f"Current Memory: {user_state[0]['agent'].memory.buffer}")
        return state, state, user_state
    
    def run_audio(self, audio_path, state, user_state):
        print(f'audio_path = {audio_path}')
        if audio_path is None or not os.path.exists(audio_path):
            state += [(None, 'No audio input. Please stop recording first and then send the audio.')]
            return state, state
        if self.audio_model is None:
            self.audio_model = whisper.load_model("small").to('cuda:0')
        text = self.audio_model.transcribe(audio_path)["text"]
        res = self.run_text(text, state, user_state)
        print(f"\nProcessed run_audio, Input transcribed audio: {text}\nCurrent state: {state}\n"
              f"Current Memory: {user_state[0]['agent'].memory.buffer}")
        return res[0], res[1], res[2]

    def upload_image(self, image, state, user_state):
        # [txt, click_img, state, user_state], [chatbot, txt, state, user_state]
        # self.reset()
        print('upload an image')
        user_state = self.clear_user_state(False, user_state)
        img = image['image']
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:6]}.png")
        image_filename = gen_new_name(image_filename, 'image')
        img.save(image_filename, "PNG")
        # self.uploaded_image_filename = image_filename
        user_state[0]['image_path'] = image_filename
        img = img.convert('RGB')

        image_caption = self.models['HuskyVQA'].inference_captioning(image_filename)
        # description = 'Debug'
        user_state[0]['image_caption'] = image_caption

        ocr_res = None
        user_state[0]['ocr_res'] = []
        if 'ImageOCRRecognition' in self.models.keys():
            ocr_res = self.models['ImageOCRRecognition'].inference(image_filename)
            ocr_res_raw = self.models['ImageOCRRecognition'].readtext(image_filename)
        if ocr_res is not None and len(ocr_res) > 0:
            Human_prompt = f'\nHuman: provide a image named {image_filename}. The description is: {image_caption} OCR result is: {ocr_res}. This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\". \n'
            user_state[0]['ocr_res'] = ocr_res_raw
        else:
            Human_prompt = f'\nHuman: provide a image named {image_filename}. The description is: {image_caption} This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\". \n'
        AI_prompt = "Received.  "
        # self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + ' AI: ' + AI_prompt
        user_state[0]['agent'].memory.buffer += Human_prompt + 'AI: ' + AI_prompt
        state = state + [(f"![](file={image_filename})*{image_filename}*", AI_prompt)]
        print(f"\nProcessed upload_image, Input image: {image_filename}\nCurrent state: {state}\n"
              f"Current Memory: {user_state[0]['agent'].memory.buffer}")

        return state, state, user_state

    def upload_video(self, video_path, state, user_state):
        # self.reset()
        print('upload a video')
        user_state = self.clear_user_state(False, user_state)
        vid_name = os.path.basename(video_path)
        # vid_name = gen_new_name(vid_name, '', vid_name.split('.')[-1])
        new_video_path = os.path.join('./image/', vid_name)
        new_video_path = gen_new_name(new_video_path, 'image', vid_name.split('.')[-1])
        shutil.copy(video_path, new_video_path)

        user_state[0]['video_path'] = new_video_path
        if "VideoCaption" in self.models.keys():
            description = self.models['VideoCaption'].inference(new_video_path)
        else:
            description = 'A video.'
        user_state[0]['video_caption'] = description
        Human_prompt = f'\nHuman: provide a video named {new_video_path}. The description is: {description}. This information helps you to understand this video, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\". \n'
        AI_prompt = f"Received video: {new_video_path} "
        # self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
        user_state[0]['agent'].memory.buffer += Human_prompt + 'AI: ' + AI_prompt

        state = state + [((new_video_path, ), AI_prompt)]
        # print('exists = ', os.path.exists("./tmp_files/1e7f_f4236666_tmp.mp4"))
        print(f"\nProcessed upload_video, Input video: `{new_video_path}`\nCurrent state: {state}\n"
              f"Current Memory: {user_state[0]['agent'].memory.buffer}")

        return state, state, user_state

    def blend_mask(self, img, mask):
        mask = mask.astype(np.uint8)
        transparency_ratio = mask.astype(np.float32) / 255 / 3
        transparency_ratio = transparency_ratio[:, :, np.newaxis]
        mask = mask[:, :, np.newaxis] 
        mask[mask != 0] = 255
        mask= mask.repeat(3, axis=2)
        mask[:,:,0] = 0
        mask[:,:,2] = 0
        new_img_arr = img * (1 - transparency_ratio) + mask * transparency_ratio
        new_img_arr = np.clip(new_img_arr, 0, 255).astype(np.uint8)
        # print(new_img_arr.shape)
        return new_img_arr

    def process_seg(self, image, state, user_state):
        Human_prompt="Please process this image based on given mask."
        if image is None or \
            user_state[0].get('image_path', None) is None or \
                not os.path.exists(user_state[0]['image_path']):
            AI_prompt = "Please upload an image for processing."
            state += [(Human_prompt, AI_prompt)]
            return None, state, state, user_state
        
        if 'SegmentAnything' not in self.models.keys():
            state += [(None, 'Please load the segmentation tool.')]
            return image['image'], state, state, user_state

        img = Image.open(user_state[0]['image_path']).convert('RGB')
        print(f'user_state[0][\'image_path\'] = {user_state[0]["image_path"]}')
        img = np.array(img, dtype=np.uint8)
        mask = image['mask'].convert('L')
        mask = np.array(mask, dtype=np.uint8)
        
        if mask.sum() == 0:
            AI_prompt = "You can click the image and ask me some questions."
            state += [(Human_prompt, AI_prompt)]
            return image['image'], state, state, user_state
        
        # if 'SegmentAnything' in self.models.keys():
        #     self.models['SegmentAnything'].clicked_region = mask
        if user_state[0].get('features', None) is None:
            user_state[0]['features'] = self.models['SegmentAnything'].get_image_embedding(img)

        res_mask = self.models['SegmentAnything'].segment_by_mask(mask, user_state[0]['features'])

        if user_state[0].get('seg_mask', None) is not None:
            res_mask = np.logical_or(user_state[0]['seg_mask'], res_mask)
        
        res_mask = res_mask.astype(np.uint8)*255
        user_state[0]['seg_mask'] = res_mask
        new_img_arr = self.blend_mask(img, res_mask)
        new_img = Image.fromarray(new_img_arr)
        res_mask_img = Image.fromarray(res_mask).convert('RGB')
        res_mask_path = gen_new_name(user_state[0]['image_path'], 'mask')
        res_mask_img.save(res_mask_path)
        AI_prompt = f"Received. The mask_path is named {res_mask_path}."
        user_state[0]['agent'].memory.buffer += '\nHuman: ' + Human_prompt + '\nAI: ' + AI_prompt
        # state = state + [(Human_prompt, f"![](file={seg_filename})*{AI_prompt}*")]
        state = state + [(Human_prompt, f'Received. The sgemented figure named `{res_mask_path}` is as follows: ')]
        state = state + [(None, (res_mask_path, ))]
        
        print(f"\nProcessed run_image, Input image: `{user_state[0]['image_path']}`\nCurrent state: {state}\n"
              f"Current Memory: {user_state[0]['agent'].memory.buffer}")
        return new_img, state, state, user_state

    def process_ocr(self, image, state, user_state):
        Human_prompt="Please process this image based on given mask."
        if image is None or \
            user_state[0].get('image_path', None) is None or \
                not os.path.exists(user_state[0]['image_path']):
            AI_prompt = "Please upload an image for processing."
            state += [(Human_prompt, AI_prompt)]
            return None, state, state, user_state

        img = np.array(image['image'])
        # img[:100+int(time.time() % 50),:100, :] = 0 
        img = Image.fromarray(img)
        # img = image['image'].convert('RGB')
        mask = image['mask'].convert('L')
        # mask.save(f'test_{int(time.time()) % 1000}.png')
        mask = np.array(mask, dtype=np.uint8)

        if mask.sum() == 0:
            AI_prompt = "You can click the image and ask me some questions."
            state += [(Human_prompt, AI_prompt)]
            return image['image'], state, state, user_state
        
        chosen_ocr_res = None
        if 'ImageOCRRecognition' in self.models.keys():
            # self.models['ImageOCRRecognition'].clicked_region = mask
            chosen_ocr_res = self.models['ImageOCRRecognition'].get_ocr_by_mask(mask, user_state[0]['ocr_res'])
        else:
            state += [Human_prompt, f'ImageOCRRecognition is not loaded.']

        if chosen_ocr_res is not None and len(chosen_ocr_res) > 0:
            AI_prompt = f'OCR result: {chosen_ocr_res}'
            # self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + ' AI: ' + AI_prompt
        else:
            AI_prompt = 'I didn\'t find any optical characters at given location.'
        
        state = state + [(Human_prompt, AI_prompt)]
        user_state[0]['agent'].memory.buffer += '\nHuman: ' + Human_prompt + '\nAI: ' + AI_prompt
        print(f"\nProcessed process_ocr, Input image: {self.uploaded_image_filename}\nCurrent state: {state}\n"
              f"Current Memory: {user_state[0]['agent'].memory.buffer}")
        return image['image'], state, state, user_state

    def process_save(self, image, state, user_state):
        if image is None:
            return None, state, state, user_state
        
        mask_image = image['mask'].convert('RGB')
        # mask = np.array(mask, dtype=np.uint8)
        # mask_image = Image.fromarray(mask).convert('RGB')
        random_name = os.path.join('image', f"{str(uuid.uuid4())[:6]}.png")
        mask_image_name = gen_new_name(random_name, 'rawmask')
        mask_image.save(mask_image_name, "PNG")
        Human_prompt="Please save the given mask."
        if np.array(mask_image, dtype=np.uint8).sum() == 0:
            AI_prompt = "I can not find the mask. Please operate on the image at first."
            state += [(Human_prompt, AI_prompt)]
            return state, state, image['image']
        
        AI_prompt = f'The saved mask is named {mask_image_name}: '
        state = state + [(Human_prompt, AI_prompt)]
        state = state + [(None, (mask_image_name, ))]
        user_state[0]['agent'].memory.buffer = user_state[0]['agent'].memory.buffer + Human_prompt + ' AI: ' + AI_prompt
        print(f"\nProcessed process_ocr, Input image: {self.uploaded_image_filename}\nCurrent state: {state}\n"
              f"Current Memory: {user_state[0]['agent'].memory.buffer}")
        return image['image'], state, state, user_state
    

    def clear_user_state(self, clear_momery, user_state):
        new_user_state = [{}]
        new_user_state[0]['agent'] = user_state[0]['agent']
        new_user_state[0]['memory'] = user_state[0]['memory']
        if clear_momery:
            new_user_state[0]['memory'].clear()
        else:
            new_user_state[0]['memory'] = user_state[0]['memory']

        return new_user_state


class ImageSketcher(gr.Image):
    """
    Code is from https://github.com/ttengwang/Caption-Anything/blob/main/app.py#L32.
    Fix the bug of gradio.Image that cannot upload with tool == 'sketch'.
    """

    is_template = True  # Magic to make this work with gradio.Block, don't remove unless you know what you're doing.

    def __init__(self, **kwargs):
        super().__init__(tool="sketch", **kwargs)

    def preprocess(self, x):
        if x is None:
            return x
        if self.tool == 'sketch' and self.source in ["upload", "webcam"]:
            # assert isinstance(x, dict)
            if isinstance(x, dict) and x['mask'] is None:
                decode_image = gr.processing_utils.decode_base64_to_image(x['image'])
                width, height = decode_image.size
                mask = np.zeros((height, width, 4), dtype=np.uint8)
                mask[..., -1] = 255
                mask = self.postprocess(mask)
                x['mask'] = mask
            elif not isinstance(x, dict):
                # print(x)
                print(f'type(x) = {type(x)}')
                decode_image = gr.processing_utils.decode_base64_to_image(x)
                width, height = decode_image.size
                decode_image.save('sketch_test.png')
                # print(width, height)
                mask = np.zeros((height, width, 4), dtype=np.uint8)
                mask[..., -1] = 255
                mask = self.postprocess(mask)
                x = {'image': x, 'mask': mask}
        x = super().preprocess(x)
        return x


class Seafoam(ThemeBase.Base):
    def __init__(
        self,
        *,
        primary_hue=colors.emerald,
        secondary_hue=colors.blue,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_md,
        text_size=sizes.text_lg,
        font=(
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono=(
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            # body_background_fill="#D8E9EB",
            body_background_fill_dark="AAAAAA",
            button_primary_background_fill="*primary_300",
            button_primary_background_fill_hover="*primary_200",
            button_primary_text_color="black",
            button_secondary_background_fill="*secondary_300",
            button_secondary_background_fill_hover="*secondary_200",
            border_color_primary="#0BB9BF",
            slider_color="*secondary_300",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_shadow="*shadow_drop_lg",
            button_large_padding="10px",
        )


css='''
#chatbot{min-height: 480px}
#image_upload:{align-items: center; min-width: 640px}
'''

def resize_800(image):
    w, h = image.size
    if w > h:
        ratio = w * 1.0 / 800
        new_w, new_h = 800, int(h * 1.0 / ratio)
    else:
        ratio = h * 1.0 / 800
        new_w, new_h = int(w * 1.0 / ratio), 800
    image = image.resize((new_w, new_h))
    return image

def cut_dialogue_history(history_memory, keep_last_n_words=500):
    if history_memory is None or len(history_memory) == 0:
        return history_memory
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"history_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    paragraphs = history_memory.split('\n')
    last_n_tokens = n_tokens
    while last_n_tokens >= keep_last_n_words:
        last_n_tokens -= len(paragraphs[0].split(' '))
        paragraphs = paragraphs[1:]
    return '\n' + '\n'.join(paragraphs)


def login_with_key(bot, debug, api_key):
    # Just for debug
    print('===>logging in')
    user_state = [{}]
    is_error = True
    if debug:
        user_state = init_agent(bot)
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False, value=''), user_state
    else:
        import openai
        from langchain.llms.openai import OpenAI
        if api_key and len(api_key) > 30:
            os.environ["OPENAI_API_KEY"] = api_key
            openai.api_key = api_key
            try:
                llm = OpenAI(temperature=0)
                llm('Hi!')
                response = 'Success!'
                is_error = False
                user_state = init_agent(bot)
            except:
                # gr.update(visible=True)
                response = 'Incorrect key, please input again'
                is_error = True
        else:
            is_error = True
            response = 'Incorrect key, please input again'
        
        return gr.update(visible=not is_error), gr.update(visible=is_error), gr.update(visible=is_error, value=response), user_state

def init_agent(bot):
    memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
    llm = OpenAI(temperature=0)
    agent = initialize_agent(
            bot.tools,
            llm,
            agent="conversational-react-description",
            verbose=True,
            memory=memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': INTERN_CHAT_PREFIX, 'format_instructions': INTERN_CHAT_FORMAT_INSTRUCTIONS,
                        'suffix': INTERN_CHAT_SUFFIX}, )
    
    user_state = [{'agent': agent, 'memory': memory}]
    return user_state
    
def change_input_type(flag):
    if flag:
        print('Using voice input.')
    else:
        print('Using text input.')
    return gr.update(visible=not flag), gr.update(visible=flag)

def ramdom_image():
    root_path = './assets/images'
    img_list = os.listdir(root_path)
    img_item = random.sample(img_list, 1)[0]
    return Image.open(os.path.join(root_path, img_item))

def ramdom_video():
    root_path = './assets/videos'
    img_list = os.listdir(root_path)
    img_item = random.sample(img_list, 1)[0]
    return os.path.join(root_path, img_item)

def process_video_tab():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

def process_image_tab():
    return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)

def add_whiteboard():
    # wb = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
    wb = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    return Image.fromarray(wb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=7862)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('--https', action='store_true')
    parser.add_argument('--load', type=str, default="HuskyVQA_cuda:0,ImageOCRRecognition_cuda:0,SegmentAnything_cuda:0")
    args = parser.parse_args()
    load_dict = {e.split('_')[0].strip(): e.split('_')[1].strip() for e in args.load.split(',')}
    bot = ConversationBot(load_dict=load_dict)
    # bot.init_agent()
    with gr.Blocks(theme=Seafoam(), css=css) as demo:
        state = gr.State([])
        # user_state is dict. Keys: [agent, memory, image_path, video_path, seg_mask, image_caption, OCR_res, ...]
        user_state = gr.State([])

        gr.HTML(
            """
            <div align='center'> <img src='/file=./assets/gvlab_logo.png' style='height:70px'/> </div>
            <p align="center"><a href="https://github.com/OpenGVLab/InternGPT"><b>GitHub</b></a>&nbsp;&nbsp;&nbsp; <a href="https://arxiv.org/pdf/2305.05662.pdf"><b>ArXiv</b></a></p>
            """)
        with gr.Row(visible=True, elem_id='login') as login:
            with gr.Column(scale=0.6, min_width=0) :
                openai_api_key_text = gr.Textbox(
                    placeholder="Input openAI API key",
                    show_label=False,
                    label="OpenAI API Key",
                    lines=1,
                    type="password").style(container=False)
            with gr.Column(scale=0.4, min_width=0):
                key_submit_button = gr.Button(value="Please log in with your OpenAI API Key", interactive=True, variant='primary').style(container=False) 
        
        with gr.Row(visible=False) as user_interface:
            with gr.Column(scale=0.5, elem_id="text_input") as chat_part:
                chatbot = gr.Chatbot(elem_id="chatbot", label="InternGPT").style(height=360)
                with gr.Row(visible=True) as input_row:
                    with gr.Column(scale=0.8, min_width=0) as text_col:
                        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(
                            container=False)
                        audio_input = gr.Audio(source="microphone", type="filepath", visible=False)
                    with gr.Column(scale=0.2, min_width=20):
                        # clear = gr.Button("Clear")
                        send_btn = gr.Button("📤 Send", variant="primary", visible=True)
            
            with gr.Column(elem_id="visual_input", scale=0.5) as img_part:
                with gr.Row(visible=True):
                        with gr.Column(scale=0.3, min_width=0):
                            audio_switch = gr.Checkbox(label="Voice Assistant", elem_id='audio_switch', info=None)
                        with gr.Column(scale=0.3, min_width=0):
                            whiteboard_mode = gr.Button("⬜️ Whiteboard", variant="primary", visible=True)
                            # whiteboard_mode = gr.Checkbox(label="Whiteboard", elem_id='whiteboard', info=None)
                        with gr.Column(scale=0.4, min_width=0, visible=True)as img_example:
                            add_img_example = gr.Button("🖼️ Give an Example", variant="primary")
                        with gr.Column(scale=0.4, min_width=0, visible=False) as vid_example:
                            add_vid_example = gr.Button("🖼️ Give an Example", variant="primary")
                with gr.Tab("Image", elem_id='image_tab') as img_tab:
                    click_img = ImageSketcher(type="pil", interactive=True, brush_radius=15, elem_id="image_upload").style(height=360)
                    with gr.Row() as vis_btn:
                        with gr.Column(scale=0.25, min_width=0):
                            process_seg_btn = gr.Button(value="👆 Pick", variant="primary", elem_id="process_seg_btn")
                        with gr.Column(scale=0.25, min_width=0):
                            process_ocr_btn = gr.Button(value="🔍 OCR", variant="primary", elem_id="process_ocr_btn")
                        with gr.Column(scale=0.25, min_width=0):
                            process_save_btn = gr.Button(value="📁 Save", variant="primary", elem_id="process_save_btn")
                        with gr.Column(scale=0.25, min_width=0):
                            clear_btn = gr.Button(value="🗑️ Clear All", elem_id="clear_btn")
                with gr.Tab("Video", elem_id='video_tab') as video_tab:
                    video_input = gr.Video(interactive=True, include_audio=True, elem_id="video_upload").style(height=360)

            login_func = partial(login_with_key, bot, args.debug)
            openai_api_key_text.submit(login_func, [openai_api_key_text], [user_interface, openai_api_key_text, key_submit_button, user_state])
            key_submit_button.click(login_func, [openai_api_key_text, ], [user_interface, openai_api_key_text, key_submit_button, user_state])

            txt.submit(
                lambda: gr.update(visible=False), [], [send_btn]).then(
                lambda: gr.update(visible=False), [], [txt]).then(
                bot.run_text, [txt, state, user_state], [chatbot, state, user_state]).then(
                lambda: gr.update(visible=True), [], [send_btn]
            ).then(lambda: "", None, [txt, ]).then(
                lambda: gr.update(visible=True), [], [txt])
            
            # send_audio_btn.click(bot.run_audio, [audio_input, state], [chatbot, state])
            send_btn.click(
                lambda: gr.update(visible=False), [], [send_btn]).then(
                lambda: gr.update(visible=False), [], [txt]).then(
                bot.run_task, [audio_switch, txt, audio_input, state, user_state], [chatbot, state, user_state]).then(
                lambda: gr.update(visible=True), [], [send_btn]).then(
                lambda: "", None, [txt, ]).then(
                lambda: gr.update(visible=True), [], [txt]
            )
            
            audio_switch.change(change_input_type, [audio_switch, ], [txt, audio_input])
            # add_img_example.click(ramdom_image, [], [click_img,]).then(
            #     bot.upload_image, [click_img, state, user_state], [chatbot, state, user_state])
            
            add_img_example.click(ramdom_image, [], [click_img,]).then(
                lambda: gr.update(visible=False), [], [send_btn]).then(
                lambda: gr.update(visible=False), [], [txt]).then(
                lambda: gr.update(visible=False), [], [vis_btn]).then( 
                bot.upload_image, [click_img, state, user_state], 
                [chatbot, state, user_state]).then(
                lambda: gr.update(visible=True), [], [send_btn]).then(
                lambda: gr.update(visible=True), [], [txt]).then(
                lambda: gr.update(visible=True), [], [vis_btn])

            # add_vid_example.click(ramdom_video, [], [video_input,]).then(
            #     bot.upload_video, [video_input, state, user_state], [chatbot, state, user_state])
            
            add_vid_example.click(ramdom_video, [], [video_input,]).then(
                lambda: gr.update(visible=False), [], [send_btn]).then(
                lambda: gr.update(visible=False), [], [txt]).then(
                lambda: gr.update(visible=False), [], [vis_btn]).then( 
                bot.upload_video, [video_input, state, user_state], 
                [chatbot, state, user_state]).then(
                lambda: gr.update(visible=True), [], [send_btn]).then(
                lambda: gr.update(visible=True), [], [txt]).then(
                lambda: gr.update(visible=True), [], [vis_btn])
            
            whiteboard_mode.click(add_whiteboard, [], [click_img, ])

            # click_img.upload(bot.upload_image, [click_img, state, txt], [chatbot, state, txt])
            click_img.upload(lambda: gr.update(visible=False), [], [send_btn]).then(
                lambda: gr.update(visible=False), [], [txt]).then(
                lambda: gr.update(visible=False), [], [vis_btn]).then( 
                bot.upload_image, [click_img, state, user_state], 
                [chatbot, state, user_state]).then(
                lambda: gr.update(visible=True), [], [send_btn]).then(
                lambda: gr.update(visible=True), [], [txt]).then(
                lambda: gr.update(visible=True), [], [vis_btn])
            
            process_ocr_btn.click(
                lambda: gr.update(visible=False), [], [vis_btn]).then(
                bot.process_ocr, [click_img, state, user_state], [click_img, chatbot, state, user_state]).then(
                lambda: gr.update(visible=True), [], [vis_btn]
            )
            # process_seg_btn.click(bot.process_seg, [click_img, state], [chatbot, state, click_img])
            process_seg_btn.click(
                lambda: gr.update(visible=False), [], [vis_btn]).then(
                bot.process_seg, [click_img, state, user_state], [click_img, chatbot, state, user_state]).then(
                lambda: gr.update(visible=True), [], [vis_btn]
            )
            # process_save_btn.click(bot.process_save, [click_img, state], [chatbot, state, click_img])
            process_save_btn.click(
                lambda: gr.update(visible=False), [], [vis_btn]).then(
                bot.process_save, [click_img, state, user_state], [click_img, chatbot, state, user_state]).then(
                lambda: gr.update(visible=True), [], [vis_btn]
            )
            video_tab.select(process_video_tab, [], [whiteboard_mode, img_example, vid_example])
            img_tab.select(process_image_tab, [], [whiteboard_mode, img_example, vid_example])
            # clear_img_btn.click(bot.reset, [], [click_img])
            clear_func = partial(bot.clear_user_state, True)
            clear_btn.click(lambda: None, [], [click_img, ]).then(
                lambda: [], None, state).then(
                clear_func, [user_state, ], [user_state, ]).then(
                lambda: None, None, chatbot
            ).then(lambda: '', None, [txt, ])
            # click_img.upload(bot.reset, None, None)
            
            # video_input.upload(bot.upload_video, [video_input, state, user_state], [chatbot, state, user_state])
            video_input.upload(lambda: gr.update(visible=False), [], [send_btn]).then(
                lambda: gr.update(visible=False), [], [txt]).then( 
                bot.upload_video, [video_input, state, user_state], 
                [chatbot, state, user_state]).then(
                lambda: gr.update(visible=True), [], [send_btn]).then(
                lambda: gr.update(visible=True), [], [txt])
            
            clear_func = partial(bot.clear_user_state, False)
            video_input.clear(clear_func, [user_state, ], [user_state, ])

        # (More detailed instructions can be found in <a href="https://www.shailab.org.cn">here</a>:</p>
        gr.HTML(
            """
            <body>
            <p style="font-family:verdana;color:#FF0000";>Tips!!! (More detailed instructions are coming soon): </p>
            </body>
            """
        )
        gr.Markdown(
            '''
            After uploading the image, you can have a **multi-modal dialogue** by sending messages like: `what is it in the image?` or `what is the background color of image?`.
            
            You also can interactively operate, edit or generate the image as follows:
            - You can click the image and press the button `Pick` to **visualize the segmented region** or press the button `OCR` to **recognize the words** at chosen position;
            - To **remove the masked reigon** in the image, you can send the message like: `remove the maked region`;
            - To **replace the masked reigon** in the image, you can send the message like: `replace the maked region with {your prompt}`;
            - To **generate a new image**, you can send the message like: `generate a new image based on its segmentation decribing {your prompt}`
            - To **create a new image by your scribble**, you can press button `Whiteboard` and drawing in the below board. After drawing, you need to press the button `Save` and send the message like: `generate a new image based on this scribble decribing {your prompt}`.
            '''
        )
        gr.HTML(
            """
            <body>
            <p style="font-family:verdana;color:#11AA00";>More features is coming soon. Hope you have fun with our demo!</p>
            </body>
            """
        )

    if args.https:
        demo.queue().launch(server_name="0.0.0.0", ssl_certfile="./certificate/cert.pem", ssl_keyfile="./certificate/key.pem", ssl_verify=False, server_port=args.port)
    else:
        demo.queue().launch(server_name="0.0.0.0", server_port=args.port)

