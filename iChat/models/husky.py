"""Inference for FastChat models."""
import abc
from typing import Optional

import os
import requests
from PIL import Image
from io import BytesIO
import numpy as np

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers import (
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    Blip2VisionConfig
)
from .husky_src.husky_chat import Blip2LlaMAForConditionalGeneration

from .husky_src.conversation import (
    conv_templates,
    get_default_conv_template,
)

from .husky_src.compression import compress_module
from .utils import prompts, gen_new_name

DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<ImageContent>"
DEFAULT_IMG_START_TOKEN = "<img>"
DEFAULT_IMG_END_TOKEN = "</img>"
IGNORE_INDEX = -100


def get_gpu_memory(max_gpus=None):
    gpu_memory = []
    num_gpus = (
        torch.cuda.device_count()
        if max_gpus is None
        else min(max_gpus, torch.cuda.device_count())
    )

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024 ** 3)
            allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory


def load_model(
        model_path, device, num_gpus, max_gpu_memory=None, load_8bit=False, debug=False
):
    kwargs = {"torch_dtype": torch.float16}

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False)
    model = Blip2LlaMAForConditionalGeneration.from_pretrained(
        model_path, low_cpu_mem_usage=True, **kwargs
    )

    if load_8bit:
        compress_module(model, device)

    if (device == "cuda" and num_gpus == 1) or device == "mps":
        model.to(device)

    if debug:
        print(model)

    model = model.eval()
    return model, tokenizer


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def build_transform(input_size):
    crop_pct = 224 / 256
    size = int(input_size / crop_pct)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize(size, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return transform


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops, encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


@torch.inference_mode()
def generate_stream(
        model, tokenizer, image_processor, params, device
):
    prompt = params["prompt"]
    images = params.get("images", None)
    temperature = float(params.get("temperature", 0.7))
    max_new_tokens = int(params.get("max_new_tokens", 1024))

    num_queries = model.config.num_query_tokens

    stop_words = ["Human: ", "Assistant: ", "###", "\n\n"]
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt')[
        'input_ids'].squeeze() for stop_word in stop_words]
    stopping_criteria = StoppingCriteriaList(
        [StoppingCriteriaSub(stops=stop_words_ids)])

    if images is not None:
        pixel_values = image_processor(load_image(images)).to(
            device)  # only support one image
        image_query = DEFAULT_IMG_START_TOKEN + \
            DEFAULT_IMAGE_TOKEN * num_queries + DEFAULT_IMG_END_TOKEN
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, image_query)
        model_inputs = tokenizer([prompt], return_tensors="pt")
        model_inputs["pixel_values"] = pixel_values
        model_inputs.pop("token_type_ids", None)
    else:
        raise NotImplementedError

    generation_config = GenerationConfig(
        bos_token_id=1,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        stopping_criteria=stopping_criteria
    )

    generation_output = model.generate(
        **model_inputs,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True
    )

    preds = generation_output.sequences
    outputs = tokenizer.batch_decode(preds, skip_special_tokens=True)
    return outputs


def resize_pos_embed(posemb, posemb_new, num_prefix_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict.
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:,
                                            :num_prefix_tokens], posemb[0, num_prefix_tokens:]
        ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))

    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    posemb_grid = posemb_grid.reshape(
        1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(
        0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb


class Blip2VisionEmbeddings(nn.Module):
    def __init__(self, config: Blip2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.num_frames = getattr(self.config, "num_frames", 16)
        self.frame_stride = 4

        self.patch_embedding = nn.Conv3d(
            in_channels=3, out_channels=self.embed_dim,
            kernel_size=(self.frame_stride, self.patch_size, self.patch_size),
            stride=(self.frame_stride, self.patch_size, self.patch_size)
        )

        self.num_patches = int(self.num_frames // self.frame_stride) * \
            (self.image_size // self.patch_size) ** 2

        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim), )
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_positions, self.embed_dim))

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values).squeeze(
            1)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(
            batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + \
            self.position_embedding[:, : embeddings.size(
                1), :].to(target_dtype)
        return embeddings


class Chat:
    def __init__(
            self,
            model_path,
            device,
            num_gpus=1,
            load_8bit=False,
            conv_template="multi_model",
            temperature=0.7,
            max_new_tokens=512,
    ):
        model, tokenizer = load_model(
            model_path, device, num_gpus, load_8bit=load_8bit
        )
        self.conv_template = conv_template
        self.model = model.to(device)
        self.tokenizer = tokenizer
        num_queries = model.config.num_query_tokens
        self.image_processor = build_transform(input_size=224)

        self.device = device
        self.dtype = model.dtype

        stop_words = ["Human: ", "Assistant: ", "###", "\n\n"]
        stop_words_ids = [tokenizer(stop_word, return_tensors='pt')[
            'input_ids'].squeeze() for stop_word in stop_words]
        stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)])

        if conv_template:
            conv = conv_templates[conv_template].copy()
        else:
            conv = get_default_conv_template(model_path).copy()

        self.conv = conv
        self.image_query = DEFAULT_IMG_START_TOKEN + \
            DEFAULT_IMAGE_TOKEN * num_queries + DEFAULT_IMG_END_TOKEN

        self.generation_config = GenerationConfig(
            bos_token_id=1,
            do_sample=True,
            top_k=20,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria
        )

    def ask(self, text, conv):
        conversations = []
        if len(conv.messages) > 0:
            conv.append_message(conv.roles[0], text)
        else:
            conv.append_message(conv.roles[0], self.image_query + "\n" + text)

        conv.append_message(conv.roles[1], None)
        conversations.append(conv.get_prompt())
        return conversations

    @torch.no_grad()
    def get_image_embedding(self, image_file):
        image = load_image(image_file)
        pixel_values = self.image_processor(image)
        pixel_values = pixel_values.unsqueeze(
            0).to(self.device, dtype=self.dtype)
        language_model_inputs = self.model.extract_feature(pixel_values)
        return language_model_inputs

    @torch.no_grad()
    def answer(self, conversations, language_model_inputs):
        model_inputs = self.tokenizer(
            conversations,
            return_tensors="pt",
        )
        model_inputs.pop("token_type_ids", None)
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)
        generation_output = self.model.generate(
            pixel_values=None,
            input_ids=input_ids,
            attention_mask=attention_mask,
            language_model_inputs=language_model_inputs,
            generation_config=self.generation_config,
            return_dict_in_generate=True,
            output_scores=True
        )

        preds = generation_output.sequences
        outputs = self.tokenizer.batch_decode(
            preds, skip_special_tokens=True)[0]
        return outputs
    
    def reset(self):
        if self.conv_template:
            self.conv = conv_templates[self.conv_template].copy()
        else:
            self.conv = get_default_conv_template(self.model_path).copy()

class HuskyVQA:
    def __init__(
        self,
        device
    ):
        model_path="model_zoo/husky-7b-v0_01"
        load_8bit=True
        max_new_tokens=512
        self.chat = Chat(
            model_path=model_path,
            device=device,
            load_8bit=load_8bit,
            max_new_tokens=max_new_tokens,
            num_gpus=1,
        )

    # @prompts(name="Visual Question Answering or Image Caption",
    #          description="useful when you want to ask some questions about this image or generate a caption for it. "
    #                      "like: describe this image in details, or what can you see in this image? "
    #                      "The input to this tool should be a string like \"{image_path},{query}\", containing the image_path and user query.")
    @prompts(name="Answer Question About The Image",
             description="useful when you need an answer for a question based on an image. "
                         "like: what is the background color of this image, or how many cats in this figure "
                         "The input to this tool should be a comma separated string of two, representing the image_path and the question")
    def inference(self, inputs):
        print(f'inputs: {inputs}')
        image_file = inputs.split(',')[0]
        query = ','.join(inputs.split(',')[1:])

        vision_feature = self.chat.get_image_embedding(image_file)
        conversations = self.chat.ask(text=query, conv=self.chat.conv)
        outputs = self.chat.answer(conversations, vision_feature)
        # NOTE: strip is important to align with the training data.
        self.chat.conv.messages[-1][1] = outputs.strip()
        # print(f'HuskyVQA: {outputs}')
        self.reset()
        print(
            f"\nProcessed HuskyVQA, Inputs: {inputs}. "
            f"Output: {outputs}")
        return outputs
    
    @prompts(name="Get Photo Description",
             description="useful when you want to know what is inside the photo. "
                         "like: describe this image in detail, what is it in this figure, "
                         "or introduce this image."
                         "The input to this tool should be a string, representing the image_path. ")
    def inference_captioning(self, inputs):
        print(f'inputs: {inputs}')
        image_file = inputs.strip()
        query = 'please describe this image in details'

        vision_feature = self.chat.get_image_embedding(image_file)

        conversations = self.chat.ask(text=query, conv=self.chat.conv)
        outputs = self.chat.answer(conversations, vision_feature)
        # NOTE: strip is important to align with the training data.
        self.chat.conv.messages[-1][1] = outputs.strip()
        self.reset()
        print(
            f"\nProcessed HuskyVQA captioning, Inputs: {inputs}. "
            f"Output: {outputs}")

        return outputs
    
    @prompts(name="Answer Question About The Masked Image",
             description="useful when you need an answer for a question based on a masked image. "
                         "like: what is the background color in the masked region, "
                         "how many cats in this masked figure or what is in this masked figure. "
                         "The input to this tool should be a comma separated string of three, "
                         "representing the image_path, mask_path and the question")
    def inference_by_mask(self, inputs):
        print(f'inputs: {inputs}')
        image_path, mask_path = inputs.split(",")[0], inputs.split(",")[1]
        question = ','.join(inputs.split(',')[2:])
        # mask_path = self.SegmentAnything.inference_by_mask(image_path)
        raw_image = Image.open(image_path).convert('RGB')
        mask_image = Image.open(mask_path).convert('RGB')
        new_image_arr = np.array(raw_image, dtype=np.uint8) // 255 * np.array(mask_image)
        new_image = Image.fromarray(new_image_arr)
        new_image_path = gen_new_name(image_path, '')
        new_image.save(new_image_path, 'PNG')
        answer = self.inference(f'{new_image_path},{question}')
        self.reset()
        print(f"\nProcessed HuskyVQA, Inputs: {inputs}, Input Question: {question}, "
              f"Output Answer: {answer}")
        return answer

    def reset(self):
        self.chat.reset()
