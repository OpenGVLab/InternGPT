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

import re
import uuid
from PIL import Image
import numpy as np
import argparse
import inspect
from functools import partial
import shutil
import whisper

import gradio as gr
import gradio.themes.base as ThemeBase
from gradio.themes.utils import colors, fonts, sizes

from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI

from iGPT.models import *
from iGPT.models.utils import gen_new_name

from iGPT.controllers import ConversationBot

import openai

# openai.api_base = 'https://closeai.deno.dev/v1'

os.makedirs('image', exist_ok=True)


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
Action Input: the input to the action, you can find all input paths in the history but must not take the tool's description as inputs.
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
        p2 = re.compile(f'(image/[-\\w]*{keyword}.(png|mp4))')
        if keyword is None or len(keyword) == 0:
            out_filenames = p1.findall(msg)
        elif not excluded:
            out_filenames = p2.findall(msg)
        elif excluded:
            all_files = p1.findall(msg)
            excluded_files = p2.findall(msg)
            out_filenames = set(all_files) - set(excluded_files)

        res = self.find_latest_image(out_filenames)
        return res

    def rectify_action(self, inputs, history_msg, agent):
        print('Rectify the action.')
        print(inputs)
        func = None
        func_name = None
        func_inputs = None
        res = None
        if 'generate' in inputs.lower() or 'beautify' in inputs.lower():
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
                mask_path = self.find_param(history_msg+inputs, 'mask')
                img_path =  self.find_parent(mask_path, history_msg+inputs)
                if img_path is None:
                    img_path = self.find_param(history_msg+inputs, 'mask', excluded=True)
                
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
            def only_chat(inputs):
                res = agent(f"You can use history message to answer this question without using any tools. {inputs}")
                res = res['output'].replace("\\", "/")
                return res
            func_name = 'ChatGPT'
            func_inputs = inputs
            func = only_chat
        
        print(f'{func_name}: {func_inputs}')
        return_res = None
        if func is None:
            res = f"I have tried to use the tool: \"{func_name}\" to acquire the results, but it is not sucessfully loaded."
        else:
            return_res = func(func_inputs)
            if os.path.exists(return_res):
                res = f"I have used the tool: \"{func_name}\" with the inputs: {func_inputs} to get the results. The result image is named {return_res}."
            else:
                res = return_res
        print(f"I have used the tool: \"{func_name}\" to obtain the results. The Inputs: {func_inputs}. Result: {return_res}.")
        return res
    
    def check_illegal_files(self, file_list):
        illegal_files = []
        for file_item in file_list:
            if not os.path.exists(file_item[0]):
                illegal_files.append(file_item[0])

        return illegal_files
    
    def find_parent(self, cur_path, history_msg):
        root_path = os.path.dirname(cur_path)
        name = os.path.basename(cur_path)
        name = name.split('.')[0]
        parent_name = name.split('_')[1]
        # p1 = re.compile(f'(image/[-\\w]*.(png|mp4))')
        p = re.compile(f'(image/{parent_name}[-\\w]*.(png|mp4))')
        out_filenames = p.findall(history_msg)
        if len(out_filenames) > 0:
            out_filenames = out_filenames[0][0]
        else:
            out_filenames = None
            
            all_file_items = os.listdir(f'{root_path}')
            for item in all_file_items:
                if item.startswith(parent_name):
                    out_filenames = os.path.join(root_path, item)
                    # out_filenames = item
                    break

        print(f'{cur_path}, parent path: {out_filenames}')
        return out_filenames
    
    def get_suggested_inputs(self, inputs, history_msg):
        image_path = self.find_param(history_msg+inputs, 'mask', excluded=True)
        mask_path = self.find_param(history_msg+inputs, 'mask')
        if image_path is None or mask_path is None:
            return inputs
            
        prompt_template2 = f"If the tool only needs image_path, image_path might be {image_path}. If the tool only needs mask_path, mask_path might be {mask_path}. "

        image_path = self.find_parent(mask_path, history_msg)
        if image_path is None:
            image_path = self.find_param(history_msg+inputs, 'mask', excluded=True)

        prompt_template1 = f"If the tool needs both image_path and mask_path as inputs, image_path might be {image_path} and mask_path might be {mask_path}. "
        prompt_template3 = 'In other cases, you could refer to history message to finish the action. '
        # prompt_template4 = 'Please finish my request using or not using tools. '
        # prompt_template4 = 'If you understand, say \"Received\". \n'
        new_inputs = prompt_template1 + prompt_template2 + prompt_template3 + inputs
        print(f'Processed by get_suggested_inputs, prompt: {new_inputs}')
        return new_inputs
    
    def check_response(self, response):
        pattern = re.compile('(image/[-\\w]*.(png|mp4))')
        # img_pattern = re.compile('(image/[-\\w]*.(png|mp4))')
        file_items = pattern.findall(response, )
        image_path = ''
        mask_path = ''
        for item in file_items:
            if len(image_path) == 0 and '_image.' in item[0]:
                image_path = item[0]
            elif len(mask_path) == 0 and '_mask.' in item[0]:
                mask_path = item[0]

        if len(image_path) == 0 or len(mask_path) == 0:
            return True
        
        res = self.find_param(response, '')
        if res == image_path:
            return True
        
        img_idx = response.find(image_path)
        mask_idx = response.find(mask_path)
        # if self.find_parent(mask_path) != image_path or \
        #     mask_idx < img_idx:
        #     return False
        if mask_idx < img_idx:
            return False

        return True

    def exec_simple_action(self, inputs, history_msg):
        print('Execute the simple action without ChatGPT.')
        print('history_msg: ', history_msg)
        print('inputs: ', inputs)
        func = None
        func_name = None
        func_inputs = None
        res = None
        if 'remove' in inputs.lower() or 'erase' in inputs.lower():
            # func = self.models['RemoveMaskedAnything']
            # cls = self.models.get('RemoveMaskedAnything', None)
            cls = self.models.get('LDMInpainting', None)
            if cls is not None:
                func = cls.inference
            mask_path = self.find_param(history_msg+inputs, 'mask')
            img_path =  self.find_parent(mask_path, history_msg+inputs)
            if img_path is None:
                img_path = self.find_param(history_msg+inputs, 'mask', excluded=True)
            # img_path = self.find_param(history_msg+inputs, 'mask', excluded=True)
            func_inputs = f'{img_path},{mask_path}'
            func_name = 'RemoveMaskedAnything'
        elif 'replace' in inputs.lower():
            cls = self.models.get('ReplaceMaskedAnything', None)
            if cls is not None:
                func = cls.inference
            mask_path = self.find_param(history_msg+inputs, 'mask')
            img_path =  self.find_parent(mask_path, history_msg+inputs)
            if img_path is None:
                img_path = self.find_param(history_msg+inputs, 'mask', excluded=True)
           
            func_inputs = f'{img_path},{mask_path},{inputs}'
            func_name = 'ReplaceMaskedAnything'
        
        print(f'{func_name}: {func_inputs}')
        
        if func is None:
            return None

        return_res = func(func_inputs)
        res = f"I have used the tool: \"{func_name}\" with the inputs: {func_inputs} to get the results. The result image is named {return_res}."
        print(res)
        return res
    
    def exec_agent(self, inputs, agent):
        # pattern = re.compile('(image/[-\\w]*.(png|mp4))')
        response = agent({"input": inputs})['output']
        response = response.replace("\\", "/")
        nonsense_words = 'I do not need to use a tool'
        if nonsense_words in response.split('.')[0] and len(response.split('.')) > 1:
            response = '.'.join(response.split('.')[1:])

        if not self.check_response(response):
            raise RuntimeError('Arguments are not matched.')

        return response

    def find_result_path(self, inputs):
        pattern = re.compile('(image/[-\\w]*.(png|mp4))')
        out_filenames = pattern.findall(inputs)
        illegal_files = self.check_illegal_files(out_filenames)
        if len(illegal_files) > 0:
            raise FileNotFoundError(f'{illegal_files} do (does) not exist.')
        res = self.find_latest_image(out_filenames)
        return res
        
    def run_text(self, text, state, user_state):
        text = text.strip()
        if text is None or len(text) == 0:
            state += [(None, 'Please input text.')]
            return state, state, user_state

        agent = user_state[0]['agent']
        agent.memory.buffer = cut_dialogue_history(agent.memory.buffer, keep_last_n_words=500)
        history_msg = agent.memory.buffer[:]
        try:
            response = self.exec_simple_action(text, history_msg)
            if response is None:
                inputs = self.get_suggested_inputs(text, history_msg)
                # inputs = text
                response = self.exec_agent(inputs, agent)
            else:
                agent.memory.buffer += f'\nHuman: {text}\n' + f'AI: {response})'
            res = self.find_result_path(response)
        except Exception as err1:
            print(f'Error in line {err1.__traceback__.tb_lineno}: {err1}')
            try:
                response = self.rectify_action(text, history_msg, agent)
                res = self.find_result_path(response)
                agent.memory.buffer += f'\nHuman: {text}\n' + f'AI: {response})'
            except Exception as err2:
                print(f'Error in line {err2.__traceback__.tb_lineno}: {err2}')
                state += [(text, 'Sorry, something went wrong inside the ChatGPT. Please check whether your image, video and message have been uploaded successfully.')]
                return state, state, user_state

        if res is not None and agent.memory.buffer.count(res) <= 1:
            state = state + [(text, response + f' `{res}` is as follows: ')]
            state = state + [(None, (res, ))]
        else:
            state = state + [(text, response)]
            
        print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
              f"Current Memory: {agent.memory.buffer}")
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
        if image is None or image.get('image', None) is None:
            return state, state, user_state
        user_state = self.clear_user_state(False, user_state)
        img = image['image']
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:6]}.png")
        image_filename = gen_new_name(image_filename, 'image')
        img.save(image_filename, "PNG")
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
        uploaded_image_filename = user_state[0]['image_path']
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
        print(f"\nProcessed process_ocr, Input image: {uploaded_image_filename}\nCurrent state: {state}\n"
              f"Current Memory: {user_state[0]['agent'].memory.buffer}")
        return image['image'], state, state, user_state

    def process_save(self, image, state, user_state):
        if image is None:
            return None, state, state, user_state
        
        uploaded_image_filename = user_state[0]['image_path']
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
        print(f"\nProcessed process_ocr, Input image: {uploaded_image_filename}\nCurrent state: {state}\n"
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
    with gr.Blocks(theme=Seafoam(), css=css) as demo:
        state = gr.State([])
        # user_state is dict. Keys: [agent, memory, image_path, video_path, seg_mask, image_caption, OCR_res, ...]
        user_state = gr.State([])

        gr.HTML(
            """
            <div align='center'> <img src='/file=./assets/gvlab_logo.png' style='height:70px'/> </div>
            <p align="center"><a href="https://github.com/OpenGVLab/InternGPT"><b>GitHub</b></a>&nbsp;&nbsp;&nbsp; <a href="https://arxiv.org/pdf/2305.05662.pdf"><b>Report</b></a>
            &nbsp;&nbsp;&nbsp; <a href="https://github.com/OpenGVLab/InternGPT/assets/13723743/8fd9112f-57d9-4871-a369-4e1929aa2593"><b>Video Demo</b></a></p>
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

            send_btn.click(
                bot.run_task, [audio_switch, txt, audio_input, state, user_state], [chatbot, state, user_state]).then(
                lambda: "", None, [txt, ])
            
            audio_switch.change(change_input_type, [audio_switch, ], [txt, audio_input])

            add_img_example.click(ramdom_image, [], [click_img,]).then(
                lambda: gr.update(visible=False), [], [send_btn]).then(
                lambda: gr.update(visible=False), [], [txt]).then(
                lambda: gr.update(visible=False), [], [vis_btn]).then( 
                bot.upload_image, [click_img, state, user_state], 
                [chatbot, state, user_state]).then(
                lambda: gr.update(visible=True), [], [send_btn]).then(
                lambda: gr.update(visible=True), [], [txt]).then(
                lambda: gr.update(visible=True), [], [vis_btn])

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
        
        gr.Markdown(
            '''
            **User Manual:**

            After uploading the image, you can have a **multi-modal dialogue** by sending messages like: `"what is it in the image?"` or `"what is the background color of the image?"`.
            
            You also can interactively operate, edit or generate the image as follows:
            - You can click the image and press the button **`Pick`** to **visualize the segmented region** or press the button **`OCR`** to **recognize the words** at chosen position;
            - To **remove the masked region** in the image, you can send the message like: `"remove the masked region"`;
            - To **replace the masked region** in the image, you can send the message like: `"replace the masked region with {your prompt}"`;
            - To **generate a new image**, you can send the message like: `"generate a new image based on its segmentation describing {your prompt}"`.
            - To **create a new image by your scribble**, you should press button **`Whiteboard`** and draw in the board. After drawing, you need to press the button **`Save`** and send the message like: `"generate a new image based on this scribble describing {your prompt}"`.
            '''
        )
        gr.HTML(
            """
            <body>
            <p style="font-family:verdana;color:#11AA00";>More features are coming soon. Hope you have fun with our demo!</p>
            </body>
            """
        )

    if args.https:
        demo.queue().launch(server_name="0.0.0.0", ssl_certfile="./certificate/cert.pem", ssl_keyfile="./certificate/key.pem", ssl_verify=False, server_port=args.port)
    else:
        demo.queue().launch(server_name="0.0.0.0", server_port=args.port)

