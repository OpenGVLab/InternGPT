import inspect
import re
import os
import numpy as np
import uuid
import shutil
import whisper
import torch
import gradio as gr
import imageio
from io import BytesIO
import requests as req

from PIL import Image

from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI

from ..models import *
from iGPT.models.utils import (gen_new_name, to_image, 
                               seed_everything, add_points_to_image)
from ..models.drag_gan import drag_gan


INTERN_GPT_PREFIX = """InternGPT is designed to be able to assist with a wide range of text and visual related tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. InternGPT is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

InternGPT is able to process and understand large amounts of text and images. As a language model, InternGPT can not directly read images, but it has a list of tools to finish different visual tasks. Each image will have a file name formed as "image/xxx.png", and InternGPT can invoke different tools to indirectly understand pictures. When talking about images, InternGPT is very strict to the file name and will never fabricate nonexistent files. When using tools to generate new image files, InternGPT is also known that the image may not be the same as the user's demand, and will use other visual question answering tools or description tools to observe the real image. InternGPT is able to use tools in a sequence, and is loyal to the tool observation outputs rather than faking the image content and image file name. It will remember to provide the file name from the last tool observation, if a new image is generated.

Human may provide new figures to InternGPT with a description. The description helps InternGPT to understand this image, but InternGPT should use tools to finish following tasks, rather than directly imagine from the description.

Overall, InternGPT is a powerful visual dialogue assistant tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 


TOOLS:
------

InternGPT  has access to the following tools:"""

INTERN_GPT_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

INTERN_GPT_SUFFIX = """You are very strict to the filename correctness and will never fake a file name if it does not exist.
You will remember to provide the image file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Since InternGPT is a text language model, InternGPT must use tools to observe images rather than imagination.
The thoughts and observations are only visible for InternGPT, InternGPT should remember to repeat important information in the final response for Human. 
Thought: Do I need to use a tool? {agent_scratchpad} Let's think step by step.
"""

INTERN_GPT_PREFIX_CN = """InternGPT 旨在能够协助完成范围广泛的文本和视觉相关任务，从回答简单的问题到提供对广泛主题的深入解释和讨论。 InternGPT 能够根据收到的输入生成类似人类的文本，使其能够进行听起来自然的对话，并提供连贯且与手头主题相关的响应。

InternGPT 能够处理和理解大量文本和图像。作为一种语言模型，InternGPT 不能直接读取图像，但它有一系列工具来完成不同的视觉任务。每张图片都会有一个文件名，格式为“image/xxx.png”，InternGPT可以调用不同的工具来间接理解图片。在谈论图片时，InternGPT 对文件名的要求非常严格，绝不会伪造不存在的文件。在使用工具生成新的图像文件时，InternGPT也知道图像可能与用户需求不一样，会使用其他视觉问答工具或描述工具来观察真实图像。 InternGPT 能够按顺序使用工具，并且忠于工具观察输出，而不是伪造图像内容和图像文件名。如果生成新图像，它将记得提供上次工具观察的文件名。

Human 可能会向 InternGPT 提供带有描述的新图形。描述帮助 InternGPT 理解这个图像，但 InternGPT 应该使用工具来完成以下任务，而不是直接从描述中想象。有些工具将会返回英文描述，但你对用户的聊天应当采用中文。

总的来说，InternGPT 是一个强大的可视化对话辅助工具，可以帮助处理范围广泛的任务，并提供关于范围广泛的主题的有价值的见解和信息。

工具列表:
------

InternGPT 可以使用这些工具:"""

INTERN_GPT_FORMAT_INSTRUCTIONS_CN = """用户使用中文和你进行聊天，但是工具的参数应当使用英文。如果要调用工具，你必须遵循如下格式:

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

INTERN_GPT_SUFFIX_CN = """你对文件名的正确性非常严格，而且永远不会伪造不存在的文件。

开始!

因为InternGPT是一个文本语言模型，必须使用工具去观察图片而不是依靠想象。
推理想法和观察结果只对InternGPT可见，需要记得在最终回复时把重要的信息重复给用户，你只能给用户返回中文句子。我们一步一步思考。在你使用工具时，工具的参数只能是英文。

聊天历史:
{chat_history}

新输入: {input}
Thought: Do I need to use a tool? {agent_scratchpad}
"""


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


class ConversationBot:
    def __init__(self, load_dict, e_mode=False, chat_disabled=False):
        print(f"Initializing InternGPT, load_dict={load_dict}")
        
        self.chat_disabled = chat_disabled
        self.models = {}
        self.audio_model = whisper.load_model("small").to('cuda:0')
        #self.audio_model = whisper.load_model("small")
        # Load Basic Foundation Models
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device,e_mode=e_mode)

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


    def init_agent(self):
        memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
        llm = OpenAI(temperature=0)
        agent = initialize_agent(
                self.tools,
                llm,
                agent="conversational-react-description",
                verbose=True,
                memory=memory,
                return_intermediate_steps=True,
                agent_kwargs={'prefix': INTERN_GPT_PREFIX, 'format_instructions': INTERN_GPT_FORMAT_INSTRUCTIONS,
                            'suffix': INTERN_GPT_SUFFIX}, )
        
        user_state = [{'agent': agent, 'memory': memory, 'StyleGAN': {}}]
        return user_state
    
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

    def rectify_action(self, inputs, history_msg, agent):
        print('Rectify the action.')
        print(inputs)
        func = None
        func_name = None
        func_inputs = None
        res = None
        if 'extract' in inputs.lower() or 'save' in inputs.lower():
            cls = self.models.get('ExtractMaskedAnything', None)
            if cls is not None:
                func = cls.inference
            
            mask_path = self.find_param(inputs, 'mask')
            if mask_path is None:
                mask_path = self.find_param(history_msg, 'mask')

            img_path =  self.find_parent(mask_path, history_msg+inputs)
            if img_path is None:
                img_path = self.find_param(history_msg+inputs, 'mask', excluded=True)
            
            func_inputs = f'{img_path},{mask_path}'
            func_name = 'ExtractMaskedAnything'
        elif 'generate' in inputs.lower() or 'beautify' in inputs.lower():
            # print('*' * 40)
            cls = self.models.get('ImageText2Image', None)
            if cls is not None:
                func = cls.inference
            
            img_path = self.find_param(inputs, '')
            if img_path is None:
                img_path = self.find_param(history_msg, '')
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
                mask_path = self.find_param(inputs, 'mask')
                if mask_path is None:
                    mask_path = self.find_param(history_msg, 'mask')
                img_path =  self.find_parent(mask_path, history_msg+inputs)
                if img_path is None:
                    img_path = self.find_param(history_msg+inputs, 'mask', excluded=True)
                
                func_inputs = f'{img_path},{mask_path},{prompt}'
            elif cls is not None: 
                prompt = inputs.strip()
                func = cls.inference
                img_path = self.find_param(inputs, 'mask', excluded=True)
                if img_path is None:
                    img_path = self.find_param(history_msg, 'mask', excluded=True)

                func_inputs = f'{img_path}'

        elif 'image' in inputs.lower() or 'figure' in inputs.lower() or 'picture' in inputs.lower():
            cls = self.models.get('HuskyVQA', None)
            func_name = 'HuskyVQA'
            if cls is not None:
                func = cls.inference
            img_path = self.find_param(inputs, 'mask', excluded=True)
            if img_path is None:
                img_path = self.find_param(history_msg, 'mask', excluded=True)
            prompt = inputs.strip()
            func_inputs = f'{img_path},{prompt}'
        else:
            def only_chat(inputs):
                if not self.chat_disabled:
                    res = agent(f"You can use history message to respond to the following question without using any tools. Request: {inputs}")
                    res = res['output'].replace("\\", "/")
                else:
                    res = "The chat-related functions is now disabled. Please try other features."
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
                res = f"I have used the tool: \"{func_name}\" with the inputs: \"{func_inputs}\" to get the results. The result image is named {return_res}."
            else:
                res = return_res
        print(f"I have used the tool: \"{func_name}\" to obtain the results. The Inputs: \"{func_inputs}\". Result: {return_res}.")
        return res
    
    def check_illegal_files(self, file_list):
        illegal_files = []
        for file_item in file_list:
            if not os.path.exists(file_item[0]):
                illegal_files.append(file_item[0])

        return illegal_files
    
    def find_parent(self, cur_path, history_msg):
        if cur_path is None:
            return None
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
        file_items = pattern.findall(response)
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

        if mask_idx < img_idx:
            return False

        return True

    def exec_simple_action(self, inputs, history_msg):
        print('Execute the simple action without ChatGPT.')
        print('history_msg: ', history_msg + inputs)
        print('inputs: ', inputs)
        func = None
        func_name = None
        func_inputs = None
        res = None
        new_inputs = inputs.replace('ReplaceMaskedAnything', 'placeholder')
        if 'remove' in inputs.lower() or 'erase' in inputs.lower():
            cls = self.models.get('LDMInpainting', None)
            if cls is not None:
                func = cls.inference
            
            mask_path = self.find_param(inputs, 'mask')
            if mask_path is None:
                mask_path = self.find_param(history_msg, 'mask')

            if mask_path is None:
                return 'I can not found the mask_path. Please check you have successfully operated on input image.'

            img_path =  self.find_parent(mask_path, history_msg+inputs)
            if img_path is None:
                img_path = self.find_param(history_msg+inputs, 'mask', excluded=True)
            
            func_inputs = f'{img_path},{mask_path}'
            func_name = 'LDMInpainting'
        elif 'replace' in new_inputs.lower():
            cls = self.models.get('ReplaceMaskedAnything', None)
            if cls is not None:
                func = cls.inference
            
            mask_path = self.find_param(inputs, 'mask')
            if mask_path is None:
                mask_path = self.find_param(history_msg, 'mask')

            if mask_path is None:
                return 'I can not found the mask_path. Please check you have successfully operated on input image.'
                
            img_path =  self.find_parent(mask_path, history_msg+inputs)
            if img_path is None:
                img_path = self.find_param(history_msg+inputs, 'mask', excluded=True)

            if img_path is None:
                return 'I can not found the image_path. Please check you have successfully uploaded an input image.'
            
            func_inputs = f'{img_path},{mask_path},{inputs}'
            func_name = 'ReplaceMaskedAnything'
        
        print(f'{func_name}: {func_inputs}')
        
        if func is None:
            return None

        return_res = func(func_inputs)
        res = f"I have used the tool: \"{func_name}\" with the inputs: \"{func_inputs}\" to get the results. The result image is named {return_res}."
        print(res)
        return res
    
    def exec_agent(self, inputs, agent):
        # pattern = re.compile('(image/[-\\w]*.(png|mp4))')
        response = agent({"input": inputs})['output']
        response = response.replace("\\", "/")
        print('response = ', response)
        using_tool_words = "I used the tool"
        if self.chat_disabled and using_tool_words not in response:
            response = "For a short period of time in the future, I cannot chat with you due to some policy requirements. I hope you can understand."
            return response

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
        print(f'The latest file is {res}.')
        return res
    
    def read_images_from_internet(self, inputs, user_state):
        urls = re.findall('(https?://[a-zA-Z0-9\.\?/%-_]*)', inputs)
        state = []
        for url in urls:
            try:
                response = req.get(url)
                bytes = BytesIO(response.content)
                image = Image.open(bytes)
                image_caption, ocr_res_raw, image_filename = self.process_image(image)
                _, user_state = self.put_image_info_into_memory(image_caption, ocr_res_raw, image_filename, user_state)
                # state += [(, None)]
                state += [(None, f"![](file={image_filename})*{image_filename}(From: {url})*")]
                inputs = inputs.replace(url, image_filename)
            except Exception as e:
                print(e)
                print(f'Error: {url} is not an Image!')

        return inputs, state, user_state
        
    def run_text(self, text, state, user_state):
        text = text.strip()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if text is None or len(text) == 0:
            state += [(None, 'Please input text.')]
            return state, state, user_state

        new_inputs, new_state, user_state = self.read_images_from_internet(text, user_state)
        agent = user_state[0]['agent']
        agent.memory.buffer = cut_dialogue_history(agent.memory.buffer, keep_last_n_words=500)
        history_msg = agent.memory.buffer[:]
        # response = self.exec_agent(new_inputs, agent)
        try:
            response = self.exec_simple_action(new_inputs, history_msg)
            if response is None:
                # inputs = self.get_suggested_inputs(text, history_msg)
                response = self.exec_agent(new_inputs, agent)
            else:
                agent.memory.buffer += f'\nHuman: {new_inputs}\n' + f'AI: {response})'
            res = self.find_result_path(response)
        except Exception as err1:
            # import pdb
            # pdb.set_trace()
            print(f'Error in line {err1.__traceback__.tb_lineno}: {err1}')
            try:
                response = self.rectify_action(new_inputs, history_msg, agent)
                res = self.find_result_path(response)
                agent.memory.buffer += f'\nHuman: {text}\n' + f'AI: {response}'
            except Exception as err2:
                print(f'Error in line {err2.__traceback__.tb_lineno}: {err2}')
                state += [(text, 'Sorry, something went wrong inside the ChatGPT. Please check whether your image, video and message have been uploaded successfully.')]
                return state, state, user_state

        state += [(text, None)] + new_state
        if res is not None and agent.memory.buffer.count(res) <= 1:
            state = state + [(None, response + f' `{res}` is as follows: ')]
            state = state + [(None, (res, ))]
        else:
            state = state + [(None, response)]
            
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
    
    def upload_audio(self, audio_path, state, user_state):
        print(f'audio_path = {audio_path}')
        if audio_path is None or not os.path.exists(audio_path):
            state += [(None, 'No audio input. Please upload audio file.')]
            return state, state

        user_state = self.clear_user_state(False, user_state)
        audio_name = os.path.basename(audio_path)
        # vid_name = gen_new_name(vid_name, '', vid_name.split('.')[-1])
        new_audio_path = os.path.join('./image/', audio_name)
        new_audio_path = gen_new_name(new_audio_path, 'audio', audio_name.split('.')[-1])
        shutil.copy(audio_path, new_audio_path)

        user_state[0]['audio_path'] = new_audio_path

        Human_prompt = f'\nHuman: provide an audio file named {new_audio_path}. You should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\". \n'
        AI_prompt = f"Received. "

        user_state[0]['agent'].memory.buffer += Human_prompt + 'AI: ' + AI_prompt

        state = state + [((new_audio_path, ), AI_prompt)]

        print(f"\nProcessed upload_video, Input Audio: `{new_audio_path}`\nCurrent state: {state}\n"
              f"Current Memory: {user_state[0]['agent'].memory.buffer}")

        return state, state, user_state
    
    def process_image(self, image):
        img = image
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:6]}.png")
        image_filename = gen_new_name(image_filename, 'image')
        img.save(image_filename, "PNG")
        img = img.convert('RGB')
        
        image_caption = None
        if 'HuskyVQA' in self.models.keys():
            image_caption = self.models['HuskyVQA'].inference_captioning(image_filename)

        ocr_res_raw = None
        if 'ImageOCRRecognition' in self.models.keys():
            # ocr_res = self.models['ImageOCRRecognition'].inference(image_filename)
            ocr_res_raw = self.models['ImageOCRRecognition'].readtext(image_filename)

        return image_caption, ocr_res_raw, image_filename

    def put_image_info_into_memory(self, image_caption, ocr_res_raw, image_filename, user_state):
        ocr_res = None
        state = []
        Human_prompt = f'\nHuman: provide a image named {image_filename}. '
        if image_caption is not None and len(image_caption) > 0:
            Human_prompt += f'The description is: {image_caption} '

        if ocr_res_raw is not None:
            ocr_res = self.models['ImageOCRRecognition'].parse_result(ocr_res_raw)

        if ocr_res is not None and len(ocr_res) > 0:
            Human_prompt += f'Recognized optical characters: {ocr_res}. '
            # user_state[0]['ocr_res'] = ocr_res_raw
            
        Human_prompt += 'This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\". \n'

        AI_prompt = "Received. "
        user_state[0]['agent'].memory.buffer += Human_prompt + 'AI: ' + AI_prompt
        state = state + [(f"![](file={image_filename})*{image_filename}*", AI_prompt)]
        return state, user_state

    def upload_image(self, image, state, user_state):
        # [txt, click_img, state, user_state], [chatbot, txt, state, user_state]
        print('upload an image')
        if image is None or image.get('image', None) is None:
            return state, state, user_state
        
        user_state = self.clear_user_state(False, user_state)
        image_caption, ocr_res_raw, image_filename = self.process_image(image['image'])
        user_state[0]['image_path'] = image_filename
        user_state[0]['ocr_res'] = ocr_res_raw
        user_state[0]['image_caption'] = image_caption
        t_state, user_state = self.put_image_info_into_memory(image_caption, ocr_res_raw, image_filename, user_state)
        state += t_state
        
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
        new_video_path = gen_new_name(new_video_path, 'video', vid_name.split('.')[-1])
        shutil.copy(video_path, new_video_path)

        user_state[0]['video_path'] = new_video_path
        if "VideoCaption" in self.models.keys():
            description = self.models['VideoCaption'].inference(new_video_path)
        else:
            description = 'A video.'
        user_state[0]['video_caption'] = description
        Human_prompt = f'\nHuman: provide a video named {new_video_path}. The description is: {description}. This information helps you to understand this video, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\". \n'
        AI_prompt = f"Received. "

        user_state[0]['agent'].memory.buffer += Human_prompt + 'AI: ' + AI_prompt

        state = state + [((new_video_path, ), AI_prompt)]

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
        # print(f'user_state[0][\'image_path\'] = {user_state[0]["image_path"]}')
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
        img = Image.fromarray(img)
        mask = image['mask'].convert('L')
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
            state += [(Human_prompt, f'ImageOCRRecognition is not loaded.')]

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
            state += [(None, 'Please upload an image or draw a mask.')]
            return None, state, state, user_state
        
        uploaded_image_filename = user_state[0].get('image_path', None)
        if image.get('mask', None) is None:
            state += [(None, 'Please upload an image or draw a mask.')]
            return None, state, state, user_state 
        
        mask_image = image['mask'].convert('RGB')
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
    
    def gen_new_image(self, state, user_state):
        model = self.models.get('StyleGAN', None)
        if model is None:
            state += [None, 'Please load StyleGAN!']
            return None, state, state, user_state
        
        if user_state[0].get('StyleGAN', None) is None:
            user_state[0]['StyleGAN'] = {}

        styleGAN_state = user_state[0]['StyleGAN']
        seed = styleGAN_state.get('seed', None)
        if seed is None:
            init_seed = 2048
            seed_everything(init_seed)
            user_state[0]['StyleGAN']['seed'] = init_seed

        device = model.device 
        e_mode = model.e_mode
        g_ema = model.g_ema
        if e_mode is True:
            g_ema.to(device=device)
        sample_z = torch.randn([1, 512], device=device)
        latent, noise = g_ema.prepare([sample_z])
        sample, F = g_ema.generate(latent, noise)
        if e_mode is True:
            g_ema.to(device="cpu")
        for i in range(len(noise)):
            if isinstance(noise[i], torch.Tensor):
                noise[i] = noise[i].to('cpu')

        gan_state = {
            'latent': latent.to('cpu'),
            'noise': noise,
            'F': F.to('cpu'),
            'sample': sample.to('cpu'),
            'history': []
        }
        
        image_arr = to_image(sample)
        new_image = Image.fromarray(image_arr)
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:6]}.png")
        image_filename = gen_new_name(image_filename, 'image')
        
        new_image.save(image_filename, "PNG")
        state = state + [(None, f"![](file={image_filename})*{image_filename}*")]
        user_state[0]['StyleGAN']['state'] = gan_state
        user_state[0]['StyleGAN']['points'] = {'end': [], 'start': []}
        user_state[0]['StyleGAN']['image_path'] = image_filename
        user_state[0]['StyleGAN']['image_size'] = model.image_size
        SIZE_TO_CLICK_SIZE = {
            1024: 15,
            256: 6
        }
        user_state[0]['StyleGAN']['click_size'] = SIZE_TO_CLICK_SIZE[model.image_size]
        Human_prompt = f'\nHuman: provide a image named {image_filename}. You should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\". \n'
        AI_prompt = "Received. "
        # self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + ' AI: ' + AI_prompt
        user_state[0]['agent'].memory.buffer += Human_prompt + 'AI: ' + AI_prompt
        return image_arr, state, state, user_state
    
    def drag_it(self, image, max_iters, state, user_state):
        model = self.models.get('StyleGAN', None)
        if model is None:
            state += [(None, 'Please load StyleGAN!')]
            return image, 0, state, state, user_state
        if user_state[0].get('StyleGAN', None) is None:
            state += [(None, 'Please click the button `New Image`.')]
            return image, 0, state, state, user_state 

        image_path = user_state[0]['StyleGAN'].get('image_path', None)
        if image_path is None:
            return image, 0, state, state, user_state
        
        points = user_state[0]['StyleGAN']['points']
        if len(points['start']) == 0:
            state += [(None, f'Please click the image.')]
            return image, 0, state, state, user_state

        if len(points['start']) != len(points['end']):
            state += [(None, f'Start points (num={len(points["start"])}) can not match end points (num={len(points["end"])})')]
            return image, 0, state, state, user_state

        click_size = user_state[0]['StyleGAN']['click_size']
        style_gan_state = user_state[0]['StyleGAN'].get('state', None)
        if style_gan_state is None:
            state += [(None, 'Please click the button `New Image`.')]
            return image, 0, state, state, user_state 
        
        max_iters = int(max_iters)
        latent = style_gan_state['latent']
        noise = style_gan_state['noise']
        F = style_gan_state['F']

        style_gan_state['history'] = []

        start_points = [torch.tensor(p).float() for p in points['start']]
        end_points = [torch.tensor(p).float() for p in points['end']]
        mask = None
            
        step = 0
        device = model.device
        e_mode = model.e_mode
        latent = latent.to(device)
        if e_mode is True:
            model.g_ema.to(device=device)
        F = F.to(device)
        for i in range(len(noise)):
            if isinstance(noise[i], torch.Tensor):
                noise[i] = noise[i].to(device)
        for sample2, latent, F, handle_points in drag_gan(model.g_ema, latent, noise, F,
                                                          start_points, end_points, mask,
                                                          device, max_iters=max_iters):
            image = to_image(sample2)
            style_gan_state['F'] = F.cpu()
            style_gan_state['latent'] = latent.cpu()
            style_gan_state['sample'] = sample2.cpu()
            points['start'] = [p.cpu().numpy().astype('int').tolist() for p in handle_points]
            org_image = image.copy()
            add_points_to_image(image, points, size=click_size)

            style_gan_state['history'].append(org_image)
            step += 1
            # print(f'step = {step}')
            if max_iters == step:
                video_name = gen_new_name(image_path, 'DragGAN', 'mp4')
                imageio.mimsave(video_name, style_gan_state['history'])
                AI_prompt = f'The editing process is saved in {video_name}: '
                state += [(None, AI_prompt)]
                # state += [None, AI_prompt]
                state += [(None, (video_name, ))]
                new_image = Image.fromarray(org_image)
                # image_filename = os.path.join('image', f"{str(uuid.uuid4())[:6]}.png")
                image_filename = gen_new_name(image_path, 'DragGAN')
                new_image.save(image_filename, "PNG")
                AI_prompt = f'The processed image is named {image_filename}: '
                state += [(None, AI_prompt)]
                state += [(None, (image_filename, ))]
                user_state[0]['StyleGAN']['state'] = style_gan_state
                Human_prompt = f'\nHuman: provide a image named {image_filename}. You should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\". \n'
                AI_prompt = "Received. "
                user_state[0]['agent'].memory.buffer += Human_prompt + 'AI: ' + AI_prompt
                del latent, sample2, F
                if e_mode is True:
                    model.g_ema.to(device="cpu")
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                yield image, step, state, state, user_state

            yield image, step, state, state, user_state
    
    def try_drag_it(self, state, user_state):
        start_point = user_state[0]['StyleGAN']['points']['start']
        end_point = user_state[0]['StyleGAN']['points']['end']
        if len(start_point) == len(end_point):
            return self.drag_it(state, user_state)
        return gr.update(visible=True), 0, state, state, user_state
    
    def save_points_for_drag_gan(self, image, user_state, evt: gr.SelectData):
        points = user_state[0]['StyleGAN']['points']
        start_point = user_state[0]['StyleGAN']['points'].get('start')
        end_point = user_state[0]['StyleGAN']['points'].get('end')
        click_size = user_state[0]['StyleGAN']['click_size']

        if len(start_point) > len(end_point):
            points['end'].append([evt.index[1], evt.index[0]])
            image = add_points_to_image(image, points, size=click_size)
            return image, user_state
        
        points['start'].append([evt.index[1], evt.index[0]])
        
        image = add_points_to_image(image, points, size=click_size)

        return image, user_state

    def reset_drag_points(self, image, user_state):
        if user_state[0].get('StyleGAN', None) is None:
            return image, user_state
        
        user_state[0]['StyleGAN']['points'] = {'end': [], 'start': []}

        gan_state = user_state[0]['StyleGAN'].get('state', None)
        sample = None
        if gan_state is not None:
            sample = gan_state.get('sample', None)

        if sample is not None:
            image = to_image(sample)
        else:
            image_path = user_state[0]['StyleGAN'].get('image_path', None)
            if image_path is not None:
                image = Image.open(image_path)

        return image, user_state

    def clear_user_state(self, clear_memory, user_state):
        new_user_state = [{}]
        new_user_state[0]['agent'] = user_state[0]['agent']
        new_user_state[0]['memory'] = user_state[0]['memory']
        if clear_memory:
            new_user_state[0]['memory'].clear()
        else:
            new_user_state[0]['memory'] = user_state[0]['memory']

        return new_user_state
