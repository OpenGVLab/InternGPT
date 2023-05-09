import inspect
import re
import os
import numpy as np
import uuid
import shutil
import whisper
import gradio as gr

from PIL import Image

from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI

from ..models import *
from iGPT.models.utils import gen_new_name

GLOBAL_SEED=1912


'''
INTERN_CHAT_PREFIX = """InternChat is designed to be able to assist with a wide range of text and visual related tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. InternChat is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

InternChat is able to process and understand large amounts of text and images. As a language model, InternChat can not directly read images, but it has a list of tools to finish different visual tasks. Each image will have a file name formed as "image/xxx.png", and InternChat can invoke different tools to indirectly understand pictures. When talking about images, InternChat is very strict to the file name and will never fabricate nonexistent files. When using tools to generate new image files, InternChat is also known that the image may not be the same as the user's demand, and will use other visual question answering tools or description tools to observe the real image. InternChat is able to use tools in a sequence, and is loyal to the tool observation outputs rather than faking the image content and image file name. It will remember to provide the file name from the last tool observation, if a new image is generated.

Human may provide new figures to InternChat with a description. The description helps InternChat to understand this image, but InternChat should use tools to finish following tasks, rather than directly imagine from the description.

Overall, InternChat is a powerful visual dialogue assistant tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 


TOOLS:
------

InternChat has access to the following tools:"""

INTERN_CHAT_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

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

INTERN_CHAT_SUFFIX = """You are very strict to the filename correctness and will never fake a file name if it does not exist.
You will remember to provide the image file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Since InternChat is a text language model, InternChat must use tools to observe images rather than imagination.
The thoughts and observations are only visible for InternChat, InternChat should remember to repeat important information in the final response for Human. 
Thought: Do I need to use a tool? {agent_scratchpad} Let's think step by step.
"""

INTERN_CHAT_PREFIX_CN = """InternChat 旨在能够协助完成范围广泛的文本和视觉相关任务，从回答简单的问题到提供对广泛主题的深入解释和讨论。 InternChat 能够根据收到的输入生成类似人类的文本，使其能够进行听起来自然的对话，并提供连贯且与手头主题相关的响应。

InternChat 能够处理和理解大量文本和图像。作为一种语言模型，InternChat 不能直接读取图像，但它有一系列工具来完成不同的视觉任务。每张图片都会有一个文件名，格式为“image/xxx.png”，InternChat可以调用不同的工具来间接理解图片。在谈论图片时，InternChat 对文件名的要求非常严格，绝不会伪造不存在的文件。在使用工具生成新的图像文件时，InternChat也知道图像可能与用户需求不一样，会使用其他视觉问答工具或描述工具来观察真实图像。 InternChat 能够按顺序使用工具，并且忠于工具观察输出，而不是伪造图像内容和图像文件名。如果生成新图像，它将记得提供上次工具观察的文件名。

Human 可能会向 InternChat 提供带有描述的新图形。描述帮助 InternChat 理解这个图像，但 InternChat 应该使用工具来完成以下任务，而不是直接从描述中想象。有些工具将会返回英文描述，但你对用户的聊天应当采用中文。

总的来说，InternChat 是一个强大的可视化对话辅助工具，可以帮助处理范围广泛的任务，并提供关于范围广泛的主题的有价值的见解和信息。

工具列表:
------

InternChat 可以使用这些工具:"""

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

因为InternChat是一个文本语言模型，必须使用工具去观察图片而不是依靠想象。
推理想法和观察结果只对InternChat可见，需要记得在最终回复时把重要的信息重复给用户，你只能给用户返回中文句子。我们一步一步思考。在你使用工具时，工具的参数只能是英文。

聊天历史:
{chat_history}

新输入: {input}
Thought: Do I need to use a tool? {agent_scratchpad}
"""
'''


VISUAL_CHATGPT_PREFIX = """Visual ChatGPT is designed to be able to assist with a wide range of text and visual related tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. Visual ChatGPT is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Visual ChatGPT is able to process and understand large amounts of text and images. As a language model, Visual ChatGPT can not directly read images, but it has a list of tools to finish different visual tasks. Each image will have a file name formed as "image/xxx.png", and Visual ChatGPT can invoke different tools to indirectly understand pictures. When talking about images, Visual ChatGPT is very strict to the file name and will never fabricate nonexistent files. When using tools to generate new image files, Visual ChatGPT is also known that the image may not be the same as the user's demand, and will use other visual question answering tools or description tools to observe the real image. Visual ChatGPT is able to use tools in a sequence, and is loyal to the tool observation outputs rather than faking the image content and image file name. It will remember to provide the file name from the last tool observation, if a new image is generated.

Human may provide new figures to Visual ChatGPT with a description. The description helps Visual ChatGPT to understand this image, but Visual ChatGPT should use tools to finish following tasks, rather than directly imagine from the description.

Overall, Visual ChatGPT is a powerful visual dialogue assistant tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 


TOOLS:
------

Visual ChatGPT  has access to the following tools:"""

VISUAL_CHATGPT_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

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

VISUAL_CHATGPT_SUFFIX = """You are very strict to the filename correctness and will never fake a file name if it does not exist.
You will remember to provide the image file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Since Visual ChatGPT is a text language model, Visual ChatGPT must use tools to observe images rather than imagination.
The thoughts and observations are only visible for Visual ChatGPT, Visual ChatGPT should remember to repeat important information in the final response for Human. 
Thought: Do I need to use a tool? {agent_scratchpad} Let's think step by step.
"""

VISUAL_CHATGPT_PREFIX_CN = """Visual ChatGPT 旨在能够协助完成范围广泛的文本和视觉相关任务，从回答简单的问题到提供对广泛主题的深入解释和讨论。 Visual ChatGPT 能够根据收到的输入生成类似人类的文本，使其能够进行听起来自然的对话，并提供连贯且与手头主题相关的响应。

Visual ChatGPT 能够处理和理解大量文本和图像。作为一种语言模型，Visual ChatGPT 不能直接读取图像，但它有一系列工具来完成不同的视觉任务。每张图片都会有一个文件名，格式为“image/xxx.png”，Visual ChatGPT可以调用不同的工具来间接理解图片。在谈论图片时，Visual ChatGPT 对文件名的要求非常严格，绝不会伪造不存在的文件。在使用工具生成新的图像文件时，Visual ChatGPT也知道图像可能与用户需求不一样，会使用其他视觉问答工具或描述工具来观察真实图像。 Visual ChatGPT 能够按顺序使用工具，并且忠于工具观察输出，而不是伪造图像内容和图像文件名。如果生成新图像，它将记得提供上次工具观察的文件名。

Human 可能会向 Visual ChatGPT 提供带有描述的新图形。描述帮助 Visual ChatGPT 理解这个图像，但 Visual ChatGPT 应该使用工具来完成以下任务，而不是直接从描述中想象。有些工具将会返回英文描述，但你对用户的聊天应当采用中文。

总的来说，Visual ChatGPT 是一个强大的可视化对话辅助工具，可以帮助处理范围广泛的任务，并提供关于范围广泛的主题的有价值的见解和信息。

工具列表:
------

Visual ChatGPT 可以使用这些工具:"""

VISUAL_CHATGPT_FORMAT_INSTRUCTIONS_CN = """用户使用中文和你进行聊天，但是工具的参数应当使用英文。如果要调用工具，你必须遵循如下格式:

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

VISUAL_CHATGPT_SUFFIX_CN = """你对文件名的正确性非常严格，而且永远不会伪造不存在的文件。

开始!

因为Visual ChatGPT是一个文本语言模型，必须使用工具去观察图片而不是依靠想象。
推理想法和观察结果只对Visual ChatGPT可见，需要记得在最终回复时把重要的信息重复给用户，你只能给用户返回中文句子。我们一步一步思考。在你使用工具时，工具的参数只能是英文。

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
    def __init__(self, load_dict):
        # load_dict = {'VisualQuestionAnswering':'cuda:0', 'ImageCaptioning':'cuda:1',...}
        print(f"Initializing VisualChatGPT, load_dict={load_dict}")
        if 'ImageCaptioning' not in load_dict:
            raise ValueError("You have to load ImageCaptioning as a basic function for i-GPT")
        # if 'SegmentAnything' not in load_dict:
        #     raise ValueError("You have to load SegmentAnything as a basic function for i-GPT")

        self.models = {}
        self.uploaded_image_filename = None
        # self.segmented_image_filename = None
        self.history_mask = None
        self.load_dict = load_dict
        # self.llm = None
        # Load Basic Foundation Models
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)
        # self.models['models'] = self.models

        # Load Template Foundation Models
        for class_name, module in globals().items():
            if getattr(module, 'template_model', False):
                template_required_names = {k for k in inspect.signature(module.__init__).parameters.keys() if k!='self'}
                loaded_names = set([type(e).__name__ for e in self.models.values()])
                if template_required_names.issubset(loaded_names):
                    self.models[class_name] = globals()[class_name](
                        **{name: self.models[name] for name in template_required_names})
                # elif 'models' in template_required_names:
                #     self.models[class_name] = globals()[class_name](
                #         **{name: self.models[name] for name in template_required_names})

        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith('inference'):
                    func = getattr(instance, e)
                    self.tools.append(Tool(name=func.name, description=func.description, func=func))
        self.llm = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
        # self.first_init=True
        self.audio_model = None

    def init_agent(self):
        self.memory.clear() #clear previous history
        self.reset()
        self.llm = OpenAI(temperature=0)
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': VISUAL_CHATGPT_PREFIX, 'format_instructions': VISUAL_CHATGPT_FORMAT_INSTRUCTIONS,
                          'suffix': VISUAL_CHATGPT_SUFFIX}, )

    def run_text(self, text, state):
        # print(f'text = {text}')
        self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        try:
            print(f'text = {text}')
            res = self.agent({"input": text.strip()})
            print('ab'* 30)
            print(res['output'])
            print('cd'* 30)
        except Exception as err:
            # Human_prompt = text
            # self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + ' AI: ' + AI_prompt
            state += [(text, 'I can not understand your instruction. Could you provide more information?')]
            print(err)
            return state, state
        
        res['output'] = res['output'].replace("\\", "/")
        # response = re.sub('(tmp_files/[-\w]*.[png|mp4])', lambda m: f'![](file={m.group(0)})*{m.group(0)}*', res['output'])

        # print("res['output'] = ", res['output'])
        # response = re.sub('(tmp_files/[-\w]*.(png|mp4))', replace_path, res['output'])
        pattern = re.compile('(image/[-\\w]*.(png|mp4))')
        out_filenames = pattern.findall(res['output'])
        response = res['output']
        state = state + [(text, response)]
        for f in out_filenames:
            state = state + [(None, f'{f[0]} is as following: ')]
            state = state + [(None, (f[0], ))]
        # if len(out_filenames) > 1:
        #     state = state + [(None, (out_filenames[-1][0], ))]
            # print('out_filename[-1][0] = ', out_filenames[-1][0])
        print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state, state
    
    def run_audio(self, audio_path, state):
        print(f'audio_path = {audio_path}')
        if self.audio_model is None:
            self.audio_model = whisper.load_model("small").to('cuda:0')
        text = self.audio_model.transcribe(audio_path)["text"]
        res = self.run_text(text, state)
        print(f"\nProcessed run_audio, Input transcribed audio: {text}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return res[0], res[1]

    def upload_image(self, image, state, txt):
        self.reset()
        img = image['image']
        image_filename = os.path.join('image/', f"{str(uuid.uuid4())[:6]}.png")
        self.uploaded_image_filename=image_filename
        img = img.convert('RGB')
        img.save(image_filename, "PNG")
        # print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
        # let some foundation models preprocess image
        # NEED_PREPROCESSING_LIST = ["SegmentAnything", "ImageOCRRecognition"]
        # for model_name in NEED_PREPROCESSING_LIST:
        #     if model_name in self.models.keys():
        #          self.models[model_name].preprocess(np.array(img), image_filename)

        description = self.models['ImageCaptioning'].inference(image_filename)
        # description = 'Debug'

        Human_prompt = f'\nHuman: provide a figure named {image_filename}. The description is: {description}. This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\". \n'
        AI_prompt = "Received. "
        self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + ' AI: ' + AI_prompt
        state = state + [(f"![](file={image_filename})*{image_filename}*", AI_prompt)]
        print(f"\nProcessed upload_image, Input image: {image_filename}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state, state, f'{txt} {image_filename} ', gr.update(visible=True), gr.update(visible=True)
    
    def upload_video(self, video_path, state, txt):
        # self.cur_file = video_path
        vid_name = os.path.basename(video_path)
        # vid_name = gen_new_name(vid_name, '', vid_name.split('.')[-1])
        new_video_path = os.path.join('./image/', f"{str(uuid.uuid4())[:6]}.mp4")
        new_video_path = gen_new_name(new_video_path, '', vid_name.split('.')[-1])
        shutil.copy(video_path, new_video_path)
        
        if "VideoCaption" in self.models.keys():
            description = self.models['VideoCaption'].inference(new_video_path)
        else:
            description = 'A video.'
        Human_prompt = f'\nHuman: provide a video named {new_video_path}. The description is: {description}. This information helps you to understand this video, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\". \n'
        AI_prompt = f"Received video: {new_video_path} "
        self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
        # state = state + [(f"![](file={new_video_path})*{new_video_path}*", AI_prompt)]
        # state = state + [(f"![](file={video_path})*{new_video_path}*", AI_prompt)]
        state = state + [((new_video_path, ), AI_prompt)]
        # print('exists = ', os.path.exists("./tmp_files/1e7f_f4236666_tmp.mp4"))
        print(f"\nProcessed upload_video, Input video: {new_video_path}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state, state, f'{txt} {new_video_path} '

    def blend_mask(self, img, mask):
        mask = mask.astype(np.uint8)
        transparency_ratio = mask.astype(np.float32) / 3
        transparency_ratio = transparency_ratio[:, :, np.newaxis]
        mask = mask[:, :, np.newaxis] * 255
        mask= mask.repeat(3, axis=2)
        mask[:,:,0] = 0
        mask[:,:,2] = 0
        new_img_arr = np.array(img) * (1 - transparency_ratio) + mask * transparency_ratio
        new_img_arr = np.clip(new_img_arr, 0, 255).astype(np.uint8)
        # print(new_img_arr.shape)
        return Image.fromarray(new_img_arr)

    def process_image(self, image, state):
        img = Image.open(self.uploaded_image_filename).convert('RGB')
        # img = image['image'].convert('RGB')
        mask = image['mask'].convert('L')
        mask = np.array(mask, dtype=np.uint8)

        Human_prompt="Please process this image based on given mask."
        if self.uploaded_image_filename is None:
            AI_prompt = "Please upload an image for processing."
            state += [(Human_prompt, AI_prompt)]
            return state, state, None
        if mask.sum() == 0:
            AI_prompt = "You can click the image in the right and ask me some questions."
            state += [(Human_prompt, AI_prompt)]
            return state, state, image['image']
        
        if self.history_mask is None:
            self.history_mask = mask
        else:
            self.history_mask = np.logical_or(self.history_mask, mask)
        
        if 'SegmentAnything' in self.models.keys():
            self.models['SegmentAnything'].clicked_region = self.history_mask
        if 'ImageOCRRecognition' in self.models.keys():
            self.models['ImageOCRRecognition'].clicked_region = mask

        # self.models['SegmentAnything'].mask = self.history_mask
        # history_mask = self.history_mask.astype(np.uint8) * 255
        res_mask = self.models['SegmentAnything'].segment_by_mask(self.history_mask)
        
        img = self.blend_mask(img, res_mask)

        AI_prompt = f"I have finished processing. Now, you can ask me some questions."
        state = state + [(Human_prompt, AI_prompt)]
        # AI_prompt = f"Received. I found {ocr_text} in this position. The sgemented figure is named {seg_filename}."
        self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + ' AI: ' + AI_prompt
        # state = state + [(Human_prompt, f"![](file={seg_filename})*{AI_prompt}*")]
        # print()
        print(f"\nProcessed run_image, Input image: {self.uploaded_image_filename}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state, state, img
    
    def reset(self, clear_history_memory=False):
        print('reset the model cache.')
        NEED_RESET_LIST = ['SegmentAnything', 'ImageOCRRecognition']
        for model_name in NEED_RESET_LIST:
            if model_name in self.models.keys():
                self.models[model_name].reset()    

        self.history_mask = None 
        self.uploaded_image_filename = None
        if clear_history_memory:
            self.agent.memory.clear()
        return None