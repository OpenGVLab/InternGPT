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
from PIL import Image
import numpy as np
import argparse
from functools import partial

import gradio as gr
import gradio.themes.base as ThemeBase
from gradio.themes.utils import colors, fonts, sizes

from openai.error import APIConnectionError

# from iGPT.models import *

from iGPT.controllers import ConversationBot

import openai
from langchain.llms.openai import OpenAI

# openai.api_base = 'https://closeai.deno.dev/v1'

os.makedirs('image', exist_ok=True)


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
        user_state = bot.init_agent()
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False, value=''), user_state
    else:
        if api_key and len(api_key) > 30:
            print(api_key)
            os.environ["OPENAI_API_KEY"] = api_key
            openai.api_key = api_key
            try:
                llm = OpenAI(temperature=0)
                llm('Hi!')
                response = 'Success!'
                is_error = False
                user_state = bot.init_agent()
            except Exception as err:
                # gr.update(visible=True)
                print(err)
                response = 'Incorrect key, please input again'
                is_error = True
        else:
            is_error = True
            response = 'Incorrect key, please input again'
        
        return gr.update(visible=not is_error), gr.update(visible=is_error), gr.update(visible=is_error, value=response), user_state

    
def change_input_type(flag):
    if flag:
        print('Using voice input.')
    else:
        print('Using text input.')
    return gr.update(visible=not flag), gr.update(visible=flag)

def random_image():
    root_path = './assets/images'
    img_list = os.listdir(root_path)
    img_item = random.sample(img_list, 1)[0]
    return Image.open(os.path.join(root_path, img_item))

def random_video():
    root_path = './assets/videos'
    vid_list = os.listdir(root_path)
    vid_item = random.sample(vid_list, 1)[0]
    return os.path.join(root_path, vid_item)

def random_audio():
    root_path = './assets/audio'
    aud_list = os.listdir(root_path)
    aud_item = random.sample(aud_list, 1)[0]
    print(os.path.join(root_path, aud_item))
    return os.path.join(root_path, aud_item)

def process_video_tab():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

def process_image_tab():
    return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

def process_audio_tab():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

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
            """
        )
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
                        audio2text_input = gr.Audio(source="microphone", type="filepath", visible=False)
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
                        with gr.Column(scale=0.4, min_width=0, visible=False) as aud_example:
                            add_aud_example = gr.Button("📻 Give an Example", variant="primary")
                        with gr.Column(scale=0.4, min_width=0, visible=False) as vid_example:
                            add_vid_example = gr.Button("📽 Give an Example", variant="primary")
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
                with gr.Tab("Audio", elem_id='audio_tab') as audio_tab:
                    audio_input = gr.Audio(source="upload", type="filepath", visible=True, elem_id="audio_upload").style(height=360)
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
                bot.run_task, [audio_switch, txt, audio2text_input, state, user_state], [chatbot, state, user_state]).then(
                lambda: "", None, [txt, ])
            
            audio_switch.change(change_input_type, [audio_switch, ], [txt, audio2text_input])

            add_img_example.click(random_image, [], [click_img,]).then(
                lambda: gr.update(visible=False), [], [send_btn]).then(
                lambda: gr.update(visible=False), [], [txt]).then(
                lambda: gr.update(visible=False), [], [vis_btn]).then( 
                bot.upload_image, [click_img, state, user_state], 
                [chatbot, state, user_state]).then(
                lambda: gr.update(visible=True), [], [send_btn]).then(
                lambda: gr.update(visible=True), [], [txt]).then(
                lambda: gr.update(visible=True), [], [vis_btn])

            add_vid_example.click(random_video, [], [video_input,]).then(
                lambda: gr.update(visible=False), [], [send_btn]).then(
                lambda: gr.update(visible=False), [], [txt]).then(
                lambda: gr.update(visible=False), [], [vis_btn]).then( 
                bot.upload_video, [video_input, state, user_state], 
                [chatbot, state, user_state]).then(
                lambda: gr.update(visible=True), [], [send_btn]).then(
                lambda: gr.update(visible=True), [], [txt]).then(
                lambda: gr.update(visible=True), [], [vis_btn])

            add_aud_example.click(random_audio, [], [audio_input,]).then(
                lambda: gr.update(visible=False), [], [send_btn]).then(
                lambda: gr.update(visible=False), [], [txt]).then(
                lambda: gr.update(visible=False), [], [vis_btn]).then( 
                bot.upload_audio, [audio_input, state, user_state], 
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
            video_tab.select(process_video_tab, [], [whiteboard_mode, img_example, aud_example, vid_example])
            img_tab.select(process_image_tab, [], [whiteboard_mode, img_example, aud_example, vid_example])
            audio_tab.select(process_audio_tab, [], [whiteboard_mode, img_example, aud_example, vid_example])
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
            
            audio_input.upload(lambda: gr.update(visible=False), [], [send_btn]).then(
                lambda: gr.update(visible=False), [], [txt]).then( 
                bot.upload_audio, [audio_input, state, user_state], 
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

