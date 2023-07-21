# coding: utf-8
import os
import numpy as np
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

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


api_base = os.environ.get('OPENAI_API_BASE', None)
if api_base is not None:
    openai.api_base = api_base

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
            body_background_fill_dark="#111111",
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
#image_upload {align-items: center; max-width: 640px}
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

def add_whiteboard():
    # wb = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
    wb = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    return Image.fromarray(wb)

def change_max_iter(max_iters):
    return gr.update(maximum=max_iters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=7862)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('--https', action='store_true')
    parser.add_argument('--load', type=str, default="HuskyVQA_cuda:0,ImageOCRRecognition_cuda:0,SegmentAnything_cuda:0")
    parser.add_argument('--tab', type=str, default="Audio,DragGAN,Image,Video")
    parser.add_argument('-e', '--e-mode', action='store_true')
    args = parser.parse_args()
    load_dict = {e.split('_')[0].strip(): e.split('_')[1].strip() for e in args.load.split(',')}
    bot = ConversationBot(load_dict=load_dict, e_mode=args.e_mode)
    with gr.Blocks(theme=Seafoam(), css=css) as demo:
        state = gr.State([])
        # user_state is dict. Keys: [agent, memory, image_path, video_path, seg_mask, image_caption, OCR_res, ...]
        user_state = gr.State([])

        gr.HTML(
            """
            <div align='center'> <img src='/file=./assets/gvlab_logo.png' style='height:70px'/> </div>
            <p align="center"><a href="https://github.com/OpenGVLab/InternGPT"><b>GitHub</b></a>
            &nbsp;&nbsp;&nbsp; <a href="https://arxiv.org/pdf/2305.05662.pdf"><b>Report</b></a>
            &nbsp;&nbsp;&nbsp; <a href="https://github.com/OpenGVLab/InternGPT/assets/13723743/8fd9112f-57d9-4871-a369-4e1929aa2593"><b>Video Demo</b></a>
            &nbsp;&nbsp;&nbsp; <a href="https://github.com/OpenGVLab/InternGPT/tree/main#imagebind_demo"><b>Video Demo with ImageBind</b></a>
            &nbsp;&nbsp;&nbsp; <a href="https://github.com/OpenGVLab/InternGPT/tree/main#draggan_demo"><b>Video Demo with DragGAN</b></a></p>
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
                    with gr.Column(min_width=0) as text_col:
                        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(
                            container=False)
                        audio2text_input = gr.Audio(source="microphone", type="filepath", visible=False)
                with gr.Row(visible=True) as input_btn:    
                    with gr.Column(scale=0.5, min_width=0):
                        audio_switch = gr.Checkbox(label="🎤 Voice Assistant", elem_id='audio_switch', info=None)
                    with gr.Column(scale=0.5, min_width=20):
                        send_btn = gr.Button("📤 Send", variant="primary", visible=True)
                    
            with gr.Column(elem_id="visual_input", scale=0.5) as img_part:
                if 'Audio' in args.tab:
                    with gr.Tab("Audio (with ImageBind)", elem_id='audio_tab') as audio_tab:
                        audio_input = gr.Audio(source="upload", type="filepath", visible=True, elem_id="audio_upload").style(height=360)
                        add_aud_example = gr.Button("📻 Audio Example", variant="primary")
                    
                    add_aud_example.click(random_audio, [], [audio_input,]).then( 
                        bot.upload_audio, [audio_input, state, user_state], 
                        [chatbot, state, user_state])
                    
                    audio_input.upload( 
                        bot.upload_audio, [audio_input, state, user_state], 
                        [chatbot, state, user_state])

                if 'DragGAN' in args.tab:
                    with gr.Tab("DragGAN", elem_id='drag_gan_tab') as drag_gan_tab:
                        drag_image = gr.Image(interactive=False).style(height=340)
                        with gr.Row(elem_id='drag_gan_btn') as drag_btn_row:   
                            with gr.Column(scale=0.33, min_width=0):
                                drag_new_img_btn = gr.Button('🖼️ New Image', variant='primary') 
                            with gr.Column(scale=0.33, min_width=0):
                                drag_btn = gr.Button('🖱︎ Drag It', variant='primary')
                            with gr.Column(scale=0.33, min_width=0):
                                drag_reset_btn = gr.Button('🧹 Clear Points', variant='primary')
                        with gr.Row(elem_id='drag_gan_progress'):   
                            with gr.Column(scale=0.5, min_width=0):
                                drag_max_iters = gr.Slider(1, 100, 25, step=1, label='Max Iterations', elem_id='drag_max_iters')
                            with gr.Column(scale=0.5, min_width=0):
                                progress = gr.Slider(value=0, maximum=25, label='Progress', interactive=False, elem_id='progress')
                    drag_new_img_btn.click(bot.gen_new_image, [state, user_state], [drag_image, chatbot, state, user_state])
                    drag_image.select(bot.save_points_for_drag_gan, [drag_image, user_state, ], [drag_image, user_state, ])
                    drag_btn.click(
                        bot.drag_it, [drag_image, drag_max_iters, state, user_state], [drag_image, progress, chatbot, state, user_state]
                    )
                    drag_max_iters.change(change_max_iter, [drag_max_iters,], [progress, ])
                    drag_reset_btn.click(bot.reset_drag_points, [drag_image, user_state], [drag_image, user_state, ])
                    drag_gan_tab.select(
                        bot.gen_new_image, [state, user_state], [drag_image, chatbot, state, user_state])


                if 'Image' in args.tab:
                    with gr.Tab("Image", elem_id='image_tab') as img_tab:
                        click_img = ImageSketcher(type="pil", interactive=True, brush_radius=15, elem_id="image_upload").style(height=360)
                        with gr.Row() as img_btn:
                            with gr.Column(scale=0.25, min_width=0):
                                process_seg_btn = gr.Button(value="👆 Pick", variant="primary", elem_id="process_seg_btn")
                            with gr.Column(scale=0.25, min_width=0):
                                process_ocr_btn = gr.Button(value="🔍 OCR", variant="primary", elem_id="process_ocr_btn")
                            with gr.Column(scale=0.25, min_width=0):
                                process_save_btn = gr.Button(value="📁 Save", variant="primary", elem_id="process_save_btn")
                            with gr.Column(scale=0.25, min_width=0):
                                clear_btn = gr.Button(value="🗑️ Clear All", elem_id="clear_btn")

                        with gr.Row(visible=True) as img_example:
                            with gr.Column(scale=0.5, min_width=0, visible=True) :
                                add_img_example = gr.Button("🖼️ Image Example", variant="primary")
                            with gr.Column(scale=0.5, min_width=0):
                                whiteboard_mode = gr.Button("⬜️ Whiteboard Mode", variant="primary", visible=True)
                        
                    add_img_example.click(random_image, [], [click_img,]).then( 
                                        bot.upload_image, [click_img, state, user_state], 
                                        [chatbot, state, user_state])
                    
                    whiteboard_mode.click(add_whiteboard, [], [click_img, ])

                    click_img.upload(lambda: gr.update(interactive=False), [], [send_btn]).then( 
                        bot.upload_image, [click_img, state, user_state], 
                        [chatbot, state, user_state]).then(
                        lambda: gr.update(interactive=True), [], [img_btn])
                    
                    process_ocr_btn.click(
                        lambda: gr.update(interactive=False), [], [img_btn]).then(
                        bot.process_ocr, [click_img, state, user_state], [click_img, chatbot, state, user_state]).then(
                        lambda: gr.update(interactive=True), [], [img_btn]
                    )
                    
                    process_seg_btn.click(
                        lambda: gr.update(interactive=False), [], [img_btn]).then(
                        bot.process_seg, [click_img, state, user_state], [click_img, chatbot, state, user_state]).then(
                        lambda: gr.update(interactive=True), [], [img_btn]
                    )
                    
                    process_save_btn.click(
                        lambda: gr.update(interactive=False), [], [img_btn]).then(
                        bot.process_save, [click_img, state, user_state], [click_img, chatbot, state, user_state]).then(
                        lambda: gr.update(interactive=True), [], [img_btn]
                    )
                    clear_func = partial(bot.clear_user_state, True)
                    clear_btn.click(lambda: None, [], [click_img, ]).then(
                        lambda: [], None, state).then(
                        clear_func, [user_state, ], [user_state, ]).then(
                        lambda: None, None, chatbot
                    ).then(lambda: '', None, [txt, ])

                if 'Video' in args.tab:
                    with gr.Tab("Video", elem_id='video_tab') as video_tab:
                        video_input = gr.Video(interactive=True, include_audio=True, elem_id="video_upload").style(height=360)
                        add_vid_example = gr.Button("📽 Video Example", variant="primary")

                    add_vid_example.click(random_video, [], [video_input,]).then( 
                    bot.upload_video, [video_input, state, user_state], 
                    [chatbot, state, user_state])

                    video_input.upload( 
                        bot.upload_video, [video_input, state, user_state], 
                        [chatbot, state, user_state])
                    clear_func = partial(bot.clear_user_state, False)
                    video_input.clear(clear_func, [user_state, ], [user_state, ])
            

            login_func = partial(login_with_key, bot, args.debug)
            openai_api_key_text.submit(login_func, [openai_api_key_text], [user_interface, openai_api_key_text, key_submit_button, user_state])
            key_submit_button.click(login_func, [openai_api_key_text, ], [user_interface, openai_api_key_text, key_submit_button, user_state])
            
            txt.submit(
                lambda: gr.update(interactive=False), [], [send_btn]).then(
                lambda: gr.update(interactive=False), [], [txt]).then(
                lambda: gr.update(interactive=False), [], [audio_switch]).then(
                bot.run_text, [txt, state, user_state], [chatbot, state, user_state]).then(
                    lambda: "", None, [txt, ]).then(
                lambda: gr.update(interactive=True), [], [txt]).then(
                    lambda: gr.update(interactive=True), [], [send_btn]
                ).then(
                    lambda: gr.update(interactive=True), [], [audio_switch]
                )

            send_btn.click(
                lambda: gr.update(interactive=False), [], [send_btn]).then(
                lambda: gr.update(interactive=False), [], [txt]).then(
                lambda: gr.update(interactive=False), [], [audio_switch]).then(
                bot.run_task, [audio_switch, txt, audio2text_input, state, user_state], [chatbot, state, user_state]).then(
                lambda: "", None, [txt, ]).then(
                lambda: gr.update(interactive=True), [], [send_btn]).then(
                    lambda: gr.update(interactive=True), [], [txt]
                ).then(
                    lambda: gr.update(interactive=True), [], [audio_switch]
                )
            
            audio_switch.change(change_input_type, [audio_switch, ], [txt, audio2text_input])
            
        gr.Markdown(
            '''
            **User Manual:**
    
            Update:

            (2023.05.24) We now support [DragGAN](https://github.com/Zeqiang-Lai/DragGAN). You can try it as follows:
            - Click the button `New Image`;
            - Click the image where blue denotes the start point and red denotes the end point;
            - Notice that the number of blue points is the same as the number of red points. Then you can click the button `Drag It`;
            - After processing, you will receive an edited image and a video that visualizes the editing process.

            <br>(2023.05.18) We now support [ImageBind](https://github.com/facebookresearch/ImageBind). If you want to generate a new image conditioned on audio, you can upload an audio file in advance:
            - To **generate a new image from a single audio file**, you can send the message like: `"generate a real image from this audio"`;
            - To **generate a new image from audio and text**, you can send the message like: `"generate a real image from this audio and {your prompt}"`;
            - To **generate a new image from audio and image**, you need to upload an image and then send the message like: `"generate a new image from above image and audio"`;

            <br>After uploading the image, you can have a **multi-modal dialogue** by sending messages like: `"what is it in the image?"` or `"what is the background color of the image?"`.

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
