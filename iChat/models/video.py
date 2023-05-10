import os
os.environ['CURL_CA_BUNDLE'] = ''

import torch
# from simplet5 import SimpleT5
import torchvision.transforms as transforms
import openai
import ffmpeg
from .tag2text import tag2text_caption
from .utils import *

from .load_internvideo import *

from .grit_model import DenseCaptioning
from .lang import SimpleLanguageModel
from scipy.io.wavfile import write as write_wav
from bark import SAMPLE_RATE, generate_audio


class VideoCaption:
    def __init__(self, device):
        self.device = device
        self.image_size = 384
        # self.threshold = 0.68
        self.video_path = None
        self.result = None
        self.tags = None
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((self.image_size,  self.image_size)), transforms.ToTensor(),self.normalize])
        self.model = tag2text_caption(pretrained="model_zoo/tag2text_swin_14m.pth", image_size=self.image_size, vit='swin_b').eval().to(device)
        self.load_video = LoadVideo()
        print("[INFO] initialize Caption model success!")

    def framewise_details(self, inputs):
        video_path = inputs.strip()
        caption = self.inference(video_path)
        frame_caption = ""
        prev_caption = ""
        start_time = 0
        end_time = 0
        for i, j in enumerate(caption):
            current_caption = f"{j}."
            current_dcs = f"{i+1}"
            if len(current_dcs) > 0:
                last_valid_dcs = current_dcs
            if current_caption == prev_caption:
                end_time = i+1
            else:
                if prev_caption:
                    frame_caption += f"Second {start_time} - {end_time}: {prev_caption}{last_valid_dcs}\n"
                start_time = i+1
                end_time = i+1
                prev_caption = current_caption
        if prev_caption:
            frame_caption += f"Second {start_time} - {end_time}: {prev_caption}{current_dcs}\n"
        total_dur = end_time
        frame_caption += f"| Total Duration: {total_dur} seconds.\n"

        print(frame_caption)
        # self.result = frame_caption
        self.video_path = video_path
        # video_prompt = f"""The tags for this vieo are: {prediction}, {','.join(tag_1)};
        # The temporal description of the video is: {frame_caption}
        # The dense caption of the video is: {dense_caption}
        # The general description of the video is: {synth_caption[0]}"""
        return frame_caption
    
    @prompts(name="Video Caption",
             description="useful when you want to generate a description for video. "
                         "like: generate a description or caption for this video. "
                         "The input to this tool should be a string, "
                         "representing the video_path")
    def inference(self, inputs):
        video_path = inputs.strip()
        data = self.load_video(video_path)
        # progress(0.2, desc="Loading Videos")
        tmp = []
        for _, img in enumerate(data):
            tmp.append(self.transform(img).to(self.device).unsqueeze(0))

        # Video Caption
        image = torch.cat(tmp).to(self.device)    
        # self.threshold = 0.68

        input_tag_list = None
        with torch.no_grad():
            caption, tags = self.model.generate(image,tag_input = input_tag_list, max_length = 50, return_tag_predict = True)
        # print(frame_caption, dense_caption, synth_caption)
        # print(caption)
        del data, image, tmp
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        self.result = caption
        self.tags = tags
        # return '. '.join(caption)
        return caption


class Summarization:
    def __init__(self, device):
        self.device = device
        self.model = SimpleT5()
        self.model.load_model(
                "t5", "./model_zoo/flan-t5-large-finetuned-openai-summarize_from_feedback", use_gpu=False)
        self.model.model = self.model.model.to(self.device)
        self.model.device = device
        
        print("[INFO] initialize Summarize model success!")

    @prompts(name="Video Summarization",
             description="useful when you want to Summarize video content for input video. "
                         "like: summarize this video. "
                         "The input to this tool should be a string, "
                         "representing the video_path")
    def inference(self, inputs):
        caption = inputs.strip()
        sum_res = self.model.predict(caption)
        return sum_res


class ActionRecognition:
    def __init__(self, device):
        self.device = device
        self.video_path = None
        # self.result = None
        self.model = load_intern_action(device)
        self.transform = transform_action()
        self.toPIL = T.ToPILImage()
        self.load_video = LoadVideo()
        print("[INFO] initialize InternVideo model success!")
    
    @prompts(name="Action Recognition",
             description="useful when you want to recognize the action category in this video. "
                         "like: recognize the action or classify this video"
                         "The input to this tool should be a string, "
                         "representing the video_path")
    def inference(self, inputs):
        video_path = inputs.strip()
        # if self.video_path == video_path:
        #     return self.result
        # self.video_path = video_path
        # data = loadvideo_decord_origin(video_path)
        data = self.load_video(video_path)

        # InternVideo
        action_index = np.linspace(0, len(data)-1, 8).astype(int)
        tmp_pred = []
        for i,img in enumerate(data):
            if i in action_index:
                tmp_pred.append(self.toPIL(img))
        action_tensor = self.transform(tmp_pred)
        TC, H, W = action_tensor.shape
        action_tensor = action_tensor.reshape(1, TC//3, 3, H, W).permute(0, 2, 1, 3, 4).to(self.device)
        with torch.no_grad():
            prediction = self.model(action_tensor)
            prediction = F.softmax(prediction, dim=1).flatten()
            prediction = kinetics_classnames[str(int(prediction.argmax()))]
        # self.result = prediction
        return prediction


class DenseCaption:
    def __init__(self, device):
        self.device = device
        self.model = DenseCaptioning(device)
        self.model.initialize_model()
        # self.model = self.model.to(device)
        self.load_video = LoadVideo()
        print("[INFO] initialize DenseCaptioe model success!")

    @prompts(name="Video Dense Caption",
             description="useful when you want to generate a dense caption for video. "
                         "like: generate a dense caption or description for this video. "
                         "The input to this tool should be a string, "
                         "representing the video_path")
    def inference(self, inputs):
        video_path = inputs.strip()
        # data = loadvideo_decord_origin(video_path)
        data = self.load_video(video_path)
        dense_caption = []
        dense_index = np.arange(0, len(data)-1, 5)
        original_images = data[dense_index,:,:,::-1]
        with torch.no_grad():
            for original_image in original_images:
                dense_caption.append(self.model.run_caption_tensor(original_image))
            dense_caption = ' '.join([f"Second {i+1} : {j}.\n" for i,j in zip(dense_index,dense_caption)])

        return dense_caption


class GenerateTikTokVideo:
    template_model = True
    def __init__(self, ActionRecognition, VideoCaption, DenseCaption):
        self.ActionRecognition = ActionRecognition
        self.VideoCaption = VideoCaption
        # self.Summarization = Summarization
        self.DenseCaption = DenseCaption
        self.SimpleLanguageModel = None

    @prompts(name="Generate TikTok Video",
             description="useful when you want to generate a video with TikTok style based on prompt."
                         "like: cut this video to a TikTok video based on prompt."
                         "The input to this tool should be a comma separated string of two, "
                         "representing the video_path and prompt")
    def inference(self, inputs):
        video_path = inputs.split(',')[0].strip()
        text = ', '.join(inputs.split(',')[1: ])
        if self.SimpleLanguageModel == None:
            self.SimpleLanguageModel = SimpleLanguageModel()
        action_classes = self.ActionRecognition.inference(video_path)
        print(f'action_classes = {action_classes}')
        dense_caption = self.DenseCaption.inference(video_path)
        print(f'dense_caption = {dense_caption}')
        caption = self.VideoCaption.inference(video_path)
        caption = '. '.join(caption)
        print(f'caption = {caption}')
        tags = self.VideoCaption.tags
        print(f'tags = {tags}')
        framewise_caption = self.VideoCaption.framewise_details(video_path)
        print(f'framewise_caption = {framewise_caption}')
        video_prompt = f"""The tags for this video are: {action_classes}, {','.join(tags)};
            The temporal description of the video is: {framewise_caption}
            The dense caption of the video is: {dense_caption}"""
        timestamp = self.run_text_with_time(video_prompt, text)
        print(f'timestamp = {timestamp}')
        if not timestamp:
            return 'Error! Please try it again.'
        start_time, end_time = min(timestamp), max(timestamp)
        print(f'start_time, end_time = = {start_time}, {end_time}')
        video_during = end_time - start_time + 1
        
        
        # prompt=f"忘记之前的回答模板，请使用中文回答这个问题。如果情节里遇到男生就叫小帅，女生就叫小美，请以’注意看，这个人叫’开始写一段的视频营销文案。尽量根据第{start_time}秒到第{end_time}秒左右的视频内容生成文案，不要生成重复句子。"
        # prompt=f"忘记之前的回答模板，请使用中文回答这个问题。如果情节里遇到男生就叫小帅，女生就叫小美，请以’注意看，这个人叫’为开头，根据第{start_time}秒到第{end_time}秒左右的视频内容生成一段视频营销文案。"
        prompt=f"忘记之前的回答模板，请使用中文回答这个问题。视频里如果出现男生就叫小帅，出现女生就叫小美，如果不确定性别，就叫大聪明。请以’注意看，这个人叫’为开头生成一段视频营销文案。"
        texts = self.run_text_with_tiktok(video_prompt, prompt).strip()
        # if texts.endswith('')
        texts += '。'
        print(f"before polishing: {texts}")
        print('*' * 40)
        # texts = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=[{"role":"user","content":f"请用润色下面的句子，去除重复的片段，但尽量保持原文内容且不许更改人物名字，并且以“注意看，这个人叫”作为开头：{texts}"}]).choices[0].message['content']
        texts = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=[{"role":"user","content":f"使用中文回答这个问题，请用润色下面的句子，去除重复的片段，并且仍以’注意看，这个人叫’为开头：{texts}"}]).choices[0].message['content']
        print(f"after polishing: {texts}")
        clipped_video_path = gen_new_name(video_path, 'tmp', 'mp4')
        wav_file = clipped_video_path.replace('.mp4', '.wav')
        audio_path = self.gen_audio(texts, wav_file)
        audio_duration = int(float(ffmpeg.probe(audio_path)['streams'][0]['duration']))+1
        os.system(f"ffmpeg -y -v quiet -ss {start_time} -t {video_during} -i {video_path} -c:v libx264 -c:a copy -movflags +faststart {clipped_video_path}")
        # output_path = self.image_filename.replace('.mp4','_tiktok.mp4')
        new_video_path = gen_new_name(video_path, 'GenerateTickTokVideo', 'mp4')
        if video_during < audio_duration:
            # 鬼畜hou
            # video_concat = os.path.join(os.path.dirname(clipped_video_path), 'concat.info')
            # video_concat = gen_new_name(clipped_video_path, '', 'info')
            video_concat = os.path.join(os.path.dirname(clipped_video_path), 'concat.info')
            video_concat = gen_new_name(video_concat, '', 'info')
            with open(video_concat,'w') as f:
                for _ in range(audio_duration//video_during+1):
                    f.write(f"file \'{os.path.basename(clipped_video_path)}\'\n")
            tmp_path = gen_new_name(video_path, 'tmp', 'mp4')
            os.system(f"ffmpeg  -y -f concat -i {video_concat} {tmp_path}")
            print(f"ffmpeg  -y -i {tmp_path} -i {wav_file} {new_video_path}")
            os.system(f"ffmpeg -y -i {tmp_path} -i {wav_file} {new_video_path}")
        else:
            print(f"ffmpeg  -y -i {clipped_video_path} -i {wav_file} {new_video_path}")
            os.system(f"ffmpeg -y -i {clipped_video_path} -i {wav_file} {new_video_path}")
        if not os.path.exists(new_video_path):
            import pdb
            pdb.set_trace()
        # state = state + [(text, f"Here is the video in *{new_file_path}*")] +[("show me the video.", (new_file_path,))]
        # print(f"\nProcessed run_video, Input video: {new_file_path}\nCurrent state: {state}\n"
        #       f"Current Memory: {self.agent.memory.buffer}")
        return (new_video_path, )
    
    def run_text_with_time(self, video_caption, text):
        # self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        prompt = "Only in this conversation, \
                  You must find the text-related start time \
                  and end time based on video caption. Your answer \
                  must end with the format {answer} [start time: end time]."
        response = self.SimpleLanguageModel(f"Video content: {video_caption}. Text: {text.strip()}." + prompt)
        # res['output'] = res['output'].replace("\\", "/")
        # print(response)
        import re
        pattern = r"\d+"
        # response = res['output']#rsplit(']')[-1] 
        try:
            # matches = re.findall(pattern, res['output'])
            matches = re.findall(pattern, response)
            start_idx , end_idx = matches[-2:]
            start_idx , end_idx = int(start_idx), int(end_idx)
        except:
            return None
            import pdb
            pdb.set_trace()
        # state = state + [(text, response)]
        print(f"\nProcessed run_text_with_time, Input text: {text}\n")
        return (start_idx, end_idx)

    def run_text_with_tiktok(self, video_content, prompt):
        # self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        inputs = f"Video description: {video_content}. {prompt}"
        
        response = self.SimpleLanguageModel(inputs)
        response = response.replace("\\", "/")
        # res = self.agent({"input":text})
        # res['output'] = res['output'].replace("\\", "/")
        # response = res['output']
        # state = state + [(prompt, response)]
        print(f"\nProcessed run_text_with_tiktok, Input text: {prompt}\n, Response: {response}")
        return response

    def gen_audio(self, text, save_path):
        audio_array = generate_audio(text)
        write_wav(save_path, SAMPLE_RATE, audio_array)
        return save_path


if __name__ == '__main__':
    # model = VideoCaption('cuda:0')
    # print(model.inference('./assets/f4236666.mp4'))
    # model = ActionRecognition('cuda:0')
    # print(model.inference('./assets/f4236666.mp4'))
    video_path = './tmp_files/f4236666.mp4'
    device = 'cuda:0'
    # caption_model = VideoCaption('cuda:0')
    # caption = caption_model.inference('./assets/f4236666.mp4')
    # sum_model = Summarize('cuda:0')
    # res = sum_model.inference(caption)
    # ds = DenseCaption(device)
    # res = ds.inference(video_path)
    from lang import SimpleLanguageModel
    model = GenerateTikTokVideo(ActionRecognition(device),
                                VideoCaption(device), 
                                DenseCaption(device)
            )
    out = model.inference(video_path+",帮我剪辑出最精彩的片段")
    print(out)