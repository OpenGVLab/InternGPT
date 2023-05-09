<p align="center"><img src="./assets/gvlab_logo.png" width="600"></p>

# InternChat

<!-- ## Description -->
**InternChat** (iChat) is an interactive visual framework. This framework integrates chatbots that have planning and reasoning capabilities, such as ChatGPT, with non-verbal instructions like pointing movements that enable users to directly manipulate images or videos on the screen. Pointing (including gestures, cursors, etc.) movements can provide more flexibility and precision in performing vision-centric tasks that require fine-grained control, editing, and generation of visual content. The name InternChat stands for **inter**action, **n**onverbal, and chatbots. Different from existing interactive systems that rely on pure language, by incorporating pointing instructions, the proposed iChat significantly improves the efficiency of communication between users and chatbots, as well as the accuracy of chatbots in vision-centric tasks, especially in complicated visual scenarios (i.e., the number of objects larger than 2). We hope this work can spark new ideas and directions for future interactive visual systems. 

## Paper
[InternChat](https://www.shlab.org.cn)


## Schedule
- [ ] Support Chinese
- [ ] More powerful foundation models in [InternImage](https://github.com/OpenGVLab/InternImage) and [InternVideo](https://github.com/OpenGVLab/InternVideo).
- [ ] More accurate interactive experience
- [ ] Web Page & Code Generation 
- [x] Support voice assistant
- [x] Support click interaction
- [x] Interactive image editing
- [x] Interactive image generation
- [x] Interactive visual question answering
- [x] Segment Anything
- [x] Image inpainting
- [x] Image caption
- [x] image matting
- [x] Optical character recognition
- [x] Action recognition
- [x] Video caption
- [x] Video dense caption
- [x] video highlight interpretation

## Online Demo
[InternChat](https://ichat.opengvlab.com/) is online. Let's try it!
<!-- <video id="video" controls="" poster="" ><source id="video" src="./assets/online_demo.mp4" type="video/mp4"></videos> -->

## System Overview
<p align="center"><img src="./assets/arch1.png" alt="Logo"></p>

## 🎁 Major Features
<!--<!-- <p align="center"><img src="./assets/online_demo.gif" alt="Logo"></p> -->  
<p align="center">A) Remove the masked object</p>
<p align="center"><img src="./assets/demo2.gif" width="500"></p>

<p align="center">B) Interactive image editing</center>
<p align="center"><img src="./assets/demo3.gif" width="500"></p>

<p align="center">C) Image generation</p>
<p align="center"><img src="./assets/demo4.gif" align='justify'  width="500"></p>

<p align="center">D) Interactive visual question answer</p>
<p align="center"><img src="./assets/demo5.gif" align='justify' width="700"></p> 

<p align="center">E) Video highlight interpretation</p>
<p align="center"><img src="./assets/demo6.jpg" align='justify' width="500"></p> 



<!-- ![alt]("./assets/demo5.gif" "title") -->


## 🛠️ Installation

### Basic requirements

- Linux
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.6+ 
- GCC & G++ 5.4+
- GPU Memory >= 17G for loading basic tools (HuskyVQA, SegmentAnything, ImageOCRRecognition)

### Install Python dependencies
```shell
pip install -r requirements.txt
```

### Model zoo
Coming soon...

## 👨‍🏫 Get Started 
Running the following shell can start a gradio service:
```shell
python -u IChatApp.py --load "HuskyVQA_cuda:0,SegmentAnything_cuda:0,ImageOCRRecognition_cuda:0" --port 3456
```

if you want to enable the voice assistant, please use `openssl` to generate the certificate:
```shell
openssl req -x509 -newkey rsa:4096 -keyout ./key.pem -out ./cert.pem -sha256 -days 365 -nodes
```

and then run:
```shell
python -u IChatApp.py --load "HuskyVQA_cuda:0,SegmentAnything_cuda:0,ImageOCRRecognition_cuda:0" --port 3456 --https
```
 



## 🎫 License

This project is released under the [Apache 2.0 license](LICENSE). 

## 🖊️ Citation

If you find this project useful in your research, please consider cite:
```BibTeX
@misc{2020mmaction2,
    title={InternChat: Solving Vision-Centric Tasks by Interacting with Chatbots Beyond Language},
    author={Zhaoyang Liu, Yinan He, Wenhai Wang, Weiyun Wang, Yi Wang, Shoufa Chen, Qinglong Zhang, Yang Yang, Qingyun Li, Jiashuo Yu, Kunchang Li, Zhe Chen, Xue Yang, Xizhou Zhu, Yali Wang, Limin Wang, Ping Luo, Jifeng Dai, Yu Qiao},
    howpublished = {\url{https://www.shlab.org.cn}},
    year={2023}
}
```

## 🤝 Acknowledgement
Thanks to the open source of the following projects:

[Hugging Face](https://github.com/huggingface) &#8194;
[LangChain](https://github.com/hwchase17/langchain) &#8194;
[TaskMatrix](https://github.com/microsoft/TaskMatrix) &#8194;
[SAM](https://github.com/facebookresearch/segment-anything) &#8194;
[Stable Diffusion](https://github.com/CompVis/stable-diffusion) &#8194; 
[ControlNet](https://github.com/lllyasviel/ControlNet) &#8194; 
[InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix) &#8194; 
[BLIP](https://github.com/salesforce/BLIP) &#8194;
[Latent Diffusion Models](https://github.com/CompVis/latent-diffusion) &#8194;
[Latent Diffusion Models](https://github.com/CompVis/latent-diffusion) &#8194;
[EasyOCR](https://github.com/JaidedAI/EasyOCR) &#8194;


