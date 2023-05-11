[[English Document](https://github.com/OpenGVLab/InternChat/blob/main/README.md)]

**[NOTE] 该项目仍在建设中，我们将继续更新，并欢迎社区的贡献/拉取请求。**

<p align="center"><img src="./assets/gvlab_logo.png" width="600"></p>

<a src="https://img.shields.io/discord/1099920215724277770?label=Discord&logo=discord" href="https://discord.gg/khWBFnCgAN">
<img src="https://img.shields.io/discord/1099920215724277770?label=Discord&logo=discord"> </a> | <a src="https://img.shields.io/badge/GPU Demo-Open-green?logo=alibabacloud" href="https://ichat.opengvlab.com">
<img src="https://img.shields.io/badge/Demo-Open-green?logo=alibabacloud"> </a> | <a src="https://img.shields.io/twitter/follow/opengvlab?style=social" href="https://twitter.com/opengvlab">
<img src="https://img.shields.io/twitter/follow/opengvlab?style=social">

# InternChat [[论文](https://arxiv.org/pdf/2305.05662.pdf)]
<!-- ## 描述 -->
**InternChat**（简称 **iChat**）是一种基于指向语言驱动的视觉交互系统，允许您使用指向设备通过点击、拖动和绘制与 ChatGPT 进行互动。InternChat 的名称代表了 **inter**action（交互）、**n**onverbal（非语言）和 **chat**bots（聊天机器人）。与依赖纯语言的现有交互系统不同，通过整合指向指令，iChat 显著提高了用户与聊天机器人之间的沟通效率，以及聊天机器人在视觉为中心任务中的准确性，特别是在复杂的视觉场景中。此外，在 iChat 中，采用辅助控制机制来提高 LLM 的控制能力，并对一个大型视觉-语言模型 **Husky** 进行微调，以实现高质量的多模态对话（在ChatGPT-3.5-turbo评测中达到 **93.89% GPT-4 质量**）。
  
## 在线Demo
  
[注意] 可能会出现排队等待较长时间。您可以clone我们的仓库并使用您自己的GPU运行。
  
[**InternChat**已上线，尝试一下！](https://ichat.opengvlab.com)


https://github.com/OpenGVLab/InternChat/assets/13723743/3270b05f-0823-4f13-9966-4010fd855643
  
## Schedule
- [ ] 支持中文
- [ ] 支持 MOSS
- [ ] 基于 InternImage 和 InternVideo 的更强大的基础模型
- [ ] 更准确的交互体验
- [ ] 网页 & 代码生成
- [x] 支持语音助手
- [x] 支持点击交互
- [x] 交互式图像编辑
- [x] 交互式图像生成
- [x] 交互式视觉问答
- [x] Segment Anything模型
- [x] 图像修复
- [x] 图像描述
- [x] 图像抠图
- [x] 光学字符识别（OCR）
- [x] 动作识别
- [x] 视频描述
- [x] 视频密集描述
- [x] 视频高光时刻截取
  
## 系统概览
<p align="center"><img src="./assets/arch1.png" alt="Logo"></p>
  
## 🎁 主要特点
<!--<!-- <p align="center"><img src="./assets/online_demo.gif" alt="Logo"></p> -->

<p align="center">(a) 移除遮盖的对象</p>
<p align="center"><img src="./assets/demo2.gif" width="500"></p>

<p align="center">(b) 交互式图像编辑</center>
<p align="center"><img src="./assets/demo3.gif" width="500"></p>

<p align="center">(c) 图像生成</p>
<p align="center"><img src="./assets/demo4.gif" align='justify'  width="500"></p>

<p align="center">(d) 交互式视觉问答</p>
<p align="center"><img src="./assets/demo5.gif" align='justify' width="700"></p>

<p align="center">(e) 交互式图像生成（你需要点击"Save"按钮保存你的笔画）</p>
<p align="center"><img width="800" alt="image" src="https://github.com/OpenGVLab/InternChat/assets/8529570/2b0da08e-af86-453d-99e5-1327f93aa917"></p>

<p align="center">(f) 视频高光解释</p>
<p align="center"><img src="./assets/demo6.jpg" align='justify' width="500"></p> 

<!-- ![alt]("./assets/demo5.gif" "title") -->

## 🛠️ 安装

### 基本要求

- Linux
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.6+
- GCC & G++ 5.4+
- GPU 内存 >= 17G 用于加载基本工具 (HuskyVQA, SegmentAnything, ImageOCRRecognition)

### 安装Python的依赖项
```shell
pip install -r requirements.txt
```

### 模型库

即将推出...

## 👨‍🏫 运行指南

运行以下 shell 可启动一个 gradio 服务：

```shell
python -u iChatApp.py --load "HuskyVQA_cuda:0,SegmentAnything_cuda:0,ImageOCRRecognition_cuda:0" --port 3456
```

如果您想启用语音助手，请使用 openssl 生成证书：

```shell
openssl req -x509 -newkey rsa:4096 -keyout ./key.pem -out ./cert.pem -sha256 -days 365 -nodes
```
然后运行：

```shell
python -u iChatApp.py --load "HuskyVQA_cuda:0,SegmentAnything_cuda:0,ImageOCRRecognition_cuda:0" --port 3456 --https
```


## 🎫 许可

该项目根据[Apache 2.0 license](LICENSE)发布。

## 🖊️ 引用

如果您在研究中发现这个项目有用，请考虑引用：
```BibTeX
@misc{2023internchat,
    title={InternChat: Solving Vision-Centric Tasks by Interacting with Chatbots Beyond Language},
    author={Zhaoyang Liu and Yinan He and Wenhai Wang and Weiyun Wang and Yi Wang and Shoufa Chen and Qinglong Zhang and Yang Yang and Qingyun Li and Jiashuo Yu and Kunchang Li and Zhe Chen and Xue Yang and Xizhou Zhu and Yali Wang and Limin Wang and Ping Luo and Jifeng Dai and Yu Qiao},
    howpublished = {\url{https://arxiv.org/abs/2305.05662}},
    year={2023}
}
```

## 🤝 致谢

感谢以下开源项目:

[Hugging Face](https://github.com/huggingface) &#8194;
[LangChain](https://github.com/hwchase17/langchain) &#8194;
[TaskMatrix](https://github.com/microsoft/TaskMatrix) &#8194;
[SAM](https://github.com/facebookresearch/segment-anything) &#8194;
[Stable Diffusion](https://github.com/CompVis/stable-diffusion) &#8194; 
[ControlNet](https://github.com/lllyasviel/ControlNet) &#8194; 
[InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix) &#8194; 
[BLIP](https://github.com/salesforce/BLIP) &#8194;
[Latent Diffusion Models](https://github.com/CompVis/latent-diffusion) &#8194;
[EasyOCR](https://github.com/JaidedAI/EasyOCR) &#8194;



如果您在试用、运行、部署中有任何问题，欢迎加入我们的微信群讨论！如果您对项目有任何的想法和建议，欢迎加入我们的微信群讨论！

<p align="center"><img width="500" alt="image" src="https://s1.ax1x.com/2023/05/11/p9rTzMq.jpg"></p> 

一群快满了，可以添加2群
<p align="center"><img width="500" alt="image" src="https://github.com/OpenGVLab/InternChat/assets/8529570/881c231d-9049-4920-a22c-680f41f0f7ee"></p> 
