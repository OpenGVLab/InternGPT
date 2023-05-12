[[English Document](https://github.com/OpenGVLab/internGPT/blob/main/README.md)]

**[NOTE] 该项目仍在建设中，我们将继续更新，并欢迎社区的贡献/拉取请求。**

<p align="center"><img src="./assets/gvlab_logo.png" width="600"></p>

<a src="https://img.shields.io/discord/1099920215724277770?label=Discord&logo=discord" href="https://discord.gg/khWBFnCgAN">
<img src="https://img.shields.io/discord/1099920215724277770?label=Discord&logo=discord"> </a> | <a src="https://img.shields.io/badge/GPU Demo-Open-green?logo=alibabacloud" href="https://ichat.opengvlab.com">
<img src="https://img.shields.io/badge/Demo-Open-green?logo=alibabacloud"> </a> | <a src="https://img.shields.io/twitter/follow/opengvlab?style=social" href="https://twitter.com/opengvlab">
<img src="https://img.shields.io/twitter/follow/opengvlab?style=social">

# InternGPT [[论文](https://arxiv.org/pdf/2305.05662.pdf)][[试玩](https://igpt.opengvlab.com)]
<!-- ## 描述 -->
**InternGPT**（简称 **iGPT**） / **InternChat**（简称 **iChat**） 是一种基于指向语言驱动的视觉交互系统，允许您使用指向设备通过点击、拖动和绘制与 ChatGPT 进行互动。internGPT 的名称代表了 **inter**action（交互）、**n**onverbal（非语言）和 Chat**GPT**。与依赖纯语言的现有交互系统不同，通过整合指向指令，iGPT 显著提高了用户与聊天机器人之间的沟通效率，以及聊天机器人在视觉为中心任务中的准确性，特别是在复杂的视觉场景中。此外，在 iGPT 中，采用辅助控制机制来提高 LLM 的控制能力，并对一个大型视觉-语言模型 **Husky** 进行微调，以实现高质量的多模态对话（在ChatGPT-3.5-turbo评测中达到 **93.89% GPT-4 质量**）。
  
## 在线Demo
  
[注意] 可能会出现排队等待较长时间。您可以clone我们的仓库并使用您自己的GPU运行。
  
[**InternGPT** 已上线，尝试一下！](https://igpt.opengvlab.com)


视频demo：https://github.com/OpenGVLab/InternGPT/assets/8529570/a02bcea5-6d1f-4e84-85a3-8a66239b8a51

## 项目规划
<details>
<summary>点击查看详情</summary>

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
</details>

  
## 系统概览
<p align="center"><img src="./assets/arch1.png" alt="Logo"></p>
  
## 🎁 主要功能
<!--<!-- <p align="center"><img src="./assets/online_demo.gif" alt="Logo"></p> -->

<details>
<summary>A) 移除遮盖的对象</summary>
<p align="center"><img src="./assets/demo2.gif" width="500"></p>
</details>

<details>
<summary>B) 交互式图像编辑</summary>
<p align="center"><img src="./assets/tower.gif" width="500"></p>
</details>

<details>
<summary>C) 图像生成</summary>
<p align="center"><img src="./assets/demo4.gif" width="500"></p>
</details>

<details>
<summary>D) 交互式视觉问答</summary>
<p align="center"><img src="./assets/demo5.gif" width="500"></p>
</details>

<details>
<summary>E) 交互式图像生成</summary>
<p align="center"><img src="https://github.com/OpenGVLab/InternGPT/assets/8529570/2b0da08e-af86-453d-99e5-1327f93aa917" width="500"></p>
</details>

<details>
<summary>F) 视频高光解说</summary>
<p align="center"><img src="./assets/demo6.jpg" width="500"></p>
</details>

<!-- ![alt]("./assets/demo5.gif" "title") -->

## 🛠️ 安装
<details>
<summary>基本要求</summary>

- Linux 
- Python 3.8+ 
- PyTorch 1.12+
- CUDA 11.6+ 
- GCC & G++ 5.4+
- GPU Memory > 17G 用于加载基本工具 (HuskyVQA, SegmentAnything, ImageOCRRecognition)
</details>

<details>
<summary>安装Python的依赖项</summary>

```shell
pip install -r requirements.txt
```
</details>


### 模型库

即将推出...

## 👨‍🏫 运行指南


<details>
<summary>运行以下 shell 可启动一个 gradio 服务（点击即可展开）：</summary>

```shell
python -u app.py --load "HuskyVQA_cuda:0,SegmentAnything_cuda:0,ImageOCRRecognition_cuda:0" --port 3456
```
如果您想启用语音助手，请使用 openssl 生成证书：

```shell
mkdir certificate
openssl req -x509 -newkey rsa:4096 -keyout certificate/key.pem -out certificate/cert.pem -sha256 -days 365 -nodes
```
然后运行：

```shell
python -u app.py --load "HuskyVQA_cuda:0,SegmentAnything_cuda:0,ImageOCRRecognition_cuda:0" --port 3456 --https
```

</details>


## 🎫 许可

该项目根据[Apache 2.0 license](LICENSE)发布。

## 🖊️ 引用

如果您在研究中发现这个项目有用，请考虑引用我们的论文：
```BibTeX
@misc{2023interngpt,
    title={InternGPT: Solving Vision-Centric Tasks by Interacting with ChatGPT Beyond Language},
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

<details>
<summary>加入微信群组二维码 (点击展开)：</summary>

<p align="center"><img width="500" alt="image" src="https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/papers/media/wechat_group.jpg"></p> 
