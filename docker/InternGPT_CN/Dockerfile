FROM  nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
    sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list

RUN apt-get clean && apt-get update && apt install -y  \
          git libass-dev cmake  libsndfile1-dev tesseract-ocr espeak-ng python3-dev python3-pip ffmpeg  \
          ninja-build ca-certificates
 
RUN ln -sv /usr/bin/python3 /usr/bin/python

RUN python3 -m pip install --no-cache-dir --upgrade pip&& pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install --no-cache-dir torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

WORKDIR /root

RUN git clone https://github.com/OpenGVLab/InternGPT.git

WORKDIR InternGPT

RUN pip  install --no-cache-dir -r requirements.txt

WORKDIR /root

RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo && \
    pip install --no-cache-dir -e detectron2_repo

RUN pip install --no-cache-dir imageio-ffmpeg

ENV OPENAI_API_BASE=https://api.openai-proxy.com/v1

RUN rm -rf InternGPT

VOLUME InternGPT

WORKDIR InternGPT

EXPOSE 7862
EXPOSE 3456
EXPOSE 8080

ENTRYPOINT ["python", "-u", "app.py", "--port","7862","--https"]
CMD ["--load","StyleGAN_cuda:0", "--tab", "DragGAN"]