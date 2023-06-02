FROM continuumio/anaconda3

RUN git clone https://github.com/Zeqiang-Lai/DragGAN.git
WORKDIR DragGAN

SHELL ["/bin/bash", "-c"]
RUN conda init bash
RUN source "/opt/conda/bin/activate"
RUN conda create -n draggan python=3.7
RUN source activate draggan
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

EXPOSE 7860

ENTRYPOINT ["python", "gradio_app.py", "--ip", "0.0.0.0"]
