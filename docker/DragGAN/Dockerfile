FROM continuumio/anaconda3

SHELL ["/bin/bash", "-c"]
RUN conda init bash
RUN source "/opt/conda/bin/activate"
RUN conda create -n draggan python=3.7
RUN source activate draggan
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
RUN pip install draggan==1.1.0b2

EXPOSE 7860

ENTRYPOINT ["python", "-m", "draggan.web", "--ip", "0.0.0.0"]
