FROM continuumio/anaconda3

RUN git clone https://github.com/OpenGVLab/InternGPT.git
WORKDIR InternGPT
SHELL ["/bin/bash", "-c"]
RUN conda init bash
RUN source "/opt/conda/bin/activate"
RUN conda create -n igpt python=3.8
RUN source activate igpt

RUN apt-get clean && apt-get update && apt-get -y install --no-install-recommends gcc g++ libjpeg-dev libpng-dev zlib1g-dev build-essential

RUN cp /opt/conda/envs/igpt/lib/libstdc++.so.6 /usr/lib/x86_64-linux-gnu
RUN conda install pytorch pytorch-cuda=11.7 cudatoolkit -c pytorch -c nvidia

RUN pip install -r requirements.txt
RUN pip install git+https://github.com/facebookresearch/detectron2.git
RUN pip install mediapipe

RUN pip uninstall -y opencv-python  
RUN pip uninstall -y opencv-python-headless
RUN pip uninstall -y opencv-contrib-python
RUN pip uninstall -y opencv-contrib-python-headless
RUN pip install opencv-python-headless==4.6.0.66
RUN pip install opencv-contrib-python==4.6.0.66

RUN apt-get -y install libgl1-mesa-glx
RUN apt-get -y install libxml2

RUN conda install -c conda-forge cudatoolkit-dev
RUN conda install --channel=numba llvmlite
RUN conda install -c numba numba

EXPOSE 7862

ENTRYPOINT ["python", "-u", "app.py", "--load", "StyleGAN_cuda:0", "--tab", "DragGAN", "--https"]
