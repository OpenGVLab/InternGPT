FROM continuumio/anaconda3

COPY . /application/igpt/
WORKDIR /application/igpt/

#RUN apt-get update && apt-get -y install --no-install-recommends gcc g++ libjpeg-dev libpng-dev zlib1g-dev build-essential
RUN apt-get clean && \
    apt-get update && \
    apt-get -y install --no-install-recommends gcc g++ libjpeg-dev libpng-dev zlib1g-dev build-essential && \
    apt-get -y install libgl1-mesa-glx && \
    apt-get -y install libxml2 && \
    apt-get clean

RUN conda create -n igpt python=3.8 -y
RUN echo "coda activate igpt" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN cp /opt/conda/envs/igpt/lib/libstdc++.so.6 /usr/lib/x86_64-linux-gnu
RUN conda activate igpt && \
    conda install pytorch=1.13.0  torchvision=0.14.0 torchaudio=0.13.0 pytorch-cuda=11.7 cudatoolkit -c pytorch -c nvidia -y && \
    pip install -r requirements.txt && \
    pip install git+https://github.com/facebookresearch/detectron2.git && \
    pip install mediapipe && \
    pip install imageio-ffmpeg && \
    pip uninstall -y opencv-python && \
    pip uninstall -y opencv-python-headless && \
    pip uninstall -y opencv-contrib-python && \
    pip uninstall -y opencv-contrib-python-headless && \
    pip install opencv-python-headless==4.6.0.66 && \
    pip install opencv-contrib-python==4.6.0.66 && \
    conda install -c conda-forge cudatoolkit-dev -y && \
    conda install --channel=numba llvmlite -y && \
    conda install -c numba numba -y && \
    pip install ipdb

EXPOSE 7862

ENTRYPOINT ["/application/igpt/entrypoint.sh"]
