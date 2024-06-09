#!/bin/bash --login
conda activate igpt
exec python -u app.py --load StyleGAN_cuda:0 --tab DragGAN
