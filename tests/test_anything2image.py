import sys
sys.path.append('../')
import torch
from iGPT.models import Audio2Image, Thermal2Image, AudioText2Image, AudioImage2Image


def test_audio2image():
    model = Audio2Image(torch.device('cuda'))
    model.inference('assets/audio/cat.wav')


def test_thermal2image():
    model = Thermal2Image(torch.device('cuda'))
    model.inference('assets/thermal/030444.jpg')
