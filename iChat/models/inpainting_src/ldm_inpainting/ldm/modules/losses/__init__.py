import torch.nn as nn

class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()