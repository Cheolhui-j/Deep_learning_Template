from torch import nn

def conv3x3(in_plain, out_plain, stride = 1):
    return nn.Conv2d(in_plain, out_plain, 3, stride=stride, padding=1, bias=False)