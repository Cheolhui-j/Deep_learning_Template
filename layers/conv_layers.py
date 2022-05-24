from torch import nn

# --------------------------------------------
# 3x3 convolution layer
# --------------------------------------------
def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """ 3x3 convolution with padding = 1

    Input shape:
        (batch size, in_planes, height, width)
    Output shape:
        (batch size, out_planes, height, width)
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups,
                     bias=False)

# --------------------------------------------
# 1x1 convolution layer
# --------------------------------------------
def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """ 1x1 convolution

    Input shape:
        (batch size, in_planes, height, width)
    Output shape:
        (batch size, out_planes, height, width)
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, groups=groups,
                     bias=False)

# --------------------------------------------
# depthwise convolution layer
# --------------------------------------------
def convdw(in_planes, out_planes, stride=1, groups=1):
    """ depthwise convolution

    Input shape:
        (batch size, in_planes, height, width)
    Output shape:
        (batch size, out_planes, height, width)
    """
    return nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=groups,
                     bias=False)
