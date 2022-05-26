import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor

import sys

sys.path.append('../')
from config import cfg
from layers.conv_layers import conv3x3, conv1x1, convdw


# ====================================== Modified Stem Block =======================================

class ModifiedStemBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ModifiedStemBlock, self).__init__()
        self.stem1 = nn.Sequential(
            conv3x3(inplanes, planes, stride=1),
            nn.BatchNorm2d(planes),
            nn.PReLU(num_parameters=planes)
        )
        self.stem2a = nn.Sequential(
            conv1x1(planes, planes//2, stride=1),
            nn.BatchNorm2d(planes//2),
            nn.PReLU(num_parameters=planes//2)
        )
        self.stem2b = nn.Sequential(
            conv3x3(planes//2, planes, stride=2),
            nn.BatchNorm2d(planes),
            nn.PReLU(num_parameters=planes)
        )
        self.stem3 = nn.Sequential(
            conv1x1(2*planes, planes, stride=1),
            nn.BatchNorm2d(planes),
            nn.PReLU(num_parameters=planes)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.stem1(x)

        branch2 = self.stem2a(out)
        branch2 = self.stem2b(branch2)
        branch1 = self.pool(out)

        out = torch.cat([branch1, branch2], 1)
        out = self.stem3(out)

        return out

class ModifiedStemBlock2(nn.Module):
    def __init__(self, inplanes, planes):
        super(ModifiedStemBlock2, self).__init__()
        self.stem1 = nn.Sequential(
            conv3x3(inplanes, planes, stride=2),
            nn.BatchNorm2d(planes),
            nn.PReLU(num_parameters=planes)
        )
        self.stem2a = nn.Sequential(
            conv1x1(planes, planes//2, stride=1),
            nn.BatchNorm2d(planes//2),
            nn.PReLU(num_parameters=planes//2)
        )
        self.stem2b = nn.Sequential(
            conv3x3(planes//2, planes, stride=2),
            nn.BatchNorm2d(planes),
            nn.PReLU(num_parameters=planes)
        )
        self.stem3 = nn.Sequential(
            conv1x1(2*planes, planes, stride=1),
            nn.BatchNorm2d(planes),
            nn.PReLU(num_parameters=planes)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.stem1(x)

        branch2 = self.stem2a(out)
        branch2 = self.stem2b(branch2)
        branch1 = self.pool(out)

        out = torch.cat([branch1, branch2], 1)
        out = self.stem3(out)

        return out

# ==================================================================================================


##############################################################################

# LeNet

class Lenet(nn.Module):
    def __init__(self) -> None:
        super(Lenet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            # nn.AvgPool2d(kernel_size = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            # nn.AvgPool2d(kernel_size = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            # nn.AvgPool2d(kernel_size = 2),
            nn.ReLU(),
        )

    def forward(self, inp: Tensor) -> Tensor:
        feature = self.feature_extractor(inp).mean(dim=[2, 3])
        return nn.functional.normalize(feature)

##############################################################################

# ResNet18

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion * planes)
            )
        

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out

class ResNet18(nn.Module):

    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.in_planes = 16

        self.conv1 = conv3x3(1, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlock, 16, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 2, stride=2)
        self.linear = nn.Linear(64 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        # out = self.linear(out)

        return out

##############################################################################

# Mobile ResNet

def _make_divisible(value, divisor=8):
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value

def _round_filters(filters, width_mult):
    if width_mult == 1.0:
        return filters
    return int(_make_divisible(filters * width_mult))

# --------------------------------------------
# DepthwiseResidualBlock
# --------------------------------------------
class DepthwiseResidualBlock(nn.Module):
    """ Implement of DepthwiseResidualBlock

    This layer creates a DepthwiseResidualBlock.

    """
    def __init__(self, inplanes, planes, stride=1, expansion=1, downsample=None):
        # e.g. inplanes=64, planes=128, input_feature_size=28x28, expansion=4
        super(DepthwiseResidualBlock, self).__init__()
        self.hiddenplanes = expansion * inplanes # hiddenplanes=4*64=256

        # pw (64 x 28 x 28) --> (256 x 28 x 28)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv1x1(inplanes, self.hiddenplanes) # channel expansion

        # dw (256 x 28 x 28) --> (256 x 14 x 14)
        self.bn2 = nn.BatchNorm2d(self.hiddenplanes)
        self.prelu1 = nn.PReLU(num_parameters=self.hiddenplanes)
        self.conv2 = convdw(self.hiddenplanes, self.hiddenplanes, stride=stride, groups=self.hiddenplanes)

        # pw-linear (256 x 14 x 14) --> (128 x 14 x 14)
        self.bn3 = nn.BatchNorm2d(self.hiddenplanes)
        self.prelu2 = nn.PReLU(num_parameters=self.hiddenplanes)
        self.conv3 = conv1x1(self.hiddenplanes, planes) # channel reduction

        self.bn4 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # pw: channel expansion
        out = self.bn1(x)
        out = self.conv1(out)

        # dw
        out = self.bn2(out)
        out = self.prelu1(out)
        out = self.conv2(out)

        # pw-linear: channel reduction
        out = self.bn3(out)
        out = self.prelu2(out)
        out = self.conv3(out)

        out = self.bn4(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

# --------------------------------------------
# Mobile Residual network (MobileResNet)
# --------------------------------------------
class fMobileResNet(nn.Module):
    """ Implement of Mobile Residual network (MobileResNet):

    This layer creates a MobileResNet model by stacking DepthwiseResidualBlocks.

    Args:
        block: block to stack in each layer - DepthwiseResidualBlock
        layers: # of stacked blocks in each layer
    """
    def __init__(self, block, layers, conf):
        # options from configuration
        self.input_opt = conf.input_opt
        self.width = conf.width
        self.expansion = conf.expansion
        super(fMobileResNet, self).__init__()
        # -------------
        # Head block
        # -------------
        self.inplanes = 64
        if self.input_opt == 'L' or self.input_opt == 'S':
            self.HeadBlock = nn.Sequential(
                conv3x3(3, self.inplanes, stride=1),
                nn.BatchNorm2d(self.inplanes),
                nn.PReLU(num_parameters=self.inplanes),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        elif self.input_opt == 'Stem-S':
            self.HeadBlock = ModifiedStemBlock(3, self.inplanes)
        elif self.input_opt == 'Stem-L':
            self.HeadBlock = ModifiedStemBlock2(3, self.inplanes)
        else:
            print("Input option error !")

        # -------------
        # Residual blocks
        # -------------
        # Layer 1: channel: 64 -> 64
        self.layer1 = self.stack_layers(block, _round_filters(64, conf.width[0]), layers[0], expansion=self.expansion)
        # Layer 2: channel: 64 -> 128
        self.layer2 = self.stack_layers(block, _round_filters(128, conf.width[1]), layers[1], stride=2, expansion=self.expansion)
        # Layer 3: channel: 128 -> 256
        self.layer3 = self.stack_layers(block, _round_filters(256, conf.width[2]), layers[2], stride=2, expansion=self.expansion)
        # Layer 4: channel: 256 -> 512
        if self.input_opt == 'L':
            self.layer4 = self.stack_layers(block, _round_filters(512, conf.width[3]), layers[3], stride=2, expansion=self.expansion)
        elif self.input_opt == 'S' or self.input_opt == 'Stem-S' or self.input_opt == 'Stem-L':
            self.layer4 = self.stack_layers(block, _round_filters(512, conf.width[3]), layers[3], stride=1, expansion=self.expansion)

        # -------------
        # Classifier
        # -------------
        self.last_channel = _round_filters(512, conf.width[3])
        self.bn2 = nn.BatchNorm2d(self.last_channel)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(self.last_channel * 7 * 7, 512)
        self.bn3 = nn.BatchNorm1d(512)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    # Stack residual blocks
    def stack_layers(self, block, planes, blocks, stride=1, expansion=1):
        downsample = None
        # Peforms downsample if stride != 1 or inplanes != planes * block.expansion:
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        # For the first residual block, stride is 1 or 2
        layers.append(block(self.inplanes, planes, stride, expansion, downsample))
        # From the second residual block, stride is 1
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.HeadBlock(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn3(x)

        return x

# --------------------------------------------
# Define MobileResNet models
# --------------------------------------------
def MobileResNet34(conf, **kwargs):
    """ Constructs a  MobileResNet34 model
    Args:
        conf: configurations
    """
    ResidualBlock = DepthwiseResidualBlock

    model = fMobileResNet(ResidualBlock, [3, 4, 6, 3], conf, **kwargs)

    if conf.pretrained:
        model.load_state_dict(torch.load(conf.pretrained_model), strict=False)

    return model

def MobileResNet50(conf, **kwargs):
    """ Constructs a  MobileResNet50 model
    Args:
        conf: configurations
    """
    ResidualBlock = DepthwiseResidualBlock

    model = fMobileResNet(ResidualBlock, [3, 4, 14, 3], conf, **kwargs)

    if conf.pretrained:
        model.load_state_dict(torch.load(conf.pretrained_model), strict=False)

    return model

def MobileResNet100(conf, **kwargs):
    """ Constructs a  MobileResNet100 model
    Args:
        conf: configurations
    """
    ResidualBlock = DepthwiseResidualBlock

    model = fMobileResNet(ResidualBlock, [3, 13, 30, 3], conf, **kwargs)

    if conf.pretrained:
        model.load_state_dict(torch.load(conf.pretrained_model), strict=False)

    return model



if __name__ == "__main__":

    # 나중엔 config에 정의된 모델별로 load할 수 있도록 수정 바람.

    model = MobileResNet100(cfg)

    print(model)