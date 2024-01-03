from collections import OrderedDict
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_planes, planes):
        super(Block, self).__init__()
        self.in_planes = in_planes
        self.planes = planes

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)

        self.res_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_planes, planes, kernel_size=1, padding=0)),
            ('bn', nn.BatchNorm2d(planes)),
        ]))

        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn(out)

        x = self.res_conv(x)
        out = self.pool(self.relu(out + x))
        return out


class ResNetProto12(nn.Module):
  def __init__(self, args, channels):
    super(ResNetProto12, self).__init__()
    self.channels = channels

    self.layer1 = Block(args.imgc, channels[0])
    self.layer2 = Block(channels[0], channels[1])
    self.layer3 = Block(channels[1], channels[2])
    self.layer4 = Block(channels[2], channels[3])
    
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.out_dim = channels[3]
    

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(
          m.weight, mode='fan_out', nonlinearity='leaky_relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.)
        nn.init.constant_(m.bias, 0.)

  def get_out_dim(self):
    return self.out_dim

  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.pool(out).flatten(1)
    return out


def resnetproto12(args):
  return ResNetProto12(args, [64, 128, 256, 512])