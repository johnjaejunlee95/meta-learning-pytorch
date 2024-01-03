from collections import OrderedDict

import torch.nn as nn

# from . import register
from .modules import *


class Block(nn.Module):
    def __init__(self, in_planes, planes, bn_args):
        super(Block, self).__init__()
        self.in_planes = in_planes
        self.planes = planes

        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, padding=1)
        self.bn1 = BatchNorm2d(planes, **bn_args)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2d(planes, **bn_args)
        self.conv3 = Conv2d(planes, planes, kernel_size=3, padding=1)
        self.bn3 = BatchNorm2d(planes, **bn_args)

        self.res_conv = Sequential(OrderedDict([
            ('conv', Conv2d(in_planes, planes, kernel_size=1, padding=0)),
            ('bn', BatchNorm2d(planes, **bn_args)),
        ]))

        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, params=None, episode=None):
      out = self.conv1(x, get_child_dict(params, 'conv1'))
      out = self.bn1(out, get_child_dict(params, 'bn1'), episode)
      out = self.relu(out)

      out = self.conv2(out, get_child_dict(params, 'conv2'))
      out = self.bn2(out, get_child_dict(params, 'bn2'), episode)
      out = self.relu(out)

      out = self.conv3(out, get_child_dict(params, 'conv3'))
      out = self.bn3(out, get_child_dict(params, 'bn3'), episode)

      x = self.res_conv(x, get_child_dict(params, 'res_conv'), episode)
      out = self.pool(self.relu(out + x))
      return out


class ResNet12(nn.Module):
  def __init__(self, args, channels, bn_args):
    super(ResNet12, self).__init__()

    episodic = bn_args.get('episodic') or []
    bn_args_ep, bn_args_no_ep = bn_args.copy(), bn_args.copy()
    bn_args_ep['episodic'] = True
    bn_args_no_ep['episodic'] = False
    bn_args_dict = dict()
    for i in [1, 2, 3, 4]:
      if 'layer%d' % i in episodic:
        bn_args_dict[i] = bn_args_ep
      else:
        bn_args_dict[i] = bn_args_no_ep

    self.layer1 = Block(args.imgc, channels[0], bn_args_dict[1])
    self.layer2 = Block(channels[0], channels[1], bn_args_dict[2])
    self.layer3 = Block(channels[1], channels[2], bn_args_dict[3])
    self.layer4 = Block(channels[2], channels[3], bn_args_dict[4])
    
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.out_dim = channels[3]
    
    self.linear = Linear(self.out_dim, args.num_ways)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(
          m.weight, mode='fan_out', nonlinearity='leaky_relu')
      elif isinstance(m, BatchNorm2d):
        nn.init.constant_(m.weight, 1.)
        nn.init.constant_(m.bias, 0.)

  def get_out_dim(self):
    return self.out_dim

  def forward(self, x, params=None, episode=None):
    out = self.layer1(x, get_child_dict(params, 'layer1'), episode)
    out = self.layer2(out, get_child_dict(params, 'layer2'), episode)
    out = self.layer3(out, get_child_dict(params, 'layer3'), episode)
    out = self.layer4(out, get_child_dict(params, 'layer4'), episode)
    out = self.pool(out).flatten(1)
    out = self.linear(out, get_child_dict(params, 'linear'))
    return out


def resnet12(args, bn_args=dict()):
  return ResNet12(args, [64, 128, 256, 512], bn_args)