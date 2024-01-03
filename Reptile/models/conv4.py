import torch
import torch.nn as nn
from collections import OrderedDict
from .modules import *


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_args):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.bn = BatchNorm2d(out_channels, **bn_args)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, params=None, episode=None):
        out = self.conv(x, get_child_dict(params, 'conv'))
        out = self.bn(out, get_child_dict(params, 'bn'), episode)
        out = self.pool(self.relu(out))
        return out


class Conv4(nn.Module):
    def __init__(self, args, bn_args):
        super(Conv4, self).__init__()
        hid_dim = args.filter_size

        episodic = bn_args.get('episodic') or []
        bn_args_ep, bn_args_no_ep = bn_args.copy(), bn_args.copy()
        
        # if 
        bn_args_ep['episodic'] = True
        bn_args_no_ep['episodic'] = False
        bn_args_dict = dict()
        for i in [1, 2, 3, 4]:
            if 'conv%d' % i in episodic:
                bn_args_dict[i] = bn_args_ep
            else:
                bn_args_dict[i] = bn_args_no_ep
            bn_args_dict[i]['n_episode'] = args.batch_size
        # print(bn_args_dict)
        self.encoder = Sequential(OrderedDict([
            ('conv1', ConvBlock(args.imgc, hid_dim, bn_args_dict[1])),
            ('conv2', ConvBlock(hid_dim, hid_dim, bn_args_dict[2])),
            ('conv3', ConvBlock(hid_dim, hid_dim, bn_args_dict[3])),
            ('conv4', ConvBlock(hid_dim, hid_dim, bn_args_dict[4])),
        ]))

        if args.datasets != "omniglot":
            self.linear = Linear(hid_dim * 5 * 5, args.num_ways)
        else:
            self.linear = Linear(hid_dim * 1 * 1, args.num_ways)
            
    def forward(self, x, params=None, episode=None):
        out = self.encoder(x, get_child_dict(params, 'encoder'), episode)
        out = out.view(out.shape[0], -1)
        out = self.linear(out, get_child_dict(params, 'linear'))
        return out


@register('convnet4')
def convnet4(args, bn_args=dict()):
    return Conv4(args, bn_args)


## easy and hard coding version

# class Conv4(nn.Module):
#     def __init__(self, args):
#         super(Conv4, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=args.imgc, out_channels=args.filter_size, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(args.filter_size)
#         self.conv2 = nn.Conv2d(args.filter_size, args.filter_size, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(args.filter_size)
#         self.conv3 = nn.Conv2d(args.filter_size, args.filter_size, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(args.filter_size)
#         self.conv4 = nn.Conv2d(args.filter_size, args.filter_size, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm2d(args.filter_size)
        
#         self.linear = nn.Linear(args.filter_size * 5 * 5, args.num_ways)


#     def forward(self, x, params=None):
        
#         if params is None:
#             params = OrderedDict(self.named_parameters())
    
#         for idx in [1, 2, 3, 4]:
#             x = F.conv2d(x, params[f'conv{idx}.weight'], params[f'conv{idx}.bias'], padding=1)
#             x = F.batch_norm(x, None, None, params[f'bn{idx}.weight'], params[f'bn{idx}.bias'], training=True)
#             x = F.relu(x)
#             x = F.max_pool2d(x, kernel_size=2, stride=2)
#         x = x.view(x.size(0), -1)
#         x = F.linear(x, params['linear.weight'], params['linear.bias'])
        
#         return x
    
    