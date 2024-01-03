
from .conv4proto import Conv4Proto
from .resnet12proto import *


def model_selection(args):
    
    if args.model == 'conv4':
        model = Conv4Proto(args).cuda()
    elif args.model == 'resnet':
        model = resnetproto12(args).cuda()
            
    return model


