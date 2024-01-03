
from .conv4 import *# Conv4
from .resnet12 import *


def model_selection(args):
    
    if args.model == 'conv4':
        model = convnet4(args).cuda() # Conv4(args).cuda()# 
    elif args.model == 'resnet':
        model = resnet12(args).cuda()
    
    return model


