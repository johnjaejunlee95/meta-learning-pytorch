import numpy as np
from .data_sampling import *
from utils.utils import *

def load_datasets(args):
    
    
    data_sampling = DataSampling(args)
    
    trainloader = cycle(data_sampling.m_dataloader['train'])
    validloader = data_sampling.m_dataloader['valid']
    
    return trainloader, validloader



