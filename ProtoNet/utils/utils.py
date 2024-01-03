import torch
import numpy as np
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    
    
def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).float().mean().item()

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v
    
def euclidean_distance(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    # logits = -torch.cdist(a, b)
    return logits

def cycle(dataloader):
    while True:
        for x in dataloader:
            yield x
            
            

def get_basic_expt_info(args):
    
    n_way = args.num_ways
    
    n_support = args.num_shots
    n_query = args.num_shots_test
    y_support = torch.from_numpy(np.repeat(range(n_way), n_support)).cuda()
    y_query = torch.from_numpy(np.repeat(range(n_way), n_query)).cuda()
    return n_way, n_support, n_query, y_support, y_query


def split_support_query(x, shots, query, ways):
    """
    x: n_sample * shape
    :param x:
    :param n_support:
    :return:
    """
    x_reshaped = x.contiguous().view(ways, shots + query, *x.shape[1:])
    x_support = x_reshaped[:, :shots].contiguous().view(ways * shots, *x.shape[1:])
    x_query = x_reshaped[:, shots:].contiguous().view(ways * query, *x.shape[1:])
    return x_support.cuda(), x_query.cuda()