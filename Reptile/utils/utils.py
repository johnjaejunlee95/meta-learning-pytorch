import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        
def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).float().mean().item()

def val_count_acc(pred, label):
    num_corrects = sum([pred == sample for pred, sample in zip(pred, label)]).item()
    return num_corrects / len(label)

def cycle(dataloader):
    while True:
        for x in dataloader:
            yield x
                
class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v
            
            
# def get_basic_expt_info(args):
    
#     n_way = args.num_ways
    
#     n_support = args.num_shots
#     n_query = args.num_shots_test
#     y_support = torch.from_numpy(np.repeat(range(n_way), n_support)).cuda()
#     y_query = torch.from_numpy(np.repeat(range(n_way), n_query)).cuda()
#     return n_way, n_support, n_query, y_support, y_query


# def split_support_query(x, shots, query, ways):
#     """
#     x: n_sample * shape
#     :param x:
#     :param n_support:
#     :return:
#     """
#     x_reshaped = x.contiguous().view(ways, shots + query, *x.shape[1:])
#     x_support = x_reshaped[:, :shots].contiguous().view(ways * shots, *x.shape[1:])
#     x_query = x_reshaped[:, shots:].contiguous().view(ways * query, *x.shape[1:])
#     return x_support.cuda(), x_query.cuda()


def sample_batch(data, labels, batch_size):
    
    num_samples = len(data)
    if num_samples < batch_size:
        raise ValueError("Batch size is larger than the number of samples.")
    indices = random.sample(range(num_samples), batch_size)
    batch_data = data[indices]
    batch_labels = labels[indices]
    
            
    return batch_data, batch_labels


def train_test_split(x, y, n_test):
    test_idxs, train_idxs = [], []
    
    for class_i in range(len(y)):
        class_ex = (y == class_i).nonzero().flatten()
        class_ex = class_ex[torch.randperm(class_ex.size(0))]

        ctest_idx = class_ex[:n_test]
        ctrain_idx = class_ex[n_test:]

        test_idxs.append(ctest_idx)
        train_idxs.append(ctrain_idx)
    
    train_idxs = torch.cat(train_idxs).detach().cpu()
    x_train, y_train = x[train_idxs], y[train_idxs]
    
    test_idxs = torch.cat(test_idxs).detach().cpu()
    x_test, y_test = x[test_idxs], y[test_idxs]
    
    return (x_train, y_train), (x_test, y_test)

     
def get_labels(args, status='train'):

    num_ways = args.num_ways
    
    if status == 'train' : 
        num_shots = args.training_shots
        labels = torch.from_numpy(np.repeat(range(num_ways), num_shots)).cuda()
    else:
        num_shots = args.num_shots
        labels = torch.from_numpy(np.repeat(range(num_ways), num_shots+1)).cuda()
    return labels


def average_params(param_list):
    mean_params = OrderedDict()

    for param_key in param_list[0].keys():
        mean_params[param_key] = torch.stack([params[param_key] for params in param_list]).mean(dim=0)

    return mean_params