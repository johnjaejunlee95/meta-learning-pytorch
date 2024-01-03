import torch 
import torch.nn.functional as F
from copy import deepcopy
from collections import OrderedDict

def test_prediction(model, x_train, y_train, x_test, y_test):

    res = []
    for test_sample in zip(x_test, y_test):
        inputs, label = (x_train, y_train)
        inputs = torch.cat((inputs, test_sample[0].unsqueeze(0)), dim=0)
        label = torch.cat((label, test_sample[1].unsqueeze(0)), dim=0)
        
        inputs = inputs.cuda()
        logits = model(inputs)
        res.append(logits.argmax(dim=-1)[-1])
    return res

def average_params(param_list):
    mean_params = OrderedDict()

    for param_key in param_list[0].keys():
        mean_params[param_key] = torch.stack([params[param_key] for params in param_list]).mean(dim=0)

    return mean_params
