import torch
import torch.nn as nn
from utils.utils import *
from collections import OrderedDict
from copy import deepcopy

class MAML(nn.Module):
    def __init__(self, model):
        super(MAML, self).__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()  
        
    def forward(self, args, batch, is_train = True):
        
        if is_train:
            update_step = args.update_step
            model_params = OrderedDict(self.model.named_parameters())
            self.model.train()
        else:
            update_step = args.update_step_test
            model_params = OrderedDict(deepcopy(self.model).named_parameters())
            self.model.eval()
        
        if args.datasets != 'omniglot':
            x_support, x_query = split_support_query(batch, shots=args.num_shots, query= args.num_shots_test, ways=args.num_ways)
            y_support, y_query= torch.from_numpy(np.repeat(range(args.num_ways), args.num_shots)).cuda(), torch.from_numpy(np.repeat(range(args.num_ways), args.num_shots_test)).cuda()
             
        else:
            x_support, y_support, x_query, y_query = batch
            x_support, y_support, x_query, y_query = x_support.cuda(), y_support.cuda(), x_query.cuda(), y_query.cuda()
        
        for i in range(update_step):
            logits = self.model(x_support, model_params)
            loss = self.loss(logits, y_support)
            grads = torch.autograd.grad(loss, model_params.values(), create_graph = not args.first_order, retain_graph=True)
            for (name, param), grad in zip(model_params.items(), grads):
                model_params[name] = param - args.update_lr*grad
                
        
        logits = self.model(x_query, model_params)
        loss = self.loss(logits, y_query)
        acc = count_acc(logits, y_query)
        
        return loss, acc
