import torch
import torch.nn as nn
from utils.utils import *


class ProtoNet(nn.Module):
    def __init__(self, model):
        super(ProtoNet, self).__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, args, batch, is_train = True):
        
        if is_train:
            num_ways = args.num_ways_proto
            self.model.train()
        else:
            num_ways = args.num_ways
            self.model.eval()
        
        
        if args.datasets != 'omniglot':
            x_support, x_query = split_support_query(batch, shots=args.num_shots, query= args.num_shots_test, ways=num_ways)
            y_support, y_query= torch.from_numpy(np.repeat(range(num_ways), args.num_shots)).cuda(), torch.from_numpy(np.repeat(range(num_ways), args.num_shots_test)).cuda()
        else:
            x_support, y_support, x_query, y_query = batch
            x_support, y_support, x_query, y_query = x_support.cuda(), y_support.cuda(), x_query.cuda(), y_query.cuda()
        
        proto = self.model(x_support)
        proto = proto.reshape(num_ways, args.num_shots, -1).mean(dim=1)
        
        logits = euclidean_distance(self.model(x_query), proto)
        loss = self.loss(logits, y_query)
        acc = count_acc(logits, y_query)

        
        return loss, acc
