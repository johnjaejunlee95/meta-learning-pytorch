import torch
from collections import OrderedDict
from copy import deepcopy
from torch import nn
from utils.utils import * 
# from utils.parameter_utils import *


class Reptile(nn.Module):
    def __init__(self, model, optimizer):

        super(Reptile, self).__init__()

        self.model = model
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optimizer
       
    def forward(self, args, train_batch):
         
        
        self.model.train()
        self.model.cuda()
        
        init_params = deepcopy(OrderedDict(deepcopy(self.model).named_parameters()))
        labels = get_labels(args, status='train')
        # params = deepcopy(OrderedDict(self.model.named_parameters()))
      
        for inner_step in range(args.update_step):
            self.optimizer.zero_grad()
            
            sample_data, sample_labels = sample_batch(train_batch, labels, args.inner_batch_size)
            sample_data, sample_labels = sample_data.cuda(), sample_labels.cuda()
            
            logits = self.model(sample_data)
            loss_ = self.loss(logits, sample_labels)
            acc_ = count_acc(logits, sample_labels)
            
            loss_.backward()
            self.optimizer.step()

        updated_params = deepcopy(OrderedDict(self.model.named_parameters()))
        self.model.load_state_dict(init_params)

        return loss_, acc_, updated_params

    
    def validation(self, args, batch):
        
        model = deepcopy(self.model).cuda()
        init_param = deepcopy(OrderedDict(model.named_parameters()))
        val_optimizer = torch.optim.Adam(model.parameters(), lr=args.update_lr, betas =(0, 0.999))
        
        model.eval()
        labels = get_labels(args, 'valid')
        (x_train, y_train), (x_test, y_test) = train_test_split(batch, labels, 1)

        for i in range(args.eval_update_step):
            val_optimizer.zero_grad()
            x_train_sample, y_train_sample = sample_batch(x_train, y_train, args.eval_inner_batch_size)
            
            x_train_sample = x_train_sample.cuda()
            y_train_sample = y_train_sample.cuda()
                                        
            logits = model(x_train_sample)
            loss_ = self.loss(logits, y_train_sample)
            
            loss_.backward()
            val_optimizer.step()
        
        model.eval()
        results = []
        for test_sample in zip(x_test, y_test):
            inputs, label = (x_train, y_train)
            inputs = torch.cat((inputs, test_sample[0].unsqueeze(0)), dim=0)
            label = torch.cat((label, test_sample[1].unsqueeze(0)), dim=0)
            
            inputs = inputs.cuda()
            logits = model(inputs)
            results.append(logits.argmax(dim=-1)[-1])
        
        with torch.no_grad():
            results = torch.stack(results)
            final_acc = val_count_acc(results, y_test)
            final_loss = self.loss(model(x_test.cuda()), y_test.cuda())
    
        model.load_state_dict(init_param)
                
        return final_loss, final_acc
    

def main():
    pass

if __name__ == "__main__":
    main()
