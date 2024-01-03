import torch
import torch.optim as optim
from protonet import ProtoNet
from models import model_selection
from datasets import load_datasets
# from datasets.omniglot_dataloader import *
from utils.utils import *
from utils.args import parse_args
from tqdm import tqdm
from copy import deepcopy

RANDOM_SEED = random.randint(0, 1000) #or specific seed number

def main(args):
    
    set_seed(RANDOM_SEED)

    model = model_selection(args)
    model_structure = deepcopy(model)
    
    protonet = ProtoNet(model)
    optimizer = optim.Adam(model.parameters(), lr=args.meta_lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    
    trainloader, validloader = load_datasets(args)
    
    best_acc = 0.0
    training_loss = Averager()
    training_acc = Averager()
    
    for epoch, (batch, _) in tqdm(enumerate(trainloader, start=1), desc='Epoch', initial=1, colour='red', total=args.epoch):
                    
        train_loss, train_acc = protonet(args, batch)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        training_loss.add(train_loss.item())
        training_acc.add(train_acc)
            
        if epoch%5000 == 0:
            print(f"Epoch {epoch} | Meta Loss: {training_loss.item():.4f} | Meta Acc: {training_acc.item():.4f}")
        
        lr_scheduler.step()
        
        if epoch % 10000 == 0:
            validation_loss = Averager()
            validation_acc = Averager()
            for batch, _ in validloader:
                val_loss, val_acc = protonet(args, batch, is_train=False)
                validation_loss.add(val_loss.item())
                validation_acc.add(val_acc)
                
            print(f"Epoch {epoch} | Validation Accuracy: {validation_acc.item():.4f} | Validation Loss: {validation_loss.item():.4f}")
            
            if best_acc < validation_acc.item():
                best_acc = validation_acc.item()
                torch.save({'model': model_structure,
                            'model_params': model.state_dict()
                            }, f'model_ckpt/ProtoNet_{args.model}_{args.datasets}_{args.num_ways}w_{args.num_shots}s_best.pth')
                
            torch.save({'model': model_structure,
                        'model_params': model.state_dict()
                        }, f'model_ckpt/ProtoNet_{args.model}_{args.datasets}_{args.num_ways}w_{args.num_shots}s_epoch{epoch}.pth')
    

if __name__ == '__main__':
    
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    main(args)