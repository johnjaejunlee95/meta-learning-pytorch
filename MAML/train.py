import torch
import torch.optim as optim
from maml import MAML
from models import model_selection
from datasets import load_datasets
from datasets.omniglot_dataloader import *
from utils.utils import *
from utils.args import parse_args
from tqdm import tqdm
from copy import deepcopy

RANDOM_SEED = random.randint(0, 1000)


def main(args):
    
    set_seed(RANDOM_SEED)
    
    model = model_selection(args)
    model_structure = deepcopy(model)
    
    maml = MAML(model)
    meta_optimizer = optim.Adam(model.parameters(), lr=args.meta_lr)
    
    trainloader, validloader = load_datasets(args)
    
    best_acc = 0.0
    training_loss = Averager()
    training_acc = Averager()
    
    for epoch in tqdm(range(1, args.epoch+1), desc='Epoch', initial=1, colour='red', total=args.epoch):
        meta_loss = 0.0
        meta_acc = 0.0
        for j in range(args.batch_size):
            batch, _ = next(trainloader)
            train_loss, train_acc = maml(args, batch)
            meta_loss += train_loss
            meta_acc += train_acc
        
        meta_acc /= args.batch_size
        meta_loss /= args.batch_size
        
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()
        
        training_loss.add(meta_loss.item())
        training_acc.add(meta_acc)
        
        if epoch % 5000 == 0:
            print(f"Epoch {epoch} | Meta Loss: {training_loss.item():.4f} | Meta Acc: {training_acc.item():.4f}")
        
        if epoch % 25000 ==0:
            validation_acc = Averager()
            validation_loss = Averager()
            for k, (batch, _) in enumerate(validloader):
                val_loss, val_acc = maml(args, batch, is_train=False)
                validation_loss.add(val_loss.item())
                validation_acc.add(val_acc)

            print(f"Epoch {epoch} | Validation Accuracy: {validation_acc.item():.4f} | Validation Loss: {validation_loss.item():.4f}")
            
            if best_acc < validation_acc.item():
                best_acc = validation_acc.item()
                torch.save({'model': model_structure,
                            'model_params': model.state_dict()
                            }, f'model_ckpt/MAML_{args.model}_{args.datasets}_{args.num_ways}w_{args.num_shots}s_best.pth')
                
            torch.save({'model': model_structure,
                        'model_params': model.state_dict()
                        }, f'model_ckpt/MAML_{args.model}_{args.datasets}_{args.num_ways}w_{args.num_shots}s_epoch{epoch}.pth')
            

if __name__ == '__main__':
    
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)

    main(args)
