import torch
import torch.optim as optim
from models import model_selection
from datasets import dataset_selection
from datasets.omniglot_dataloader import *
from utils.utils import *
from utils.args import parse_args
from tqdm import tqdm
from reptile import Reptile
from utils.variables import *
from copy import deepcopy
# from utils.parameter_utils import *

RANDOM_SEED = random.randint(0, 1000)


def main(args):
    
    set_seed(RANDOM_SEED)
    
    model = model_selection(args)
    model_structure = deepcopy(model)
    optimizer = optim.Adam(model.parameters(), lr=args.update_lr, betas =(0, 0.999))

    reptile = Reptile(model, optimizer)
    
    trainloader, validloader = dataset_selection(args)
    
    best_acc = 0.0
    training_loss = Averager()
    training_acc = Averager()
        
    for epoch in tqdm(range(1, args.epoch+1), desc='Epoch', initial=1, colour='blue', total=args.epoch):
        meta_loss = 0.0
        meta_acc = 0.0
        
        frac_done = epoch / args.epoch
        epsilon = (1 - frac_done)
        init_params = deepcopy(OrderedDict(model.named_parameters()))
        new_params = []
        for _ in range(args.batch_size):
            batch, _ = next(trainloader)
                             
            train_loss, train_acc, params = reptile.forward(args, batch)
            new_params.append(params)
            meta_loss += train_loss
            meta_acc += train_acc
            
        mean_params = average_params(new_params)
        model = apply_params(model, init_params, mean_params, epsilon)
        
        meta_acc /= args.batch_size
        meta_loss /= args.batch_size
        
        training_loss.add(meta_loss.item())
        training_acc.add(meta_acc)
        
        if epoch%10000 == 0:
            print(f"Epoch {epoch} | Meta Loss: {training_loss.item():.4f} | Meta Acc: {training_acc.item():.4f}")
        
        if epoch%25000 ==0:
            validation_acc = Averager()
            validation_loss = Averager()
            for k, (batch, _) in enumerate(validloader):

                val_loss, val_acc = reptile.validation(args, batch)
                validation_loss.add(val_loss.item())
                validation_acc.add(val_acc)

            print(f"Epoch {epoch} | Validation Accuracy: {validation_acc.item():.4f} | Validation Loss: {validation_loss.item():.4f}")
            
            if best_acc < validation_acc.item():
                best_acc = validation_acc.item()
                torch.save({'model': model_structure,
                            'model_params': model.state_dict()
                            }, f'model_ckpt/Reptile_{args.model}_{args.datasets}_{args.num_ways}w_{args.num_shots}s_best.pth')
                
            torch.save({'model': model_structure,
                        'model_params': model.state_dict()
                        }, f'model_ckpt/Reptile_{args.model}_{args.datasets}_{args.num_ways}w_{args.num_shots}s_epoch{epoch}.pth')
        
       
if __name__ == "__main__":
    
    arg = parse_args()
    torch.cuda.set_device(arg.gpu_id)
    main(arg)