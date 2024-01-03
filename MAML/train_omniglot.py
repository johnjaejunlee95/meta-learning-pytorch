import torch
import torch.optim as optim
from maml import MAML
from models import model_selection
from datasets.omniglot_dataloader import *
from utils.utils import *
from utils.args import parse_args
from tqdm import tqdm


RANDOM_SEED = random.randint(0, 1000) #or specific seed number


def main(args):
    
    set_seed(RANDOM_SEED)
    
    args.datasets = 'omniglot'
    args.imgc = 1
    
    model = model_selection(args)
    maml = MAML(model)
    
    meta_optimizer = optim.Adam(model.parameters(), lr=args.meta_lr)
    
    trainloader = get_omniglot_dataloader('train', args.batch_size, args.num_ways, args.num_shots, args.num_shots_test, args.batch_size*args.epoch)
    validloader = get_omniglot_dataloader('val', 1, args.num_ways, args.num_shots, args.num_shots_test, args.max_test_task)
    
    best_acc = 0.0
    
    training_loss = Averager()
    training_acc = Averager()
    
    for epoch, batch in tqdm(enumerate(trainloader, start=1), desc='Epoch', initial=1, colour='red', total=args.epoch):
        meta_loss = 0.0
        meta_acc = 0.0
        for j, (x_support, y_support, x_query, y_query) in enumerate(batch):
            batch_single = (x_support, y_support, x_query, y_query)
            train_loss, train_acc = maml.forward(args, batch_single)
            meta_loss += train_loss
            meta_acc += train_acc
        
        meta_acc /= args.batch_size
        meta_loss /= args.batch_size
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()
        
        training_loss.add(meta_loss.item())
        training_acc.add(meta_acc)
        
        if (epoch+1)%1000 == 0:
            print(f"Epoch {epoch+1} | Meta Loss: {training_loss.item():.4f} | Meta Acc: {training_acc.item():.4f}")
        
        if (epoch+1)%5000 ==0:
            validation_acc = Averager()
            validation_loss = Averager()
            for batch in validloader:
                val_loss, val_acc = maml.forward(args, batch, is_train=False)
                validation_loss.add(val_loss.item())
                validation_acc.add(val_acc)

            print(f"Epoch {epoch+1} | Validation Accuracy: {validation_acc.item():.4f} | Validation Loss: {validation_loss.item():.4f}")
            
            if best_acc < validation_acc.item():
                best_acc = validation_acc
                torch.save({'model': model,
                            'model_params': model.state_dict()
                            }, f'model_ckpt/MAML_omniglot_{args.model}_{args.datasets}_{args.num_ways}w_{args.num_shots}s_best.pth')
                
            torch.save({'model': model,
                        'model_params': model.state_dict()
                        }, f'model_ckpt/MAML_omniglot_{args.model}_{args.datasets}_{args.num_ways}w_{args.num_shots}s_epoch{epoch+1}.pth')

if __name__ == '__main__':
    
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    main(args)
