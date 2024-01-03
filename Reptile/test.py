import torch
import torch.optim as optim
from datasets.data_sampling import *
from utils.utils import *
from datasets.omniglot_dataloader import *
from utils.utils import *
from utils.args import parse_args
from tqdm import tqdm
from reptile import Reptile
from utils.variables import *
from copy import deepcopy


def main(args):
    
    data_sampling = DataSampling(args)
    testloader = data_sampling.m_dataloader['test']
    
    checkpoint = torch.load(f'model_ckpt/Reptile_{args.model}_{args.datasets}_{args.num_ways}w_{args.num_shots}s_best.pth')
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_params'])
    
    reptile = Reptile(model)
    
    test_loss = Averager()
    test_acc = Averager()
    for test_batch, _ in testloader:
        loss, acc = reptile.validation(args, test_batch)
        test_loss.add(loss.item())
        test_acc.add(acc)
    print(f"Test Accuracy: {test_acc.item():.4f} | Test Loss: {test_loss.item():.4f}")
    

if __name__ == '__main__':
    
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)

    main(args)