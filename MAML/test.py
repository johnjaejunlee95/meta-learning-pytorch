import torch
from datasets.data_sampling import *
from datasets.omniglot_dataloader import *
from utils.utils import *
from utils.args import parse_args
from maml import MAML


def main(args):
    
    data_sampling = DataSampling(args)
    testloader = data_sampling.m_dataloader['test']
    
    checkpoint = torch.load(f'model_ckpt/MAML_{args.model}_{args.datasets}_{args.num_ways}w_{args.num_shots}s_best.pth')
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_params'])
    
    maml = MAML(model)
    
    test_loss = Averager()
    test_acc = Averager()
    for test_batch, _ in testloader:
        loss, acc = maml(args, test_batch, is_train=False)
        test_loss.add(loss.item())
        test_acc.add(acc)
    print(f"Test Accuracy: {test_acc.item():.4f} | Test Loss: {test_loss.item():.4f}")
    

if __name__ == '__main__':
    
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)

    main(args)