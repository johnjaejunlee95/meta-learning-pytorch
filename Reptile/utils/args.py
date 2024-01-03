from argparse import ArgumentParser
import argparse

def parse_args(default=False):
    """Command-line argument parser for train."""

    parser = ArgumentParser(
        description='PyTorch implementation of Multi-MAML'
    )

    argparser = argparse.ArgumentParser()
    
    argparser.add_argument("--epoch", type=int, help="number of training tasks", default=100000)
    argparser.add_argument("--num_ways", type=int, help="n way", default=5)
    argparser.add_argument("--training_shots", type=int, help="training shots", default=15)
    argparser.add_argument("--num_shots", type=int, help="k shot for support set", default=5)
    argparser.add_argument("--imgc", type=int, help="imgc", default=3)
    argparser.add_argument("--filter_size", type=int, help="size of filters of convblock", default=64)
    
    argparser.add_argument("--batch_size", type=int, help="meta batch size: task num", default=5)
    argparser.add_argument("--inner_batch_size", type=int, help="inner loop batch size", default=10)
    argparser.add_argument("--eval_inner_batch_size", type=int, help="inner loop iter", default=15)    
    
    argparser.add_argument("--update_lr", type=float, help="inner-loop update learning rate", default=0.001)
    
    argparser.add_argument("--num_sampling", type=int, help="eval task number", default=100)
    argparser.add_argument("--update_step", type=int, help="inner-loop update steps", default=8)
    argparser.add_argument("--eval_update_step", type=int, help="eval_num_update_steps", default=50)
    
    argparser.add_argument("--update", type=str, help="update method: maml, anil, boil", default="maml")
    argparser.add_argument("--gpu_id", type=int, help="gpu device number", default=3)
    argparser.add_argument("--model", type=str, help="model architecture", default="conv4")
    argparser.add_argument("--datasets", type=str, help="meta dataset", default="mini")
    argparser.add_argument("--version", type=str, help="version", default="0")
    args = argparser.parse_args()

    if default:
        return parser.parse_args('')
    else:
        return args