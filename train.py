import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter

from data_processor import load_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FB15k-237', help='Dataset name')
    parser.add_argument('--model', type=str, default='HypE', help='Model name')

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

    parser.add_argument('--cuda', type=int, default=0, help='GPU ID')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--emb_dim', type=int, default=200, help='Embedding dimension')

    parser.add_argument('--test', type=bool, default=False, help='Test mode')
    parser.add_argument('--pretrained', type=bool, default=False, help='Load pretrained model')


    args = parser.parse_args()

    if args.test:
        test(model, args)
    else:
        train(model, args)





