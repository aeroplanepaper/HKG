import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
import time
import math

from torch_geometric.nn.conv import MessagePassing, GCNConv, GATConv
from torch_scatter import scatter
from torch_geometric.utils import softmax

model_hyperparam = {
    'HypE': {
        'in_channels': 1,
        'out_channels': 6,
        'filt_h': 1,
        'filt_w': 1,
        'stride': 2,
        'hidden_drop': 0.2,
        'max_arity': 7,
    }
}


class HypE(torch.nn.Module):
    def __init__(self, args):
        super(HypE, self).__init__()
        self.in_channels = model_hyperparam[args.kg_model]["in_channels"]
        self.out_channels = model_hyperparam[args.kg_model]["out_channels"]
        self.filt_h = model_hyperparam[args.kg_model]["filt_h"]
        self.filt_w = model_hyperparam[args.kg_model]["filt_w"]
        self.stride = model_hyperparam[args.kg_model]["stride"]
        self.hidden_drop_rate = model_hyperparam[args.kg_model]["hidden_drop"]
        self.emb_dim = args.emb_dim
        self.max_arity = model_hyperparam[args.kg_model]["max_arity"]

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.inp_drop = torch.nn.Dropout(0.2)

        fc_length = (1-self.filt_h+1)*math.floor((self.emb_dim-self.filt_w)/self.stride + 1)*self.out_channels

        self.bn2 = torch.nn.BatchNorm1d(fc_length)
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)
        # Projection network
        self.fc = torch.nn.Linear(fc_length, self.emb_dim)
        self.device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

        # size of the convolution filters outputted by the hypernetwork
        fc1_length = self.in_channels*self.out_channels*self.filt_h*self.filt_w
        # Hypernetwork
        self.fc1 = torch.nn.Linear(self.emb_dim + self.max_arity + 1, fc1_length)
        self.fc2 = torch.nn.Linear(self.max_arity + 1, fc1_length)



    def convolve(self, r, e, pos):

        e = e.view(-1, 1, 1, self.emb_dim)
        # r = self.R(r_idx)
        x = e
        x = self.inp_drop(x)
        one_hot_target = (pos == torch.arange(self.max_arity + 1).reshape(self.max_arity + 1)).float().to(self.device)
        poses = one_hot_target.repeat(r.shape[0]).view(-1, self.max_arity + 1)
        one_hot_target.requires_grad = False
        poses.requires_grad = False
        k = self.fc2(poses)
        k = k.view(-1, self.in_channels, self.out_channels, self.filt_h, self.filt_w)
        k = k.view(e.size(0)*self.in_channels*self.out_channels, 1, self.filt_h, self.filt_w)
        x = x.permute(1, 0, 2, 3)
        x = F.conv2d(x, k, stride=self.stride, groups=e.size(0))
        x = x.view(e.size(0), 1, self.out_channels, 1-self.filt_h+1, -1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(e.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, r, E):
        '''
        r: relation embedding
        E: entity embedding (each row is an entity embedding, containing |r| rows)
        '''
        # r = self.R(r_idx)

        for i in range(E.shape[1]):
            e = self.convolve(r, E[:,i], i).view(-1, 1)
            if i == 0:
                x = e
            else:
                x = x * e

        x = x * r
        x = self.hidden_drop(x)
        x = torch.sum(x, dim=1)
        return x



class CEGAT(MessagePassing):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers,
                 heads,
                 output_heads,
                 dropout,
                 Normalization='bn'
                 ):
        super(CEGAT, self).__init__()
        self.convs = nn.ModuleList()
        self.normalizations = nn.ModuleList()

        if Normalization == 'bn':
            self.convs.append(GATConv(in_dim, hid_dim, heads))
            self.normalizations.append(nn.BatchNorm1d(hid_dim))
            for _ in range(num_layers-2):
                self.convs.append(GATConv(heads*hid_dim, hid_dim))
                self.normalizations.append(nn.BatchNorm1d(hid_dim))

            self.convs.append(GATConv(heads*hid_dim, out_dim,
                                      heads=output_heads, concat=False))
        else:  # default no normalizations
            self.convs.append(GATConv(in_dim, hid_dim, heads))
            self.normalizations.append(nn.Identity())
            for _ in range(num_layers-2):
                self.convs.append(GATConv(hid_dim*heads, hid_dim))
                self.normalizations.append(nn.Identity())

            self.convs.append(GATConv(hid_dim*heads, out_dim,
                                      heads=output_heads, concat=False))

        self.dropout = dropout

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()
        for normalization in self.normalizations:
            if normalization.__class__.__name__ != 'Identity':
                normalization.reset_parameters()

    def forward(self, x, edges, edge_weight, training=True):
        #         Assume edge_index is already V2V
        #        x: [N, in_dim] (all vertices)
        #        edge_index: [2, E]

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x=x, edge_index=edges)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i](x)
            x = F.dropout(x, p=self.dropout, training=training)
        x = self.convs[-1](x, edges)
        return x


class CEGCN(MessagePassing):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers,
                 dropout,
                 Normalization='bn'
                 ):
        super(CEGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.normalizations = nn.ModuleList()

        if Normalization == 'bn':
            self.convs.append(GCNConv(in_dim, hid_dim, normalize=False))
            self.normalizations.append(nn.BatchNorm1d(hid_dim))
            for _ in range(num_layers-2):
                self.convs.append(GCNConv(hid_dim, hid_dim, normalize=False))
                self.normalizations.append(nn.BatchNorm1d(hid_dim))

            self.convs.append(GCNConv(hid_dim, out_dim, normalize=False))
        else:  # default no normalizations
            self.convs.append(GCNConv(in_dim, hid_dim, normalize=False))
            self.normalizations.append(nn.Identity())
            for _ in range(num_layers-2):
                self.convs.append(GCNConv(hid_dim, hid_dim, normalize=False))
                self.normalizations.append(nn.Identity())

            self.convs.append(GCNConv(hid_dim, out_dim, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ is 'Identity'):
                normalization.reset_parameters()

    def forward(self, x, edges, edge_weight, training=True):
        #         Assume edge_index is already V2V
        #        x: [N, in_dim] (all vertices)
        #        edge_index: [2, E]

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edges, edge_weight)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i](x)
            x = F.dropout(x, p=self.dropout, training=training)
        x = self.convs[-1](x, edges)
        return x


class HKGAT(nn.Module):
    def __init__(self, args, edges, relations, edge_weight):
        super(HKGAT, self).__init__()
        self.HKG = HypE(args)
        self.GAT = CEGAT(in_dim=args.emb_dim,
                      hid_dim=args.hid_dim,  # Use args.enc_hidden to control the number of hidden layers
                      out_dim=args.emb_dim,
                      num_layers=args.num_layers,
                      heads=args.heads,
                      output_heads=args.output_heads,
                      dropout=args.dropout,
                      Normalization=args.normalization)

        # self.GAT = CEGCN(in_dim=args.emb_dim,
        #                 hid_dim=args.hid_dim,  # Use args.enc_hidden to control the number of hidden layers
        #                 out_dim=args.emb_dim,
        #                 num_layers=args.num_layers,
        #                 dropout=args.dropout,
        #                 Normalization=args.normalization)


        self.E = nn.Embedding(args.entity_num, args.emb_dim, padding_idx=0)
        self.R = nn.Embedding(args.relation_num, args.rel_emb_dim, padding_idx=0)

        # self.E.weight.data[0] = torch.ones(args.emb_dim)
        # self.R.weight.data[0] = torch.ones(args.rel_emb_dim)
        # xavier_uniform_(self.E.weight.data[1:])
        # xavier_uniform_(self.R.weight.data[1:])
        self.E.weight.data = torch.randn(args.entity_num, args.emb_dim)
        self.R.weight.data = torch.randn(args.relation_num, args.rel_emb_dim)


        self.edges = edges
        self.relations = relations # Contains all relations in the dataset
        self.edge_weight = edge_weight


    def forward(self, index, mode):
        if mode == 'train_hkg':
            r_idx = self.relations[index]['relation']
            e_idx = self.relations[index]['entity']
            r = self.R(r_idx)
            E = self.E(e_idx)
            return self.HKG(r,E)
        elif mode == 'train_gat':
            # print(self.E.weight[0])
            # return self.GAT(x, self.edges, self.edge_weight, training=True)
            return self.GAT(self.E.weight, self.edges, self.edge_weight, training=True)
        elif mode == 'predict':
            # return self.GAT(x, self.edges, self.edge_weight, training=False)
            return self.GAT(self.E.weight, self.edges, self.edge_weight, training=False)
            # return self.GAT(self.E.weight, self.edges, self.edge_weight,training=False)



