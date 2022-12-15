import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
import time
import math

from torch_geometric.nn.conv import MessagePassing, GCNConv, GATConv
from torch_geometric.nn import HypergraphConv
from torch_scatter import scatter, scatter_add
from torch_geometric.utils import softmax

model_hyperparam = {
    'HypE': {
        'in_channels': 1,
        'out_channels': 3,
        'filt_h': 1,
        'filt_w': 1,
        'stride': 2,
        'hidden_drop': 0.2,
    },

    'HSimplE': {
        'hidden_drop': 0.2,
    }
}


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class HSimplE(nn.Module):
    def __init__(self, args):
        super(HSimplE, self).__init__()
        self.emb_dim = args.emb_dim
        self.max_arity = 6

        self.hidden_drop_rate = model_hyperparam[args.kg_model]["hidden_drop"]

        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)


    def shift(self, v, sh):
        y = torch.cat((v[:, sh:], v[:, :sh]), dim=1)
        return y

    # def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):
    #     r = self.R(r_idx)
    #     e1 = self.E(e1_idx)
    #     e2 = self.shift(self.E(e2_idx), int(1 * self.emb_dim/self.max_arity))
    #     e3 = self.shift(self.E(e3_idx), int(2 * self.emb_dim/self.max_arity))
    #     e4 = self.shift(self.E(e4_idx), int(3 * self.emb_dim/self.max_arity))
    #     e5 = self.shift(self.E(e5_idx), int(4 * self.emb_dim/self.max_arity))
    #     e6 = self.shift(self.E(e6_idx), int(5 * self.emb_dim/self.max_arity))
    #     x = r * e1 * e2 * e3 * e4 * e5 * e6
    #     x = self.hidden_drop(x)
    #     x = torch.sum(x, dim=1)
    #     return x

    def forward(self, r, E, ms, bs):
        '''
        r: relation embedding
        E: entity embedding (each row is an entity embedding, containing |r| rows)
        '''

        for i in range(E.shape[1]):
            e = self.shift(E[:,i], int((i + 1) * self.emb_dim/self.max_arity)) * ms[:, i].view(-1, 1) + bs[:, i].view(-1, 1)
            if i == 0:
                x = e
            else:
                x = x * e
        x = x * r
        x = self.hidden_drop(x)
        x = torch.sum(x, dim=1)
        return x



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
        self.max_arity = args.arity

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.inp_drop = torch.nn.Dropout(0.2)

        fc_length = (1-self.filt_h+1)*math.floor((self.emb_dim-self.filt_w)/self.stride + 1)*self.out_channels

        self.bn2 = torch.nn.BatchNorm1d(fc_length)
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)
        # Projection network
        self.fc = torch.nn.Linear(fc_length, self.emb_dim)
        self.device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

        self.one_hot_target = [(pos == torch.arange(self.max_arity).reshape(self.max_arity)).float().to(self.device) for pos in range(self.max_arity)]

        # size of the convolution filters outputted by the hypernetwork
        fc1_length = self.in_channels*self.out_channels*self.filt_h*self.filt_w
        # Hypernetwork
        # self.fc1 = torch.nn.Linear(self.emb_dim + self.max_arity + 1, fc1_length)
        self.fc2 = torch.nn.Linear(self.max_arity, fc1_length)



    def convolve(self, r, e, pos):

        e = e.view(-1, 1, 1, self.emb_dim)
        # r = self.R(r_idx)
        x = e
        x = self.inp_drop(x)
        # one_hot_target = (pos == torch.arange(self.max_arity + 1).reshape(self.max_arity + 1)).float().to(self.device)
        one_hot_target = self.one_hot_target[pos]
        poses = one_hot_target.repeat(r.shape[0]).view(-1, self.max_arity)
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

    def forward(self, r, E, ms, bs):
        '''
        r: relation embedding
        E: entity embedding (each row is an entity embedding, containing |r| rows)
        '''

        for i in range(E.shape[1]):
            e = self.convolve(r, E[:,i], i) * ms[:, i].view(-1, 1) + bs[:, i].view(-1, 1)
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
            if normalization.__class__.__name__ != 'Identity':
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
        return


class HCHA(nn.Module):
    """
    This model is proposed by "Hypergraph Convolution and Hypergraph Attention" (in short HCHA) and its convolutional layer
    is implemented in pyg.
    """

    def __init__(self, args):
        super(HCHA, self).__init__()

        self.num_layers = args.num_layers
        self.dropout = args.dropout  # Note that default is 0.6

#         Note that add dropout to attention is default in the original paper
        self.convs = nn.ModuleList()
        self.convs.append(HypergraphConv(in_channels=args.emb_dim, out_channels=args.hid_dim,
                                            heads=args.heads, dropout=args.dropout, use_attention=args.use_attention))

        for _ in range(self.num_layers-2):
            self.convs.append(HypergraphConv(in_channels=args.hid_dim, out_channels=args.hid_dim,
                                            heads=args.heads, dropout=args.dropout, use_attention=args.use_attention))

        self.convs.append(HypergraphConv(in_channels=args.hid_dim, out_channels=args.emb_dim,
                                            heads=args.heads, dropout=args.dropout, use_attention=args.use_attention))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edges, edge_weight, training=True):

        for i, conv in enumerate(self.convs[:-1]):

            x = F.elu(conv(x=x, hyperedge_index=edges))
            x = F.dropout(x, p=self.dropout, training=training)

#         x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x=x, hyperedge_index=edges)

        return x

class Sequential_model(nn.Module):
    def __init__(self, args):
        super(Sequential_model, self).__init__()
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.encode_attn = nn.MultiheadAttention(args.emb_dim, args.heads, dropout=args.dropout)
        self.transform = nn.Sequential(nn.Linear(args.emb_dim, args.hid_dim),
                                       nn.ReLU(),
                                       nn.Linear(args.hid_dim, args.emb_dim))
        self.decode_attn = nn.MultiheadAttention(args.emb_dim, args.heads, dropout=args.dropout)

    def forward(self, check_ins, mask, poi_emb, training=True):
        check_ins = check_ins.permute(1, 0, 2)
        poi_emb = poi_emb.permute(1, 0, 2)
        check_ins = self.encode_attn(check_ins, check_ins, check_ins, attn_mask=mask)[0]
        check_ins = self.transform(check_ins)
        interests = self.decode_attn(poi_emb, check_ins, check_ins)[0]
        interests = F.dropout(interests, p=self.dropout, training=training)
        return interests


class HKGAT(nn.Module):
    def __init__(self, args, edges, relations, edge_weight):
        super(HKGAT, self).__init__()
        self.gat_model = args.gat_model
        self.HKG = HypE(args) if args.kg_model == 'HypE' else HSimplE(args)

        if args.gat_model == 'CEGAT':
            self.GAT = CEGAT(in_dim=args.emb_dim,
                          hid_dim=args.hid_dim,  # Use args.enc_hidden to control the number of hidden layers
                          out_dim=args.emb_dim,
                          num_layers=args.num_layers,
                          heads=args.heads,
                          output_heads=args.output_heads,
                          dropout=args.dropout,
                          Normalization=args.normalization)
        elif args.gat_model == 'CEGCN':
            self.GAT = CEGCN(in_dim=args.emb_dim,
                            hid_dim=args.hid_dim,  # Use args.enc_hidden to control the number of hidden layers
                            out_dim=args.emb_dim,
                            num_layers=args.num_layers,
                            dropout=args.dropout,
                            Normalization=args.normalization)
        elif args.gat_model == 'HCHA':
            self.GAT = HCHA(args)

        self.sequential_model = Sequential_model(args)

        self.entity_num = args.entity_num

        self.E = nn.Embedding(args.entity_num + 1, args.emb_dim, padding_idx=0)
        self.R = nn.Embedding(args.relation_num, args.rel_emb_dim, padding_idx=0)

        self.check_in_encoder = nn.Sequential(
            nn.Linear(7 * args.emb_dim, args.emb_dim)
        )

        # if args.task == 'check_in':
        #     self.time_encoder_user = nn.Sequential(
        #         nn.Linear(3 * args.emb_dim, args.emb_dim),
        #         nn.ReLU(),
        #         nn.Linear(3 * args.num_poi, args.num_poi),
            # )

        self.time_encoder_poi = nn.Sequential(
            nn.Linear(3 * args.emb_dim, args.emb_dim),
        #     # nn.ReLU(),
        #     # nn.Linear(3 * args.num_poi, args.num_poi),
        )


        self.E.weight.data = torch.randn(args.entity_num + 1, args.emb_dim)
        self.R.weight.data = torch.randn(args.relation_num, args.rel_emb_dim)
        # nn.init.xavier_uniform_(self.E.weight.data)
        # nn.init.xavier_uniform_(self.R.weight.data)
        self.E_GNN = nn.Embedding(args.entity_num + 1, args.emb_dim, padding_idx=0)
        self.E_GNN.weight.data = torch.randn(args.entity_num + 1, args.emb_dim)
        self.edges = edges
        self.relations = relations # Contains all relations in the dataset
        self.edge_weight = edge_weight
        self.poi_index = torch.range((args.poi_index['start']), (args.poi_index['end']), 1).long()


    def forward(self, index, mode, ms=None, bs=None):
        if mode == 'kg':
            r_index = index[:, 0]
            e_index = index[:, 1:]
            r = self.R(r_index)
            E = self.E(e_index)

            return self.HKG(r, E, ms, bs)
        elif mode == 'train_gat_friendship':
            # print(self.E.weight[0])
            # return self.GAT(x, self.edges, self.edge_weight, training=True)
            # self.E.requires_grad_(False)
            # self.R.requires_grad_(False)
            # print(self.E.weight[0])
            return self.GAT(self.E.weight.data, self.edges, self.edge_weight, training=True)

        elif mode == 'predict_friendship':
            # return self.GAT(x, self.edges, self.edge_weight, training=False)
            # self.E.requires_grad_(False)
            # self.R.requires_grad_(False)
            return self.GAT(self.E.weight.data, self.edges, self.edge_weight, training=False)
        elif mode == 'train_gat_check_in':
            # user = index[:, 0]
            last_check_in = index[:, -1, :]
            labels = last_check_in[:, 1]
            time1 = last_check_in[:, 4]
            time2 = last_check_in[:, 5]
            time3 = last_check_in[:, 6]

            index = index[:, :-1,:]

            # poi = index[:, 3]
            self.E.requires_grad_(False)
            x = self.GAT(torch.cat((self.E.weight, self.E_GNN.weight)), self.edges, self.edge_weight, training=True)
            # print(self.E.weight.data[1])
            check_in_emb = torch.cat((x[index[:,:, 0]], x[index[:, :, 1]], x[index[:, :, 2]], x[index[:, :, 3]],
                                      x[index[:, :, 4]], x[index[:, :, 5]], x[index[:, :, 6]]), dim=2)
            check_in_emb = self.check_in_encoder(check_in_emb).squeeze()
            poi_emb = x[self.poi_index]
            poi_emb = torch.concat((poi_emb, time1.repeat(poi_emb.shape[0]), time2.repeat(poi_emb.shape[0]), time3.repeat(poi_emb.shape[0])), dim=1)
            poi_emb = self.time_encoder_poi(poi_emb)

            interests = self.sequential_model(check_in_emb, ms, poi_emb, training=True)

            similarity = torch.mm(interests, poi_emb.t())
            return similarity, labels

        elif mode == 'test_gat_check_in':
            user = index[:, 0]
            time1 = index[:, 1]
            time2 = index[:, 2]

            x = self.GAT(torch.cat((self.E.weight, self.E_GNN.weight)), self.edges, self.edge_weight, training=False)
            # user_emb = torch.cat((x[user], x[time1] + x[time2]), dim=1)
            # user_emb = self.time_encoder_user(user_emb)
            user_emb = torch.cat((x[user], x[time1], x[time2]), dim=1)
            user_emb = self.time_encoder_user(user_emb)
            # poi_emb = torch.cat((x[self.poi_index], x[time1], x[time2]), dim=1)
            # poi_emb = self.time_encoder_user(poi_emb)
            poi_emb = x[self.poi_index]
            # norm = (user_emb.norm(dim=1, keepdim=True) * poi_emb.norm(dim=1, keepdim=True).t())

            similarity = torch.matmul(user_emb, poi_emb.t())

            return similarity
            #
            # similarity = torch.matmul(user_emb, poi_emb.t()) / (user_emb.norm(dim=1, keepdim=True) * poi_emb.norm(dim=1, keepdim=True).t())
            #
            # return similarity




