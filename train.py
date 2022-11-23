import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
import argparse
import sklearn.metrics
import pickle

from data_processor import *
from model import HKGAT
import utils

def get_model(args, expanded_edges, all_relations, edge_weight):
    if args.model == 'HKGAT':
        model = HKGAT(args, expanded_edges, all_relations, edge_weight)
    else:
        raise NotImplementedError
    return model


def test(model,friendship_test_data, friendship_test_label, friendship_test, friendship_all, user_index, poi_index, device):
    model.eval()
    with torch.no_grad():
        out = model(index=None, mode='predict')
        # friendship_test_data, friendship_test_label = utils.negative_sample_friendship_test(friendship_test, friendship_all,
        #                                                                                user_index)
        # friendship_test_data = friendship_test_data.to(device)
        # friendship_test_label = friendship_test_label.to(device)
        predictions = F.cosine_similarity(out[friendship_test_data[0]], out[friendship_test_data[1]])

        top_k = utils.test_topK_friendship(out, user_index, friendship_test, friendship_all, topK=10)
        acc = sklearn.metrics.accuracy_score(friendship_test_label.cpu().detach().numpy(),
                                             predictions.cpu().detach().numpy() > 0.5) * 100

        auc = sklearn.metrics.roc_auc_score(friendship_test_label.cpu().detach().numpy(), predictions.cpu().detach().numpy())
        ap = sklearn.metrics.average_precision_score(friendship_test_label.cpu().detach().numpy().astype(int),
                                                     predictions.cpu().detach().numpy())
        print('Test',
              'topK: {:.4f}'.format(top_k),
              'acc: {:.4f}'.format(acc),
              'auc: {:.4f}'.format(auc),
              'ap: {:.4f}'.format(ap))


def test_temp(model, friendship_test_data, friendship_test_label, device):
    model.eval()
    with torch.no_grad():
        out = model(index=None, mode='predict')
        # friendship_test_data, friendship_test_label = utils.negative_sample_friendship(friendship_test, friendship_all, user_index, sample_ratio=1)
        friendship_test_data = friendship_test_data.to(device)
        friendship_test_label = friendship_test_label.to(device)

        predictions = F.cosine_similarity(out[friendship_test_data[0]], out[friendship_test_data[1]])

        # top_k = utils.test_topK_friendship(out, user_index, friendship_test, friendship_all, topK=10)
        acc = sklearn.metrics.accuracy_score(friendship_test_label.cpu().detach().numpy(),
                                             predictions.cpu().detach().numpy() > 0.5) * 100

        auc = sklearn.metrics.roc_auc_score(friendship_test_label.cpu().detach().numpy(), predictions.cpu().detach().numpy())
        ap = sklearn.metrics.average_precision_score(friendship_test_label.cpu().detach().numpy().astype(int),
                                                     predictions.cpu().detach().numpy())
        print('Test',
              # 'topK: {:.4f}'.format(top_k),
              'acc: {:.4f}'.format(acc),
              'auc: {:.4f}'.format(auc),
              'ap: {:.4f}'.format(ap))



def train(args):
    all_relations, expanded_edges, edge_weight, test_data, train_data, user_index, poi_index = process_data(args)
    # save the data
    with open('data/{}_data_ablation.pkl'.format(args.city), 'wb') as f:
        pickle.dump([all_relations, expanded_edges, edge_weight, test_data, train_data, user_index, poi_index], f)
    # load the data
    # with open('data/{}_data_ablation.pkl'.format(args.city), 'rb') as f:
    #     all_relations, expanded_edges, edge_weight, test_data, train_data, user_index, poi_index = pickle.load(f)

    # with open('data/{}_data_ablation.pkl'.format(args.city), 'rb') as f:
    #     all_relations, expanded_edges, edge_weight, test_data, train_data, user_index, poi_index = pickle.load(f)
    #




    args.relation_num = 8
    friendship_train = train_data['friendship']
    friendship_train_inverse = [(b, a) for a, b in friendship_train]
    friendship_train = friendship_train + friendship_train_inverse
    friendship_test = test_data['friendship']
    friendship_test_inverse = [(b, a) for a, b in friendship_test]
    friendship_test = friendship_test + friendship_test_inverse

    friendship_all = friendship_test + friendship_train

    # with open('data/lbsn_data.pkl', 'rb') as f:
    #     test_f_data, test_f_labels, friendship_train, expanded_edges = pickle.load(f)

    args.entity_num = int(max(max(expanded_edges[0]), max(expanded_edges[1])) + 1)

    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

    expanded_edges = expanded_edges.to(device)
    edge_weight = edge_weight.to(device)

    friendship_test_data, friendship_test_label = utils.negative_sample_friendship_test(friendship_test, friendship_all,
                                                                                   user_index)
    friendship_test_data = friendship_test_data.to(device)
    friendship_test_label = friendship_test_label.to(device)

    # x = torch.randn(args.entity_num, args.emb_dim).to(device)

    model = get_model(args, expanded_edges, all_relations, edge_weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    criterion_kg = nn.CrossEntropyLoss()
    criterion_gat = nn.BCEWithLogitsLoss()

    if args.pretrained:
        model.load_state_dict(torch.load('model/{}_model.pt'.format(args.model)))

    for epoch in range(args.num_epochs):
        model.train()

        total_loss = 0

        # Train KG first


        # Train GAT
        optimizer.zero_grad()
        out = model(index=None, mode='train_gat')
        friendship_train_data, friendship_train_label = utils.negative_sample_friendship_train(friendship_train, friendship_all, user_index, sample_ratio=3)
        friendship_train_data = friendship_train_data.to(device)
        friendship_train_label = friendship_train_label.to(device)

        predictions = F.cosine_similarity(out[friendship_train_data[0]], out[friendship_train_data[1]])
        acc = sklearn.metrics.accuracy_score(friendship_train_label.cpu().detach().numpy(),
                                             predictions.cpu().detach().numpy() > 0.5) * 100

        gat_loss = criterion_gat(predictions, friendship_train_label)
        auc = sklearn.metrics.roc_auc_score(friendship_train_label.cpu().detach().numpy(), predictions.cpu().detach().numpy())
        ap = sklearn.metrics.average_precision_score(friendship_train_label.cpu().detach().numpy().astype(int),
                                                     predictions.cpu().detach().numpy())
        gat_loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'gat_loss: {:.4f}'.format(gat_loss.item()),
                  'acc: {:.4f}'.format(acc),
                  'auc: {:.4f}'.format(auc),
                  'ap: {:.4f}'.format(ap))

        total_loss += gat_loss.item()

        if epoch % 40 == 0:
            test(model, friendship_test_data, friendship_test_label, friendship_test, friendship_all, user_index, poi_index, device)

            # test_temp(model, test_f_data, test_f_labels, device)
        # if (epoch + 1) % args.save_model == 0:
        #     torch.save(model.state_dict(), 'model/{}_model.pt'.format(args.model))






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='NYC', help='City name,choices=[NYC, TKY, SP, JK, KL]')
    parser.add_argument('--model', type=str, default='HKGAT', help='Model name')

    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training ratio')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

    parser.add_argument('--cuda', type=int, default=8, help='GPU ID')
    parser.add_argument('--num_epochs', type=int, default=6000, help='Number of epochs')
    
    # KG parameters
    parser.add_argument('--kg_model', type=str, default='HypE', help='KG model, choices=[TransE, DistMult, ComplEx, HypE]')
    
    #GAT parameters
    parser.add_argument('--emb_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--rel_emb_dim', type=int, default=256, help='Relation embedding dimension')
    parser.add_argument('--heads', type=int, default=1, help='Number of heads')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--normalization', default='ln', help='Normalization method')
    parser.add_argument('--hid_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--output_heads', type=int, default=1, help='Output heads')

    #test parameters
    parser.add_argument('--test', type=bool, default=False, help='Test mode')
    parser.add_argument('--pretrained', type=bool, default=False, help='Load pretrained model')
    parser.add_argument('--save_model', type=int, default=100, help='Save model every x epochs')
    parser.add_argument('--ablation', type=bool, default=True, help='Ablation study')


    args = parser.parse_args()
    args.ablation_list = ['poi_price', 'poi_count', 'poi_category_one', 'category']

    if args.test:
        test(args)
    else:
        train(args)

if __name__ == '__main__':
    main()





