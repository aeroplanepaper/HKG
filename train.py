import datetime
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
import argparse
import sklearn.metrics
import pickle
# from tensorboard import
from torch.utils.tensorboard import SummaryWriter

from data_processor import *
from model import HKGAT
import utils

record = {}

def get_model(args, expanded_edges, all_relations, edge_weight):
    if args.model == 'HKGAT':
        model = HKGAT(args, expanded_edges, all_relations, edge_weight)
    else:
        raise NotImplementedError
    return model


def test(model, friendship_test_data, friendship_test_label, friendship_test, friendship_all, user_index, poi_index,
         writer, epoch, run):
    model.eval()
    with torch.no_grad():
        out = model(index=None, mode='predict')
        predictions = F.cosine_similarity(out[friendship_test_data[0]], out[friendship_test_data[1]])

        top_k = utils.test_topK_friendship(out, user_index, friendship_test, friendship_all, topK=10)
        acc = sklearn.metrics.accuracy_score(friendship_test_label.cpu().detach().numpy(),
                                             predictions.cpu().detach().numpy() > 0.5) * 100

        auc = sklearn.metrics.roc_auc_score(friendship_test_label.cpu().detach().numpy(),
                                            predictions.cpu().detach().numpy())
        ap = sklearn.metrics.average_precision_score(friendship_test_label.cpu().detach().numpy().astype(int),
                                                     predictions.cpu().detach().numpy())
        writer.add_scalar('Test/Accuracy', acc, epoch)
        writer.add_scalar('Test/AUC', auc, epoch)
        writer.add_scalar('Test/AP', ap, epoch)
        writer.add_scalar('Test/Top10', top_k, epoch)

        if 'top_k' not in record:
            record['top_k'] = [top_k]
            record['acc'] = [acc]
            record['auc'] = [auc]
            record['ap'] = [ap]
        else:
            if len(record['top_k']) < run + 1:
                record['top_k'].append(top_k)
                record['acc'].append(acc)
                record['auc'].append(auc)
                record['ap'].append(ap)
            elif top_k > record['top_k'][run]:
                record['top_k'][run] = top_k
                record['acc'][run] = acc
                record['auc'][run] = auc
                record['ap'][run] = ap
                print('New record at epoch %d' % epoch)

        print('Test',
              'topK: {:.4f}'.format(top_k),
              'acc: {:.4f}'.format(acc),
              'auc: {:.4f}'.format(auc),
              'ap: {:.4f}'.format(ap))

#
# def test_temp(model, friendship_test_data, friendship_test_label, device):
#     model.eval()
#     with torch.no_grad():
#         out = model(index=None, mode='predict')
#         # friendship_test_data, friendship_test_label = utils.negative_sample_friendship(friendship_test, friendship_all, user_index, sample_ratio=1)
#         friendship_test_data = friendship_test_data.to(device)
#         friendship_test_label = friendship_test_label.to(device)
#
#         predictions = F.cosine_similarity(out[friendship_test_data[0]], out[friendship_test_data[1]])
#
#         # top_k = utils.test_topK_friendship(out, user_index, friendship_test, friendship_all, topK=10)
#         acc = sklearn.metrics.accuracy_score(friendship_test_label.cpu().detach().numpy(),
#                                              predictions.cpu().detach().numpy() > 0.5) * 100
#
#         auc = sklearn.metrics.roc_auc_score(friendship_test_label.cpu().detach().numpy(),
#                                             predictions.cpu().detach().numpy())
#         ap = sklearn.metrics.average_precision_score(friendship_test_label.cpu().detach().numpy().astype(int),
#                                                      predictions.cpu().detach().numpy())
#         print('Test',
#               # 'topK: {:.4f}'.format(top_k),
#               'acc: {:.4f}'.format(acc),
#               'auc: {:.4f}'.format(auc),
#               'ap: {:.4f}'.format(ap))


def train(args):
    # all_relations, relation_count, friendship_relation_idx, expanded_edges, edge_weight, test_data, train_data, user_index, poi_index = process_data(args)
    # save the data
    # with open('data/{}_data.pkl'.format(args.city), 'wb') as f:
    #     pickle.dump([all_relations, relation_count, friendship_relation_idx, expanded_edges, edge_weight, test_data, train_data, user_index, poi_index], f)
    # load the data
    with open('data/{}_data.pkl'.format(args.city), 'rb') as f:
        all_relations, relation_count, friendship_relation_idx, expanded_edges, edge_weight, test_data, train_data, user_index, poi_index = pickle.load(f)

    # all_relations = [[relation[0] - 1, relation[1]] for relation in all_relations if relation[0] != 0]

    inverse_relations = [[relation[0], (relation[1][1], relation[1][0])] for relation in all_relations if relation[0] == 1]
    all_relations = all_relations + inverse_relations

    # with open('data/{}_data_ablation.pkl'.format(args.city), 'rb') as f:
    #     all_relations, expanded_edges, edge_weight, test_data, train_data, user_index, poi_index = pickle.load(f)
    #

    args.relation_num = max(relation[0] for relation in all_relations) + 1

    ########################################
    # prepare friendship data
    friendship_train = train_data['friendship']
    friendship_train_inverse = [(b, a) for a, b in friendship_train]
    friendship_train = friendship_train + friendship_train_inverse
    friendship_test = test_data['friendship']
    # friendship_test_inverse = [(b, a) for a, b in friendship_test]
    # friendship_test = friendship_test + friendship_test_inverse
    friendship_all = friendship_test + friendship_train
    # friendship_test_data, friendship_test_label = utils.negative_sample_friendship_test(friendship_test, friendship_all,
    #                                                                                     user_index)
    # with open('data/{}_friendship_test_data.pkl'.format(args.city), 'wb') as f:
    #     pickle.dump([friendship_test_data, friendship_test_label], f)
    with open('data/{}_friendship_test_data.pkl'.format(args.city), 'rb') as f:
        friendship_test_data, friendship_test_label = pickle.load(f)

    ########################################

    ########################################
    # prepare knowledge graph data
    total_relations = len(all_relations)
    max_arity = max([len(relation[1]) for relation in all_relations])
    args.arity = max_arity
    #######################################

    # args.entity_num = int(max(max(expanded_edges[0]), max(expanded_edges[1])) + 1)
    args.entity_num = 8751
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

    expanded_edges = expanded_edges.to(device)
    edge_weight = edge_weight.to(device)





    # if args.pretrained:
    #     model.load_state_dict(torch.load('models/{}_model.pt'.format(args.model)))

    for run in range(args.runs):
        model = get_model(args, expanded_edges, all_relations, edge_weight).to(device)
        optimizer_kg = optim.Adam(model.parameters(), lr=args.kg_lr)
        optimizer_gat = optim.Adam(model.parameters(), lr=args.gat_lr)
        # optimizer_gat = optim.SGD(model.parameters(), lr=args.gat_lr, momentum=0.9)
        criterion_kg = nn.CrossEntropyLoss()
        criterion_gat = nn.BCEWithLogitsLoss()
        lr_scheduler_gat = optim.lr_scheduler.StepLR(optimizer_kg, step_size=200, gamma=0.8)

        writer = SummaryWriter('runs/{}_{}_useKG_{}_kg_train{}_{}'.format(args.city, run, args.use_kg,args.num_epochs, datetime.datetime.now()))
        print('KG training')
        if args.use_kg:
            for epoch in range(args.num_epochs):
                model.train()

                total_loss = 0

                # Train KG first
                random.shuffle(all_relations)
                count = 0
                for iteration in range(total_relations // args.batch_size + 1):

                    last_iteration = iteration == total_relations // args.batch_size
                    batch_pos = all_relations[iteration * args.batch_size: (iteration + 1) * args.batch_size] \
                        if not last_iteration else all_relations[iteration * args.batch_size:]

                    batch, labels, ms, bs = utils.sample_negative_relation_batch(batch_pos, args.entity_num,
                                                                                 neg_ratio=args.neg_ratio, max_arity=max_arity)
                    number_of_positive = len(np.where(labels > 0)[0])

                    batch = torch.LongTensor(batch).to(device)
                    ms = torch.FloatTensor(ms).to(device)
                    bs = torch.FloatTensor(bs).to(device)
                    # labels = torch.LongTensor(labels).to(device)
                    # index[]
                    optimizer_kg.zero_grad()
                    predictions = model(index=batch, mode='train_hkg', ms=ms, bs=bs)

                    predictions = utils.padd_and_decompose(labels, predictions, args.neg_ratio * max_arity)
                    targets = torch.zeros(number_of_positive).long().to(device)
                    kg_loss = criterion_kg(predictions, targets)
                    kg_loss.backward()
                    optimizer_kg.step()
                    total_loss += kg_loss.item()
                    count += 1
                    if iteration % 100 == 0 or last_iteration:
                        print('Epoch {}, iteration {}, kg_loss: {:.4f}'.format(epoch, iteration, total_loss / count))
                        total_loss = 0
                        count = 0

            # Train GAT
        print('GAT training')
        for epoch in range(args.gat_train_iter):
            optimizer_gat.zero_grad()
            out = model(index=None, mode='train_gat')
            friendship_train_data, friendship_train_label = utils.negative_sample_friendship_train(friendship_train,
                                                                                                   friendship_all,
                                                                                                   user_index,
                                                                                                   sample_ratio=3)
            friendship_train_data = friendship_train_data.to(device)
            friendship_train_label = friendship_train_label.to(device)

            predictions = F.cosine_similarity(out[friendship_train_data[0]], out[friendship_train_data[1]])
            acc = sklearn.metrics.accuracy_score(friendship_train_label.cpu().detach().numpy(),
                                                 predictions.cpu().detach().numpy() > 0.5) * 100

            gat_loss = criterion_gat(predictions, friendship_train_label)
            auc = sklearn.metrics.roc_auc_score(friendship_train_label.cpu().detach().numpy(),
                                                predictions.cpu().detach().numpy())
            ap = sklearn.metrics.average_precision_score(friendship_train_label.cpu().detach().numpy().astype(int),
                                                         predictions.cpu().detach().numpy())
            gat_loss.backward()
            optimizer_gat.step()
            if args.use_kg:
                lr_scheduler_gat.step()

            writer.add_scalar('gat_loss', gat_loss.item(), epoch)
            writer.add_scalar('Train/ACC', acc, epoch)
            writer.add_scalar('Train/AUC', auc, epoch)
            writer.add_scalar('Train/AP', ap, epoch)


            if epoch % 10 == 0:
                print('Epoch: {:04d}'.format(epoch + 1),
                      'gat_loss: {:.4f}'.format(gat_loss.item()),
                      'acc: {:.4f}'.format(acc),
                      'auc: {:.4f}'.format(auc),
                      'ap: {:.4f}'.format(ap))

            if epoch % 40 == 0:
                test(model, friendship_test_data, friendship_test_label, friendship_test, friendship_all, user_index,
                     poi_index, writer, epoch, run)

                # test_temp(model, test_f_data, test_f_labels, device)
            if (epoch + 1) % args.save_model == 0:
                torch.save(model.state_dict(), 'models/{}_model_iter{}.pt'.format(args.model, epoch + 1))

        writer.close()
    top_k = np.array(record['top_k'])
    acc = np.array(record['acc'])
    auc = np.array(record['auc'])
    ap = np.array(record['ap'])
    print('top_k:{:.4f} ± {:.4f}'.format(top_k.mean(axis=0), top_k.std(axis=0)))
    print('acc:{:.4f} ± {:.4f}'.format(acc.mean(axis=0), acc.std(axis=0)))
    print('auc:{:.4f} ± {:.4f}'.format(auc.mean(axis=0), auc.std(axis=0)))
    print('ap:{:.4f} ± {:.4f}'.format(ap.mean(axis=0), ap.std(axis=0)))
    with open('results/{}_useKG_{}_kg_train{}.txt'.format(args.city, args.use_kg, args.num_epochs), 'w') as f:
        f.write('top_k:{:.4f} ± {:.4f}\n'.format(top_k.mean(axis=0), top_k.std(axis=0)))
        f.write('acc:{:.4f} ± {:.4f}\n'.format(acc.mean(axis=0), acc.std(axis=0)))
        f.write('auc:{:.4f} ± {:.4f}\n'.format(auc.mean(axis=0), auc.std(axis=0)))
        f.write('ap:{:.4f} ± {:.4f}\n'.format(ap.mean(axis=0), ap.std(axis=0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='NYC', help='City name,choices=[NYC, TKY, SP, JK, KL]')
    parser.add_argument('--model', type=str, default='HKGAT', help='Model name')

    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training ratio')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs')

    parser.add_argument('--cuda', type=int, default=8, help='GPU ID')
    parser.add_argument('--gat_train_iter', type=int, default=4000, help='Number of GAT training iterations')

    # KG parameters
    parser.add_argument('--kg_model', type=str, default='HSimplE',
                        help='KG model, choices=[TransE, DistMult, ComplEx, HypE, HSimplE]')
    parser.add_argument('--kg_lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--neg_ratio', type=int, default=10, help='Negative ratio')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--use_kg', type=bool, default=True, help='Use KG or not')

    # GAT parameters
    parser.add_argument('--gat_lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--emb_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--rel_emb_dim', type=int, default=256, help='Relation embedding dimension')
    parser.add_argument('--heads', type=int, default=1, help='Number of heads')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--normalization', default='ln', help='Normalization method')
    parser.add_argument('--hid_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--output_heads', type=int, default=1, help='Output heads')

    # test parameters
    parser.add_argument('--test', type=bool, default=False, help='Test mode')
    parser.add_argument('--pretrained', type=bool, default=False, help='Load pretrained model')
    parser.add_argument('--save_model', type=int, default=500, help='Save model every x epochs')
    parser.add_argument('--ablation', type=bool, default=False, help='Ablation study')

    args = parser.parse_args()
    args.ablation_list = ['poi_price', 'poi_count', 'poi_category_one', 'category']

    if args.test:
        test(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
