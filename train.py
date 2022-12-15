import collections
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

import data_processor
from data_processor import *
from model import HKGAT
import utils

record = {}

def get_model(args, edges, all_relations, edge_weight):
    if args.model == 'HKGAT':
        model = HKGAT(args, edges, all_relations, edge_weight)
    else:
        raise NotImplementedError
    return model



def kg_eval(args, model, friendship_test_data, friendship_test_label, friendship_test, poi_index, writer, epoch, device):
    model.eval()
    with torch.no_grad():
        friendship_test = [[1, friendship_test_data[0].cpu().detach().numpy()[i], friendship_test_data[1].cpu().detach().numpy()[i]] + [args.entity_num] * 5 for i in range(len(friendship_test_data[0]))]
        ms = np.array([[1,1,0,0,0,0,0]])
        ms = np.repeat(ms, len(friendship_test), axis=0)
        bs = np.array([[0,0,1,1,1,1,1]])
        bs = np.repeat(bs, len(friendship_test), axis=0)
        ms = torch.LongTensor(ms).to(device)
        bs = torch.LongTensor(bs).to(device)
        friendship_test = torch.LongTensor(friendship_test).to(device)
        predictions = model(index=friendship_test, mode='kg', ms=ms, bs=bs)
        acc = sklearn.metrics.accuracy_score(friendship_test_label.cpu().detach().numpy(),
                                             torch.sigmoid(predictions).cpu().detach().numpy() > 0.5) * 100

        auc = sklearn.metrics.roc_auc_score(friendship_test_label.cpu().detach().numpy(),
                                            predictions.cpu().detach().numpy())
        ap = sklearn.metrics.average_precision_score(friendship_test_label.cpu().detach().numpy().astype(int),
                                                     predictions.cpu().detach().numpy())
        print('KG Test at epoch %d' % epoch,
              'acc: {:.4f}'.format(acc),
              'auc: {:.4f}'.format(auc),
              'ap: {:.4f}'.format(ap))
        writer.add_scalar('KG/acc', acc, epoch)
        writer.add_scalar('KG/auc', auc, epoch)
        writer.add_scalar('KG/ap', ap, epoch)

def test_friendship(model, friendship_test_data, friendship_test_label, friendship_test, friendship_all, user_index, poi_index,
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


def test_check_in(model, check_in_test_data, check_in_test_label,check_in_test_data_least,check_in_test_data_least_label, epoch,run, writer):
    model.eval()
    with torch.no_grad():
        predictions_all= model(index=check_in_test_data, mode='test_gat_check_in')

        top1 = utils.check_in_topK(predictions_all, check_in_test_label, topK=1)
        top5 = utils.check_in_topK(predictions_all, check_in_test_label, topK=5)
        top10 = utils.check_in_topK(predictions_all, check_in_test_label, topK=10)
        top20 = utils.check_in_topK(predictions_all, check_in_test_label, topK=20)
        mrr = utils.check_in_MRR(predictions_all, check_in_test_label)
        avg_rank = utils.check_in_avgrank(predictions_all, check_in_test_label)
        loss = F.cross_entropy(predictions_all, check_in_test_label)
        print('Test at epoch %d on all test data' % epoch,
              'loss: {:.4f}'.format(loss),
              'top1: {:.4f}'.format(top1),
              'top5: {:.4f}'.format(top5),
              'top10: {:.4f}'.format(top10),
              'top20: {:.4f}'.format(top20),
              'MRR: {:.4f}'.format(mrr),
              'avg_rank: {:.4f}'.format(avg_rank))

        predictions_least = model(index=check_in_test_data_least, mode='test_gat_check_in')
        top1_least = utils.check_in_topK(predictions_least, check_in_test_data_least_label, topK=1)
        top5_least = utils.check_in_topK(predictions_least, check_in_test_data_least_label, topK=5)
        top10_least = utils.check_in_topK(predictions_least, check_in_test_data_least_label, topK=10)
        top20_least = utils.check_in_topK(predictions_least, check_in_test_data_least_label, topK=20)
        mrr_least = utils.check_in_MRR(predictions_least, check_in_test_data_least_label)
        avg_rank_least = utils.check_in_avgrank(predictions_least, check_in_test_data_least_label)
        loss_least = F.cross_entropy(predictions_least, check_in_test_data_least_label)


        print('Test at epoch %d on least visited poi test data' % epoch,
                'loss: {:.4f}'.format(loss_least),
                'top1: {:.4f}'.format(top1_least),
                'top5: {:.4f}'.format(top5_least),
                'top10: {:.4f}'.format(top10_least),
                'top20: {:.4f}'.format(top20_least),
                'MRR: {:.4f}'.format(mrr_least),
                'avg_rank: {:.4f}'.format(avg_rank_least))

        writer.add_scalar('Run:{}/Test/Top1'.format(run), top1, epoch)
        writer.add_scalar('Run:{}/Test/Top5'.format(run), top5, epoch)
        writer.add_scalar('Run:{}/Test/Top10'.format(run), top10, epoch)
        writer.add_scalar('Run:{}/Test/Top20'.format(run), top20, epoch)
        writer.add_scalar('Run:{}/Test/MRR'.format(run), mrr, epoch)
        writer.add_scalar('Run:{}/Test/AvgRank'.format(run), avg_rank, epoch)
        writer.add_scalar('Run:{}/Test/Loss'.format(run), loss, epoch)

        writer.add_scalar('Run:{}/Test/Top1_least'.format(run), top1_least, epoch)
        writer.add_scalar('Run:{}/Test/Top5_least'.format(run), top5_least, epoch)
        writer.add_scalar('Run:{}/Test/Top10_least'.format(run), top10_least, epoch)
        writer.add_scalar('Run:{}/Test/Top20_least'.format(run), top20_least, epoch)
        writer.add_scalar('Run:{}/Test/MRR_least'.format(run), mrr_least, epoch)
        writer.add_scalar('Run:{}/Test/AvgRank_least'.format(run), avg_rank_least, epoch)
        writer.add_scalar('Run:{}/Test/Loss_least'.format(run), loss_least, epoch)

        if 'top1' not in record:
            record['top1'] = [top1]
            record['top5'] = [top5]
            record['top10'] = [top10]
            record['top20'] = [top20]
            record['mrr'] = [mrr]
            record['avg_rank'] = [avg_rank]
            record['loss'] = [loss]
            record['top1_least'] = [top1_least]
            record['top5_least'] = [top5_least]
            record['top10_least'] = [top10_least]
            record['top20_least'] = [top20_least]
            record['mrr_least'] = [mrr_least]
            record['avg_rank_least'] = [avg_rank_least]
            record['loss_least'] = [loss_least]
        else:
            if len(record['top1']) < run + 1:
                record['top1'].append(top1)
                record['top5'].append(top5)
                record['top10'].append(top10)
                record['top20'].append(top20)
                record['mrr'].append(mrr)
                record['avg_rank'].append(avg_rank)
                record['loss'].append(loss)
                record['top1_least'].append(top1_least)
                record['top5_least'].append(top5_least)
                record['top10_least'].append(top10_least)
                record['top20_least'].append(top20_least)
                record['mrr_least'].append(mrr_least)
                record['avg_rank_least'].append(avg_rank_least)
                record['loss_least'].append(loss_least)
            elif record['mrr'][run] < mrr:
                record['top1'][run] = top1
                record['top5'][run] = top5
                record['top10'][run] = top10
                record['top20'][run] = top20
                record['mrr'][run] = mrr
                record['avg_rank'][run] = avg_rank
                record['loss'][run] = loss
                record['top1_least'][run] = top1_least
                record['top5_least'][run] = top5_least
                record['top10_least'][run] = top10_least
                record['top20_least'][run] = top20_least
                record['mrr_least'][run] = mrr_least
                record['avg_rank_least'][run] = avg_rank_least
                record['loss_least'][run] = loss_least
                print('New record at epoch %d' % epoch)

def train(args):
    # all_relations, relation_count, friendship_relation_idx, edge_index, test_data, train_data, user_index, poi_index, hyperedge_num, hypernode_num = process_data(args)
    # with open('data/{}_data.pkl'.format(args.city), 'wb') as f:
    #     pickle.dump([all_relations, relation_count, friendship_relation_idx, edge_index, test_data, train_data, user_index, poi_index, hyperedge_num, hypernode_num], f)

    # load the data
    with open('data/{}_data.pkl'.format(args.city), 'rb') as f:
        all_relations, relation_count, friendship_relation_idx, edge_index, test_data, train_data, user_index, poi_index, hyperedge_num, hypernode_num = pickle.load(f)

    if args.gat_model in ['CEGCN', 'CEGAT']:
        expanded_edges, edge_weight = data_processor.ConstructV2V(edge_index)
        expanded_edges = torch.LongTensor(expanded_edges)
        edge_weight = torch.FloatTensor(edge_weight)
        expanded_edges, edge_weight = data_processor.norm_contruction(expanded_edges, edge_weight)
        # edge_weight = torch.FloatTensor(edge_weight)
    elif args.gat_model in ['HCHA']:
        pass
        # edge_index, hyperedge_num = data_processor.add_selfloops(edge_index,hypernode_num, hyperedge_num)

    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')


    all_relations = [[relation[0] - 1, relation[1]] for relation in all_relations if relation[0] != 0]

    inverse_relations = [[relation[0], (relation[1][1], relation[1][0])] for relation in all_relations if relation[0] == 0]
    all_relations = all_relations + inverse_relations

    # with open('data/{}_data_ablation.pkl'.format(args.city), 'rb') as f:
    #     all_relations, expanded_edges, edge_weight, test_data, train_data, user_index, poi_index = pickle.load(f)
    #

    args.relation_num = max(relation[0] for relation in all_relations) + 1
    args.entity_num = hypernode_num
    args.hyperedge_num = hyperedge_num

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

    #######################################
    # prepare poi data
    check_in_train = train_data['check_in']
    check_in_test = test_data['check_in']
    check_in_train = utils.extract_trajectories(check_in_train)
    check_in_test = utils.extract_trajectories(check_in_test)
    check_in_train_label = [v[1] - poi_index['start'] for v in check_in_train]
    check_in_test_label = [v[1] - poi_index['start'] for v in check_in_test]
    check_in_count = collections.Counter(check_in_test_label + check_in_train_label)
    # find 10% least popular pois
    check_in_count = sorted(check_in_count.items(), key=lambda x: x[1])
    check_in_count = check_in_count[:int(len(check_in_count) * 0.3)]
    check_in_least = set([v[0] for v in check_in_count])
    check_in_test_data_least = [v for v in check_in_test if v[1] - poi_index['start'] in check_in_least]
    check_in_test_data_least_label = [v[1] - poi_index['start'] for v in check_in_test_data_least]


    check_in_train = [[v[0], v[5], v[6]] for v in check_in_train]
    check_in_test = [[v[0], v[5], v[6]] for v in check_in_test]
    check_in_test_data_least = [[v[0], v[5], v[6]] for v in check_in_test_data_least]
    check_in_all = check_in_train + check_in_test
    check_in_all = [tuple(v) for v in check_in_all]
    check_in_train = np.array(check_in_train)
    check_in_test = np.array(check_in_test)

    time1_index = {}
    time1_index['start'] = min(v[1] for v in check_in_all)
    time1_index['end'] = max(v[1] for v in check_in_all)

    time2_index = {}
    time2_index['start'] = min(v[2] for v in check_in_all)
    time2_index['end'] = max(v[2] for v in check_in_all)

    args.num_poi = poi_index['end'] - poi_index['start'] + 1
    args.poi_index = poi_index

    check_in_train = torch.LongTensor(check_in_train).to(device)
    check_in_test = torch.LongTensor(check_in_test).to(device)
    check_in_train_label = torch.LongTensor(check_in_train_label).to(device)
    check_in_test_label = torch.LongTensor(check_in_test_label).to(device)
    check_in_test_data_least = torch.LongTensor(check_in_test_data_least).to(device)
    check_in_test_data_least_label = torch.LongTensor(check_in_test_data_least_label).to(device)
    ########################################
    # prepare knowledge graph data
    total_relations = len(all_relations)
    max_arity = max([len(relation[1]) for relation in all_relations])
    args.arity = max_arity
    #######################################

    # args.entity_num = int(max(max(expanded_edges[0]), max(expanded_edges[1])) + 1)



    if args.gat_model in ['CEGCN', 'CEGAT']:
        expanded_edges = expanded_edges.to(device)
        edge_weight = edge_weight.to(device)
    elif args.gat_model in ['HCHA']:
        edge_index = torch.LongTensor(edge_index).to(device)
        # if args.use_attention:
        #     edge_weight = torch.ones((args.hyperedge_num, args.emb_dim)).to(device)
        # else:
        edge_weight = None


    # if args.pretrained:
    #     model.load_state_dict(torch.load('models/{}_model.pt'.format(args.model)))
    writer = SummaryWriter('runs/task_{}_{}_useKG_{}_GAT_model_{}_kg_train{}_gat_lr_{}_{}'.format(args.task,args.city, args.use_kg,args.gat_model, args.num_epochs,args.gat_lr, datetime.datetime.now()))

    for run in range(args.runs):
        if args.gat_model in ['CEGCN', 'CEGAT']:
            model = get_model(args, expanded_edges, all_relations, edge_weight).to(device)
        elif args.gat_model in ['HCHA']:
            model = get_model(args, edge_index, all_relations, edge_weight).to(device)

        optimizer_kg = optim.Adam(model.parameters(), lr=args.kg_lr)
        optimizer_gat = optim.Adam(model.parameters(), lr=args.gat_lr, weight_decay=args.gat_weight_decay)
        # optimizer_gat = optim.SGD(model.parameters(), lr=args.gat_lr, momentum=0.9)
        criterion_kg = nn.CrossEntropyLoss()
        if args.task == 'friendship':
            criterion_gat = nn.BCEWithLogitsLoss()
        elif args.task == 'check_in':
            criterion_gat = nn.CrossEntropyLoss()
            # criterion_gat = nn.BCEWithLogitsLoss()

        lr_scheduler_gat = optim.lr_scheduler.StepLR(optimizer_kg, step_size=200, gamma=0.8)

        print('KG training')
        if args.use_kg:
            for epoch in range(args.num_epochs):
                model.train()

                total_loss = 0
                # kg_eval(args, model, friendship_test_data, friendship_test_label, friendship_test, poi_index, writer, epoch, device)
                # Train KG first
                random.shuffle(all_relations)
                count = 0
                for iteration in range(total_relations // args.kg_batch_size + 1):

                    last_iteration = iteration == total_relations // args.kg_batch_size
                    if last_iteration:
                        continue
                    batch_pos = all_relations[iteration * args.kg_batch_size: (iteration + 1) * args.kg_batch_size]


                    batch, labels, ms, bs = utils.sample_negative_relation_batch(batch_pos, args.entity_num,
                                                                                 neg_ratio=args.neg_ratio, max_arity=max_arity)
                    number_of_positive = len(np.where(labels > 0)[0])

                    batch = torch.LongTensor(batch).to(device)
                    ms = torch.FloatTensor(ms).to(device)
                    bs = torch.FloatTensor(bs).to(device)
                    # labels = torch.LongTensor(labels).to(device)
                    # index[]
                    optimizer_kg.zero_grad()
                    predictions = model(index=batch, mode='kg', ms=ms, bs=bs)

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
            if args.task == 'friendship':
                out = model(index=None, mode='train_gat_friendship')
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
                # if args.use_kg:
                #     lr_scheduler_gat.step()

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
                    test_friendship(model, friendship_test_data, friendship_test_label, friendship_test, friendship_all, user_index,
                         poi_index, writer, epoch, run)

                    # test_temp(model, test_f_data, test_f_labels, device)
                if (epoch + 1) % args.save_model == 0:
                    torch.save(model.state_dict(), 'models/{}_model_iter{}.pt'.format(args.model, epoch + 1))
            elif args.task == 'check_in':
                check_in_train_idx = np.arange(len(check_in_train))
                np.random.shuffle(check_in_train_idx)
                for iteration in range(len(check_in_train) // args.gat_batch_size + 1):
                    last_iteration = iteration == len(check_in_train) // args.gat_batch_size
                    if last_iteration:
                        continue
                    batch_idx = check_in_train_idx[iteration * args.gat_batch_size: (iteration + 1) * args.gat_batch_size] \
                        if not last_iteration else check_in_train_idx[iteration * args.gat_batch_size:]
                    batch = check_in_train[batch_idx]
                    labels = check_in_train_label[batch_idx]

                    # add label to batch

                    # batch, labels = utils.negative_sample_check_in_train(batch, check_in_all, poi_index, time1_index, time2_index, sample_ratio=3)
                    # batch = batch.to(device)
                    # labels = labels.to(device)

                    predictions = model(index=batch, mode='train_gat_check_in')
                    gat_loss = criterion_gat(predictions, labels)
                    top_1 = utils.check_in_topK(predictions, labels, 1)
                    top_5 = utils.check_in_topK(predictions, labels, 5)
                    top_10 = utils.check_in_topK(predictions, labels, 10)
                    top_20 = utils.check_in_topK(predictions, labels, 20)
                    mrr = utils.check_in_MRR(predictions, labels)
                    avg_rank = utils.check_in_avgrank(predictions, labels)
                    #
                    # acc = sklearn.metrics.accuracy_score(labels.cpu().detach().numpy(),
                    #                                         predictions.cpu().detach().numpy() > 0.5) * 100
                    #
                    labels = labels.cpu().detach()
                    one_hot_labels = torch.zeros(len(labels), poi_index['end'] - poi_index['start'] + 1).scatter_(1, labels.unsqueeze(1), 1)
                    normalized_predictions = F.softmax(predictions, dim=1)
                    # auc = sklearn.metrics.roc_auc_score(one_hot_labels.numpy(),
                    #                                     normalized_predictions.cpu().detach().numpy(), multi_class='ovo')
                    # ap = sklearn.metrics.average_precision_score(one_hot_labels.numpy().astype(int),
                    #                                             normalized_predictions.cpu().detach().numpy(), average='macro')
                    # f1 = sklearn.metrics.f1_score(one_hot_labels.numpy().astype(int),
                    #                                             normalized_predictions.cpu().detach().numpy() > 0.5, average='macro')


                    writer.add_scalar('Run{}/gat_loss'.format(run), gat_loss.item(), iteration)
                    writer.add_scalar('Run{}/Train/Top1'.format(run), top_1, iteration)
                    writer.add_scalar('Run{}/Train/Top5'.format(run), top_5, iteration)
                    writer.add_scalar('Run{}/Train/Top10'.format(run), top_10, iteration)
                    writer.add_scalar('Run{}/Train/Top20'.format(run), top_20, iteration)
                    writer.add_scalar('Run{}/Train/MRR'.format(run), mrr, iteration)
                    writer.add_scalar('Run{}/Train/AvgRank'.format(run), avg_rank, iteration)
                    # writer.add_scalar('run{}/Train/AUC'.format(run), auc, iteration)
                    # writer.add_scalar('run{}/Train/AP'.format(run), ap, iteration)
                    # writer.add_scalar('run{}/Train/F1'.format(run), f1, iteration)
                    #
                    # gat_loss = criterion_gat(predictions, torch.ones(len(batch)).to(device))
                    gat_loss.backward()
                    optimizer_gat.step()
                    if iteration % 20 == 0 or last_iteration:
                        # print('Epoch {}, iteration {}, gat_loss: {:.4f}, acc: {:.4f}, auc: {:.4f}, ap: {:.4f}'.format(epoch, iteration, gat_loss.item(), acc, auc, ap))

                        print('Epoch {}, iteration {}, gat_loss: {:.4f}, top_1: {:.4f}, top_5: {:.4f}, top_10: {:.4f}, top_20:{:.4f}, MRR:{:.4f}, AvgRank:{:.4f}'.format(
                            epoch, iteration, gat_loss.item(), top_1, top_5, top_10, top_20, mrr, avg_rank))
                        # print('Epoch {}, iteration {}, gat_loss: {:.4f}, top_1: {:.4f}, top_5: {:.4f}, top_10: {:.4f}, top_20:{:.4f}, MRR:{:.4f}, AUC:{:.4f}, AP:{:.4f}, F1:{:.4f}'.format(
                        #     epoch, iteration, gat_loss.item(), top_1, top_5, top_10, top_20, mrr, auc, ap, f1))
                if epoch % 5 == 0:
                    test_check_in(model, check_in_test, check_in_test_label, check_in_test_data_least, check_in_test_data_least_label, epoch,run, writer)
                if epoch == args.gat_train_iter - 1:
                    torch.save(model.state_dict(), 'models/{}_use_KG_{}_runs_{}.pt'.format(args.model, args.use_kg, run))



    writer.close()
    top_1 = np.array(record['top1'])
    top_5 = np.array(record['top5'])
    top_10 = np.array(record['top10'])
    top_20 = np.array(record['top20'])
    mrr = np.array(record['mrr'])
    avg_rank = np.array(record['avg_rank'])
    top_1_least = np.array(record['top1_least'])
    top_5_least = np.array(record['top5_least'])
    top_10_least = np.array(record['top10_least'])
    top_20_least = np.array(record['top20_least'])
    mrr_least = np.array(record['mrr_least'])
    avg_rank_least = np.array(record['avg_rank_least'])

    # print('top_k:{:.4f} ± {:.4f}'.format(top_k.mean(axis=0), top_k.std(axis=0)))
    # print('acc:{:.4f} ± {:.4f}'.format(acc.mean(axis=0), acc.std(axis=0)))
    # print('auc:{:.4f} ± {:.4f}'.format(auc.mean(axis=0), auc.std(axis=0)))
    # print('ap:{:.4f} ± {:.4f}'.format(ap.mean(axis=0), ap.std(axis=0)))
    # print('f1:{:.4f} ± {:.4f}'.format(f1.mean(axis=0), f1.std(axis=0)))
    print('top_1:{:.4f} ± {:.4f}'.format(top_1.mean(axis=0), top_1.std(axis=0)))
    print('top_5:{:.4f} ± {:.4f}'.format(top_5.mean(axis=0), top_5.std(axis=0)))
    print('top_10:{:.4f} ± {:.4f}'.format(top_10.mean(axis=0), top_10.std(axis=0)))
    print('top_20:{:.4f} ± {:.4f}'.format(top_20.mean(axis=0), top_20.std(axis=0)))
    print('mrr:{:.4f} ± {:.4f}'.format(mrr.mean(axis=0), mrr.std(axis=0)))
    print('avg_rank:{:.4f} ± {:.4f}'.format(avg_rank.mean(axis=0), avg_rank.std(axis=0)))
    print('top_1_least:{:.4f} ± {:.4f}'.format(top_1_least.mean(axis=0), top_1_least.std(axis=0)))
    print('top_5_least:{:.4f} ± {:.4f}'.format(top_5_least.mean(axis=0), top_5_least.std(axis=0)))
    print('top_10_least:{:.4f} ± {:.4f}'.format(top_10_least.mean(axis=0), top_10_least.std(axis=0)))
    print('top_20_least:{:.4f} ± {:.4f}'.format(top_20_least.mean(axis=0), top_20_least.std(axis=0)))
    print('mrr_least:{:.4f} ± {:.4f}'.format(mrr_least.mean(axis=0), mrr_least.std(axis=0)))
    print('avg_rank_least:{:.4f} ± {:.4f}'.format(avg_rank_least.mean(axis=0), avg_rank_least.std(axis=0)))

    with open('results/{}_useKG_{}_GAT_model_{}_kg_train{}_gat_lr_{}.txt'.format(args.city, args.use_kg,args.gat_model, args.num_epochs, args.gat_lr), 'w') as f:
        f.write('top_1:{:.4f} ± {:.4f}'.format(top_1.mean(axis=0), top_1.std(axis=0)))
        f.write('top_5:{:.4f} ± {:.4f}'.format(top_5.mean(axis=0), top_5.std(axis=0)))
        f.write('top_10:{:.4f} ± {:.4f}'.format(top_10.mean(axis=0), top_10.std(axis=0)))
        f.write('top_20:{:.4f} ± {:.4f}'.format(top_20.mean(axis=0), top_20.std(axis=0)))
        f.write('mrr:{:.4f} ± {:.4f}'.format(mrr.mean(axis=0), mrr.std(axis=0)))
        f.write('avg_rank:{:.4f} ± {:.4f}'.format(avg_rank.mean(axis=0), avg_rank.std(axis=0)))
        f.write('top_1_least:{:.4f} ± {:.4f}'.format(top_1_least.mean(axis=0), top_1_least.std(axis=0)))
        f.write('top_5_least:{:.4f} ± {:.4f}'.format(top_5_least.mean(axis=0), top_5_least.std(axis=0)))
        f.write('top_10_least:{:.4f} ± {:.4f}'.format(top_10_least.mean(axis=0), top_10_least.std(axis=0)))
        f.write('top_20_least:{:.4f} ± {:.4f}'.format(top_20_least.mean(axis=0), top_20_least.std(axis=0)))
        f.write('mrr_least:{:.4f} ± {:.4f}'.format(mrr_least.mean(axis=0), mrr_least.std(axis=0)))
        f.write('avg_rank_least:{:.4f} ± {:.4f}'.format(avg_rank_least.mean(axis=0), avg_rank_least.std(axis=0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='NYC', help='City name,choices=[NYC, TKY, SP, JK, KL]')
    parser.add_argument('--model', type=str, default='HKGAT', help='Model name')
    parser.add_argument('--task', type=str, default='check_in', help='Task name,choices=[friendship, check_in]')

    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training ratio')

    parser.add_argument('--runs', type=int, default=3, help='Number of runs')

    parser.add_argument('--cuda', type=int, default=2, help='GPU ID')


    # KG parameters
    parser.add_argument('--kg_model', type=str, default='HypE',
                        help='KG model, choices=[TransE, DistMult, ComplEx, HypE, HSimplE]')
    parser.add_argument('--kg_lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--neg_ratio', type=int, default=10, help='Negative ratio')
    parser.add_argument('--num_epochs', type=int, default=100 , help='Number of epochs')
    parser.add_argument('--kg_batch_size', type=int, default=512, help='KG Batch size')
    parser.add_argument('--use_kg', type=bool, default=False, help='Use KG or not')

    # GAT parameters
    parser.add_argument('--gat_model', type=str, default='HCHA',
                        help='GAT model, choices=[HCHA, CEGAT, CEGCN]')
    parser.add_argument('--gat_train_iter', type=int, default=500, help='Number of GAT training iterations')
    parser.add_argument('--gat_batch_size', type=int, default=512, help='GAT Batch size')
    parser.add_argument('--gat_lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--gat_weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--emb_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--rel_emb_dim', type=int, default=256, help='Relation embedding dimension')
    parser.add_argument('--heads', type=int, default=1, help='Number of heads')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--normalization', default='ln', help='Normalization method')
    parser.add_argument('--hid_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--output_heads', type=int, default=1, help='Output heads')
    parser.add_argument('--use_attention', type=bool, default=False, help='Use attention or not')

    # test parameters
    parser.add_argument('--test', type=bool, default=False, help='Test mode')
    parser.add_argument('--pretrained', type=bool, default=False, help='Load pretrained model')
    parser.add_argument('--save_model', type=int, default=500, help='Save model every x epochs')
    parser.add_argument('--ablation', type=bool, default=False, help='Ablation study')

    args = parser.parse_args()
    args.ablation_list = ['poi_price', 'poi_count', 'poi_category_one', 'category']

    if args.test:
        # test(args)
        pass
    else:
        train(args)


if __name__ == '__main__':
    main()
