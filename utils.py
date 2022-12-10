import math
import random

import numpy as np
import torch
from torch.nn import functional as F

def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic

def negative_sample_friendship_test(friendship, friendship_all, user_index):
    friendship_data = [[], []]
    conflict = set(friendship_all)
    samples = len(friendship)
    neg_samples = []
    labels = [1] * len(friendship)

    for i in range(0, samples):
        # x = friendship[random.randint(0, len(friendship) - 1)][random.randint(0, 1)]
        # x = random.randint(user_index['start'], user_index['end'])
        x = friendship[i][0]
        y = random.randint(user_index['start'], user_index['end'])

        while (x, y) in conflict or (y, x) in conflict or x == y:
            # x = friendship[random.randint(0, len(friendship) - 1)][random.randint(0, 1)]
            # x = random.randint(user_index['start'], user_index['end']
            y = random.randint(user_index['start'], user_index['end'])

        neg_samples.append((x, y))
        # neg_samples.append((y, x))
        # labels.append(0)
        labels.append(0)

    sampled_friendship = friendship + neg_samples
    # shuffle the training data(list)
    index = [i for i in range(len(sampled_friendship))]
    random.shuffle(index)
    sampled_friendship = [sampled_friendship[i] for i in index]
    friendship_data[0] = [i[0] for i in sampled_friendship]
    friendship_data[1] = [i[1] for i in sampled_friendship]
    friendship_data = torch.LongTensor(friendship_data)
    labels = torch.FloatTensor([labels[i] for i in index])

    return friendship_data, labels

def negative_sample_friendship_train(friendship, friendship_all,user_index, sample_ratio=1):
    friendship_data = [[], []]
    conflict = set(friendship_all)
    samples = len(friendship) * sample_ratio
    neg_samples = []
    labels = [1] * len(friendship)

    for i in range(0, samples):
        # x = friendship[random.randint(0, len(friendship) - 1)][random.randint(0, 1)]
        x = random.randint(user_index['start'], user_index['end'])
        # x = friendship[i][0]
        y = random.randint(user_index['start'], user_index['end'])

        while (x, y) in conflict or (y, x) in conflict or x == y:
            # x = friendship[random.randint(0, len(friendship) - 1)][random.randint(0, 1)]
            x = random.randint(user_index['start'], user_index['end'])
            y = random.randint(user_index['start'], user_index['end'])
        neg_samples.append((x, y))
        neg_samples.append((y, x))
        labels.append(0)
        labels.append(0)

    sampled_friendship = friendship + neg_samples
    # shuffle the training data(list)
    index = [i for i in range(len(sampled_friendship))]
    random.shuffle(index)
    sampled_friendship = [sampled_friendship[i] for i in index]
    friendship_data[0] = [i[0] for i in sampled_friendship]
    friendship_data[1] = [i[1] for i in sampled_friendship]
    friendship_data = torch.LongTensor(friendship_data)
    labels = torch.FloatTensor([labels[i] for i in index])

    return friendship_data, labels

def negative_sample_check_in_train(check_ins, check_in_all, poi_index, time1_index, time2_index, sample_ratio=1):

    conflict = set(check_in_all)
    neg_samples = []
    labels = [1] * len(check_ins)

    for check_in in check_ins:
        for k in range(0, sample_ratio):
            poi = random.randint(poi_index['start'], poi_index['end'])
            time1 = random.randint(time1_index['start'], time1_index['end'])
            time2 = random.randint(time2_index['start'], time2_index['end'])

            while (check_in[0], check_in[1], check_in[2], poi) in conflict:
                poi = random.randint(poi_index['start'], poi_index['end'])

            neg_samples.append([check_in[0], check_in[1], check_in[2], poi])

            while (check_in[0], check_in[1], time1, check_in[3]) in conflict:
                time1 = random.randint(time1_index['start'], time1_index['end'])

            neg_samples.append([check_in[0], check_in[1], time1, check_in[3]])

            while (check_in[0], time2, check_in[2], check_in[3]) in conflict:
                time2 = random.randint(time2_index['start'], time2_index['end'])

            neg_samples.append([check_in[0], time2, check_in[2], check_in[3]])

            labels.append(0)
            labels.append(0)
            labels.append(0)

    check_in_data = np.vstack((check_ins, neg_samples))

    check_in_data = torch.LongTensor(check_in_data)
    labels = torch.FloatTensor(labels)

    return check_in_data, labels

def test_topK_friendship(v, user_index, friendship, friendship_all, topK=10):
    user_number = user_index['end'] - user_index['start'] + 1
    v = v[0:user_number]

    v_norm = torch.norm(v, dim=-1, keepdim=True)

    dot_numerator = torch.mm(v, v.t())
    dot_denominator = torch.mm(v_norm, v_norm.t())
    sim = (dot_numerator / dot_denominator)

    # vidx = args.vidx.cpu().numpy().tolist()

    k = topK
    friend_list = [[], []]

    for i in range(len(friendship)):
        friend_list[0].append(friendship[i][0])
        friend_list[1].append(friendship[i][1])

    # for i in range(0, 2 * args.friend_edge_num, 2):
    #     sim[vidx[i]][vidx[i + 1]] = -1
    #     sim[vidx[i + 1]][vidx[i]] = -1

    true_friend = set(zip(friend_list[0], friend_list[1]))

    for i in range(len(friendship_all)):
        if (friendship_all[i][0], friendship_all[i][1]) not in true_friend:
            sim[friendship_all[i][0]][friendship_all[i][1]] = -1
            sim[friendship_all[i][1]][friendship_all[i][0]] = -1

    for i in range(0, user_number):
        sim[i][i] = -1

    real = 0
    tot = user_number
    for i in range(0, user_number):
        sim_i = sim[i]
        s = sim_i.argsort()[-k:]
        if i not in friend_list[0] and i not in friend_list[1]:
            tot -= 1
            continue
        l1 = [i] * k
        l2 = s.cpu().detach().numpy().tolist()
        predict = list(zip(l1, l2))
        for (friend_1, friend_2) in predict:
            if (friend_1, friend_2) in true_friend or (friend_2, friend_1) in true_friend:
                real += 1
                break

    topK = real / tot

    return topK

def check_in_topK(predictions, labels, topK=10):
    predictions = predictions.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    real = 0
    tot = len(labels)
    for i in range(0, len(labels)):
        sim_i = predictions[i]
        s = sim_i.argsort()[-topK:]
        if labels[i] in s:
            real += 1
    topK = real / tot
    return topK



def sample_negative_relation_batch(pos_batch, entity_num, neg_ratio, max_arity):
    relation_type = [relation[0] for relation in pos_batch]
    relation = [list(relation[1]) for relation in pos_batch]
    relation = [np.array([relation_type[i]] + relation[i] + [0]) for i in range(len(relation))]
    # pos_batch = np.append(relation, np.zeros((len(relation), 1)), axis=1).astype("int")
    arities = [len(r) - 2 for r in relation]

    neg_batch = []
    for i, c in enumerate(relation):
        c = np.array(list(c) + [entity_num] * (max_arity - arities[i]))
        neg_batch.append(neg_each(np.repeat([c], neg_ratio * arities[i] + 1, axis=0), arities[i], entity_num, neg_ratio))
    labels = []
    batch = []
    arities_new = []
    for i in range(len(neg_batch)):
        labels.append(1)
        labels = labels + [0] * (neg_ratio * arities[i])
        arities_new = arities_new + [arities[i]] * (neg_ratio * arities[i] + 1)
        # labels.append([1] + [0] * (neg_ratio * arities[i]))
        for j in range(len(neg_batch[i])):
            batch.append(neg_batch[i][j][:-1])
    labels = np.array(labels)
    batch = np.array(batch)

    ms = np.zeros((len(batch), max_arity))
    bs = np.ones((len(batch), max_arity))
    for i in range(len(batch)):
        ms[i][0:arities_new[i]] = 1
        bs[i][0:arities_new[i]] = 0
    return batch, labels, ms, bs

def neg_each(arr, arity, entity_num, neg_ratio):
    for a in range(arity):
        arr[a* neg_ratio + 1:(a + 1) * neg_ratio + 1, a + 1] = np.random.randint(low=1, high=entity_num, size=neg_ratio)
    return arr

def padd(a, max_length):
    b = F.pad(a, (0, max_length - len(a)), 'constant', -math.inf)
    return b

def decompose_predictions(targets, predictions, max_length):
    positive_indices = np.where(targets > 0)[0]
    seq = []
    for ind, val in enumerate(positive_indices):
        if(ind == len(positive_indices)-1):
            seq.append(padd(predictions[val:], max_length))
        else:
            seq.append(padd(predictions[val:positive_indices[ind + 1]], max_length))
    return seq

def padd_and_decompose(targets, predictions, max_length):
    seq = decompose_predictions(targets, predictions, max_length)
    return torch.stack(seq)



