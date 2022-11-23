import random
import torch
import torch.nn as nn


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

    