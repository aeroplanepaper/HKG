import copy
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import scipy.io as scio
import os
import json
import time
import geohash
from itertools import combinations

from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn as nn
import torch

import utils


def load_data(city):
    # 1). Load basic info included in the dataset.
    # def load_basic_info():
    #     df_checkins = pd.read_csv('./data/raw/dataset_WWW2019/dataset_WWW_Checkins_anonymized.txt', sep='\t',
    #                               header=None)
    #     df_checkins.columns = ['user_id', 'Venue_id', 'utc_time', 'Timezone_offset']
    #
    #     df_poi = pd.read_csv('./data/raw/dataset_WWW2019/raw_POIs.txt', sep='\t', header=None)
    #     df_poi.columns = ['Venue_id', 'latitude', 'longitude', 'category', 'Country_code']
    #
    #     df_friendship_old = pd.read_csv('./data/raw/dataset_WWW2019/dataset_WWW_friendship_old.txt', sep='\t',
    #                                     header=None)
    #     df_friendship_old.columns = ['user_id1', 'user_id2']
    #
    #     df_friendship_new = pd.read_csv('./data/raw/dataset_WWW2019/dataset_WWW_friendship_new.txt', sep='\t',
    #                                     header=None)
    #     df_friendship_new.columns = ['user_id1', 'user_id2']
    #     df_friendship = pd.merge(df_friendship_old, df_friendship_new, how='outer')
    #
    #     # print basic info
    #     print('Number of users: ', len(df_checkins['user_id'].unique()))
    #     print('Number of POIs: ', len(df_checkins['Venue_id'].unique()))
    #     print('Number of checkins: ', len(df_checkins))
    #     print('Number of friendship_old: ', len(df_friendship_old))
    #     print('Number of friendship_new: ', len(df_friendship_new))
    #     print('Number of friendship: ', len(df_friendship))
    #     print('Number of countries: ', len(df_poi['Country_code'].unique()))
    #     print('Number of categories: ', len(df_poi['category'].unique()))
    #
    #     return df_checkins, df_poi, df_friendship

    # def load_city_info():
    #     # 2). Load the checkins data (Only keep the checkins in the 4 selected cities)
    #     nyc_data = scio.loadmat('./data/raw/mat/dataset_connected_NYC.mat')
    #     tky_data = scio.loadmat('./data/raw/mat/dataset_connected_TKY.mat')
    #     sp_data = scio.loadmat('./data/raw/mat/dataset_connected_SaoPaulo.mat')
    #     jk_data = scio.loadmat('./data/raw/mat/dataset_connected_Jakarta.mat')
    #
    #     return nyc_data, tky_data, sp_data, jk_data
    #
    # def process_city_info(city_data, all_poi: pd.DataFrame):
    #     df_checkin = pd.DataFrame(city_data['selected_checkins'])  # (105961, 4)
    #     df_checkin.columns = ['user_id', 'time', 'Venue_id', 'Venue_category']
    #
    #     df_friend_new = pd.DataFrame(city_data['friendship_new'])  # (10545, 2)
    #     df_friend_old = pd.DataFrame(city_data['friendship_old'])  # (8723, 2)
    #     df_friend = pd.merge(df_friend_new, df_friend_old, how='outer')  # (19268, 2)
    #
    #     for i in range(len(df_checkin['Venue_id'])):
    #         df_checkin['Venue_id'][i] = (all_poi.iloc[df_checkin['Venue_id'][i]][0], df_checkin['Venue_id'][i])
    #         # df_checkin['Venue_category'][i] = all_poi.iloc[df_checkin['Venue_id'][i][1]][3]
    #
    #     return df_checkin, df_friend, df_checkin['Venue_id'].unique()

    def load_city_data(city):
        path = "./data/raw/follow_mat/" + city + "/"
        check_in = pd.read_csv(path + "checkins_" + city + "_follow_mat.csv")
        poi_data = pd.read_csv(path + "POI_" + city + "_follow_mat.csv")

        print(city + " check_in number: ", len(check_in))
        users = set(list(check_in['userid']))
        print(city + " user number: ", len(users))
        venues = set(list(check_in['Venue_id']))
        print(city + " venues number: ", len(venues))

        friend_old = np.load(path + "friend_old.npy")
        friend_new = np.load(path + "friend_new.npy")
        friend = np.vstack([friend_old, friend_new])
        # inverse_friend = np.array([friend[:, 1], friend[:, 0]]).T
        # friend = np.vstack([friend, inverse_friend])
        return check_in, poi_data, users, venues, friend

    def process_city_data(city, check_in, poi_data, users, venues, friend):
        # process friendship
        friend = pd.DataFrame(friend)
        friend.columns = ['user1', 'user2']
        friend.drop_duplicates(subset=['user1', 'user2'], keep='first', inplace=True)
        friend = friend.reset_index(drop=True)

        friend_list = []
        for index, row in friend.iterrows():
            if row['user1'] != row['user2'] and row['user1'] in users and row['user2'] in users:
                friend_list.append([row['user1'], row['user2']])
        print(city + " friendship number: ", len(friend_list))

        # recode users
        users_sort = list(sorted(list(users)))
        users_dic = {}
        for i in range(len(users_sort)):
            users_dic[users_sort[i]] = i

        user_index = {}
        user_index['start'] = 0
        user_index['end'] = len(users_sort) - 1

        # recode venues
        venues_sort = list(sorted(list(venues)))
        venues_dic = {}
        for i in range(len(venues_sort)):
            venues_dic[venues_sort[i]] = i + len(users_sort)

        poi_index = {}
        poi_index['start'] = len(users_sort)
        poi_index['end'] = len(users_sort) + len(venues_sort) - 1

        current_index = len(users_sort) + len(venues_sort) - 1
        # recode friendship
        friend_list_index = []
        for i in range(len(friend_list)):
            friend_list_index.append((users_dic[friend_list[i][0]], users_dic[friend_list[i][1]]))

        poi_data['venues_index'] = None
        poi_data_venue_index = []
        for index, row in poi_data.iterrows():
            poi_data_venue_index.append(venues_dic[row['Venue_id']])
        poi_data['venues_index'] = poi_data_venue_index

        poi_data = poi_data.sort_values('Venue_id', ascending=True, inplace=False)
        poi_data = poi_data.reset_index(drop=True)
        poi_lat_lon = {}  # {poi_index:(lat,lon)}
        for index, row in poi_data.iterrows():
            poi_lat_lon[row['venues_index']] = (row['latitude'], row['longitude'])

        poi_location = {}
        all_geo_4 = {}
        all_geo_5 = {}
        all_geo_6 = {}

        geo_relations = {}
        count = 0
        for poi in poi_lat_lon.keys():
            location = poi_lat_lon[poi]
            geohash_4, geohash_5, geohash_6 = transform_location(location)
            if geohash_4 not in all_geo_4:
                all_geo_4[geohash_4] = len(all_geo_4) + 1
            if geohash_5 not in all_geo_5:
                all_geo_5[geohash_5] = len(all_geo_5) + 1
            if geohash_6 not in all_geo_6:
                all_geo_6[geohash_6] = len(all_geo_6) + 1
            poi_location[poi] = {}
            poi_location[poi]['geohash_4'] = geohash_4
            poi_location[poi]['geohash_5'] = geohash_5
            poi_location[poi]['geohash_6'] = geohash_6
            geo_relations[count] = [geohash_4, geohash_5, geohash_6]
            count += 1

        for geo_4 in all_geo_4.keys():
            all_geo_4[geo_4] = all_geo_4[geo_4] + current_index
        current_index += len(all_geo_4)

        for geo_5 in all_geo_5.keys():
            all_geo_5[geo_5] = all_geo_5[geo_5] + current_index
        current_index += len(all_geo_5)

        for geo_6 in all_geo_6.keys():
            all_geo_6[geo_6] = all_geo_6[geo_6] + current_index
        current_index += len(all_geo_6)

        for poi in poi_location.keys():
            poi_location[poi]['geohash_4'] = all_geo_4[poi_location[poi]['geohash_4']]
            poi_location[poi]['geohash_5'] = all_geo_5[poi_location[poi]['geohash_5']]
            poi_location[poi]['geohash_6'] = all_geo_6[poi_location[poi]['geohash_6']]

        for geo in geo_relations.keys():
            geo_relations[geo][0] = all_geo_4[geo_relations[geo][0]]
            geo_relations[geo][1] = all_geo_5[geo_relations[geo][1]]
            geo_relations[geo][2] = all_geo_6[geo_relations[geo][2]]


    # encode the lat and lon to geohash

        check_in = check_in.sort_values('Venue_id', ascending=True, inplace=False)
        check_in = check_in.reset_index(drop=True)
        check_in['users_index'] = None
        check_in['venues_index'] = None
        # check_in['lat_lon'] = None
        check_in_users_index = []
        check_in_venues_index = []
        # check_in_lat_lon = []
        check_in_geohash_4 = []
        check_in_geohash_5 = []
        check_in_geohash_6 = []

        for index, row in check_in.iterrows():
            check_in_users_index.append(users_dic[row['userid']])
            check_in_venues_index.append(venues_dic[row['Venue_id']])
            # check_in_lat_lon.append(poi_lat_lon[venues_dic[row['Venue_id']]])
            check_in_geohash_4.append(poi_location[venues_dic[row['Venue_id']]]['geohash_4'])
            check_in_geohash_5.append(poi_location[venues_dic[row['Venue_id']]]['geohash_5'])
            check_in_geohash_6.append(poi_location[venues_dic[row['Venue_id']]]['geohash_6'])


        check_in['users_index'] = check_in_users_index
        check_in['venues_index'] = check_in_venues_index
        # check_in['lat_lon'] = check_in_lat_lon
        check_in['geohash_4'] = check_in_geohash_4
        check_in['geohash_5'] = check_in_geohash_5
        check_in['geohash_6'] = check_in_geohash_6

        # check_in['local_year'] = None
        check_in['local_month'] = None
        check_in['hour_period'] = None
        # check_in_local_year = []
        check_in_local_month = []
        check_in_hour_period = []

        for index, row in check_in.iterrows():
            time_1 = row['utc_time'][:-10] + row['utc_time'][-4:]
            timezone_offset = row['Timezone_offset']
            struct_time = time.mktime(time.strptime(time_1, "%a %b %d  %H:%M:%S %Y")) + timezone_offset * 60
            localtime = time.localtime(struct_time)  # 返回元组
            # check_in_local_year.append(localtime[0])
            check_in_local_month.append(localtime[1])
            check_in_hour_period.append(time_partition(localtime[6], localtime[3], localtime[4]))

        time_hour_dic = {}
        tot_hour_period = max(check_in_hour_period)

        for i in range(len(check_in_hour_period)):
            if check_in_hour_period[i] not in time_hour_dic:
                time_hour_dic[check_in_hour_period[i]] = check_in_hour_period[i] + current_index
            check_in_hour_period[i] = time_hour_dic[check_in_hour_period[i]]

        current_index += tot_hour_period

        time_month_dic = {}
        tot_month = max(check_in_local_month)

        for i in range(len(check_in_local_month)):
            if check_in_local_month[i] not in time_month_dic:
                time_month_dic[check_in_local_month[i]] = check_in_local_month[i] + current_index
            check_in_local_month[i] = time_month_dic[check_in_local_month[i]]

        current_index += tot_month

        # check_in['local_year'] = check_in_local_year
        check_in['local_month'] = check_in_local_month
        check_in['hour_period'] = check_in_hour_period

        user_trajectory_index = {}
        check_in = check_in.sort_values('users_index', ascending=True, inplace=False)
        user_group = check_in.groupby("users_index")
        user_group = list(user_group)
        for i in range(len(user_group)):  # 每一个group2 都是用户
            df = pd.DataFrame(user_group[i][1])  # 每个用户的访问轨迹
            df = df.reset_index(drop=True)
            user_trajectory_index[df['users_index'][0]] = set(list(df['venues_index']))
        check_in = check_in.drop(columns=['userid', 'Venue_id', 'utc_time', 'Timezone_offset'])
        return check_in, friend_list_index, user_trajectory_index, venues_dic, time_hour_dic, time_month_dic, \
               geo_relations, current_index, user_index, poi_index

    def time_partition(day, hour, min):
        # days [0-6] per 1 day, hours [0-23] per 1 hour, minutes [0-1] per 30 minutes
        # the time will be partied into 7 * 24 * 2 index
        if 0 <= min < 30:
            return day * 48 + (hour + 1) * 2 - 1
        else:
            return day * 48 + (hour + 1) * 2

    def dump_city_data(city, check_in, friend_list_index, user_trajectory_index):
        path = './data/processed' + city + '/'
        f = open(path + 'check_in.pkl')
        pickle.dump(check_in, f)
        f.close()
        f = open(path + 'friend_list_index.pkl')
        pickle.dump(friend_list_index, f)
        f.close()
        f = open(path + 'user_trajectory_index.pkl')
        pickle.dump(user_trajectory_index, f)
        f.close()

    def load_extra_poi_info(city, venues, venues_dic, time_hour_dic, time_month_dic, current_index):
        base_dir = './data/raw/Venue_detail/'
        path = base_dir + city + '/'
        file_names = os.listdir(path)
        file_names = zip(file_names, range(len(file_names)))
        file_names = dict(file_names)
        all_side_info = {}

        all_contact = {}
        all_category_level_one = {}
        all_category_level_two = {}

        all_tip_count = []
        all_price_tier = {}
        all_like_count = []
        all_rating = []
        all_photos_count = []

        tip_count_cut_num = 6
        like_count_cut_num = 6
        rating_cut_num = 6
        photos_count_cut_num = 6

        for file_name in file_names:
            # load extra poi info from json file
            with open(path + file_name, 'r') as f:

                extra_poi_info = json.load(f)
                side_info = {}
                # exact useful poi information
                # might useful:
                # id,
                # contact{phone, twitter, facebook, instagram},
                # categories[{name, icon{prefix".../cate1/cate2"}}],
                # stats{tipCount},
                # price{tier},
                # likes{count},
                # rating,
                # photos{count},
                # hours{timeframes[{days, open[{renderedtime}]}]}
                # popular{timeframes[{days, open[{renderedtime}]}]}
                # if 'id' not in extra_poi_info:
                #     # todo: 4ad83e6ff964a520eb1021e3.json airport 只有category
                #     print(file_name)
                #     continue

                id = extra_poi_info['id']
                if id not in venues or id in all_side_info:
                    continue

                contacts_load = extra_poi_info['contact']
                contacts_processed = set()

                for contact in contacts_load.keys():
                    if contact == 'facebookUsername' or contact == 'facebookName' or contact == 'formattedPhone':
                        continue
                    if contact not in all_contact:
                        all_contact[contact] = len(all_contact) + 1
                    contacts_processed.add(all_contact[contact])

                if len(contacts_processed) == 0:
                    contacts_processed.add(0) # 5 is the index of 'None'

                categories = extra_poi_info['categories']
                category_level_one_processed = set()
                category_level_two_processed = set()

                for category_info in categories:
                    category_level_one = category_info['icon']['prefix'].split('/')[5]
                    if category_level_one not in all_category_level_one:
                        all_category_level_one[category_level_one] = len(all_category_level_one) + 1
                    category_level_one_processed.add(all_category_level_one[category_level_one])

                    category_level_two = category_info['name']
                    if category_level_two not in all_category_level_two:
                        all_category_level_two[category_level_two] = {}
                        all_category_level_two[category_level_two]['index'] = len(all_category_level_two)
                        all_category_level_two[category_level_two]['parent'] = all_category_level_one[
                            category_level_one]
                    category_level_two_processed.add(all_category_level_two[category_level_two]['index'])

                tip_count = extra_poi_info['stats']['tipCount']
                all_tip_count.append(tip_count)
                # tip_count_processed = process_tip_count(tip_count)

                price_tier = extra_poi_info['price']['tier'] if 'price' in extra_poi_info.keys() else None
                if price_tier not in all_price_tier:
                    all_price_tier[price_tier] = 1
                else:
                    all_price_tier[price_tier] += 1
                price_tier_processed = 1 if price_tier is None else price_tier + 1

                like_count = extra_poi_info['likes']['count'] if 'likes' in extra_poi_info.keys() else None
                if like_count is None or like_count == -1:
                    like_count = 0
                all_like_count.append(like_count)

                rating = extra_poi_info['rating'] if 'rating' in extra_poi_info.keys() else None
                all_rating.append(rating)

                photos_count = extra_poi_info['photos']['count']
                all_photos_count.append(photos_count)

                # todo: hours, popular

                side_info['contact'] = contacts_processed
                side_info['category_level_one'] = category_level_one_processed
                side_info['category_level_two'] = category_level_two_processed
                side_info['tip_count'] = tip_count
                side_info['price_tier'] = price_tier_processed
                side_info['like_count'] = like_count
                side_info['rating'] = rating
                side_info['photos_count'] = photos_count
                all_side_info[venues_dic[id]] = side_info

        all_tip_count = np.array(all_tip_count)
        all_like_count = np.array(all_like_count)
        all_rating = np.array(all_rating)
        all_photos_count = np.array(all_photos_count)

        tip_count_cut = cut_all_count(all_tip_count, cut_num=tip_count_cut_num)
        like_count_cut = cut_all_count(all_like_count, cut_num=like_count_cut_num)
        rating_cut = cut_all_count(all_rating[all_rating != None], cut_num=rating_cut_num)
        photos_count_cut = cut_all_count(all_photos_count, cut_num=photos_count_cut_num)

        for key in all_side_info.keys():
            all_side_info[key]['tip_count'] = process_counts(all_side_info[key]['tip_count'], tip_count_cut)
            all_side_info[key]['like_count'] = process_counts(all_side_info[key]['like_count'], like_count_cut)
            if all_side_info[key]['rating'] is None:
                all_side_info[key]['rating'] = 1
            else:
                all_side_info[key]['rating'] = process_counts(all_side_info[key]['rating'], rating_cut) + 1
            all_side_info[key]['photos_count'] = process_counts(all_side_info[key]['photos_count'], photos_count_cut)

        # recode side info

        # print(current_index)
        for key in all_side_info.keys():
            all_side_info[key]['contact'] = {current_index + val + 1 for val in all_side_info[key]['contact']}

        current_index += len(all_contact) + 1

        for key in all_category_level_one:
            all_category_level_one[key] = current_index + all_category_level_one[key]

        for key in all_category_level_two:
            all_category_level_two[key]['parent'] = current_index + all_category_level_two[key]['parent']

        for key in all_side_info.keys():
            all_side_info[key]['category_level_one'] = {current_index + val for val in
                                                        all_side_info[key]['category_level_one']}

        current_index += len(all_category_level_one)

        for key in all_category_level_two:
            all_category_level_two[key]['index'] = current_index + all_category_level_two[key]['index']

        for key in all_side_info.keys():
            all_side_info[key]['category_level_two'] = {current_index + val for val in
                                                        all_side_info[key]['category_level_two']}

        current_index += len(all_category_level_two)

        for key in all_side_info.keys():
            all_side_info[key]['tip_count'] = current_index + all_side_info[key]['tip_count']

        current_index += tip_count_cut_num

        for key in all_side_info.keys():
            all_side_info[key]['price_tier'] = current_index + all_side_info[key]['price_tier']

        current_index += 5  # 5 price_tier in tot(1 is None)

        for key in all_side_info.keys():
            all_side_info[key]['like_count'] = current_index + all_side_info[key]['like_count']

        current_index += like_count_cut_num

        for key in all_side_info.keys():
            all_side_info[key]['rating'] = current_index + all_side_info[key]['rating']

        current_index += rating_cut_num

        for key in all_side_info.keys():
            all_side_info[key]['photos_count'] = current_index + all_side_info[key]['photos_count']

        current_index += photos_count_cut_num

        print("Total vertex number", current_index)

        print(city + ' side information number: ', len(all_side_info))
        return all_side_info, all_category_level_two, current_index

    def cut_all_count(all_count, cut_num):
        all_count = np.sort(all_count)
        cut_len = len(all_count) // cut_num
        cuts = []
        total = 0
        for count in all_count:
            total += 1
            if total == cut_len:
                cuts.append(count)
                total = 0
        return cuts

    def process_counts(count, cuts):
        index = 1
        for cut in cuts:
            if count > cut:
                index += 1
            else:
                break
        return index

    def transform_location(location):
        lat, lng = location[0], location[1]
        hash_4 = geohash.encode(lat, lng, precision=4)
        hash_5 = geohash.encode(lat, lng, precision=5)
        hash_6 = geohash.encode(lat, lng, precision=6)
        return hash_4, hash_5, hash_6

    def extract_relations(city, check_ins:pd.DataFrame, friend,
                      time_hour_dic, time_month_dic, poi_details, all_categories):
        relations = {}
        hyper_edges = {}
        total_relation = 0
        check_in_relation = {}
        friendship_relation = {}
        categories_relation = {}
        time_relation = {}
        contact_relation = {}
        poi_category_one_relation = {}
        poi_category_two_relation = {}
        poi_counts_relation = {}
        poi_price_relation = {}

        check_ins = np.array(check_ins)
        for i in range(len(check_ins)):
            check_in_relation[i] = check_ins[i]
        relations['check_in'] = check_in_relation
        hyper_edges['check_in'] = check_in_relation
        # type 1 : check in
        total_relation += len(check_in_relation)

        for i in range(len(friend)):
            friendship_relation[i] = friend[i]
        relations['friendship'] = friendship_relation
        hyper_edges['friendship'] = friendship_relation
        # type 2 : friendship
        total_relation += len(friendship_relation)

        count = 0
        for key in all_categories.keys():
            category_level2 = all_categories[key]['index']
            category_level1 = all_categories[key]['parent']
            categories_relation[count] = [category_level1, category_level2]
            count += 1
        relations['category'] = categories_relation
        # type 3 : category
        total_relation += count

        count = 0
        for month in time_month_dic.keys():
            month_node = time_month_dic[month]
            for period in time_hour_dic.keys():
                time_relation[count] = [month_node, time_hour_dic[period]]
                count += 1
        relations['time'] = time_relation
        total_relation += count
        # type 4 : time

        count = 0
        for poi in poi_details.keys():
            # poi = poi_details[key]['index']
            contacts = poi_details[poi]['contact']
            category_level_one = poi_details[poi]['category_level_one']
            category_level_two = poi_details[poi]['category_level_two']
            tip_count = poi_details[poi]['tip_count']
            price_tier = poi_details[poi]['price_tier']
            like_count = poi_details[poi]['like_count']
            rating = poi_details[poi]['rating']
            photos_count = poi_details[poi]['photos_count']

            contact_relation[count] = [poi] + [contact for contact in contacts]
            poi_category_one_relation[count] = [poi] + [category for category in category_level_one]
            poi_category_two_relation[count] = [poi] + [category for category in category_level_two]
            poi_counts_relation[count] = [poi, tip_count, like_count, rating, photos_count]
            poi_price_relation[count] = [poi, price_tier]
            count += 1

        relations['poi_category_one'] = poi_category_one_relation
        relations['poi_category_two'] = poi_category_two_relation
        relations['poi_counts'] = poi_counts_relation
        relations['poi_price'] = poi_price_relation

        total_relation += count * 5
        return relations, hyper_edges, total_relation


    def load_all(city):
        # all_checkins, all_poi, all_friendships = load_basic_info()
        # all_checkin_relations = process_checkin(all_checkins)
        # cities = ['NYC', 'TKY', 'SP', 'JK', 'KL']
        # cities = ['NYC']
        # cities = ['IST']  # 数据很多有问题，只有极少的信息

        # for city in cities:
        check_in, poi_data, users, venues, friend = load_city_data(city)

        check_in_processed, friend_processed, user_trajectory_processed, venues_dic, time_hour_dic, \
            time_month_dic, geo_relations, current_index, user_index, poi_index = process_city_data(city, check_in, poi_data, users, venues, friend)

        poi_details, all_categories, total_nodes = load_extra_poi_info(city, venues, venues_dic, time_hour_dic, time_month_dic, current_index)
        relations, hyper_edges, total_relation_num = extract_relations(city, check_in_processed, friend_processed,
                                      time_hour_dic, time_month_dic, poi_details, all_categories)
        relations['poi_geo'] = geo_relations

        # hyper_edges = copy.deepcopy(relations)
        hyper_edges['trajectory'] = user_trajectory_processed

        total_edge_num = total_relation_num + len(user_trajectory_processed)
        total_relation_num += len(geo_relations)

        print('Total edge number: ', total_edge_num)
        print('Total relation number:{}'.format(total_relation_num))
        print('load success')
        return relations, hyper_edges, total_relation_num, total_edge_num, user_index, poi_index, total_nodes

    #load from file
    try:
        # raise Exception
        relations = pickle.load(open('./data/processed/'+city+'/relations.pkl', 'rb'))
        hyper_edges = pickle.load(open('./data/processed/'+city+'/hyper_edges.pkl', 'rb'))
        total_relation_num = pickle.load(open('./data/processed/'+city+'/total_relation_num.pkl', 'rb'))
        total_edge_num = pickle.load(open('./data/processed/'+city+'/total_edge_num.pkl', 'rb'))
        user_index = pickle.load(open('./data/processed/'+city+'/user_index.pkl', 'rb'))
        poi_index = pickle.load(open('./data/processed/'+city+'/poi_index.pkl', 'rb'))
        total_nodes = pickle.load(open('./data/processed/'+city+'/total_nodes.pkl', 'rb'))
        print('load data from file success')
        return relations, hyper_edges, total_relation_num, total_edge_num, user_index, poi_index, total_nodes
    except:
        # print('load failed')
        print('load data from file failed, load from raw data')
        relations, hyper_edges, total_relation_num, total_edge_num, user_index, poi_index, total_nodes = load_all(city)
        pickle.dump(relations, open('./data/processed/'+city+'/relations.pkl', 'wb'))
        pickle.dump(hyper_edges, open('./data/processed/'+city+'/hyper_edges.pkl', 'wb'))
        pickle.dump(total_relation_num, open('./data/processed/'+city+'/total_relation_num.pkl', 'wb'))
        pickle.dump(total_edge_num, open('./data/processed/'+city+'/total_edge_num.pkl', 'wb'))
        pickle.dump(user_index, open('./data/processed/'+city+'/user_index.pkl', 'wb'))
        pickle.dump(poi_index, open('./data/processed/'+city+'/poi_index.pkl', 'wb'))
        pickle.dump(total_nodes, open('./data/processed/'+city+'/total_nodes.pkl', 'wb'))
        print('load data from raw data success')
        return relations, hyper_edges, total_relation_num, total_edge_num, user_index, poi_index, total_nodes

def ConstructV2V(edge_index):
    # Assume edge_index = [V;E], sorted
    """
    For each he, clique-expansion. Note that we DONT allow duplicated edges.
    Instead, we record its corresponding weights.
    We default no self loops so far.
    """
    
    edge_weight_dict = {}
    for he in np.unique(edge_index[1, :]):
        nodes_in_he = np.sort(edge_index[0, :][edge_index[1, :] == he])
        if len(nodes_in_he) == 1:
            continue  # skip self loops
        combs = combinations(nodes_in_he, 2)
        for comb in combs:
            if not comb in edge_weight_dict.keys():
                edge_weight_dict[comb] = 1
            else:
                edge_weight_dict[comb] += 1

# # Now, translate dict to edge_index and norm
#
    new_edge_index = np.zeros((2, len(edge_weight_dict)))
    new_norm = np.zeros((len(edge_weight_dict)))
    cur_idx = 0
    for edge in edge_weight_dict:
        new_edge_index[:, cur_idx] = edge
        new_norm[cur_idx] = edge_weight_dict[edge]
        cur_idx += 1
        
    return new_edge_index, new_norm

def norm_contruction(edge_index,edge_weight, TYPE='V2V'):
    if TYPE == 'V2V':
        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, add_self_loops=True)
    return edge_index, edge_weight

# def add_selfloops(edge_index, entity_num, hyperedge_num):
#     # update so we dont jump on some indices
#     # Assume edge_index = [V;E]. If not, use ExtractV2E()
#
#
#     hyperedge_appear_fre = Counter(edge_index[1])
#     # store the nodes that already have self-loops
#     skip_node_lst = []
#     for edge in hyperedge_appear_fre:
#         if hyperedge_appear_fre[edge] == 1:
#             skip_node = edge_index[0][torch.where(
#                 edge_index[1] == edge)[0].item()]
#             if(skip_node in skip_node_lst):
#                 continue
#             skip_node_lst.append(skip_node.item())
#
#     new_edge_idx = edge_index[1].max() + 1
#     new_edges = torch.zeros(
#         (2, entity_num - len(skip_node_lst)), dtype=edge_index.dtype)
#     tmp_count = 0
#     for i in range(entity_num):
#         if i not in skip_node_lst:
#             new_edges[0][tmp_count] = i
#             new_edges[1][tmp_count] = new_edge_idx
#             new_edge_idx += 1
#             tmp_count += 1
#
#     hyperedge_num = hyperedge_num + entity_num - len(skip_node_lst)
#     edge_index = torch.cat((edge_index, new_edges), dim=1)
#     # Sort along w.r.t. nodes
#     _, sorted_idx = torch.sort(edge_index[0])
#     edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)
#     return edge_index, hyperedge_num


def process_data(args):
    # load data
    relations, edges, total_relations, total_edges, user_index, poi_index, total_nodes = load_data(args.city)
    all_relations = []
    del edges['trajectory']
    
    test_data = {}
    train_data = {}
    test_index = {}
    test_index['friendship'] = set(np.random.choice(len(relations['friendship']), int(len(relations['friendship']) * (1-args.train_ratio)), replace=False))
    test_index['check_in'] = set(np.random.choice(len(relations['check_in']), int(len(relations['check_in']) * (1-args.train_ratio)), replace=False))


    relation_count = 0
    for relation in relations.keys():
        if relation == 'friendship' or relation == 'check_in':
            friendship_relation_idx = relation_count
            test_data[relation] = []
            train_data[relation] = []
            for i in range(len(relations[relation])):
                if i in test_index[relation]:
                    test_data[relation].append(relations[relation][i])
                else:
                    train_data[relation].append(relations[relation][i])
                    all_relations.append([relation_count, relations[relation][i]])
        elif relation not in args.ablation_list:
            for idx in relations[relation].keys():
                all_relations.append([relation_count, relations[relation][idx]])
        relation_count += 1
    # combine all relations

    v_index = []
    e_index = []
    count = 0
    for edge in edges.keys():
        if edge == 'friendship' or edge == 'check_in':
            for i in range(len(edges[edge])):
                if i not in test_index[edge]:
                    # all_edges.append(train_data[edge][i])
                    for v in edges[edge][i]:
                        v_index.append(v)
                        e_index.append(count)
                    count += 1
        else:
            for idx in edges[edge].keys():
                # all_edges.append(edges[edge][idx])
                for v in edges[edge][idx]:
                    v_index.append(v)
                    e_index.append(count)
                count += 1
    # combine all edges

    hyperedge_num = count + 1
    # hypernode_num = np.array(all_relations)

    v_e = np.array([v_index, e_index])
    v_e = v_e.T[np.lexsort(v_e[::-1, :])].T

    return all_relations, relation_count, friendship_relation_idx, v_e, test_data, train_data, user_index, poi_index, hyperedge_num, total_nodes

