import pickle

import numpy as np
import pandas as pd
import scipy.io as scio
import os
import json
import time


def load_data():
    # 1). Load basic info included in the dataset.
    def load_basic_info():
        df_checkins = pd.read_csv('./data/raw/dataset_WWW2019/dataset_WWW_Checkins_anonymized.txt', sep='\t',
                                  header=None)
        df_checkins.columns = ['user_id', 'Venue_id', 'utc_time', 'Timezone_offset']

        df_poi = pd.read_csv('./data/raw/dataset_WWW2019/raw_POIs.txt', sep='\t', header=None)
        df_poi.columns = ['Venue_id', 'latitude', 'longitude', 'category', 'Country_code']

        df_friendship_old = pd.read_csv('./data/raw/dataset_WWW2019/dataset_WWW_friendship_old.txt', sep='\t',
                                        header=None)
        df_friendship_old.columns = ['user_id1', 'user_id2']

        df_friendship_new = pd.read_csv('./data/raw/dataset_WWW2019/dataset_WWW_friendship_new.txt', sep='\t',
                                        header=None)
        df_friendship_new.columns = ['user_id1', 'user_id2']
        df_friendship = pd.merge(df_friendship_old, df_friendship_new, how='outer')

        # print basic info
        print('Number of users: ', len(df_checkins['user_id'].unique()))
        print('Number of POIs: ', len(df_checkins['Venue_id'].unique()))
        print('Number of checkins: ', len(df_checkins))
        print('Number of friendship_old: ', len(df_friendship_old))
        print('Number of friendship_new: ', len(df_friendship_new))
        print('Number of friendship: ', len(df_friendship))
        print('Number of countries: ', len(df_poi['Country_code'].unique()))
        print('Number of categories: ', len(df_poi['category'].unique()))

        return df_checkins, df_poi, df_friendship

    def load_city_info():
        # 2). Load the checkins data (Only keep the checkins in the 4 selected cities)
        nyc_data = scio.loadmat('./data/raw/mat/dataset_connected_NYC.mat')
        tky_data = scio.loadmat('./data/raw/mat/dataset_connected_TKY.mat')
        sp_data = scio.loadmat('./data/raw/mat/dataset_connected_SaoPaulo.mat')
        jk_data = scio.loadmat('./data/raw/mat/dataset_connected_Jakarta.mat')

        return nyc_data, tky_data, sp_data, jk_data

    def process_city_info(city_data, all_poi: pd.DataFrame):
        df_checkin = pd.DataFrame(city_data['selected_checkins'])  # (105961, 4)
        df_checkin.columns = ['user_id', 'time', 'Venue_id', 'Venue_category']

        df_friend_new = pd.DataFrame(city_data['friendship_new'])  # (10545, 2)
        df_friend_old = pd.DataFrame(city_data['friendship_old'])  # (8723, 2)
        df_friend = pd.merge(df_friend_new, df_friend_old, how='outer')  # (19268, 2)

        for i in range(len(df_checkin['Venue_id'])):
            df_checkin['Venue_id'][i] = (all_poi.iloc[df_checkin['Venue_id'][i]][0], df_checkin['Venue_id'][i])
            # df_checkin['Venue_category'][i] = all_poi.iloc[df_checkin['Venue_id'][i][1]][3]

        return df_checkin, df_friend, df_checkin['Venue_id'].unique()

    def load_city_data(city):
        path = "./data/raw/follow_mat/" + city + "/"
        check_in = pd.read_csv(path + "checkins_" + city.lower() + "_follow_mat.csv")
        poi_data = pd.read_csv(path + "POI_" + city.lower() + "_follow_mat.csv")

        print(city + " check_in number: ", len(check_in))
        users = set(list(check_in['userid']))
        print(city + " user number: ", len(users))
        venues = set(list(check_in['Venue_id']))
        print(city + " venues number: ", len(venues))

        friend_old = np.load(path + "friend_old.npy")
        friend_new = np.load(path + "friend_new.npy")
        friend = np.vstack([friend_old, friend_new])

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

        # recode venues
        venues_sort = list(sorted(list(venues)))
        venues_dic = {}
        for i in range(len(venues_sort)):
            venues_dic[venues_sort[i]] = i + len(users_sort)

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

        check_in = check_in.sort_values('Venue_id', ascending=True, inplace=False)
        check_in = check_in.reset_index(drop=True)
        check_in['users_index'] = None
        check_in['venues_index'] = None
        check_in['lat_lon'] = None
        check_in_users_index = []
        check_in_venues_index = []
        check_in_lat_lon = []

        for index, row in check_in.iterrows():
            check_in_users_index.append(users_dic[row['userid']])
            check_in_venues_index.append(venues_dic[row['Venue_id']])
            check_in_lat_lon.append(poi_lat_lon[venues_dic[row['Venue_id']]])
        check_in['users_index'] = check_in_users_index
        check_in['venues_index'] = check_in_venues_index
        check_in['lat_lon'] = check_in_lat_lon

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

        return check_in, friend_list_index, user_trajectory_index, venues_dic, time_hour_dic, time_month_dic, current_index

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
                # id = extra_poi_info['id']
                # if id == '4ad83e6ff964a520eb1021e3':
                #     continue
                #
                #
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
            all_side_info[key]['contact'] = {current_index + val for val in all_side_info[key]['contact']}

        current_index += len(all_contact)

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
        return all_side_info

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

    def load_all():
        # all_checkins, all_poi, all_friendships = load_basic_info()
        # all_checkin_relations = process_checkin(all_checkins)
        # cities = ['NYC', 'TKY', 'SP', 'JK', 'KL']
        cities = ['NYC']
        # cities = ['IST']  # 数据很多有问题，只有极少的信息

        for city in cities:
            check_in, poi_data, users, venues, friend = load_city_data(city)
            check_in_processed, friend_processed, user_trajectory_processed, venues_dic, time_hour_dic, time_month_dic, current_index \
                = process_city_data(city, check_in, poi_data, users, venues, friend)
            # dump_city_data(city, check_in_processed, friend_processed, user_trajectory_processed)

            poi_details = load_extra_poi_info(city, venues, venues_dic, time_hour_dic, time_month_dic, current_index)
            dump_city_data(city, check_in_processed, friend_processed, user_trajectory_processed, poi_details)

    load_all()
    print('load success')


if __name__ == '__main__':
    load_data()
