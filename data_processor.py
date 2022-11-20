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
            if row['user1'] != row['user2']:
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

        # recode friendship
        friend_list_index = []
        for i in range(len(friend_list)):
            friend_list_index.append((users_dic[friend_list[i][0]], users_dic[friend_list[i][1]]))

        poi_data['venues_index'] = None
        for index, row in poi_data.iterrows():
            poi_data['venues_index'][index] = venues_dic[row['Venue_id']]

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

        check_in['local_month'] = None
        check_in['hour_period'] = None
        check_in_local_month = []
        check_in_hour_period = []

        for index, row in check_in.iterrows():
            time_1 = row['utc_time'][:-10] + row['utc_time'][-4:]
            timezone_offset = row['Timezone_offset']
            struct_time = time.mktime(time.strptime(time_1, "%a %b %d  %H:%M:%S %Y")) + timezone_offset * 60
            localtime = time.localtime(struct_time)  # 返回元组
            check_in_local_month.append(localtime[1])
            check_in_hour_period.append(time_partition(localtime[6], localtime[3], localtime[4]))
        check_in['local_month'] = check_in_local_month
        check_in['hour_period'] = check_in_hour_period

        co_occurrence_list_index = {}
        check_in = check_in.sort_values('users_index', ascending=True, inplace=False)
        user_group = check_in.groupby("users_index")
        user_group = list(user_group)
        for i in range(len(user_group)):  # 每一个group2 都是用户
            df = pd.DataFrame(user_group[i][1])  # 每个用户的访问轨迹
            df = df.reset_index(drop=True)
            co_occurrence_list_index[df['users_index'][0]] = set(list(df['venues_index']))

        return check_in, friend_list_index, co_occurrence_list_index

    def time_partition(day, hour, min):
        # days [1-7] per 1 day, hours [0-23] per 1 hour, minutes [0-1] per 30 minutes
        # the time will be partied into 7 * 24 * 2 index
        if 0 <= min < 30:
            return day * hour * 2
        else:
            return day * (hour * 2 + 1)

    def load_extra_poi_info(city):
        base_dir = './data/raw/Venue_detail/'
        file_names = os.listdir(base_dir + city + '/')
        file_names = zip(file_names, range(len(file_names)))
        file_names = dict(file_names)
        all_side_info = []

        for file_name in file_names:
            # load extra poi info from json file
            with open(dir + file_name, 'r') as f:
                extra_poi_info = json.load(f)
                side_info = {}
                # exact useful poi information
                # might useful:
                # id,
                # contact{phone, twitter, facebook, instagram},
                # categories[{name, icon{prefix".../cate1/cate2"}}],
                # stats{tipCount},
                # price{tier},
                # likes{count}, rating,
                # photos{count},
                # hours{timeframes[{days, open[{renderedtime}]}]}
                # popular{timeframes[{days, open[{renderedtime}]}]}
                id = extra_poi_info['id']

                name = extra_poi_info['name']
                contact = extra_poi_info['contact'] if 'contact' in extra_poi_info.keys() else ''
                address = extra_poi_info['location']['address'] if 'address' in extra_poi_info.keys() else ''
                city = extra_poi_info['location']['city'] if 'city' in extra_poi_info.keys() else ''
                country = extra_poi_info['location']['country'] if 'country' in extra_poi_info.keys() else ''
                categories = extra_poi_info['categories'] if 'categories' in extra_poi_info.keys() else ''
                stats = extra_poi_info['stats'] if 'stats' in extra_poi_info.keys() else ''
                like_count = extra_poi_info['likes']['count'] if 'likes' in extra_poi_info.keys() else ''
                rating = extra_poi_info['rating'] if 'rating' in extra_poi_info.keys() else ''
                # print(id, name, contact, address, city, country, categories, stats, like_count, rating)

                side_info.append({1: {id, name}})
                side_info.append({2, {id, contact}})
                side_info.append({3, {id, address}})
                side_info.append({4, {id, city}})
                side_info.append({5, {id, country}})
                side_info.append({6, {id, categories}})
                side_info.append({7, {id, stats}})
                side_info.append({8, {id, like_count}})
                side_info.append({9, {id, rating}})

                all_side_info.append(side_info)

        return all_side_info

    def load_all():
        # all_checkins, all_poi, all_friendships = load_basic_info()
        # all_checkin_relations = process_checkin(all_checkins)
        cities = ['NYC', 'TKY', 'SP', 'JK']

        for city in cities:
            check_in, poi_data, users, venues, friend = load_city_data(city)
            check_in_processed, friend_processed, co_occurence_processed = process_city_data(city, check_in, poi_data, users, venues, friend)
            poi_details = load_extra_poi_info(city)

    load_all()
    print('load success')


if __name__ == '__main__':
    load_data()
