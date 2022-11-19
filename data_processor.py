import numpy as np
import pandas as pd
import scipy.io as scio
import os
import json
import jsonpath as jp

def load_data():
    # 1). Load basic info included in the dataset.
    def load_basic_info():
        df_checkins = pd.read_csv('./data/raw/dataset_WWW2019/dataset_WWW_Checkins_anonymized.txt', sep='\t', header=None)
        df_checkins.columns = ['user_id', 'Venue_id', 'utc_time', 'Timezone_offset']

        df_poi = pd.read_csv('./data/raw/dataset_WWW2019/raw_POIs.txt', sep='\t', header=None)
        df_poi.columns = ['Venue_id', 'latitude', 'longitude', 'category', 'Country_code']

        df_friendship_old = pd.read_csv('./data/raw/dataset_WWW2019/dataset_WWW_friendship_old.txt', sep='\t', header=None)
        df_friendship_old.columns = ['user_id1', 'user_id2']

        df_friendship_new = pd.read_csv('./data/raw/dataset_WWW2019/dataset_WWW_friendship_new.txt', sep='\t', header=None)
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

    def process_city_data(city_data, all_poi: pd.DataFrame):
        df_checkin = pd.DataFrame(city_data['selected_checkins'])  # (105961, 4)
        df_checkin.columns = ['user_id', 'time', 'Venue_id', 'Venue_category']
        df_friend_new = pd.DataFrame(city_data['friendship_new'])  # (10545, 2)
        df_friend_old = pd.DataFrame(city_data['friendship_old'])  # (8723, 2)
        df_friend = pd.merge(df_friend_new, df_friend_old, how='outer')  # (19268, 2)

        for i in range(len(df_checkin['Venue_id'])):
            df_checkin['Venue_id'][i] = (all_poi.iloc[df_checkin['Venue_id'][i]][0], df_checkin['Venue_id'][i])
            df_checkin['Venue_category'][i] = all_poi.iloc[df_checkin['Venue_id'][i][1]][3]

        return df_checkin, df_friend, df_checkin['Venue_id'].unique()

    # def combine_extra_poi_info(df_poi: pd.DataFrame, dir):
        # get file names in the directory
    def load_extra_poi_info(dir):
        file_names = os.listdir(dir)
        file_names = zip(file_names, range(len(file_names)))
        file_names = dict(file_names)
        all_side_info = []

        for file_name in file_names:
            # load extra poi info from json file
            with open(dir + file_name, 'r') as f:
                extra_poi_info = json.load(f)
                side_info = []
                # exact useful poi information
                # might useful:
                # id, name,
                # contact{phone, twitter, facebook},
                # location{address, city. country},
                # categories[{id, name}],
                # stats{tipCount, usersCount, checkinCount, visitsCount},
                # likes{count}, rating, specials ?,
                # reasons{count, items[{summary, reasonName}]} ?,
                # open ?
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

                side_info.append({id, name})
                side_info.append({id, contact})
                side_info.append({id, address})
                side_info.append({id, city})
                side_info.append({id, country})
                side_info.append({id, categories})
                side_info.append({id, stats})
                side_info.append({id, like_count})
                side_info.append({id, rating})

                all_side_info.append(side_info)

        return all_side_info

    def process_checkin(all_checkins):
        checkin_relations = []
        for i in range(len(all_checkins)):
            checkin_relations.append({all_checkins['user_id'][i], all_checkins['Venue_id'][i], all_checkins['utc_time'][i], all_checkins['Timezone_offset'][i]})
        return checkin_relations

    def load_all():
        all_checkins, all_poi, all_friendships = load_basic_info()
        # all_checkin_relations = process_checkin(all_checkins)
        # cities = ['NYC', 'TKY', 'SP', 'JK']
        nyc_data, tky_data, sp_data, jk_data = load_city_info()
        # # print(nyc_data)
        nyc_checkins, nyc_friendship, nyc_pois = process_city_data(nyc_data, all_poi)
        tky_checkins, tky_friendship, tky_pois = process_city_data(tky_data, all_poi)
        sp_checkins, sp_friendship, sp_pois = process_city_data(sp_data, all_poi)
        jk_checkins, jk_friendship, jk_pois = process_city_data(jk_data, all_poi)
        # print(nyc_checkins)
        base_dir = './data/raw/Venue_detail/'
        nyc_poi_details = load_extra_poi_info(base_dir + 'NYC' + '/')
        tky_poi_details = load_extra_poi_info(base_dir + 'TKY' + '/')
        sp_poi_details = load_extra_poi_info(base_dir + 'SP' + '/')
        jk_poi_details = load_extra_poi_info(base_dir + 'JK' + '/')
        print('Number of all POI side info: ', len(nyc_poi_details) + len(tky_poi_details) + len(sp_poi_details) + len(jk_poi_details))

    load_all()
    print('load success')


if __name__ == '__main__':
    load_data()
