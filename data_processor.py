import numpy as np
import pandas as pd
import scipy.io as scio
import os
import json


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
    def combine_extra_poi_info(dir):
        file_names = os.listdir(dir)
        file_names = zip(file_names, range(len(file_names)))
        file_names = dict(file_names)

        for file_name in file_names:
            # load extra poi info from json file
            with open(dir + file_name, 'r') as f:
                extra_poi_info = json.load(f)
            print(extra_poi_info)



    # all_checkins, all_poi, all_friendships = load_basic_info()
    # nyc_data, tky_data, sp_data, jk_data = load_city_info()
    # # print(nyc_data)
    # nyc_checkins, nyc_friendship, nyc_pois = process_city_data(nyc_data, all_poi)
    # tky_checkins, tky_friendship, tky_pois = process_city_data(tky_data, all_poi)
    # sp_checkins, sp_friendship, sp_pois = process_city_data(sp_data, all_poi)
    # jk_checkins, jk_friendship, jk_pois = process_city_data(jk_data, all_poi)

    base_dir = './data/raw/Venue_detail/'
    cities = ['NYC', 'TKY', 'SP', 'JK']
    combine_extra_poi_info(base_dir + cities[0] + '/')




if __name__ == '__main__':
    load_data()