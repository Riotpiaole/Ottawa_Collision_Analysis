import re
import os
import numpy as np
import pandas as pd
from time import time
from constant import ontario_station, desire_cols, rename_cols
from multiprocessing import Pool, cpu_count


climate_dir = './datasets/climates/'
collision_dir = './datasets/collisions/'
regex = re.compile(r'\w+\.csv')


def ls_files(dirs, regex=regex):
    return filter(regex.match, os.listdir(dirs))


def get_province_path(province, path=None, prefix=None):
    if prefix:
        return os.path.join('./datasets/climates/%s' % province, prefix + path)
    if path:
        return os.path.join('./datasets/climates/%s' % province, path)
    return './datasets/climates/%s' % province


def read_weather(path, province='Ontario'):
    print("Preprocessing file (%s)" % (path))
    start = time()
    data = pd.read_csv(get_province_path(province, path), low_memory=False)
    result = data[np.logical_and(
        data['Year'].isin([2014, 2015, 2016, 2017]),
        data['X.U.FEFF..Station.Name.'].isin(ontario_station))]

    print("[%s]" % path, result.Year.unique())
    result = result.dropna(subset=['X.U.FEFF..Station.Name.'])
    del data
    end = time()
    result.to_csv(get_province_path(province, path, '14-17'), index=False)
    print("Preprocessing %s completed elapse %d s" %
          (path, round((end - start), 2)))


def combine_reduce(province):
    new_path = re.compile(r'14-17\w+.csv').match
    all_path = filter(
        new_path,
        os.listdir('./datasets/climates/{}'
                   .format(province)))

    df_lists = [
        pd.read_csv(
            os.path.join(
                get_province_path('Ontario'),
                item),
            low_memory=False) for item in all_path]
    df_lists = pd.concat(df_lists, ignore_index=True, sort=True)
    df_lists.dropna(subset=['Temp...C.'], inplace=True)
    df_lists.dropna(subset=['Hmdx'], inplace=True)
    df_lists.dropna(subset=['Stn.Press..kPa.'], inplace=True)
    df_lists.to_csv('./datasets/climates/ontario.csv', index=False)
    return df_lists


def shrink(province):
    '''shrink the csv with'''
    def func(x): return os.path.join(get_province_path(province), x)
    if province.lower() == 'ontario':
        all_paths = list(
            filter(
                regex.match,
                os.listdir('./datasets/climates/Ontario')))

    with Pool(cpu_count()) as pool:
        pool.map(read_weather, all_paths)
        pool.close()


def combine_ontario():

    new_columns_dict = dict(zip(desire_cols, rename_cols))

    if os.path.isfile('./datasets/climates/ontario.csv'):

        print("ontario.csv found. loading ontario weather data")
        datasets = pd.read_csv('./datasets/climates/ontario.csv')[desire_cols]
        datasets.fillna('Unknown', inplace=True)

        # rename the columns to new cols
        datasets = datasets.rename(index=str, columns=new_columns_dict)

        return datasets[datasets['Name'].str.contains('OTTAWA')],\
            datasets[datasets['Name'].str.contains('TORONTO')]

    if not os.path.isfile(
        get_province_path(
            'Ontario',
            'Ontario_2_2.csv',
            '14-17')):
        raise FileNotFoundError("Please shrink the weather data")

    def func(x): return os.path.join(get_province_path('Ontario'), '14-17' + x)
    all_paths = list(
        map(func, ['ontario_2_1.csv', 'Ontario_2_2.csv', 'Ontario_4.csv']))
    df_list = []
    for path in all_paths:
        df_list.append(
            pd.read_csv(path).dropna(
                subset=['X.U.FEFF..Station.Name.']))

    data = pd.concat(df_list, sort=True, ignore_index=True)[[desire_cols]]
    data.to_csv(os.path.join(climate_dir, 'ontario.csv'), index=False)
    data = data.rename(index=str, columns=new_columns_dict)
    data.fillna('Unknown', inplace=True)
    return data

if __name__ == "__main__":
    ottawa, toronto = combine_ontario()
