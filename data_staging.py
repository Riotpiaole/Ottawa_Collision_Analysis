import os
import re
import sys
import utm
import holidays
import numpy as np
import pandas as pd
from math import pi
from os.path import join as join_path
from numpy import arctan2, sin, cos, sqrt
from multiprocessing import Pool, cpu_count
from datetime import datetime, time, timedelta

# multi-process mapper
# Global variables
pool = Pool(processes=cpu_count())


def collision_dir(city):
    return os.path.join('./datasets/collisions/', city)


climate_dir = './datasets/climates/'
station = pd.read_csv(
    climate_dir +
    'Station-Inventory-EN.csv',
    skiprows=3)


# selecting only Province in ALBERTA and ONTARIO
station = station[station['Province'].isin(['ALBERTA', 'ONTARIO'])]

regex = re.compile(r'\w+\.?(xls|xlsx)')


def read_csv(filename: 'str') -> 'pd.DataFrame':
    return pd.read_csv(filename)


def climate():
    pass

    # "X.Date.Time","Year","Month","Day","Time","Temp...C.","Temp.Flag","Dew.Point.Temp...C.","Dew.Point.Temp.Flag","Rel.Hum....","Rel.Hum.Flag","Wind.Dir..10s.deg.","Wind.Dir.Flag","Wind.Spd..km.h.","Wind.Spd.Flag","Visibility..km.","Visibility.Flag","Stn.Press..kPa.","Stn.Press.Flag","Hmdx","Hmdx.Flag","Wind.Chill","Wind.Chill.Flag","Weather.","X.U.FEFF..Station.Name.","X.Province."
    # return station_inventory


def collision():
    all_collision = {}
    Canadian_Holidays = holidays.Canada()

    hour_dimension = None

    ottawa, calgory, toronto = read_data()

    # ===================================================
    # handling time after 12 am would parse it as string
    # ===================================================

    ottawa.Time = ottawa.Time.apply(
        lambda x: x if isinstance(
            x, time) else datetime.strptime(
            x, '%H:%M:%S').time())

    # combining date and time
    ottawa['Datetime'] = ottawa.Date.combine(
        ottawa.Time, lambda x, y: datetime.combine(x, y))
    # rounding up 30 minutes time region
    ottawa['Datetime'] = ottawa.Datetime.apply(
        lambda x: pd.Timestamp(x)).dt.round('30min')

    # Hour dimension
    ottawa['year'] = ottawa['Date'].apply(lambda x: x.year)
    ottawa['month'] = ottawa['Date'].apply(lambda x: x.month)
    ottawa['day'] = ottawa['Date'].apply(lambda x: x.day)

    ottawa['day_of_the_week'] = ottawa['Date'].apply(
        lambda x: x.weekday())

    ottawa['weekend'] = ottawa['day_of_the_week'].apply(
        lambda x: 1 if x in [5, 6] else 0)

    ottawa['holiday'] = ottawa['Date'].apply(
        lambda x: 1 if x in Canadian_Holidays else 0)

    ottawa['holiday_name'] = ottawa['Date'].apply(
        lambda x: Canadian_Holidays.get(x)
        if Canadian_Holidays.get(x) else 'Unknown')
    hour_dimension = ottawa.groupby(
        ['year', 'month', 'day',
         'Datetime', 'day_of_the_week',
         'weekend', 'holiday', 'holiday_name'])\
        .size().reset_index()

    # ===================================================
    #  Weather Dimension
    # ===================================================

    return ottawa, hour_dimension


def read_data()->'tuple(pd.DataFrame)|3':
    '''Obtain all of the citys data'''
    # TODO add preprocessing of Toronto and Calgory to same as star schema

    ottawa, calgary, toronto = None, None, None

    # obtain ottawa
    ottawa_files = os.listdir(collision_dir('Ottawa'))
    ottawa_file_names = map(
        lambda x: join_path(collision_dir('Ottawa'), x),
        list(filter(regex.match, ottawa_files)))

    df_list = pool.map(pd.read_excel, ottawa_file_names)

    if 'Year' in df_list[-1].columns.values:
        df_list[-1].drop('Year', axis=1, inplace=True)
        df_list[-1].to_excel(result[-1])

    ottawa = pd.concat(df_list, sort=True, ignore_index=True)


#     # obtain calgary
#     calgary_files = os.listdir(collision_dir('Calgary'))
#     calgary = pd.read_csv(join_path(collision_dir('Calgary'), calgary_files[0]))


#     # obtain toronto
#     toronto_files = os.listdir(collision_dir('Toronto'))
#     toronto = pd.read_csv(
#         join_path(collision_dir('Toronto'),toronto_files[0]))

#     convert_impact_type = np.vectorize(lambda x: '01 - Fatal injury' if x > 0 else '02 - Non-fatal injury')
#     impact_type = convert_impact_type(toronto.FATAL_NO)

#     toronto['Collision_Classification'] = impact_type.reshape((impact_type.shape[0],1))

    return ottawa, calgary, toronto


def test_lat_lon_dist(dist: 'func'):
    pt1 = (38.897147, -77.043934)
    pt2 = (38.898556, -77.037852)
    result = 0.613  # km
    assert(result == round(dist(pt1, pt2), 3))


@np.vectorize
def degree2rad(x):
    return x * pi / 180

degree_rad = degree2rad(
        station[['Latitude (Decimal Degrees)',
                'Longitude (Decimal Degrees)']].values[:].copy())

def vectorize_dist(location):
    lat1, lon1 = list(location)
    lat2, lon2 = degree_rad[:, 0], degree_rad[:, 1]
    a = sin(lat1 - lat2)**2 + cos(lat1) * cos(lat2) * sin((lon1 - lon2) / 2)**2
    c = 2 * arctan2(sqrt(a), sqrt(1 - a))
    return tuple(station['Name'].iloc[np.argpartition(6371 * c,1)[:4]].values)

if __name__ == "__main__":
    # tables = climate()
    ottawa, hour = collision()
    tmp = station[['Latitude (Decimal Degrees)',
                   'Longitude (Decimal Degrees)']]
    test_unit = np.array([45.299792, -75.453557])

    # ottawa['ClosestStation'] = compute_dist(ottawa[['Latitude', 'Longitude']].values)
    a = ottawa[['Latitude', 'Longitude']].values
    result = np.apply_along_axis(vectorize_dist,-1,a)


# "X.Date.Time","Year","Month","Day","Time","Temp...C.","Temp.Flag","Dew.Point.Temp...C.","Dew.Point.Temp.Flag","Rel.Hum....","Rel.Hum.Flag","Wind.Dir..10s.deg.","Wind.Dir.Flag","Wind.Spd..km.h.","Wind.Spd.Flag","Visibility..km.","Visibility.Flag","Stn.Press..kPa.","Stn.Press.Flag","Hmdx","Hmdx.Flag","Wind.Chill","Wind.Chill.Flag","Weather.","X.U.FEFF..Station.Name.","X.Province."
