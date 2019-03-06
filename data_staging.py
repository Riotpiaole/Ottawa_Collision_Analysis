import os
import re
import holidays
import numpy as np
import pandas as pd
from os.path import join as join_path
from multiprocessing import Pool , cpu_count
from datetime import datetime , time , timedelta

# multi-process mapper
pool = Pool(processes=cpu_count())
collision_dir = lambda city: os.path.join('./datasets/collisions/',city)
regex = re.compile(r'\w+\.?(xls|xlsx)')

def read_csv(filename:'str') -> 'pd.DataFrame':
    return pd.read_csv(filename)

def climate():
    pass


    # "X.Date.Time","Year","Month","Day","Time","Temp...C.","Temp.Flag","Dew.Point.Temp...C.","Dew.Point.Temp.Flag","Rel.Hum....","Rel.Hum.Flag","Wind.Dir..10s.deg.","Wind.Dir.Flag","Wind.Spd..km.h.","Wind.Spd.Flag","Visibility..km.","Visibility.Flag","Stn.Press..kPa.","Stn.Press.Flag","Hmdx","Hmdx.Flag","Wind.Chill","Wind.Chill.Flag","Weather.","X.U.FEFF..Station.Name.","X.Province."
    # return station_inventory

def collision():
    all_collision = {}
    Canadian_Holidays = holidays.Canada()

    ottawa , calgory , toronto = read_data()

    # ===================================================
    # handling time after 12 am would parse it as string
    # ===================================================

    ottawa.Time = ottawa.Time.apply(
        lambda x: x if isinstance(x,time) else datetime.strptime(x,'%H:%M:%S').time())

    # combining date and time
    ottawa['Datetime'] = ottawa.Date.combine(ottawa.Time, lambda x , y: datetime.combine(x,y))
    # rounding up 30 minutes time region
    ottawa['Datetime'] = ottawa.Datetime.apply(lambda x: pd.Timestamp(x)).dt.round('30min')

    # Hour dimension
    ottawa['year'] = ottawa['Date'].apply(lambda x:x.year)
    ottawa['month'] = ottawa['Date'].apply(lambda x:x.month)
    ottawa['day'] = ottawa['Date'].apply(lambda x:x.day)

    ottawa['day_of_the_week'] = ottawa['Date'].apply(
            lambda x:x.weekday())

    ottawa['weekend'] = ottawa['day_of_the_week'].apply(
            lambda x: 1 if x in [5,6] else 0)

    ottawa['holiday'] = ottawa['Date'].apply(
            lambda x: 1 if x in Canadian_Holidays else 0)

    ottawa['holiday_name'] = ottawa['Date'].apply(
            lambda x: Canadian_Holidays.get(x)
                if Canadian_Holidays.get(x) else 'Unknown')
    hour_dimension = ottawa.groupby(
            ['year','month','day',
            'Datetime','day_of_the_week',
            'weekend','holiday','holiday_name'])\
                .size().reset_index()

    # ===================================================
    #  Weather Dimension
    # ===================================================

    climate_dir = './datasets/climates/'
    station_inventory = pd.read_csv(climate_dir + 'Station-Inventory-EN.csv',skiprows=3)

    # selecting only Province in ALBERTA and ONTARIO
    station_inventory = station_inventory[
        station_inventory['Province'].isin([ 'ALBERTA', 'ONTARIO'])]
#     ottawa['closest_station']
    return ottawa , hour_dimension , station_inventory


def read_data()->'tuple(pd.DataFrame)|3':
    '''Obtain all of the citys data'''
    #TODO add preprocessing of Toronto and Calgory to same as star schema
    # obtain ottawa
    ottawa_files = os.listdir(collision_dir('Ottawa'))
    ottawa_file_names = map(
        lambda x : join_path(collision_dir('Ottawa'),x),
            list(filter(regex.match, ottawa_files)))


    df_list = pool.map(pd.read_excel,ottawa_file_names)

    if 'Year' in df_list[-1].columns.values:
        df_list[-1].drop('Year',axis=1,inplace=True)
        df_list[-1].to_excel(result[-1])

    ottawa = pd.concat(df_list, sort=True,ignore_index=True)

    calgary , toronto = None , None

    # obtain calgary
    calgary_files = os.listdir(collision_dir('Calgary'))
    calgary = pd.read_csv(join_path(collision_dir('Calgary'), calgary_files[0]))


    # obtain toronto
    toronto_files = os.listdir(collision_dir('Toronto'))
    toronto = pd.read_csv(
        join_path(collision_dir('Toronto'),toronto_files[0]))

    convert_impact_type = np.vectorize(lambda x: '01 - Fatal injury' if x > 0 else '02 - Non-fatal injury')
    impact_type = convert_impact_type(toronto.FATAL_NO)

    toronto['Collision_Classification'] = impact_type.reshape((impact_type.shape[0],1))

    return ottawa , calgary , toronto


if __name__ == "__main__":
    # tables = climate()
    ottawa , hour , station = collision()
