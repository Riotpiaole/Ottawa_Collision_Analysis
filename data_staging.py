import os
import re
import sys
import utm
import holidays
import datetime
import numpy as np
import pandas as pd
from math import pi
from os.path import join as join_path
from multiprocessing import Pool, cpu_count
from datetime import datetime, time, timedelta
from numpy import arcsin, sin, cos, sqrt, power
from shrink import combine_ontario

# multi-process mapper
# Global variables


def collision_dir(city):
    return os.path.join('./datasets/collisions/', city)


climate_dir = './datasets/climates/'
station = pd.read_csv(
    climate_dir +
    'Station-Inventory-EN.csv',
    skiprows=3)

ottawa_weather, toronto_weather = combine_ontario()


station = station[station['Province'].isin(['ALBERTA', 'ONTARIO'])]
station_final = station[station.Name.str.contains('OTTAWA|TORONTO|calgary')]

station_ottawa = station_final[station_final.Name.str.contains('OTTAWA')]

station_calgary = station_final[station_final.Name.str.contains('calgary')]
station_toronto = station_final[station_final.Name.str.contains('TORONTO')]

station_ottawa = station_ottawa[
    station_ottawa.Name.isin(ottawa_weather.Name.unique())]

# generate station long and lat to ottawa_weather
ottawa_weather['Latitude'] = ottawa_weather.Name.apply(
    lambda name: station_ottawa[station_ottawa.Name == name].Latitude.values[0] / 10000000)
ottawa_weather['Longitude'] = ottawa_weather.Name.apply(
    lambda name: station_ottawa[station_ottawa.Name == name].Longitude.values[0] / 10000000)

# add dummy data to ensure ottawa weather
ottawa_weather = ottawa_weather.append(
    ['Unknown'],
    verify_integrity=True,
    ignore_index=True)

station_toronto = station_toronto[
    station_toronto.Name.isin(toronto_weather.Name.unique())]

# generate station long and lat to toronto_weather
toronto_weather['Latitude'] = toronto_weather.Name.apply(
    lambda name: station_toronto[station_toronto.Name == name].Latitude.values[0] / 10000000)
toronto_weather['Longitude'] = toronto_weather.Name.apply(
    lambda name: station_toronto[station_toronto.Name == name].Longitude.values[0] / 10000000)

ottawa_station_lookup = pd.Index(station_ottawa.Name)
toronto_station_lookup = pd.Index(station_toronto.Name)
calgary_station_lookup = pd.Index(station_calgary.Name)


# obtain all of the holidays
Canadian_Holidays = holidays.Canada()


def read_data()->'tuple(pd.DataFrame)|3':
    '''Obtain all of the citys data'''
    ottawa, calgary, toronto = None, None, None
    print("Retrieveing data from collision datasets")
    if os.path.isfile('./datasets/tmp/ottawa.csv'):
        ottawa = pd.read_csv('./datasets/tmp/ottawa.csv')
    else:
        # obtain ottawa
        ottawa_files = os.listdir(collision_dir('Ottawa'))
        ottawa_file_names = list(map(
            lambda x: join_path(collision_dir('Ottawa'), x),
            filter(regex.match, ottawa_files)))

        df_list = [
            pd.read_csv(fn, encoding="ISO-8859-1") for fn in ottawa_file_names]
        df_list[-1].drop(columns=['Year'], inplace=True)
        ottawa = pd.concat(df_list, sort=True, ignore_index=True)
        ottawa.dropna(inplace=True)
        ottawa.to_csv('./datasets/tmp/ottawa.csv', index=False)

    # obtain calgary
    calgary_files = os.listdir(collision_dir('calgary'))
    calgary = pd.read_csv(
        join_path(
            collision_dir('calgary'),
            calgary_files[0]))

    # obtain toronto
    toronto_files = os.listdir(collision_dir('Toronto'))
    toronto = pd.read_csv(
        join_path(collision_dir('Toronto'), toronto_files[0]))

    return ottawa, calgary, toronto

# selecting only Province in ALBERTA and ONTARIO


regex = re.compile(r'\w+\.csv')


@np.vectorize
def degree2rad(x):
    return float(x) * pi / 180.


def compute_distance(
        location: 'np.ndarray',
        station: 'np.ndarray',
        df_station: 'station.DataFrame',
        top=1)->'np.ndarray(dtype:str)':
    '''compute GIS distance of a location against all of weather station'''
    lat1, lon1 = list(location)
    lat2, lon2 = station[:, 0], station[:, 1]
    a = power(sin(lat1 - lat2) / 2, 2)
    b = power(sin(lon1 - lon2) / 2, 2)
    return df_station.iloc[
        np.argpartition(
            6371 * 2 * arcsin(
                sqrt(a + cos(lat1) * cos(lat2) * b)), 1
        )[:top]]['Name']


def vectorizes_compute_dist(
        table: 'collision_table',
        city: 'str',
        top)->'ndarray':
    '''compute the distance with relevant city '''
    stations = None
    assert(city.lower() in ['ottawa', 'toronto', 'calgary'])

    if city.lower() == 'ottawa':
        stations = station_ottawa

    elif city.lower() == 'toronto':
        stations = station_toronto

    else:
        stations = station_calgary

    stations_values = degree2rad(
        stations[
            ['Latitude',
             'Longitude']] / 10000000)

    return np.apply_along_axis(
        compute_distance, -1, table, stations_values, stations, top)

def suggregate_weather_station(
        row: 'clloision_row',
        lookup: 'pd.DataFrame',
        weather_table: 'pd.DataFrame')-> 'pd.DataFrame':
    '''generating weather dimension suggregate key'''
    key = 0
    rt, rn = row
    try:
        key = lookup.get_loc((str(rt)[:-3], rn))
    except KeyError:
        key = lookup.shape[0] - 1
    return key


def vectorize_weather_station(
        table: 'clloision_table',
        weather_table: 'ndarray')-> 'pd.DataFrame':
    '''merge weather station table vectorizly'''
    weather_table
    lookup = pd.Index(weather_table[['DateTime', 'Name']])
    # check_for_exists
    return np.apply_along_axis(
        suggregate_weather_station,
        -1, table[['DateTime', '1_closest']],
        lookup,
        weather_table)

def locaton_row(func: 'function'):
    def location_and_call(*args, **kwargs):
        result = func(*args, **kwargs)
        size = len(result)
        if size == 2:
            # case2
            return ['Unknown', result[0], result[1]]
        elif size == 3:
            return result
        elif size == 1:
            return ['Unknown', 'Unknown', 'Unknown']
        elif size >= 4:
            return [result[0], result[1], '/'.join(result[2:])]
    return location_and_call


def location_preprocess(location_cols):
    #                           streetName , Intersect 1 , intersect 2,
    # case1: s1 btwn i1 & i2 ->   s1       , I1          , I2
    # case2: s1 @ s2         ->   nan      , s1          , s2
    # case3: No loca         ->
    @locaton_row
    def mapper_function(x):
        regex = re.compile(r'and|\&|\/')
        if 'btwn' in x:
            if regex.search(x):
                return re.split(r'btwn|BTWN|\&|\/', x)
            else:
                return x
        elif '@' in x:
            return re.split(r'@', x)
        elif 'No Location Given' in x:
            return ['Unknown']
        else:
            return x
    return np.vstack(location_cols.apply(mapper_function).values)


def toronto_preprocess(st1, st2):
    if st2 == 'Unknown':
        return [st1, 'Unknown', 'Unknown']
    else:
        return [st1, st2, 'Unkown']


def ottawa_dimensions(ottawa):
    if os.path.isfile('./datasets/tmp/ottawa_star_schema.csv'):
        ottawa = pd.read_csv('./datasets/tmp/ottawa.csv', low_memory=False)[
            ['hour_key',
             'location_key',
             'Accident_key',
             'weather_key',
             'Is_Fatal',
             'Is_Intersection']]
        ottawa.to_csv('./datasets/tmp/ottawa_star_schema.csv', index=False)
        return ottawa
    ottawa.dropna(inplace=True)
    # generate is fatal columns
    ottawa['Is_Fatal'] = ottawa.Collision_Classification.apply(
        lambda collision_class: 1 if collision_class == '01 - Fatal injury' else 0)

    ottawa['Is_Intersection'] = ottawa.Location.apply(
        lambda location: 1 if '@' in location else 0)

    # handling time after 12 am would parse it as string
    ottawa['DateTime'] = ottawa.Date.combine(
        ottawa.Time, lambda date, time:
            datetime.strptime(date + ' ' + time, '%Y-%m-%d %I:%M:%S %p')
        if len(time) > 5 else
            datetime.strptime(date + ' ' + time, '%Y-%m-%d %H:%M'))
    ottawa['Accident_time'] = ottawa.DateTime.apply(
        lambda datetime: datetime.time())
    ottawa['Date'] = ottawa.DateTime.apply(
        lambda datetime: datetime.date())
    # combining date and time

    # rounding up 1 hour time region upper bound

    ottawa['year'] = ottawa['Date'].apply(lambda x: x.year)
    ottawa['month'] = ottawa['Date'].apply(lambda x: x.month)
    ottawa['day'] = ottawa['Date'].apply(lambda x: x.day)
    ottawa['hour'] = ottawa['Accident_time'].apply(lambda x: x.hour)

    # ===================================================
    # Hour dimension
    # ===================================================
    if os.path.isfile('./datasets/tmp/hour_dimension.csv'):
        print("hour_dimension file found")
        hour_dimension = pd.read_csv(
            './datasets/tmp/hour_dimension.csv',
            low_memory=False)
    else:
        print("Ottawa datasets preprocessing")
        print("Staging hour_dimension")

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
            ['year', 'month', 'day', 'hour',
             'Date', 'day_of_the_week', 'weekend',
             'holiday', 'holiday_name'], as_index=False)\
            .size().reset_index()

        hour_dimension.drop(columns=hour_dimension.columns[-1], inplace=True)
        hour_dimension.drop_duplicates(inplace=True)
        hour_dimension.to_csv(
            './datasets/tmp/hour_dimension.csv')
        ottawa.to_csv('./datasets/tmp/ottawa.csv',index=False)
    # ===================================================
    #  Accident Dimension
    # ===================================================

    key = [
        'Time',
        'Environment',
        'Road_Surface',
        'Traffic_Control',
        'Impact_type']

    value = [
        'Accident-time',
        'Environment',
        'Road_Surface',
        'Traffic_Control',
        'Impact_type']
    kv_dict = dict(zip(key, value))

    # group by combination of unique cases
    accident_dimension = ottawa.groupby(key).size().reset_index()
    accident_dimension = accident_dimension.rename(index=str, columns=kv_dict)
    accident_dimension.drop(
        columns=accident_dimension.columns[-1], inplace=True)
    accident_dimension.drop_duplicates(inplace=True)
    accident_dimension.to_csv('./datasets/tmp/accident_dimension.csv')
    ottawa.to_csv('./datasets/tmp/ottawa.csv',index=False)

    # ===================================================
    #  Weather Dimension
    # ===================================================
    weather_dimension = None
    if os.path.isfile('./datasets/tmp/weather_dimension.csv'):
        print("Weather Dimension found loading the file")
        weather_dimension = pd.read_csv(
            './datasets/tmp/weather_dimension.csv', low_memory=False)
    else:
        print("Preprocessing weather dimension")
        # generating compute the distance against all of the ottawa stations
        closest_weather_stations = vectorizes_compute_dist(
            degree2rad(ottawa[['Latitude', 'Longitude']].values),
            'ottawa', 3)
        # find the top three closest stations
        ottawa['1_closest'] = closest_weather_stations[:, 0]

        # weather_dimension

        ottawa['weather_key'] = vectorize_weather_station(
            ottawa, ottawa_weather)

        weather_dimension = ottawa_weather[ottawa_weather.index.isin(
            ottawa['weather_key'].unique())]

        # inverting index for remove un-needed
        weather_dimension['Inverted_index'] = weather_dimension.index
        weather_dimension.reset_index(inplace=True, drop=True)

        weather_index_lookup = pd.Index(weather_dimension['Inverted_index'])
        ottawa['weather_key'] = ottawa['weather_key'].apply(
            lambda x: weather_index_lookup.get_loc(x))
        weather_dimension.drop_duplicates(inplace=True)

        weather_dimension.to_csv(
            './datasets/tmp/weather_dimension.csv')
        ottawa.to_csv('./datasets/tmp/ottawa.csv', index=False)

    # ===================================================
    # Location
    # ===================================================
    location_indexing = location_preprocess(ottawa.Location)
    ottawa['Street-Name'] = location_indexing[:, 0]
    ottawa['Intersection-1'] = location_indexing[:, 1]
    ottawa['Intersection-2'] = location_indexing[:, 2]


    neighbourhoods = ottawa.Neighborhood

    location_dimension = ottawa.groupby(
        ['Street-Name', 'Intersection-1', 'Intersection-2',
            'Neighborhood', 'Latitude', 'Longitude']).size().reset_index()

    location_dimension.drop(
        columns=location_dimension.columns[-1], inplace=True)
    location_dimension.drop_duplicates(inplace=True)
    location_index = pd.Index(location_dimension)
    ottawa['location_key'] = [location_index.get_loc(
        tuple(i)) for i in ottawa[location_dimension.columns].values]
    location_dimension['City'] = [
        'OTTAWA' for i in range(
            location_dimension.shape[0])]

    location_dimension.to_csv('./datasets/tmp/location_dimension.csv')
    # ===================================================
    # Grouping suggorgate key
    # ===================================================

    ac_index = pd.Index(accident_dimension[value])

    ottawa_accident_value = []

    for row in ottawa[key].values:
        ottawa_accident_value.append(
            ac_index.get_loc(tuple(row)))

    ottawa['Accident_key'] = np.array(ottawa_accident_value)
    index = pd.Index(hour_dimension[['hour', 'Date']])

    def hour_parser(hr, date):
        result = hour_dimension[
            (hour_dimension['Date'] == date) &
            (hour_dimension['hour'] == hr)]
        return result.index.values[0]

    ottawa['hour_key'] = ottawa.hour.combine(
        ottawa.Date,
        hour_parser)
    ottawa.to_csv('./datasets/tmp/ottawa.csv',index=False)
    return ottawa


def calgary_dimension(calgary):
    # ==========================
    # Accident_Dim
    # ==========================
    # Environment :-> Visibility
    # Traffic-Control :-> TRAFFCTL
    # Road-Surface :-> RDSFCOND
    # Light:-> LIGHT
    # ImpactType :-> IMPACTTYPE
    # Neighbour :-> Comm

    if os.path.isfile('./datasets/tmp/calgary.csv'):
        return pd.read_csv('./datasets/tmp/calgary.csv')

    key = [
        'DATE',
        'COLLISION_LOCATION',
        'COLLISION_SEVERITY',
        'COMM_NAME'
        'LATITUDE',
        'LONGITUDE']

    val = [
        'Date',
        'Location',
        'Collision_Classification',
        'Neighborhood'
        'Latitude',
        'Longitude']

    new_cols = dict(zip(key, val))
    calgary.rename(index=str, columns=new_cols, inplace=True)

    calgary['Is_Fatal'] = calgary.Collision_Classification.apply(
        lambda collision_class: 1 if 'FATAL' == collision_class else 0)

    calgary['Date'] = calgary.Date.apply(
        lambda date: datetime.strptime(
            date, '%Y/%m/%d'))

    return calgary


def toronto_dimension(toronto):

    toronto = toronto[toronto.YEAR.isin([2014, 2015, 2016, 2017])]
    all_impact_types = {
        'SMV Other': '07 - SMV other',
        'Angle': '02 - Angle',
        'Turning Movement': '05 - Turning movement',
        'SMV Unattended Vehicle': '06 - SMV unattended vehicle',
        'Cyclist Collisions': '99 - Other',
        'Pedestrian Collisions': '99 - Other',
        'Other': '99 - Other',
        'Sideswipe': '04 - Sideswipe',
        'Approaching': '01 - Approaching',
        'Rear End': '03 - Rear end'}

    all_environment_type = {
        'Rain': '02 - Rain',
        'Clear': '01 - Clear',
        'Snow': '03 - Snow',
        ' ': '99 - Other',
        'Other': '99 - Other',
        'Freezing Rain': '04 - Freezing Rain',
        'Fog, Mist, Smoke, Dust': '07 - Fog, mist, smoke, dust',
    }

    all_traffic_type = {
        'Traffic Signal': '01 - Traffic signal',
        'No Control': '10 - No control',
        'Stop Sign': '02 - Stop sign',
        'Pedestrian Crossover': '04 - Ped. crossover'
    }

    all_road_surface_type = {
        'Dry': '01 - Dry',
        'Wet': '02 - Wet',
        'Other': '99 - Other',
        'Loose Snow': '03 - Loose snow',
        'Slush': '04 - Slush',
        ' ': '99 - Other',
        'Ice': '06 - Ice'
    }

    toronto.drop(
        columns=[
            'FID',
            'INITDIR',
            'VEHTYPE',
            'MANOEUVER',
            'DRIVACT',
            'DRIVCOND',
            'PEDTYPE',
            'PEDACT',
            'PEDCOND',
            'CYCLISTYPE',
            'CYCACT',
            'CYCCOND',
            'PEDESTRIAN',
            'FATAL_NO',
            'OFFSET',
            'CYCLIST',
            'AUTOMOBILE',
            'MOTORCYCLE',
            'TRUCK',
            'TRSN_CITY_VEH',
            'EMERG_VEH',
            'PASSENGER',
            'SPEEDING',
            'AG_DRIV',
            'REDLIGHT',
            'ALCOHOL',
            'INVAGE',
            'DISABILITY',
            'Division',
            'Ward_Name',
            'Ward_ID',
            'Hood_ID',
            'INVTYPE',
            'ACCLASS',
            'Hood_Name'],
        inplace=True)

    # ====================================================
    # Accident_Dimension
    # ====================================================

    def parse_time_string(x):
        int_4_str = (4 - len(str(x))) * '0' + str(x)
        return int_4_str[:2] + ':' + int_4_str[2:] + ':00'

    toronto['Impact_type'] = toronto['IMPACTYPE'].apply(
        lambda x: all_impact_types[x])

    toronto['Environment'] = toronto['VISIBILITY'].apply(
        lambda x: all_environment_type[x])

    toronto['Road_Surface'] = toronto['RDSFCOND'].apply(
        lambda x: all_road_surface_type[x])

    toronto['Traffic_Control'] = toronto['TRAFFCTL'].apply(
        lambda x: all_traffic_type[x])

    toronto['Accident-time'] = toronto['TIME'].apply(
        lambda x: parse_time_string(x))

    extra_dimension = []

    if os.path.isfile('./datasets/tmp/accident_dimension.csv'):
        accident_dimension = pd.read_csv(
            './datasets/tmp/accident_dimension.csv',
            low_memory=False)
        accident_dimension.drop(
            columns=accident_dimension.columns[0],
            inplace=True)
        toronto_accident = toronto.groupby(
            ['Accident-time',
             'Environment',
             'Road_Surface',
             'Traffic_Control',
             'Impact_type']).size().reset_index()

        retrieval_cols = [
            'Accident-time',
            'Environment',
            'Road_Surface',
            'Traffic_Control',
            'Impact_type']
        toronto_index = pd.Index(accident_dimension[retrieval_cols]).values

        def check_exists(row):
            if tuple(row) not in toronto_index:
                extra_dimension.append(row)
        apple = [check_exists(x)
                 for x in toronto_accident[retrieval_cols].values]

        toronto_accident.drop(
            columns=toronto_accident.columns[-1], inplace=True)
        accident_dimension = pd.concat(
            [accident_dimension, toronto_accident], ignore_index=True)
        accident_dimension.drop_duplicates(inplace=True)
        toronto['Accident_key'] = [
            accident_dimension[
                np.all(
                    (value == accident_dimension[:]).values, 1)].index[0]
            for value in toronto[accident_dimension.columns].values]
        accident_dimension.to_csv('./datasets/tmp/accident_dimension.csv')
    else:
        raise FileNotFoundError("Preprocessed Ottaw dimension first")

    # =================================================
    # weather_dimension
    # =================================================

    toronto['1_closest'] = vectorizes_compute_dist(
        degree2rad(toronto[['LATITUDE', 'LONGITUDE']].values),
        'toronto', 1)[:, 0]
    toronto['DateTime'] = toronto.DATE.combine(
        toronto['Accident-time'],
        lambda date, time:
            date[:10] + ' ' + time[:5])
    cond = True
    if os.path.isfile('./datasets/tmp/weather_dimension.csv'):
        weather_dimension = pd.read_csv(
            './datasets/tmp/weather_dimension.csv',)
        toronto_weathers = toronto_weather[toronto_weather.DateTime.isin(
            toronto.DateTime.unique())]
        toronto_weathers['City'] = [
            'Toronto' for i in range(
                toronto_weathers.shape[0])]
        try:
            weather_dimension.drop(
                columns=[
                    'Unnamed: 0',
                    '0',
                    'Inverted_index'],
                inplace=True)
        except KeyError:
            cond = False
        if cond:
            weather_dimension = pd.concat([
                weather_dimension,
                toronto_weathers[weather_dimension.columns]], ignore_index=True)

            weather_dimension.drop_duplicates(inplace=True)
            weather_dimension.fillna('Unknown', inplace=True)
            weather_dimension.to_csv('./datasets/tmp/weather_dimension.csv')
    else:
        raise FileNotFoundError('Missing preprocessed dimensions')

    def parse_weather_key(datetime, station_name):
        result = weather_dimension[
            (weather_dimension['DateTime'] == datetime) &
            (weather_dimension['Name'] == station_name)
        ]
        if result.shape[0] == 0:
            return 636
        else:
            return result.index

    toronto['weather_key'] = toronto.DateTime.combine(
        toronto['1_closest'],
        parse_weather_key)

    toronto['Is_Fatal'] = [1 for i in range(toronto.shape[0])]

    # =================================================
    # location_dimension
    # =================================================

    toronto.STREET2.replace(' ', 'Unknown', inplace=True)
    toronto['Is_Intersection'] = toronto.STREET2.apply(
        lambda street2:
            1 if street2 != 'Unknown' else 0)

    location_result = np.vstack(toronto.STREET1.combine(
        toronto.STREET2,
        lambda st1, st2:
            toronto_preprocess(st1, st2)))
    toronto['Street-Name'] = location_result[:, 0]
    toronto['Intersection-1'] = location_result[:, 1]
    toronto['Intersection-2'] = location_result[:, 2]
    toronto.rename(index=str, columns={
        'District': 'Neighborhood',
        'LONGITUDE': 'Longitude',
        'LATITUDE': 'Latitude',
    }, inplace=True)
    toronto_location_dim = toronto.groupby(
        ['Street-Name', 'Intersection-1', 'Intersection-2',
            'Neighborhood', 'Latitude', 'Longitude']).size().reset_index()
    toronto_location_dim.drop(
        columns=toronto_location_dim.columns[-1], inplace=True)
    toronto_location_dim['City'] = [
        'Toronto' for i in range(
            toronto_location_dim.shape[0])]

    if os.path.isfile('./datasets/tmp/location_dimension.csv'):
        location_dim = pd.read_csv('./datasets/tmp/location_dimension.csv')
        location_dim.drop(columns=[location_dim.columns[0]], inplace=True)
        location_dim = pd.concat(
            [
                toronto_location_dim[location_dim.columns],
                location_dim
            ], ignore_index=True)
        location_dim.drop_duplicates(inplace=True)
        location_dim.to_csv('./datasets/tmp/location_dimension.csv')
    else:
        raise FileNotFoundError('location_dimension is not ready')
    location_index = pd.Index(location_dim[['Street-Name', 'Intersection-1']])
    toronto['location_key'] = toronto['Street-Name'].combine(
        toronto['Intersection-1'],
        lambda s1, i1:
            location_dim[
                (location_dim['Street-Name'] == s1) &
                (location_dim['Intersection-1'] == i1)].index[0])

    # ====================================================
    # hour_dimension
    # ====================================================
    toronto['Date'] = toronto.DATE.apply(
        lambda date: datetime.strptime(
            date[:10], '%Y-%m-%d'))

    toronto['month'] = toronto.Date.apply(
        lambda date: date.month)

    toronto['day'] = toronto.Date.apply(
        lambda date: date.day)
    toronto['day_of_the_week'] = toronto.Date.apply(
        lambda date: date.weekday())
    toronto['weekend'] = toronto.day_of_the_week.apply(
        lambda date: 1 if date in [5, 6] else 0)
    toronto['holiday'] = toronto.Date.apply(
        lambda date: 1 if date in Canadian_Holidays else 0)
    toronto['holiday_name'] = toronto.Date.apply(
        lambda date: Canadian_Holidays.get(date)
        if Canadian_Holidays.get(date) else 'Unknown')
    toronto.rename(
        index=str,
        columns={
            'YEAR': 'year',
            'Hour': 'hour'
        },
        inplace=True)
    toronto['Date'] = toronto.Date.apply(
        lambda date: str(date)[:10])
    if os.path.isfile('./datasets/tmp/hour_dimension.csv'):
        hour = pd.read_csv('./datasets/tmp/hour_dimension.csv')
        toronto_hour = toronto.groupby(
            ['year', 'month', 'day', 'hour',
                'Date', 'day_of_the_week', 'weekend',
                'holiday', 'holiday_name'], as_index=False)\
            .size().reset_index()
        toronto_hour.drop(columns=[toronto_hour.columns[-1]], inplace=True)
        hour = pd.concat([hour, toronto_hour], ignore_index=True)
        hour.drop_duplicates(inplace=True)
        hour.to_csv('./datasets/tmp/hour_dimension.csv')
        toronto['hour_key'] = toronto.Date.combine(
            toronto.hour,
            lambda date, hr:
                hour[
                    (hour['Date'] == date) &
                    (hour['hour'] == hr)].index.values[0])
    final_cols = [
        'hour_key',
        'location_key',
        'Accident_key',
        'weather_key',
        'Is_Fatal',
        'Is_Intersection']
    ottawa = pd.read_csv(
        './datasets/tmp/ottawa.csv',
        low_memory=False)[final_cols]

    mix_result = pd.concat(
        [ottawa, toronto[ottawa.columns]], ignore_index=True)
    mix_result.to_csv('./datasets/tmp/ottawa_star_schema.csv', index=False)
    return mix_result


def staging():
    ottawa, calgary, toronto = read_data()

    ottawa = ottawa_dimensions(ottawa)

    toronto = toronto_dimension(toronto)

    return ottawa, toronto



if __name__ == "__main__":
    toronto = staging()
