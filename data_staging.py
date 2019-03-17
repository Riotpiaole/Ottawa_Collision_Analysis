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

pool = Pool(processes=cpu_count())


def collision_dir(city):
    return os.path.join('./datasets/collisions/', city)


climate_dir = './datasets/climates/'
station = pd.read_csv(
    climate_dir +
    'Station-Inventory-EN.csv',
    skiprows=3)

station = station[station['Province'].isin(['ALBERTA', 'ONTARIO'])]
station_final = station[station.Name.str.contains('OTTAWA|TORONTO|calgary')]

station_ontario = station[station.Name.str.contains('OTTAWA|TORONTO')]

station_ottawa = station[station.Name.str.contains('OTTAWA')]
station_calgary = station[station.Name.str.contains('calgary')]
station_toronto = station[station.Name.str.contains('TORONTO')]

ottawa_station_lookup = pd.Index(station_ottawa.Name)
toronto_station_lookup = pd.Index(station_ottawa.Name)
calgary_station_lookup = pd.Index(station_ottawa.Name)


ottawa_weather, toronto_weather = combine_ontario()


def round_minutes(dt, resoultion):
    new_minute = (dt.minute // resolution + 1)


def read_data()->'tuple(pd.DataFrame)|3':
    '''Obtain all of the citys data'''
    ottawa, calgary, toronto = None, None, None
    print("Retrieveing data from collision datasets")
    if os.path.isfile('./datasets/result/ottawa.csv'):
        ottawa = pd.read_csv('./datasets/result/ottawa.csv')
    else:
        # obtain ottawa
        ottawa_files = os.listdir(collision_dir('Ottawa'))
        ottawa_file_names = map(
            lambda x: join_path(collision_dir('Ottawa'), x),
            list(filter(regex.match, ottawa_files)))

        df_list = pool.map(pd.read_excel, ottawa_file_names)

        if 'Year' in df_list[-1].columns.values:
            df_list[-1].drop('Year', axis=1, inplace=True)
            df_list[-1].to_excel(df_list[-1])

        ottawa = pd.concat(df_list, sort=True, ignore_index=True)
        ottawa.dropna(inplace=True)
        pool.close()

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


regex = re.compile(r'\w+\.?(xls|xlsx)')


def test_lat_lon_dist(dist: 'func'):
    pt1 = (38.897147, -77.043934)
    pt2 = (38.898556, -77.037852)
    result = 0.613  # km
    assert(result == round(dist(pt1, pt2), 3))


@np.vectorize
def degree2rad(x):
    return x * pi / 180


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
        lookup: 'pd.DataFrame')-> 'pd.DataFrame':
    '''generating weather dimension suggregate key'''
    rt, rn = row
    return lookup.get_loc((str(rt)[:-3], rn))


def vectorize_weather_station(
        table: 'clloision_table',
        weather_table: 'ndarray')-> 'pd.DataFrame':
    '''merge weather station table vectorizly'''
    lookup = pd.Index(weather_table[['DateTime', 'Name']])

    return np.apply_along_axis(
        suggregate_weather_station,
        -1, table[['Datetime', '1_closest']],
        lookup)


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

def ottawa_dimensions(ottawa):
    if os.path.isfile('./datasets/result/ottawa.csv'):
        ottawa = pd.read_csv('./datasets/result/ottawa.csv')[
            ['hour_key',
                'location_key',
                'Accident_key',
                'weather_key',
                'Is_Fatal',
                'Is_Intersection']]
        ottawa.to_csv('./datasets/result/ottawa_star_schema.csv', index=False)
        return ottawa

    # generate is fatal columns
    ottawa['Is_Fatal'] = ottawa.Collision_Classification.apply(
        lambda collision_class: 1 if collision_class == '01 - Fatal injury' else 0)
    ottawa['Is_Intersection'] = ottawa.Location.apply(
        lambda location: 1 if '@' in location else 0)


    # handling time after 12 am would parse it as string
    ottawa.Time = ottawa.Time.apply(
        lambda x: x if isinstance(
            x, time) else datetime.strptime(
            x, '%H:%M:%S').time())

    # combining date and time
    ottawa['Datetime'] = ottawa.Date.combine(
        ottawa.Time, lambda x, y: datetime.combine(x, y))

    # rounding up 1 hour time region upper bound
    ottawa['Datetime'] = ottawa.Datetime.apply(
        lambda x: pd.Timestamp(x)).dt.round('H')

    ottawa['year'] = ottawa['Date'].apply(lambda x: x.year)
    ottawa['month'] = ottawa['Date'].apply(lambda x: x.month)
    ottawa['day'] = ottawa['Date'].apply(lambda x: x.day)
    ottawa['hour'] = ottawa['Time'].apply(lambda x: x.hour)

    # ===================================================
    # Hour dimension
    # ===================================================
    if os.path.isfile('./datasets/result/hour_dimension.csv'):
        print("hour_dimension file found")
        hour_dimension = pd.read_csv('./datasets/result/hour_dimension.csv')
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
             'holiday', 'holiday_name'])\
            .size().reset_index()

        hour_dimension.drop(columns=hour_dimension.columns[-1], inplace=True)
        hour_dimension.to_csv(
            './datasets/result/hour_dimension.csv')

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

    # ===================================================
    #  Weather Dimension
    # ===================================================

    weather_dimension = None
    if os.path.isfile('./datasets/result/weather_dimension.csv'):
        print("Weather Dimension found loading the file")
        weather_dimension = pd.read_csv(
            './datasets/result/weather_dimension.csv')

    else:
        print("Preprocessing weather dimension")
        # generating compute the distance against all of the ottawa stations
        closest_weather_stations = vectorizes_compute_dist(
            degree2rad(ottawa[['Latitude', 'Longitude']].values),
            'ottawa', 3)
        # find the top three closest stations

        ottawa['1_closest'] = closest_weather_stations[:, 0]

        # remove the ottawa_weather that is exists in ottawa
        ottawa_weathers = ottawa_weather[
            ottawa_weather.DateTime.isin(
                ottawa['Datetime'].apply(lambda x: str(x)[:-3]).values)
        ].reset_index(drop=True)

        weather_dimension = ottawa_weathers[
            ottawa_weathers.Name.isin(
                ottawa['1_closest'].unique())
        ].reset_index(drop=True)

        ottawa['weather_key'] = vectorize_weather_station(
            ottawa, weather_dimension)

        weather_dimension = weather_dimension[weather_dimension.index.isin(
            ottawa['weather_key'].unique())]

        # inverting index for remove un-needed
        weather_dimension['Inverted_index'] = weather_dimension.index
        weather_dimension.reset_index(inplace=True, drop=True)

        weather_dimension['Latitude'] = weather_dimension.Name.apply(
            lambda x: station_ottawa.iloc[
                ottawa_station_lookup.get_loc(x)]
            ['Latitude']) / 10000000

        weather_dimension['Longitude'] = weather_dimension.Name.apply(
            lambda x: station_ottawa.iloc[
                ottawa_station_lookup.get_loc(x)]
            ['Longitude']) / 10000000

        weather_index_lookup = pd.Index(weather_dimension['Inverted_index'])
        ottawa['weather_key'] = ottawa['weather_key'].apply(
            lambda x: weather_index_lookup.get_loc(x))

        weather_dimension.to_csv(
            './datasets/result/weather_dimension.csv')

    # ===================================================
    # Location
    # ===================================================
    location_indexing = location_preprocess(ottawa.Location)
    ottawa['Street-Name'] = location_indexing[:, 0]
    ottawa['Intersection-1'] = location_indexing[:, 1]
    ottawa['Intersection-2'] = location_indexing[:, 2]

    all_csv = filter(re.compile(r'\w+\.csv').match, os.listdir('.'))
    neighbourhoods = [pd.read_csv(i, engine='python') for i in all_csv]
    neighbourhoods = pd.concat(neighbourhoods, sort=True, ignore_index=True)

    # replacing unicode error
    neighbourhoods.replace(np.nan, 'unknown', inplace=True)
    neighbourhoods.Neighborhood = neighbourhoods.Neighborhood.combine(
        neighbourhoods.Record,
        lambda neighbour, record:
            record if neighbour == 'unknown' else neighbour)

    #TODO jackline lat long issues
    ottawa['Neighborhood'] = neighbourhoods.Neighborhood[:-45]

    location_dimension = ottawa.groupby(
        ['Street-Name', 'Intersection-1', 'Intersection-2',
            'Neighborhood', 'Latitude', 'Longitude']).size().reset_index()

    location_dimension.drop(
        columns=location_dimension.columns[-1], inplace=True)

    location_index = pd.Index(location_dimension)
    ottawa['location_key'] = [location_index.get_loc(
        tuple(i)) for i in ottawa[location_dimension.columns].values]

    location_dimension.to_csv('./datasets/result/location_dimension.csv')

    # ===================================================
    #TODO Event
    # ===================================================

    # event_data = pd.read_csv('./events_annual_Ottawa_14-17.csv')
    # event_data.drop(columns=['Event-key'], inplace=True)

    # def convert_str_date(date): return datetime.strptime(date, '%Y-%m-%d')

    # event_data['Event-start-date'] = event_data['Event-start-date'].apply(
    #     convert_str_date)
    # event_data['Event-end-date'] = event_data['Event-end-date'].apply(
    #     convert_str_date)

    # ===================================================
    # Grouping suggorgate key
    # ===================================================

    ac_index = pd.Index(accident_dimension[value])

    ottawa_accident_value = []

    for row in ottawa[key].values:
        ottawa_accident_value.append(
            ac_index.get_loc(tuple(row)))

    ottawa['Accident_key'] = np.array(ottawa_accident_value)

    hour_index = pd.Index(hour_dimension[['hour', 'Date']])
    ottawa['hour_key'] = ottawa.hour.combine(
        ottawa.Date,
        lambda hour, date:
            hour_index.get_loc((hour, str(date)[:-9])))



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

    if os.path.isfile('./datasets/result/calgary.csv'):
        return pd.read_csv('./datasets/result/calgary.csv')

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
        'SMV Other':'07 - SMV other',
        'Angle':'02 - Angle',
        'Turning Movement':'05 - Turning movement',
        'SMV Unattended Vehicle':'06 - SMV unattended vehicle',
        'Cyclist Collisions':'99 - Other',
        'Pedestrian Collisions':'99 - Other',
        'Other':'99 - Other',
        'Sideswipe':'04 - Sideswipe',
        'Approaching':'01 - Approaching',
        'Rear End':'03 - Rear end'}

    all_environment_type ={
        'Rain':'02 - Rain',
        'Clear':'01 - Clear',
        'Snow':'03 - Snow',
        ' ':'99 - Other',
        'Other':'99 - Other',
        'Freezing Rain':'04 - Freezing Rain',
        'Fog, Mist, Smoke, Dust':'07 - Fog, mist, smoke, dust',
    }

    all_traffic_type = {
        'Traffic Signal':'01 - Traffic signal',
        'No Control':'10 - No control',
        'Stop Sign':'02 - Stop sign',
        'Pedestrian Crossover':'04 - Ped. crossover'
    }

    all_road_surface_type = {
        'Dry':'01 - Dry',
        'Wet':'02 - Wet',
        'Other':'99 - Other',
        'Loose Snow':'03 - Loose snow',
        'Slush':'04 - Slush',
        ' ':'99 - Other',
        'Ice':'06 - Ice'
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
        return int_4_str[:2]+ ':' + int_4_str[2:] + ':00'

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

    if os.path.isfile('./datasets/result/accident_dimension.csv'):
        accident_dimension = pd.read_csv('./datasets/result/accident_dimension.csv')
        toronto_accident = toronto.groupby(
            ['Accident-time',
            'Environment',
            'Road_Surface',
            'Traffic_Control',
            'Impact_type']).size().reset_index()

        retrieval_cols = ['Accident-time','Environment','Road_Surface','Traffic_Control','Impact_type']
        toronto_index = pd.Index(accident_dimension[retrieval_cols]).values
        def check_exists(row):
            if tuple(row) not in toronto_index:
                extra_dimension.append(row)
        apple = [ check_exists(x) for x in toronto_accident[retrieval_cols].values]

        toronto_accident.drop(columns=toronto_accident.columns[-1],inplace=True)


    # toronto['closest_station'] = vectorize_weather_station(
    #     degree2rad(toronto[['LATITUDE','LONGITUDE']].values),
    #     'toronto')[:,0]

    # toronto['is_fatal'] = [1 for i in range(toronto.shape[0])]
    return toronto , toronto_accident, extra_dimension


def staging():
    # obtain all of the holidays
    Canadian_Holidays = holidays.Canada()

    ottawa, calgary, toronto = read_data()

    ottawa = ottawa_dimensions(ottawa)
    calgary = calgary_dimension(calgary)
    toronto , toronto_accident , extra_dimension= toronto_dimension(toronto)

    return ottawa, calgary, toronto , toronto_accident ,extra_dimension


if __name__ == "__main__":
    ottawa, calgary, toronto, toronto_accident , extra_dimension = staging()
    # not_change = datetime.time(8,30)
    # change = datetime.time(8,14)
