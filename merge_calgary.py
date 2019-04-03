import re
import os
import pandas as pd
from constant import desire_cols, rename_cols


def merge_table(old, new, path=''):
    result = pd.concat([old, new[old.columns]],ignore_index=True)
    result.drop_duplicates(inplace=True)
    result.to_csv('./datasets/new/' + path)
    return result


csv = re.compile(r'\w+.csv').match

weather, collision, hour, location = [
    pd.read_csv(os.path.join('./datasets/calgary_result', item), encoding="ISO-8859-1")
    for item in filter(csv, os.listdir('./datasets/calgary_result'))]

hour_final, location_final = pd.read_csv(
    './datasets/new/hour_dimension.csv'), pd.read_csv('./datasets/new/location_dimension.csv')
weather_final, fact_table = pd.read_csv(
    './datasets/new/weather_dimension.csv'), pd.read_csv('./datasets/new/ottawa_star_schema.csv')

rename_weather_cols = {
    'DATE': 'DateTime',
    'Wind.Spd..km.h.': 'WindSpeed',
    'Wind.Chill': 'WindChill',
    'Stn.Press..kPa.': 'Preasure',
    'lat': 'Latitude',
    'lon': 'Longitude',
    'Nearest Station': 'Name',
    'Visibility..km.': 'Vis',
    'Wind.Dir..10s.deg.': 'WindDir',
    'Temp...C.': 'Temp',
}


weather.rename(index=str, columns=rename_weather_cols, inplace=True)

weather['Province'] = ['Alberta' for i in range(weather.shape[0])]
weather['humidity'] = ['Unknown' for i in range(weather.shape[0])]
weather['DateTime'] = weather.DateTime.apply(
    lambda datetime: datetime + ' 10:00')

collision.fillna('Unknown',inplace=True)
collision['Is_Intersection'] = [ 1 if street == 'Unknown' else 0 for street in collision.STREET.values ]
collision['Is_Fatal'] = collision.COLLISION_SEVERITY.apply(
    lambda x : 1 if  'FATAL' == x else 0)

hour.hour.fillna(10,inplace=True)

# weather_final= merge_table(weather_final[ weather_final.columns[1:]],weather,'weather_dimension.csv')

# hour_final= merge_table(hour_final[hour_final.columns[1:]],hour,'hour_dimension.csv')

# location_final= merge_table(location[location_final.columns[1:]],location,'location_dimension.csv')


collision['Accident_key'] = [ 35364 for i in range(collision.shape[0])]
collision['hour_key'] =  collision.DATE.apply(
    lambda date:
        hour_final[
            (hour_final.Date == date)&
            (hour_final.hour == 10)].index.values[0])
collision['location_key'] = collision.STREET.combine(
    collision['INTERSECTION-1'],
    lambda street , I1:
        location_final[
            (location_final['Intersection-1']== street) &
            (location_final['Intersection-2'] == I1)].index.values[0])

def parse_weather(station,date):
    key = weather[(weather['Name'] == station)&
                (weather['DateTime'] == date + ' 10:00')].index
    if key.empty:
        return 636
    else:
        return int(key.values[0]) + 642

collision['weather_key'] = collision['Nearest Station'].combine(
    collision.DATE,
    parse_weather)

