import re
import os
import pandas as pd
from multiprocessing import Pool, cpu_count

climate_dir = './datasets/climates/'
collision_dir = './datasets/collisions/'
regex = re.compile(r'\w+\.csv')


def climate_dirs(path):
    return os.path.join('./datasets/climates/', path)

def read_weather(path):
    print("Process check pid[%d]"% os.getpid())
    data = pd.read_csv(path)
    result = data[data['Year'].isin([2014,2015,2016,2017])].copy()
    del data
    print("[%d] processing complete"%os.getpid())
    return result

def shrink(province):
    all_paths = [
        os.path.join(climate_dirs(province),path)
        for path in list(filter(
        regex.match
        ,os.listdir(climate_dirs(province))))]
    with Pool(processes=cpu_count()) as pool:
        df_list = pool.map(read_weather, all_paths)
    in_date_climate = pd.concat(df_list, sort=True, ignore_index=True)
    in_date_climate.to_csv(climate_dirs("14-17_ontario.csv"),index=False)
    # return in_date_climate


if __name__ == "__main__":
    datasets = shrink('Ontario')

