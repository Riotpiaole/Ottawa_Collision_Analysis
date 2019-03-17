import os
import psycopg2
import pandas as pd

if __name__ == "__main__":
    conn = psycopg2.connect(
        host="137.122.24.222",
        database="group_22",
        user="rlian072",
        password="Monday6867",
        port=15432)
    result_dir = './datasets/result/'
    all_table = {}

    def dir_concat(x): return os.path.join(result_dir, x)
    for path in os.listdir(result_dir):
        all_table[path[:-4]] = pd.read_csv(dir_concat(path))
