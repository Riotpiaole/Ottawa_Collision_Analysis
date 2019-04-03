import re
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


collision_2014 = pd.read_csv(
    './2014collisionsfinal.csv', low_memory=False)
scaler = preprocessing.MinMaxScaler()

rear_end = collision_2014[collision_2014.Impact_type ==
                          '03 - Rear end']

x_norm, y_norm = scaler.fit_transform(rear_end.X.values.reshape(-1, 1)),\
    scaler.fit_transform(rear_end.Y.values.reshape(-1, 1))

rear_end['X_norm'] = x_norm
rear_end['Y_norm'] = y_norm

rear_end[['X_norm','Y_norm']].to_csv('./norm_x.csv',index=False)
from sklearn.cluster import DBSCAN
# db = DBSCAN(
#     eps=.002,
#     min_samples=30,
#     n_jobs=4
# ).fit(
#     rear_end[['X_norm', 'Y_norm']])

db = DBSCAN(
    eps = .0055,
    min_samples = 10,
    n_jobs = -2 # Use all CPUs except 1
).fit(
    rear_end[['X_norm', 'Y_norm']])

df = pd.DataFrame({'label': db.labels_})




# Visualization
rear_end['cluster'] = db.labels_

import matplotlib.pyplot as plt

def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

cmap = get_cmap(len(rear_end['cluster'].unique()) * 1.25)

color_a = []
for i in range(len(rear_end['cluster'].unique())):
    color_a.append(cmap(i))
del color_a[len(color_a) - 1]
color_a.append('black')

color_array = np.asarray(color_a)

plt.scatter(rear_end['X_norm'],
            rear_end['Y_norm'],
            c=color_array[rear_end['cluster']],
            alpha= 0.3)
plt.xlabel('X_norm')
plt.ylabel('Y_norm')



# Evaluation and finding the best parameters using Silhouette Coefficient
from sklearn import metrics

X = rear_end[['X_norm', 'Y_norm']]

labels = db.labels_
score_scaled = metrics.silhouette_score(X, labels)
print('==================')
print('==================')
print('Current silhouette_score is:')
print(score_scaled)
print('==================')


### Find the best parameters using Silhouette Coefficient
###### Already using the best parameters ######
###### Time consuming, please uncomment following block is you want to use ######
###### ==================
###### ==================
###### the best silhouette_score is:
###### 0.07149377325615387
###### whose parameters [eps,min_samples] are:
###### [0.005500000000000001, 10]
###### ==================
###### ==================
# p_eps = .001
# scores_scaled = []
# score_information = []
# for p_min_samples in range(10,50):
#     while p_eps < .01:
#         labels = DBSCAN(
#             eps = p_eps,
#             min_samples = p_min_samples,
#             n_jobs = -2 # Use all CPUs except 1
#         ).fit(X).labels_
#
#         score = metrics.silhouette_score(X, labels)
#         scores_scaled.append(score)
#         score_information.append([p_eps,p_min_samples])
#         p_eps += .0005
#
# max = max(scores_scaled)
# index = scores_scaled.index(max)
#
# print('==================')
# print('the best silhouette_score is:')
# print(max)
# print('whose parameters [eps,min_samples] are:')
# print(score_information[index])
# print('==================')
# print('==================')

###### End of the block ######
##############################
##############################



gg = df.groupby(
    by=df.label)



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

location_index = location_preprocess(rear_end.Location)

rear_end['Street-Name'] = location_index[:, 0]
rear_end['Intersection-1'] = location_index[:, 1]
rear_end['Intersection-2'] = location_index[:, 2]

print("======================================================================================")
print(gg.size().sort_values(ascending=False))

# 2 1 3
cols = ['Street-Name','Intersection-1','Intersection-2']


# print(street_name2.groupby(cols[0]).size().sort_values(ascending=False))
# print("======================================================================================")
# print(street_name1.groupby(cols).size().sort_values(ascending=False))
# print("======================================================================================")
# print(street_name3.groupby(cols).size().sort_values(ascending=False))
# print("======================================================================================")
# print(street_name5.groupby(cols).size().sort_values(ascending=False))
# print("======================================================================================")
# print(street_name15.groupby(cols).size().sort_values(ascending=False))
# print("======================================================================================")

# plt.show()

def street_count():
    count = 0
    ggs = []

    for i in [0, 19, 26, 5 ,15]:
        [ ggs.append(j) for j in list(gg.groups[i].values) ]
    ggs = list(set(ggs))
    return ggs

#     # print(rear_end.iloc[ggs].groupby(['Street-Name']).size().sort_values(ascending=False))

groups = street_count()
count = {}

def count_street(street):
    if street == 'Unknown':
        return
    if street not in count.keys():
        count[street] = 1
        return
    count[street] +=1


for row in rear_end.iloc[groups][cols].values:
    for streets in row:
        count_street(streets)
print("======================================================================================")
result_1 = pd.DataFrame(
    {
        'street_name':list(count.keys()),
        'count':list(count.values())
    }).sort_values(['count'],ascending=False)
print(result_1)
print("======================================================================================")
result_2 = rear_end.iloc[groups].groupby(
    cols[1:]).size().sort_values(ascending=False)
print(result_2)
print("======================================================================================")
result_3 = rear_end.iloc[groups].groupby(
    cols).size().sort_values(ascending=False)
print(result_3)
print("======================================================================================")