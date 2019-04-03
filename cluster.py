import pandas as pd
import numpy as np
from sklearn import preprocessing
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler

collision_2014 = pd.read_csv(
    './2014collisionsfinal.csv', low_memory=False)


collision_2014.dropna(inplace=True)

(m, n) = collision_2014.shape
min_max_scaler = preprocessing.MinMaxScaler()

# normalizing the lat and long of the collision datasets
# min max normalizing for converting the datasets in range[0,1]
lat_scaled = min_max_scaler.fit_transform(
    collision_2014.Latitude.values.reshape(m, 1))
long_scaled = min_max_scaler.fit_transform(
    collision_2014.Longitude.values.reshape(m, 1))


collision_2014['Lat'] = lat_scaled
collision_2014['long'] = long_scaled


# one hot encoding each span of each desire columns
datasets = pd.get_dummies(
    collision_2014[['Collision_Classification', 'Traffic_Control', 'Lat', 'long']])

desire_cols = [
    'Latitude',
    'Longitude',
    'Fatal_Injury',
    'Non-fatal_injury',
    'P.D._only',
    'Traffic_signal',
    'Stop_sign',
    'Yield_sign',
    'School_bus',
    'Traffic_gate',
    'Traffic_controller',
    'No_control',
    'Roundabout']

new_col_dict = dict(zip(
    datasets.columns.values,
    desire_cols))

datasets = datasets.rename(
    index=str,
    columns=new_col_dict)

# db scan
from sklearn.cluster import DBSCAN
db = DBSCAN(
    n_jobs=4
).fit(
    datasets[desire_cols[2:]].values)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
datasets['labels'] = db.labels_

unique_labels = set(db.labels_)
labels = db.labels_

colors = ['o', 'v', '^', '<', '>', '1', '8', 's', 'p', '*', 'h', 'd', 'x']
dicts = zip(unique_labels, colors)

colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)


ax = plt.axes(projection='3d')
# ploting z with label and x y in lat and long
for index, (label, color) in enumerate(dicts):
    curr_data = datasets[datasets.labels == label]
    ax.scatter3D(
        curr_data.Latitude.values,
        curr_data.Longitude.values,
        curr_data.labels.values,
        label='class {}'.format(label),
        marker=color[0]
    )
plt.show()

# K-mean cluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(datasets[desire_cols[2:]])
print(pca.explained_variance_ratio_)
 k_mean = KMeans(
    n_clusters=2,
    random_state=0).fit(
        datasets[datasets.columns[3:]])

labels = k_mean.labels_
unique_labels = set(k_mean.labels_)

datasets['k_labels'] = labels

dicts = zip(unique_labels, colors)

closest, _ = pairwise_distances_argmin_min(k_mean.cluster_centers_, datasets[datasets.columns[3:-1]])


for index, (label, color) in enumerate(dicts):
    curr_data = datasets[datasets.k_labels == labels][:10]
    ax.scatter3D(
        curr_data.Latitude.values,
        curr_data.Longitude.values,
        curr_data.k_labels.values,
        label='class {}'.format(index),
        marker=color[0]
    )
plt.show()
