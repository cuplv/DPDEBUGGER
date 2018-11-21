"""
=========================================================
Comparing different clustering algorithms on toy datasets
=========================================================

This example aims at showing characteristics of different
clustering algorithms on datasets that are "interesting"
but still in 2D. The last dataset is an example of a 'null'
situation for clustering: the data is homogeneous, and
there is no good clustering.

While these examples give some intuition about the algorithms,
this intuition might not apply to very high dimensional data.

The results could be improved by tweaking the parameters for
each clustering strategy, for instance setting the number of
clusters for the methods that needs this parameter
specified. Note that affinity propagation has a tendency to
create many clusters. Thus in this example its two parameters
(damping and per-point preference) were set to mitigate this
behavior.
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import argparse
import pandas as pd
from math import exp

argparser = argparse.ArgumentParser()

argparser.add_argument("--filename", help="input_file", required=False)

argparser.add_argument("--measurements", help="is 10-measurements provided", default = "yes", required=False)

argparser.add_argument("--clusters", help="number of clusters", default = 2, required=False)

argparser.add_argument("--featurex", help="the cluster variable feature", default = 'id', required=False)

argparser.add_argument("--output", help="name of output file", required=False)

args = argparser.parse_args()

filename = args.filename
df = pd.read_csv(filename)
measurements = args.measurements
cluster_num = args.clusters
featurex = args.featurex
cluster_image = args.output
cluster_output = args.output
if(cluster_num == ""):
    cluster_numbers = 2
else:
    try:
        cluster_numbers = int(cluster_num)
    except ValueError:
        print("Clusrer_number should be integer")

if(measurements == "yes" or measurements == "y" or measurements == ""):
    df_T = df[['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10']]
    df['mean'] = (df['T1'] + df['T2'] + df['T3'] + df['T4'] + df['T5'] + df['T6'] + df['T7'] + df['T8'] + df['T9'] + df['T10'])/(10)
    df['mean'] = [int(x) for x in df['mean']]
X = df[featurex]
y = df['mean']


df['mean'] = [i for i in df['mean']]
df[featurex] = [i for i in df[featurex]]
mean_id = np.array(df[[featurex,'mean']]).reshape(-1,2);
mean_id_original = mean_id
np.random.seed(0)

clusters = [0 for x in df['mean']]
data_new_id = [0 for x in df['mean']]
weight = [100 for x in df['mean']]

colors = np.array([x for x in 'brgcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)


clustering_names = ['SpectralClustering']
plot_num = 1

start = time.time()

datasets = [mean_id]
for i_dataset, dataset in enumerate(datasets):

    # normalize dataset for easier parameter selection
    mean_id = StandardScaler().fit_transform(mean_id)
    spectral = cluster.SpectralClustering(
    n_clusters=cluster_numbers, eigen_solver='arpack',
    gamma =1.0,affinity="alignment")
    clustering_algorithms = [spectral]

    for name, algorithm in zip(clustering_names, clustering_algorithms):
        # predict cluster memberships
        t0 = time.time()
        algorithm.fit(np.asarray(mean_id))
        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(mean_id)
        end = time.time()
        print "Clustering takes (seconds): " + str(end - start)
        plt.rc('font', weight='bold')
        plt.rc('xtick', labelsize=24)
        plt.rc('ytick', labelsize=24)
        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Number of data points',fontsize=28, fontweight='bold',color='b')
        ax.set_ylabel('Time (S)',fontsize=28, fontweight='bold',color='b')
    
        plt.scatter(mean_id_original[:, 0], mean_id_original[:, 1], color=colors[y_pred].tolist(), s=75, linewidths=2)
        print "Cluster 0 is blue, cluster 1 is red, cluster 2 is green, cluster 3 is cyan, cluster 4 is purple."
        
        counter = 0
        for x in mean_id:
            clusters[counter] = y_pred[counter]
            counter = counter + 1
        
        df['label'] = np.array(clusters).reshape(-1,1)
        df['weight'] = np.array(weight).reshape(-1,1)
        df.to_csv(cluster_output+".csv")

        plot_num += 1

plt.show()
