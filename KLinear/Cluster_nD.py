# Clustering and other steps :)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from statistics import mean
import random
from scipy.stats import norm
import argparse
import time
import math
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn.linear_model import Ridge
import sys

# max_iter = 50 for micro-benchmarks
class K_Means:
    def __init__(self, k=2, tol=0.01, max_iter=50):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):
        self.centroids_line = {}

        data_len = len(data)
        data_x = data[:,2:]
        data_y = data[:,1]
        id = data[:,0]
        for i in range(self.k):
            self.centroids_line[i] = []

        for i in range(self.k):
            n1 = np.random.randint(low=0,high=data_len/2)
            n2 = np.random.randint(low=(data_len/2) + 1,high=data_len-1)
            row=[n1,n2]
            rows = data_x[np.ix_(row)]
            cols = data_y[np.ix_(row)]
            lr = linear_model.LinearRegression()
            lr.fit(rows, cols)
            for item in lr.coef_:
                self.centroids_line[i].append(item)
            self.centroids_line[i].append(lr.intercept_)

        for i in range(self.max_iter):
            self.line_classification = {}
            
            for i in range(self.k):
                self.line_classification[i] = []
            
            for j in range(len(data)):
                data_j = data_x[j]
                data_j = np.append(data_j,[1])
                distances_line = [np.linalg.norm(data_y[j]-(np.dot(self.centroids_line[k],data_j))) for k in range(self.k)]
                line_classification = distances_line.index(min(distances_line))
                self.line_classification[line_classification].append(data[j])

            for classification in self.line_classification:
                X = []
                y = []
                for record in self.line_classification[classification]:
                    X.append(record[2:])
                    y.append(record[1])
                lr = Ridge()
                lr.fit(X, y)
                self.centroids_line[classification] = []
                for item in lr.coef_:
                    self.centroids_line[classification].append(item)
                self.centroids_line[classification].append(lr.intercept_)

        return self.centroids_line
#
#    def predict(self,data):
#        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
#        classification = distances.index(min(distances))
#        return classification



argparser = argparse.ArgumentParser()

argparser.add_argument("--filename", help="input_file", required=False)

argparser.add_argument("--clusters", help="number of clusters", default = 2, required=False)

argparser.add_argument("--iteration", help="number of iterations", default = 50, required=False)

argparser.add_argument("--output", help="name of output file", required=False)

args = argparser.parse_args()

filename = args.filename
df = pd.read_csv(filename)
cluster_num = args.clusters
iters_number = int(args.iteration)
cluster_image = args.output
cluster_output = args.output
if(cluster_num == ""):
    cluster_numbers = 2
else:
    try:
        cluster_numbers = int(cluster_num)
    except ValueError:
        print("Clusrer_number should be integer")

id = df['id']

col_nums = len(df.columns)
col_nums = col_nums - 1     # exclude id
if(col_nums > 3):
    print "plot is not possible due to numbder of variables"

startTime = int(round(time.time() * 1000))

clf = K_Means(k=cluster_numbers,max_iter=iters_number)
centroids = clf.fit(np.array(df))

endTime = int(round(time.time() * 1000))
rTime = endTime - startTime

print("The clustering computation time is " + str(rTime))
#
#colors = 10*["g","r","c","k","b","m","y"]
#
for i in clf.centroids_line:
    print "Cluster " + str(i) + " is: "
    j = 0
    for m in clf.centroids_line[i]:
        if j != len(clf.centroids_line[i])-1:
            print str(m) + " * x" + str(j) + " "
        else:
             print str(m)
        j += 1
    print "\n"
    

clusters = [0 for x in id]
data_new_id = [0 for x in id]
weight = [100 for x in id]
clusters.append(0)
data_new_id.append(0)
counter = 1;

colors = 10*["r","g","c","k","b","m","y"]
markers =  10 * ["o","D","*","^","<",">","s"]

if(col_nums <= 3):
    plt.rc('font', weight='bold')
    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)
    fig = plt.figure()
    if col_nums == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('',fontsize=28, fontweight='bold',color='b')
        ax.set_ylabel('',fontsize=28, fontweight='bold',color='b')
        ax.set_zlabel('',fontsize=28, fontweight='bold',color='b')
    elif col_nums == 2:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('image size (B)',fontsize=28, fontweight='bold',color='b')
        ax.set_ylabel('Time (ms)',fontsize=28, fontweight='bold',color='b')


for classification in clf.line_classification:
    color = colors[classification]
    mark = markers[classification]
    for featureset in clf.line_classification[classification]:
        if col_nums == 3:
            ax.scatter(featureset[2], featureset[3], featureset[1], marker=mark, color=color, s=75, linewidths=2)
        if col_nums == 2:
            plt.scatter(featureset[2], featureset[1], marker=mark, color=color, s=75, linewidths=2)
        clusters[counter] = classification
        data_new_id[counter] = featureset[0]
        counter = counter + 1


del clusters[0]
del data_new_id[0]

df.drop(['id'],1,inplace=True)
for name in df.columns.values:
    df.drop([name],1,inplace=True)
df.reset_index(drop=True)
df['id'] = np.array(data_new_id).reshape(-1,1)
df['label'] = np.array(clusters).reshape(-1,1)
df['weight'] = np.array(weight).reshape(-1,1)


df.to_csv(cluster_output+".csv")

if col_nums <= 3:
    plt.savefig(cluster_image+"_line.png")
    plt.show()
