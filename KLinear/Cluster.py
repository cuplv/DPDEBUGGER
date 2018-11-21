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
from sklearn.linear_model import Ridge

class regression_line():
    def __init__(self):
        self.m = 0
        self.b = 0
    
    def best_fit_slope(self,xs,ys):
        m = mean(xs) * mean(ys)
        m = m - mean([a*b for a,b in zip(xs,ys)])
        m = m / ((mean(xs)**2)- mean([a*b for a,b in zip(xs,xs)]))
        self.m = m
        return m

    def best_intercept(self,xs,ys):
        b = mean(ys) - (self.m * mean(xs))
        self.b = b
        return b;
    
    def squared_errors(self,ys_orig,ys_line):
        return sum([(a-b)**2 for a,b in zip(ys_orig,ys_line)])
    
    def coefficient_of_determination(self,ys_orig,ys_line):
        # line of mean for values in y
        y_mean_line = [mean(ys_orig) for y in ys_orig]
        squared_error_reg = self.squared_errors(ys_orig,ys_line)
        squared_error_mean = self.squared_errors(ys_orig,y_mean_line)
        return 1 - (squared_error_reg / squared_error_mean)


class K_Means:
    def __init__(self, k=2, tol=0.01, max_iter=50):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.class_line = regression_line()
    
    def fit(self,data):
        self.centroids_line = {}
        self.m = {}
        self.b = {}
        
        data_len = len(data)
        print data_len
        for i in range(self.k):
            n1 = np.random.randint(low=0,high=data_len/2)
            n2 = np.random.randint(low=data_len/2+1,high=data_len-1)
            self.m[i] = (data[n1][1] - data[n2][1])/(data[n1][0] - data[n2][0])
            self.b[i] = ((data[n1][1] + data[n2][1])/2) - (self.m[i] * ((data[n1][0] + data[n2][0])/2))
            self.centroids_line[i] = [self.m[i],self.b[i]]
        
        for i in range(self.max_iter):
            self.line_classification = {}
            
            for i in range(self.k):
                self.line_classification[i] = []
            
            for featureset in data:
                distances_line = [np.linalg.norm(featureset[1]-(self.m[centroid]*featureset[0]+self.b[centroid])) for centroid in self.centroids_line]
                line_classification = distances_line.index(min(distances_line))
                self.line_classification[line_classification].append(featureset)
                
            prev_centroids_lines = dict(self.centroids_line)
            

            for classification in self.line_classification:
                xs = []
                ys = []
                for x, y in self.line_classification[classification]:
                    xs.append(x)
                    ys.append(y)
                lr = Ridge().fit(np.asarray(xs).reshape(len(xs),1), np.asarray(ys).reshape(len(ys),1))
                self.m[classification] = lr.coef_[0]
                self.b[classification] = lr.intercept_
                self.centroids_line[classification] = [self.m[classification],self.b[classification]]

        return self.centroids_line

    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification



argparser = argparse.ArgumentParser()

argparser.add_argument("--filename", help="input_file", required=False)

argparser.add_argument("--measurements", help="is 10-measurements provided", default = "yes", required=False)

argparser.add_argument("--clusters", help="number of clusters", default = 2, required=False)

argparser.add_argument("--featurex", help="the cluster variable feature", default = 'id', required=False)

argparser.add_argument("--output", help="name of output file", required=False)

args = argparser.parse_args()
if(args.filename == None):
    filename = raw_input("Enter the name of your input data set (.csv) without file type: ")
    print("\n **Please make sure your data set include id feature** \n")
    measurements = raw_input("Are 10-measurements included in file as features T1 ... T10 (yes(y)/no(n))? ")
    print("\n **In case of \'No\', you should put *mean* and *std* (standard deviation) for each record. Header of features should be mean and std respectfully** \n")
    cluster_num= raw_input("Enter number of clusters to divide data set (default is 2): ")
    if(cluster_num == ""):
        cluster_numbers = 2
    else:
        try:
            cluster_numbers = int(cluster_num)
        except ValueError:
            print("That's not an int!")
    featurex = raw_input("Specify the variable of clustering function (default is \'id\')?")
    cluster_image = raw_input("Enter the name of plot file for clustering: ")
    cluster_output = raw_input("Enter the name of output data set (.csv) file without file type: ")

    df = pd.read_csv(filename)
else:
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

id = df[featurex]
Mean = np.array(df['mean'].reshape(-1,1))
np.round(Mean,2)
mean_id = np.array(df[[featurex,'mean']]).reshape(-1,2);
np.round(mean_id,2)

reg_line = regression_line()
xs = []
ys = []
for x, y in mean_id:
    xs.append(x)
    ys.append(y)

startTime = int(round(time.time() * 1000))

clf = K_Means(k=cluster_numbers)
centroids = clf.fit(mean_id)

endTime = int(round(time.time() * 1000))
rTime = endTime - startTime

print("The clustering computation time is " + str(rTime))

colors = 10*["g","r","c","k","b","m","y"]


clusters = [0 for x in id]
feature_x = [0 for x in id]
clusters.append(0)
feature_x.append(0)
counter = 1;
markers =  10 * ["o","D","*","^","<",">","s"]
max_X = 0;
max_Y = 0;

plt.rc('font', weight='bold')
plt.rc('xtick', labelsize=24)
plt.rc('ytick', labelsize=24)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('N (Integer)',fontsize=28, fontweight='bold',color='b')
ax.set_ylabel('Time (S)',fontsize=28, fontweight='bold',color='b')

for classification in clf.line_classification:
    color = colors[classification]
    mark = markers[classification]
    for featureset in clf.line_classification[classification]:
        temp = float(featureset[1])
        plt.scatter(featureset[0], temp, marker=mark, color=color, s=75, linewidths=2)
        clusters[counter] = classification
        feature_x[counter] = featureset[0]
        counter = counter + 1
        if(featureset[0] > max_X):
            max_X = featureset[0]
        if(temp > max_Y):
            max_Y = temp


for i in clf.centroids_line:
    color = colors[i]
    regression_line = [(clf.centroids_line[i][0]*x) + clf.centroids_line[i][1] for x in xs]
    print("Cluster " + str(i) + " is: " + str(clf.centroids_line[i][0]) + " * x + " + str(clf.centroids_line[i][1]))
    plt.plot(xs,regression_line,color=color,linewidth = '3.0')

del clusters[0]
del feature_x[0]

if(measurements == "yes" or measurements == "y" or measurements == ""):
    df.drop(['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','mean'],1,inplace=True)
else:
    df.drop(['mean'],1,inplace=True)

df.drop(['id'],1,inplace=True)
if(featurex!='id'):
    df.drop([featurex],1,inplace=True)
df['label'] = np.array(clusters).reshape(-1,1)

df['featurex'] = np.array(feature_x).reshape(-1,1)
df.to_csv(cluster_output+".csv")

ax.axis([0, max_X+2, 0, max_Y+1])

plt.savefig(cluster_image+"_line.png")
plt.show()
