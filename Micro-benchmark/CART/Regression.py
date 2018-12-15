
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
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing, neighbors, tree
from sklearn.model_selection import cross_val_score
import pydotplus
import time
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit

argparser = argparse.ArgumentParser()

argparser.add_argument("--filename", help="input_file", required=False)
argparser.add_argument("--kfolds", help="number of k", default = "10", required=False)
argparser.add_argument("--output", help="name of output file", required=False)
argparser.add_argument("--depth", help="depth of tree", default = "", required=False)
argparser.add_argument("--measurements", help="is 10 times measurement given?", default = "yes", required=False)
argparser.add_argument("--split", help="min sample split", default = "", required=False)
args = argparser.parse_args()

filename = args.filename
df = pd.read_csv(filename)
max_depth = args.depth
if(max_depth == ""):
    max_depth_tree = None
else:
    try:
        max_depth_tree = int(max_depth)
    except ValueError:
        print("Max depth should be an integer!")
min_split = args.split
kfolds = args.kfolds
if(kfolds == ""):
    kfolds_numbers = 10
else:
    try:
        kfolds_numbers = int(kfolds)
    except ValueError:
        print("K-fold should be an integer!")

if(min_split == ""):
    min_split_tree = 2
else:
    try:
        min_split_tree = int(min_split)
    except ValueError:
        print("Min Split should be an integer!")

if(args.measurements=="yes"):
    df_T = df[['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10']]
    Mean = (df['T1'] + df['T2'] + df['T3'] + df['T4'] + df['T5'] + df['T6'] + df['T7'] + df['T8'] + df['T9'] + df['T10'])/(10)
    df = df.drop(['T1'],1)
    df = df.drop(['T2'],1)
    df = df.drop(['T3'],1)
    df = df.drop(['T4'],1)
    df = df.drop(['T5'],1)
    df = df.drop(['T6'],1)
    df = df.drop(['T7'],1)
    df = df.drop(['T8'],1)
    df = df.drop(['T9'],1)
    df = df.drop(['T10'],1)
else:
    Mean = df['mean']
    df = df.drop(['mean'],1)

np.round(Mean,2)
id = df['id']
df = df.drop(['id'],1)

header = list(df.columns.values)

X = np.array(df)
y = np.array(Mean)

kf = ShuffleSplit(n_splits=kfolds_numbers,test_size=0.1)
iteration = 0
startTime = int(round(time.time() * 1000))

accuracy_avg = 0
best_r2_score = 0

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
        
    regr_1 = DecisionTreeRegressor(max_depth=max_depth_tree,min_samples_split=min_split_tree)
        
    regr_1.fit(X_train, y_train)
    r2_score = regr_1.score(X_test, y_test)
    accuracy_avg = accuracy_avg + r2_score
    if(r2_score > best_r2_score):
        best_r2_score = r2_score
        regr = regr_1

print("The coefficient of determination is: " + str(accuracy_avg/kfolds_numbers))
unique_leaf = np.unique(regr.apply(X))
print("Leaf nodes in number: " + str(len(unique_leaf)))
dp = regr.decision_path(X)
max_len = 0
for key in dp:
    key_vec = key.toarray()[0]
    leng = np.sum(key_vec)
    if leng > max_len:
        max_len = leng
print("Depth of tree is: " + str(max_len))

endTime = int(round(time.time() * 1000))
rTime = endTime - startTime
print('Time of computation for tree :' + str(rTime))


if(args.filename==None):
    out = filename +'_tree.dot'
else:
    out = args.output +'_tree.dot'
tree.export_graphviz(regr_1,out_file=out,feature_names=header)

if(args.filename==None):
    print("\n The program generates three trees with highest accuracy. Please run: dot -Tpng " + filename +"_tree.dot" + " -o tree.png to see the final decision tree. Please note that treen is tree0, tree1, or tree2. \n")
else:
    print("\n The program generates three trees with highest accuracy. Please run: dot -Tpng " + args.output +"_tree.dot" + " -o tree.png to see the final decision tree. Please note that treen is tree. \n")
