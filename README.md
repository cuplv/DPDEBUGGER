# DPDEBUGGER
The repository for the implementations of performance debugging with machine learning tools
Paper can be found here: [Differential Performance Debugging With Discriminant Regression Trees](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16647), AAAI'18, 2468-2475.

DPDEBUGGER consists of two steps:
1) Functional Clustering: The goal is to cluster traces with similar timing behaviors. The similarity definition needs to consider properties of inputs such as the size of inputs. This step uses two novel functional clustering algorithms: 1) KLinear Clustering that is an extension of KMeans algorithm, 2) Alignment Kernel that uses Spectral Clustering with a new similarity kernel between points.
2) Classification: Decision Tree algorithm has used to explain what properties of program internals are common in the same cluster and what properties are separating different clusters. 

### KLinear Clustering
```
cd KLinear/
python Cluster.py --filename SnapBuddy/snapBuddy_for_clustering.csv --measurements no --featurex size --clusters 5 --output SnapBuddy/test_output
```
### Spectral Clustering with Alignment Kernel
you need to modify the file in /Your/Path/To/scikit-learn/sklearn/metrics/pairwise.py
Add the function inside Alignment_Kernel.py to the pairwise.py file. Then,
```
cd /Your/Path/To/scikit-learn
sudo python setup.py install
```
Then, you can use Alignment Clsutering.
```
cd Spectral_Alignment/
python SpectralClustering_1.py --filename fop/result_time.csv --measurements no --clusters 2 --featurex size --output ./fop/result_time_spectral.csv
```

### Classification Tree
```
cd Classification/
python Classify.py --filename SnapBuddy/SnapBuddy_for_classification.csv --output SnapBuddy/output
dot -Tpng SnapBuddy/output_tree1.dot -o SnapBuddy/tree.png
```
