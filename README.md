# DPDEBUGGER
The repository for the implementations of performance debugging with machine learning tools
Paper can be found here: [Differential Performance Debugging With Discriminant Regression Trees](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16647), AAAI'18, 2468-2475.

### KLinear Clustering
```
cd KLinear/
python Cluster.py --filename SnapBuddy_AAAI/snapBuddy_for_clustering.csv --measurements no --featurex size --clusters 5 --output test_output
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
python Classify.py --filename SnapBuddy_AAAI/SnapBuddy_final.csv --output SnapBuddy_AAAI/output
dot -Tpng SnapBuddy_AAAI/output_tree1.dot -o SnapBuddy_AAAI/tree.png
```
