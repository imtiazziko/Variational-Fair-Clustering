# Clustering-with-fairness-constraints
This is the code for the paper **Clustering With Fairness Constraints -- A Flexible and Scalable approach**. This clustering method helps you to find clusters with specefied proportions of different groups of pertaining to a sensitive attribute of the dataset (e.g. race, gender or sex etc.) for different well-known clustering method such as K-means or Spectral clustering (Normalized cut) etc in a flexible and scalable way.

## Prerequisites

The code is based on python 3.6 and need the Pandas and Matplotlib libraries. We have provided the saved model for our reported results in the paper in the [data](./data) directory.

To evaluate the code simply run the following which put the outputs in the [outputs](./outputs) folder.
```
sh evaluate_Fair_clustering.sh
```
For different dataset and parameters change the above script according to the required arguments described in [test_fair_clustering.py](./test_fair_clustering.py).



