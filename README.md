# Linear classification (IRLS and probabilistic generative model)

We consider the data set vehicle.pkl. The task is to classify a given silhouette as
one of two types of vehicles, i.e., SAAB or VAN, using a set of features extracted from
the silhouette. You are required to use Python for this assignment. The goal will be to
minimize misclassification rate on the test set. The data has been splitted into a training
set and a test set. First, extract from both of these sets only data points of the classes
SAAB (C = 2) and VAN (C = 4).

For the whole set of instructions please see [the included PDF](NN19_task2.pdf)

## Solution

The code for the probabilistic general model is in [genmod.py](genmod.py), 
included with accuracies for different number of features used [genmod.csv](genmod.csv)

The code for the IRLS algorithm is in [irls.py](irls.py).

The graphs of accuracies for each model using a certain amount of features is available
in the [report.pdf](report.pdf).