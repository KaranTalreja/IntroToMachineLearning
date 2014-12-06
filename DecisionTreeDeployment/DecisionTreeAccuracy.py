import sys
sys.path.append("../GaussianDeployment/")
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()

def submitAccuracies():
    return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3),
          "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}


########################## DECISION TREE #################################


### your code goes here--now create 2 decision tree classifiers,
### one with min_samples_split=2 and one with min_samples_split=50
### compute the accuracies on the testing data and store
### the accuracy numbers to acc_min_samples_split_2 and
### acc_min_samples_split_5, respectively
from classifyDT import classify
from sklearn.metrics.metrics import accuracy_score
clf = classify(features_train, labels_train,50.0)
pred = clf.predict_proba(features_test)
roundedNumber = []
for i in range(0,len(pred)):
    roundedNumber.append(round(pred[i,1]))
acc_min_samples_split_50 = accuracy_score(labels_test,roundedNumber)### you fill this in!


clf = classify(features_train, labels_train,2.0)

pred = clf.predict_proba(features_test)
acc_min_samples_split_2 = accuracy_score(labels_test,pred[:,1])### you fill this in!
print submitAccuracies()
