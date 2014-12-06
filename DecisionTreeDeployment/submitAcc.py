import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()



#################################################################################


########################## DECISION TREE #################################



#### your code goes here
from classifyDT import classify
from sklearn.metrics.metrics import accuracy_score
clf = classify(features_train, labels_train,50.0)

clf.fit(features_train,labels_train)
pred = clf.predict_proba(features_test)
roundedNumber = []
for i in range(0,len(pred)):
    roundedNumber.append(round(pred[i,1]))
acc = accuracy_score(labels_test,roundedNumber)### you fill this in!
print acc### be sure to compute the accuracy on the test set


    
def submitAccuracies():
    return {"acc":round(acc,3)}
