import sys
sys.path.append("../GaussianDeployment/")
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()



#################################################################################


########################## DECISION TREE #################################



#### your code goes here
from sklearn.metrics.metrics import accuracy_score
from sklearn import tree
clf = tree.DecisionTreeClassifier()

clf.fit(features_train,labels_train)
pred = clf.predict_proba(features_test)


acc = accuracy_score(labels_test,pred[:,1])### you fill this in!


    
def submitAccuracies():
    return {"acc":round(acc,3)}
