# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 22:24:30 2020

@author: Dillon Morse
"""

import pandas as pd
import numpy as np
import Process_and_Cluster as process


CH_features = process.CH_features()
CH_labels = process.Label_CH()['Label']
Boulder_features = process.Boulder_features().iloc[:,:-2]



from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split( CH_features,
                                                     CH_labels,
                                                     test_size=0.33,
                                                     random_state=42
                                                     )

knn = KNeighborsClassifier(n_jobs = -1)
parameters = {'n_neighbors': np.arange(2,11,1),
                  'weights': ('uniform', 'distance') 
             }

clf = GridSearchCV(estimator = knn,
                   param_grid = parameters
                   )


clf.fit(X_train, y_train)
acc_test = clf.score(X_test, y_test)
print('Accuracy is {:.3f}.'.format(acc_test))


Boulder_labels = clf.predict(Boulder_features)

def Label_Boulder():
    return Boulder_labels