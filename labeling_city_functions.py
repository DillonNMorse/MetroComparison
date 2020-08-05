# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 16:08:15 2020

@author: Dillon Morse
"""

def cluster(city1_df, city2_df, num_clusters = 5):
    from sklearn.cluster import AgglomerativeClustering, KMeans
    
    df1 = city1_df.copy()

    clusterer = AgglomerativeClustering( n_clusters = num_clusters,
                                         linkage = 'complete',
                                         affinity = 'cosine')
    
# =============================================================================
#     clusterer = KMeans(n_clusters = num_clusters,
#                        algorithm = 'auto')
# =============================================================================
    
    city1_labels = clusterer.fit_predict(city1_df)
    df1['Label'] = city1_labels
    
    return df1




def label_city2(city1, city2):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
    import numpy as np
    
    df2 = city2.copy()

    X = city1.iloc[:,:-1]
    y = city1.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split( X, y,
                                                         test_size = 0.33,
                                                         random_state = 12
                                                         )
# =============================================================================
#     knn = KNeighborsClassifier(n_jobs = -1)
#     parameters = {'n_neighbors': np.arange(2,21,1),
#                       'weights': ('uniform', 'distance') 
#                  }
#     # max is 0.606
# =============================================================================
    
    clf = SVC(kernel = 'rbf',
              probability = True,
              class_weight = 'balanced',
              C = 2.0235896477)
# =============================================================================
#     parameters = {'C': np.logspace(-3, 3, 50),
#                   'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#                   'degree': [1, 2, 3, 4, 5, 6, 7]
#                   }
#     # max is 0.798    
# =============================================================================



# =============================================================================
#     rfc = RandomForestClassifier()
#     parameters = {'n_estimators': [327], #[int(k) for k in np.linspace(325, 332, 7)],
#                   'criterion': ['entropy'],
#                   'max_depth': [162], #[int(k) for k in np.linspace(160, 165, 5)],
#                   'min_samples_split': [4], #[3,4,5],
#                   #'max_features': ['auto', 'sqrt']
#                   }
#     # max is 0.610
# =============================================================================


# =============================================================================
#     clf = GridSearchCV(estimator = svc,
#                              param_grid = parameters,
#                              verbose = 1,
#                              n_jobs = -1
#                              )   
# =============================================================================
    clf.fit(X_train, y_train)
    acc_test = clf.score(X_test, y_test)
    
    city2_labels = clf.predict(city2)  
    df2['Label'] = city2_labels
    
    return df2, acc_test#, clf.best_estimator_




def count_labels(city1, city2, name1, name2):
    import pandas as pd
    
    df1 = city1.groupby( ['Label'] ).count().iloc[:,0].rename(name1)
    df2 = city2.groupby( ['Label'] ).count().iloc[:,0].rename(name2)
    
    df_total = pd.concat([df1, df2], axis = 1 )
    
    return df_total