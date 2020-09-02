# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 15:27:56 2020

@author: Dillon Morse
"""

import encode_category_data2 as cats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = cats.get_labeled_cats()


# =============================================================================
# Look at data
# =============================================================================
category_totals = pd.Series(index = data.columns[1:], 
                            data = [sum(data[cat]) for cat in data.columns[1:]]
                            ).sort_values(ascending = False)
fig, ax = plt.subplots(1,1)
bars = plt.bar(category_totals.index, category_totals, alpha = 0.6)
plt.xticks(rotation=45)
plt.title('Prevalance of Labeled Venue Types')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
ax.xaxis.set_ticks_position('none')
for bar in bars:
    height = bar.get_height()
    ax.annotate('{}'.format(height),
                xy=(bar.get_x() + bar.get_width() / 2, height), # center text
                xytext=(0, -12),  # drop text down in to bars
                textcoords="offset points",
                ha='center', va='bottom')
plt.show()





df = pd.read_pickle('encoded_category_definitions.pickle')

print('\nTraining feature extractor.')
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, 
                               random_state = 13, 
                               test_size = 0.20, 
                               shuffle = True)

print(train.shape)
print(test.shape)


from sklearn.feature_extraction.text import TfidfVectorizer

# =============================================================================
# vectorizer = TfidfVectorizer(strip_accents = 'unicode', 
#                              analyzer = 'word', 
#                              ngram_range = (1,3), 
#                              norm = 'l2'
#                              )
# =============================================================================
vectorizer = TfidfVectorizer(ngram_range = (1,1) )

vectorizer.fit(df['CategoryKeyWords'])



X_train = vectorizer.transform(train['CategoryKeyWords'])
y_train = train.drop(labels = ['Category', 'CategoryKeyWords'], 
                     axis=1
                     )

X_test = vectorizer.transform( test['CategoryKeyWords'])
y_test = test.drop(labels = ['Category', 'CategoryKeyWords'], 
                   axis=1
                   )


feature_names = vectorizer.get_feature_names()


dense = X_test.todense()
denselist = dense.tolist()
word_df = pd.DataFrame(denselist, columns=feature_names)






print('\nTraining classifier.')
# =============================================================================
# 
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
# from sklearn.decomposition import TruncatedSVD
# from sklearn.pipeline import Pipeline
# =============================================================================



from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier

dim_reduct = TruncatedSVD(n_components = 75)

X_train = dim_reduct.fit_transform(X_train)
X_test = dim_reduct.transform(X_test)

print('Explained variance ratio: ', sum( dim_reduct.explained_variance_ratio_) )


nn = MLPClassifier(hidden_layer_sizes = (500, 500, 100, 100, 50),
                   alpha = 1e-4,
                   activation = 'relu',
                   solver = 'adam',
                   max_iter = 2000
                    )
clf = OneVsRestClassifier(nn, n_jobs = -1)
clf.fit(X_train, y_train)

def pred_accuracy( threshold = 0.5, clf = clf):
    from sklearn.metrics import accuracy_score
    prediction = (clf.predict_proba(X_test) > threshold).astype(int)
    prediction = pd.DataFrame(data = prediction, 
                          columns = y_train.columns
                          )
    print('Test accuracy is {:.3f}'.format(accuracy_score(y_test, prediction)))

    return None

def get_pred( threshold = 0.5, clf = clf):
    prediction = (clf.predict_proba(X_test) > threshold).astype(int)
    prediction = pd.DataFrame(data = prediction, 
                          columns = y_train.columns
                          )
    prediction['Category'] = train['Category'].reset_index(drop = True)
    
    preds = {}
    for k in prediction.index:
        preds[ prediction.loc[k,'Category'] ] = []
        for j in prediction.columns[:-1]:
            if prediction.loc[k,j] != 0:
                preds[prediction.loc[k,'Category']].append(j)
    
    return preds

pred_accuracy()


# =============================================================================
# LogReg = LogisticRegression(n_jobs = -1,
#                             solver = 'lbfgs',
#                             max_iter = 200,
#                             class_weight = 'uniform',
#                             C = 7943282
#                             )           # acc = 0.560, n_comp = 115 (dim_reduct)
# 
# RanFor = RandomForestClassifier(min_samples_split = 2,
#                                 min_samples_leaf = 2,
#                                 class_weight = 'balanced',
#                                 n_estimators = 110,
#                                 max_depth = 82
#                                 )       # acc = 0.420
# 
# KNei = KNeighborsClassifier(n_jobs = -1,
#                             n_neighbors = 5,
#                             weights = 'uniform' 
#                             )           # acc = 0.520
# 
# =============================================================================

# =============================================================================
# clf = OneVsRestClassifier(LogReg,
#                           n_jobs = -1)
# =============================================================================



# =============================================================================
# dist = {'estimator__C': np.logspace(3,9,41),
#         }
# 
# grid =       GridSearchCV(clf,
#                           param_grid = dist,
#                           n_jobs = -1,
#                           )
# 
# 
# 
# clf.fit(X_train, y_train)
# =============================================================================

# =============================================================================
# print('\nBest params are ', grid.best_params_)
# best = grid.best_params_
# =============================================================================

# =============================================================================
# prediction = clf.predict(X_test)
# prediction = pd.DataFrame(data = prediction, 
#                           columns = y_train.columns
#                           )
# print('Test accuracy is {:.3f}'.format(accuracy_score(y_test, prediction)))
# =============================================================================
#prediction['Category'] = test['Category'].reset_index(drop = True)


#target_names = list(df.columns[1:-1])


#print(classification_report(y_test, prediction, target_names=target_names))

# =============================================================================
# for category in df.columns[1:]:
#     print('**Processing {} Categories...**'.format(category))
#     
#     # Training logistic regression model on train data
#     clf.fit(X_train, y_train)
#     
#     # calculating test accuracy
#     prediction = clf.predict(X_test)
#     print('Test accuracy is {:.3f}'.format(accuracy_score(y_test, prediction)))
#     print("\n")
# =============================================================================
