# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 18:30:06 2020

@author: Dillon Morse
"""

import pandas as pd

def apply_pca(df, city, vars_to_keep):
    from sklearn.decomposition import PCA
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    decomp =  PCA() 
    X = decomp.fit_transform( df )
    X = pd.DataFrame( data = X, index = df.index )
    
    princ_components = list( range(vars_to_keep) )
    
    vari = decomp.explained_variance_ratio_
    sum_vari = [sum(vari[:k]) for k in princ_components]
    
    fig, ax = plt.subplots(1,1)
    sns.barplot(x = princ_components, y = sum_vari)
    sns.lineplot(x = princ_components, y = [0.9]*vars_to_keep)
    plt.xticks( np.linspace(0, vars_to_keep, 10) )
    ax.set_xticklabels( [int(k) for k in np.linspace(0, vars_to_keep, 10)] )
    plt.xlabel('Number of principal components kept')
    plt.ylabel('Percentage of total variance accounted for')
    ax.annotate('90% of variance', xy = (1,0.86))
    plt.title('Information Retained After\nDimensional Reduction for ' + city)
    
    return X.iloc[:,:vars_to_keep]



def apply_scaling(df1, df2):
    from sklearn.preprocessing import StandardScaler
    
    col_names = {}
    for j, name in enumerate( df1.columns ):
        col_names[j] = name
    
    scaler = StandardScaler()
    city1_data = scaler.fit_transform( df1 )
    city2_data = scaler.transform(df2 )
    
    city1_scaled = ( pd.DataFrame( city1_data,
                                   index = df1.index 
                                   )
                       .rename(columns = col_names )
                       ) 
    
    city2_scaled = ( pd.DataFrame( city2_data,
                                   index = df2.index
                                   )
                       .rename(columns = col_names )
                       )
    
    return city1_scaled, city2_scaled