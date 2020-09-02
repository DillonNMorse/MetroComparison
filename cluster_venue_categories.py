# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 15:10:04 2020

@author: Dillon Morse
"""


def cluster_cats(rebuild = True, num_syns = 0, plot = False):
    
    import pandas as pd
    
    
    # =========================================================================
    # Re-build definitions dictionary with different numbers of synonyms used. 
    #   Can be slow - a minute or more depending on how many synonyms are 
    #   included
    # =========================================================================
    
    if rebuild:
        num_syns_ = num_syns
        import build_dict_for_category_synonyms_and_definitions as build
        all_defs = build.all_category_defs(num_syns = num_syns)
        all_defs = build.manual_defs(all_defs)
        cleaned_defs = build.clean_defs( all_defs, stem = False, lem = True )
        build.vectorize_defs( cleaned_defs  )
        
    
    # =========================================================================
    # Read in vectorized definitions from pickle file
    # =========================================================================
    word_vecs = pd.read_pickle('vectorized_defs_dataframe.pickle')
    
    
    # =========================================================================
    # Cluster the definition word vectors
    # =========================================================================
    from sklearn.cluster import KMeans
    
    ap = KMeans(n_clusters = 10)
    ap.fit( word_vecs.iloc[:,:-1] )
    cluster_labels = ap.labels_
    cluster_labels = pd.DataFrame(cluster_labels, columns=['ClusterLabel'])
    cluster_labels['Category'] = word_vecs['Category']
    cluster_labels.to_pickle('clustered_categories.pickle')
    
    if plot:
        # =====================================================================
        # Use PCA to reduce dimensions, plot results color-coded by the 
        #   clusters above
        # =====================================================================
        from sklearn.decomposition import PCA
        
        red_word_vecs = (pd.DataFrame( data = PCA(n_components = 3)
                           .fit_transform( word_vecs.iloc[:, :-1] ) )
                         )
        red_word_vecs['Category'] = word_vecs['Category']
        red_word_vecs['Cluster'] = cluster_labels['ClusterLabel']
        
        import matplotlib
        matplotlib.use('nbagg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        from IPython import get_ipython
        get_ipython().run_line_magic('matplotlib', 'qt')
        
        x = red_word_vecs.iloc[:,0]
        y = red_word_vecs.iloc[:,1]
        z = red_word_vecs.iloc[:,2]
        
        fig = plt.figure()
        ax = Axes3D(fig)
        
        ax.scatter(x, y, z,
                    c = red_word_vecs.iloc[:,4],
                    cmap = plt.cm.tab10,
                    alpha = 1,
                    s = 150)
        
        plt.show()