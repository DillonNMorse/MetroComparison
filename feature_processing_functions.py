# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 19:59:54 2020

@author: Dillon Morse
"""

# =============================================================================
# Loads data for a given city, keeping only those regions in that city that
# contain a minimum number of venues
# =============================================================================

def fetch_data(cityname, min_venues):
    import pandas as pd
    
    file = cityname + '_venue_data.csv'
    columns = [0, 1, 3, 7]
    
    city_data = pd.read_csv(file, index_col = 0, usecols= columns)
        
    city_data['City'] = cityname[:2] #+ city1_data['CircleNum'].astype(str)    
    
    city_populated_filter = ( city_data['CircleNum']
                              .isin( populated_regions(city_data, min_venues)
                                    ) 
                              )
    city_data = city_data[ city_populated_filter ].reset_index(drop = True)
    
    
    
    return city_data[['City', 'CircleNum', 'Name', 'Category']]



# =============================================================================
# Keeps only those regions containing some min number of venues. 
# Output: list of index values to keep 
# =============================================================================

def populated_regions(df, min_venues = 4):

    count =( df.groupby('CircleNum')
               .count()
               .drop(columns = 'City')
               .rename(columns = {'Category':'NumVenues'})
               )
    first_region = count.index[0]
    last_region = count.index[-1]
    missing_regions = [x for x in range(first_region, last_region + 1)  
                                  if x not in count.index]    
    sparse_regions = [k for k in count.index 
                              if count.loc[k,'NumVenues'] < min_venues ] 
    dropped_regions = missing_regions + sparse_regions

    return [k for k in range(last_region+1) if k not in dropped_regions ]



# =============================================================================
# One-hot-encode information, group by region of the cities
# =============================================================================

def encode(df):
    import pandas as pd
    
    venue_per_region = ( pd.get_dummies(df, 
                                        columns = ['Category', 'Cluster'],
                                        prefix = ['Cat', 'Clus']
                                        )
                           .groupby(['City', 'CircleNum'])
                           .mean()
                           )
    city1 = df.iloc[ 0, 0]
    city2 = df.iloc[-1, 0]

    return venue_per_region.loc[city1], venue_per_region.loc[city2]


# =============================================================================
# Print some informative descriptions of the data
# =============================================================================

def describe(df, city1, city2):
    import numpy as np
    
    categories = df['Category'].unique()
    num_venues =len( df['Name'].unique() )
    
    grouped = df.groupby(['City', 'CircleNum']).count()

    city1_count = grouped.loc[city1[:2]].shape[0]
    city2_count = grouped.loc[city2[:2]].shape[0]
    

    print('\nThere are {} popuated regions in {}.'.format(city1_count, city1))
          
    print('There are {} populated regions in {}.'.format(city2_count, city2))
          
    print('There are {} unique venues between the cities.'.format(num_venues))
    
    print('There are {} unique category labels between the two cities.'.format(len(categories)))
     
    
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-deep')
    
    fig, ax = plt.subplots(1)
    bins = range(0,110,10)
    ax.hist( [ grouped.loc[ city1[:2] ]['Name'],
               grouped.loc[ city2[:2] ]['Name'] 
               ],
             alpha = 0.6,
             label = [city1, city2],
             bins = bins,
             density = True,
             )
    plt.legend(loc='upper right')
    plt.title('Venue Densities Between Cities')
    plt.xlabel('Number of venues per city sub-region')
    plt.ylabel('Density Function')
    plt.yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks( [k-5 for k in bins[1:] if k%20 == 0 ] )
    ax.set_xticklabels([k for k in bins[1:] if k%20 == 0] )
    
    fig.set_size_inches(18.5*0.4, 10.5*0.4)
    plt.savefig('Venue_Density_Plot.png',
                bbox_inches = 'tight',
                dpi = 100)
    
    describe_cat_clusters(df)
 
    return


# =============================================================================
# Describe category clustering
# =============================================================================

def describe_cat_clusters(all_data):
    
    print('\nThe category label clustering yeilds the following clusters:')
    
    cat_clusters = all_data.copy()[['Category', 'Cluster']]
    cat_clusters['Count'] = 1
    cat_clusters = cat_clusters.groupby('Cluster').sum()
    print( cat_clusters )
    
    #find_nearest_category(all_data)
    
    return 

# =============================================================================
# Find closest word to this vector
# =============================================================================

def find_nearest_category(all_data):
    import spacy 
    import numpy as np
    import pandas as pd    
    from time import time
    
    t0 = time()

    clusters = list( all_data['Cluster'].unique() )
    
    cat_data = ['Cluster_'+str(j) for j in clusters]
    
    for j in clusters:
        cat_data[j] = all_data[ all_data['Cluster'] == j ]['Category']

    cluster_strings = []
    
    for j in clusters:
        string = ''
        for cat in cat_data[j]:
            string += cat
        cluster_strings.append(string)

    
    vect_dim = 300
    all_feat_vectors = np.zeros( (len(cluster_strings), vect_dim) )
    
    nlp = spacy.load('en_core_web_lg')
    for i, string in enumerate( cluster_strings ):
        all_feat_vectors[i,:] = nlp(string).vector
     
    t1 = time()
    
    print('Took {:.1f} seconds to load and vectorize strings.'.format(t1-t0) )
    
    
        
    return





