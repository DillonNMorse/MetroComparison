# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 17:13:11 2020

@author: Dillon Morse
"""

import pandas as pd
import numpy as np
import Dataframe_clean_functions as clean

# =============================================================================
# Load data sets, keep only regions with at least 3 venues
# =============================================================================
CH_data = pd.read_csv('CH_venue_data.csv', index_col = 0)
CH_data['City'] = 'CH_' + CH_data['CircleNum'].astype(str)
CH_data = CH_data[ CH_data['CircleNum']
                   .isin( clean.populated_regions(CH_data) )  ]

Boulder_data = pd.read_csv('Boulder_venue_data.csv', index_col = 0)
Boulder_data['City'] = 'Bo_' + Boulder_data['CircleNum'].astype(str)
Boulder_data = Boulder_data[ Boulder_data['CircleNum']
                            .isin( clean.populated_regions(Boulder_data) )  ]

# =============================================================================
# Combine data from cities to ensure that the clustering algorithm sees the 
# same feature label columns for both
# =============================================================================
frames = [CH_data, Boulder_data]
all_data = ( pd.concat( frames )
               .reset_index(drop = True)
               )
reduced_data = all_data[ ['City',
                          'Category'
                          ]
                        ]


# =============================================================================
# Create some descitpive statistics to better understand the data sets
# =============================================================================

categories = reduced_data['Category'].unique()
num_venues =len( all_data['Name'].unique() )

CH_count = ( CH_data.groupby('City')
                    .count()
                    .rename(columns = {'ID':'NumVenues'})['NumVenues']
                    )
Boulder_count = ( Boulder_data.groupby('City')
                              .count()
                              .rename(columns = {'ID':'NumVenues'})['NumVenues']
                              )

if __name__ == '__main__':

    print('There are {} popuated regions in Chapel Hill.'.format(CH_count.shape[0] ))
    print('There are {} populated regions in Boulder.'.format(Boulder_count.shape[0] ))
    print('There are {} unique venues between the cities.'.format(num_venues))
    print('There are {} unique category labels between the two cities.'.format(len(categories)))
     
    
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-deep')
    
    fig, ax = plt.subplots(1)
    ax.hist( [CH_count, Boulder_count],
             alpha = 0.6,
             label = ['Chape Hill', 'Boulder']
             )
    plt.legend(loc='upper right')
    plt.title('Venue Densities Between Cities')
    plt.xlabel('Number of venues per city region')
    plt.ylabel('Frequency')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.set_size_inches(18.5*0.6, 10.5*0.6)
    plt.savefig('Venue_Density_Plot.jpeg',
                bbox_inches = 'tight',
                dpi = 100)
 
 
# =============================================================================
# One-hot-encode "Category" feature to remove categorical data for processing
# 
# For each region within the cities take the mean for each venue type (might
# try sum instead) to create region-level data
#
# Split cities apart afterwards for clustering-algorithm training
# =============================================================================
venue_per_region = ( pd.get_dummies(reduced_data, columns = ['Category'])
                       .groupby('City')
                       .mean()
                       )
CH_filter = venue_per_region.index.str[:2] == 'CH'
Boulder_filter = venue_per_region.index.str[:2] == 'Bo'

CH_venue_per_region = venue_per_region[ CH_filter ]
Boulder_venue_per_region = venue_per_region[ Boulder_filter ]


# =============================================================================
# Train
# =============================================================================

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
CH_scaled = scaler.fit_transform( CH_venue_per_region )
Boulder_scaled = scaler.transform( Boulder_venue_per_region )

num_clusters = 5
Clusterer = AgglomerativeClustering( n_clusters = num_clusters,
                                     linkage = 'ward')
CH_labels = Clusterer.fit_predict(CH_scaled)
# =============================================================================
# Boulder_labels = Clusterer.predict(Boulder_scaled)        
# =============================================================================

CH_labeled = pd.DataFrame( data = {'Region': CH_count.index,
                                    'Label': CH_labels} 
                          )
# =============================================================================
# Boulder_labeled = pd.DataFrame( data = {'Region': Boulder_count.index,
#                                          'Label': Boulder_labels} 
#                                )   
# =============================================================================

if __name__ == '__main__':
    print('The number of regions in each cluster is:')
    print(CH_labeled.groupby('Label').count())



CH_labeled['CircleNum'] = CH_labeled['Region'].str[3:].astype(int)

Boulder_scaled = pd.DataFrame(data = Boulder_scaled)
Boulder_scaled['Region'] = Boulder_venue_per_region.index
Boulder_scaled['CircleNum'] = Boulder_scaled['Region'].str[3:].astype(int)


def CH_features():
    return CH_scaled

def Boulder_features():
    return Boulder_scaled

def Label_CH():
    return CH_labeled


