# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 17:01:23 2020

@author: Dillon Morse
"""



def distance_between(lat1, lng1, lat2, lng2):
    
    import numpy as np
    
    dlat = np.deg2rad( lat2 - lat1 )
    dlng = np.deg2rad( lng2 - lng1 )
    l1 = np.deg2rad(lat1)
    l2 = np.deg2rad(lat2)
    
    x = np.sin(dlat/2)**2 + np.cos(l1)*np.cos(l2)*np.sin(dlng/2)**2
    d = 2*np.arcsin( np.sqrt(x) )*6373*1000
    
    return d




def get_centers(city, region_radius):
    import numpy as np

    coords = {  'denver': {'tl': [40.079, -105.289],
                           'br': [39.530, -104.700],
                           }, 
              'triangle': {'tl': [36.121,  -79.200],
                           'br': [35.680,  -78.512],
                           }
              }
    
    tl_lat = coords[city]['tl'][0]
    tl_lng = coords[city]['tl'][1]
    
    br_lat = coords[city]['br'][0]
    br_lng = coords[city]['br'][1]
    
    delta_lat = (2*region_radius/111111) 
    delta_lng = ( (2*region_radius/111111)*
                  (1/(np.cos(np.deg2rad( (tl_lat+br_lat)/2 ))))
                 )
    
    delta_x = distance_between(tl_lat, tl_lng, tl_lat, br_lng)
    delta_y = distance_between(tl_lat, tl_lng, br_lat, tl_lng)
    
 
    n_lat = int( delta_y/(2*region_radius) )
    n_lng = int( delta_x/(2*region_radius) )
    
    circle_centers = {}
    l = 0
    for j in range(n_lat):
        for k in range(n_lng):
            circle_centers[l] =( [tl_lat - j*delta_lat, tl_lng + k*delta_lng] )
            l += 1
    
    return circle_centers




# =============================================================================
# Keeps only those regions containing some min number of venues. 
# Output: list of index values to keep 
# =============================================================================

def populated_regions(df, min_venues = 4):

    count =( df.groupby('CircleNum')
               .count()
               .drop(columns = ['City', 'Name'])
               .rename(columns = {'Category':'NumVenues'})
               )
    first_region = count.index[0]
    last_region = count.index[-1]
    missing_regions = [x for x in range(first_region, last_region + 1)  
                                  if x not in count.index]    
    sparse_regions = [k for k in count.index 
                              if count.loc[k,'NumVenues'] < min_venues ] 
    dropped_regions = missing_regions + sparse_regions

    return [k for k in range(first_region, last_region+1) if k not in dropped_regions ]




# =============================================================================
# Loads data for a given city, keeping only those regions in that city that
# contain a minimum number of venues
# =============================================================================

def fetch_data(cityname, min_venues, search_radius, region_radius):
    import pandas as pd
    import numpy as np

    file = cityname + '_venue_data.csv'
    columns = [0, 1, 3, 4, 5, 7]
    
    city_data = pd.read_csv(file, index_col = 0, usecols= columns)
    city_data.drop_duplicates(subset = ['Lat', 'Lng', 'Name'], inplace = True)
    city_data.dropna(how = 'any', inplace = True)
    city_data.reset_index(drop = True, inplace = True)          
    
    regions = get_centers(cityname, region_radius)


    venue_lats = np.array( city_data['Lat'] )
    venue_lngs = np.array( city_data['Lng'] )
    
    region_lats = np.array( [regions[k][0] for k in regions] )[:,None]
    region_lngs = np.array( [regions[k][1] for k in regions] )[:,None]
    
    
    d_filters = ( distance_between(venue_lats,
                                   venue_lngs,
                                   region_lats,
                                   region_lngs) <= search_radius )
    # Every row corresponds to a region
    # Every column corresponds to a venue
    
    
    
    ##########################################################################
    # Here is where the slowdown seems to be - filtering and re-grouping all
    # of the data.
    
    df_list = []
    for row in range( d_filters.shape[0] ):
        
        radius_filter = d_filters[row, :]
                        
        df = city_data[ radius_filter ][['Name', 'Category']]
        df['CircleNum'] = row
        
        df_list.append(df)    

    city_data_regrouped = pd.concat(df_list)
    ##########################################################################
                 
        
    city_data_regrouped['City'] = cityname[:2]   
    
    city_populated_filter = ( city_data_regrouped['CircleNum']
                              .isin( populated_regions(city_data_regrouped, 
                                                       min_venues
                                                       ) ) )
    
    city_data_filtered = (city_data_regrouped[ city_populated_filter ]
                          .reset_index(drop = True)
                          )
    
    
    return city_data_filtered[['City', 'CircleNum', 'Name', 'Category']]