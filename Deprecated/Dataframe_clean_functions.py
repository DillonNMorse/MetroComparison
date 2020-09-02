# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 19:59:54 2020

@author: Dillon Morse
"""

def populated_regions(df):
    min_venues = 3
    count =( df.groupby('CircleNum')
               .count()
               )
    first_region = count.index[0]
    last_region = count.index[-1]
    missing_regions = [x for x in range(first_region, last_region + 1)  
                                  if x not in count.index]    
    sparse_regions = [k for k in count.index if count.loc[k,'ID'] <= min_venues ] 
    dropped_regions = missing_regions + sparse_regions

    return [k for k in range(last_region+1) if k not in dropped_regions ]