# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:10:39 2020

@author: Dillon Morse
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
import folium
import data_and_region_setup as drs

def make_map(df, city, num_clusters):
    import GetCircleCenters as CC   # Deperecated function
    
    city_coords = {'denver': [39.8, -105] ,
                   'triangle': [35.928028, -78.8] ,
                   }
    
    lat = city_coords[city][0]
    lng = city_coords[city][1]
    map_city = folium.Map(location=[lat, lng],
                          zoom_start = 10,
                          tiles = 'Stamen Toner')
    
    city_labeled = df.copy()
    
    city_initials = city[:2]
    centers_string_call = 'build_' + city_initials + '_circle_centers'
    city_centers = getattr( CC, centers_string_call )()
    
    
    city_labeled['lnglat'] = [ city_centers[k] for k in city_labeled.index]

    radius = 500
    
    
    #num_clusters = len( city_labeled['Label'].unique() )
    x = np.arange( num_clusters )
    ys = [i + x + (i*x)**2 for i in range(num_clusters)]
    colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
    rainbow = [colors.rgb2hex(i) for i in colors_array]
    
    for k in city_labeled.index:
        lnglat = city_labeled.loc[k,'lnglat']
        Label = city_labeled.loc[k,'Label']
        (folium.Circle( lnglat,
                        radius = radius,
                        fill = True,
                        color = rainbow[Label],
                        weight = 1
                        )
               .add_to(map_city)
               )
    
    filename = city + '_map.html'
    map_city.save(filename)
    
    return map_city



def make_map2(df, city, num_clusters, region_radius):
    
    city_coords = {'denver': [39.8, -105] ,
                   'triangle': [35.928028, -78.8] ,
                   }
    
    lat = city_coords[city][0]
    lng = city_coords[city][1]
    map_city = folium.Map(location=[lat, lng],
                          zoom_start = 10,
                          tiles = 'Stamen Toner')
    
    city_labeled = df.copy()
    
    #city_initials = city[:2]
    #centers_string_call = 'build_' + city_initials + '_circle_centers'
    #city_centers = getattr( CC, centers_string_call )()
    
    city_centers = getattr(drs, 'get_centers' )(city, region_radius)
    
    city_labeled['lnglat'] = [ city_centers[k] for k in city_labeled.index]

    radius = region_radius
    
    
    #num_clusters = len( city_labeled['Label'].unique() )
    x = np.arange( num_clusters )
    ys = [i + x + (i*x)**2 for i in range(num_clusters)]
    colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
    rainbow = [colors.rgb2hex(i) for i in colors_array]
    
    for k in city_labeled.index:
        lnglat = city_labeled.loc[k,'lnglat']
        Label = city_labeled.loc[k,'Label']
        (folium.Circle( lnglat,
                        radius = radius,
                        fill = True,
                        color = rainbow[Label],
                        weight = 1
                        )
               .add_to(map_city)
               )
    print('Map made! Now saving.')
    filename = city + '_map.html'
    map_city.save(filename)
    
    return map_city