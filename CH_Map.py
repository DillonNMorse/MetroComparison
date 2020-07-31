# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 12:51:43 2020

@author: Dillon Morse
"""

import folium
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
import GetCircleCenters as CC
import Dataframe_clean_functions as clean
import Process_and_Cluster as cluster


# =============================================================================
# Create map centered at these coordinates
# =============================================================================
lat = 35.928028
lng = -79.045645
map_CH = folium.Map(location=[lat, lng], zoom_start = 11)

# =============================================================================
# Read in both city data and list of all region-centers (called 'markers' in
# the map context). Remove those markers which don't show up in the city data
# (i.e. due to too-low a venue population in the region)
# =============================================================================


CH_labeled = cluster.Label_CH()

marker_centers = CC.build_CH_circle_centers()
surviving_centers = [marker_centers[k] for k in CH_labeled['CircleNum'].unique() ]

radius = 500

CH_labeled['Center'] = surviving_centers

num_clusters = len( CH_labeled['Label'].unique() )
x = np.arange( num_clusters )
ys = [i + x + (i*x)**2 for i in range(num_clusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

for k in CH_labeled.index:
    lnglat = CH_labeled.loc[k,'Center']
    Label = CH_labeled.loc[k,'Label']
    (folium.Circle( lnglat,
                    radius = radius,
                    fill = True,
                    color = rainbow[Label]
                    )
           .add_to(map_CH)
           )


map_CH.save('CH_map.html')

def make_map():
    return map_CH