# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 14:10:23 2020

@author: Dillon Morse
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 12:51:43 2020

@author: Dillon Morse
"""

import folium
import GetCircleCenters as CC
import data_and_region_setup as drs


# =============================================================================
# Create map centered at these coordinates
# =============================================================================
lat_tr = 35.928028
lng_tr = -78.8#-79.045645
map_tr = folium.Map(location=[lat_tr, lng_tr], zoom_start = 10)

lat_de = 39.8#40.019824
lng_de = -105#-105.262982
map_de = folium.Map(location=[lat_de, lng_de], zoom_start=10)

# =============================================================================
# Read in both city data, make markers
# =============================================================================
radius = 250

centers_tr = drs.get_centers('triangle', 250)  # CC.build_tr_circle_centers()
centers_de = drs.get_centers('denver',   250) #CC.build_de_circle_centers()

print('Centers loaded. Starting first map with ', len(centers_tr), ' circles.')
for circnum in centers_tr:
    lnglat = centers_tr[circnum]
    (folium.Circle( lnglat,
                    radius = radius,
                    fill = True,
                    weight = 1
                    )
           .add_to(map_tr)
           )
print('Done with first map. Making second with ', len(centers_de), ' circles.')
for circnum in centers_de:
    lnglat = centers_de[circnum]
    (folium.Circle( lnglat,
                    radius = radius,
                    fill = True,
                    weight = 1
                    )
           .add_to(map_de)
           )
print('Both maps made. Now to save them.')
map_tr.save('triangle_map.html')
map_de.save('denver_map.html')

