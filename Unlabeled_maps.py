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
radius = 500

centers_tr = CC.build_tr_circle_centers()
centers_de = CC.build_de_circle_centers()


for lnglat in centers_tr:
    (folium.Circle( lnglat,
                    radius = radius,
                    fill = True,
                    weight = 1
                    )
           .add_to(map_tr)
           )
for lnglat in centers_de:
    (folium.Circle( lnglat,
                    radius = radius,
                    fill = True,
                    weight = 1
                    )
           .add_to(map_de)
           )

map_tr.save('triangle_map.html')
map_de.save('denver_map.html')

