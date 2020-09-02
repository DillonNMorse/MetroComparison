# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:55:54 2020

@author: Dillon Morse
"""


import pandas as pd
import numpy as np

df = pd.concat( [pd.read_csv('denver_venue_data.csv'),
                 pd.read_csv('denver_venue_data.csv') ]).reset_index(drop = True)

# todo: drop 'Moving Target' category
df = df[ df['Category'] != 'Moving Target' ]

all_cats = df['Category'].unique()


prim = {0: 'outdoor',
        1: 'services',
        2: 'entertainment',
        3: 'airport',
        4: 'restaurant',
        5: 'school',
        6: 'shopping',
        7: 'bar',
        }

sec = {prim[0]: {0: 'trail', 1:'preserve', 2: 'stables', 3:'campground',
                 4: 'water transport', 5: 'biking', 6: 'farmers market',
                 7: 'park', 8: 'skating', 9: 'pool', 10: 'farm', 11: 'dog',
                 12: 'golf', 13: 'playground'},
       prim[1]: {0: 'home', 1: 'hair', 2: 'print', 3: 'business', 4: 'law',
                 5: 'b&b', 6: 'locksmith', 7: 'bank', 8: 'massage', 9: 'gym',
                 10: 'health and beauty', 11: 'construction', 12: 'landscaping',
                 13: 'photography', 14: 'bus'},
       prim[2]: {0: 'arcade', 1: 'rec center', 2: 'spa', 3: 'art', 4:'theater',
                 },
       prim[3]: {0: 'service'},
       prim[4]: {0: 'breakfast', 1: 'pizza', 2: 'burger', 3: 'sandwich',
                 4: 'coffee', 5: 'fast food', 6: 'ice cream', 7: 'truck',
                 8: 'diner'},
       prim[5]: {0: 'elementary'},
       prim[6]: {0: 'pharmacy', 1: 'grocery', 2: 'liquor', 3: 'marijuana'},
       prim[7]: {0: 'bar'}
       }



cats = {'Trail': [ [prim[0]], [sec[prim[0]][0]] ],
        'Home Service': [ [prim[1]], [sec[prim[1]][0]] ],
        'Arcade': [ [prim[2]], [sec[prim[2]][0]] ],
        'Nature Preserve': [ [prim[0]], [sec[prim[0]][1]] ],
        'Airport Service': [ [prim[3]], [sec[prim[3]][0]] ],
        'Stables': [ [prim[0]], [sec[prim[0]][2]] ],
        'Campground': [ [prim[0]], [sec[prim[0]][3]] ],
        'Boat or Ferry': [ [prim[0]], [sec[prim[0]][4]] ],
        'Bike Trail': [ [prim[0]], [sec[prim[0]][0], sec[prim[0]][5]] ],
        'Farmers Market': [ [prim[0]], [sec[prim[0]][6]] ],
        'Park': [ [prim[0]], [sec[prim[0]][7]] ],
        'Salon / Barbershop': [ [prim[1]], [sec[prim[1]][1]] ],
        'Breakfast Spot': [ [prim[4]], [sec[prim[4]][0]] ],
        'Pizza Place': [ [prim[4]], [sec[prim[4]][1]] ],
        'Print Shop': [ [prim[1]], [sec[prim[1]][2]] ],
        'Elementary School': [ [prim[5]], [sec[prim[5]][0]] ],
        'Business Service': [ [prim[1]], [sec[prim[1]][3]] ],
        'Burger Joint': [ [prim[4]], [sec[prim[4]][2]] ],
        'Skate Park': [ [prim[0]], [sec[prim[0]][7], sec[prim[0]][8]] ],
        'Pool': [ [prim[0]], [sec[prim[0]][9]] ],
        'Lawyer': [ [prim[1]], [sec[prim[1]][4]] ],
        'Recreation Center': [ [prim[2]], [sec[prim[2]][1]] ],
        'Bed & Breakfast': [ [prim[1]], [sec[prim[1]][5]] ],
        'Farm': [ [prim[0]], [sec[prim[0]][10]] ],
        'Dog Run': [ [prim[0]], [sec[prim[0]][11]] ],
        'Spa': [ [prim[2]], [sec[prim[2]][2]] ],
        'Pharmacy': [ [prim[6]], [sec[prim[6]][0]] ],
        'Grocery Store': [ [prim[6]], [sec[prim[6]][1]] ],
        'Coffee Shop': [ [prim[4]], [sec[prim[4]][4]] ],
        'Sandwich Place': [ [prim[4]], [sec[prim[4]][3]] ],
        'Locksmith': [ [prim[1]], [sec[prim[1]][6]] ],
        'Fast Food Restaurant': [ [prim[4]], [sec[prim[4]][5]] ],
        'Bank': [ [prim[1]], [sec[prim[1]][7]] ],
        'Ice Cream Shop': [ [prim[4]], [sec[prim[4]][6]] ],
        'Liquor Store': [ [prim[6]], [sec[prim[6]][2]] ],
        'Massage Studio': [ [prim[1]], [sec[prim[1]][8]] ],
        'Gym / Fitness Center': [ [prim[1]], [sec[prim[1]][9]] ],
        'Health & Beauty Service': [ [prim[1]], [sec[prim[1]][10]] ],
        'Bar': [ [prim[7]], [sec[prim[7]][0]] ],
        'Golf Course': [ [prim[0]], [sec[prim[0]][12]] ],
        'Construction & Landscaping': [ [prim[1]], [sec[prim[1]][11],sec[prim[1]][12] ] ],
        'Marijuana Dispensary': [ [prim[6]], [sec[prim[6]][3]] ],
        'Playground': [ [prim[0]], [sec[prim[0]][13]] ],
        'Photography Studio': [ [prim[1]], [sec[prim[1]][13]] ],
        'Food Truck': [ [prim[4]], [sec[prim[4]][7]] ],
        'Art Gallery': [ [prim[2]], [sec[prim[2]][3]] ],
        'Gym': [ [prim[1]], [sec[prim[1]][9]] ],
        'Theater': [ [prim[2]], [sec[prim[2]][4]] ],
        'Bus Stop': [ [prim[1]], [sec[prim[1]][14]] ],
        'Diner': [ [prim[4]], [sec[prim[4]][8]] ],
        }







def extract_nested_values(it):
    if isinstance(it, list):
        for sub_it in it:
            yield from extract_nested_values(sub_it)
    elif isinstance(it, dict):
        for value in it.values():
            yield from extract_nested_values(value)
    else:
        yield it

all_prim_labels = [prim[k] for k in prim]
all_sec_labels = set( extract_nested_values(sec))



test_cats = all_cats[:50]



prim_hot = np.zeros( (len(test_cats), len(prim)) )
sec_hot = np.zeros( (len(test_cats), len(all_sec_labels)) )


prim_hot = pd.DataFrame(0, index = test_cats, columns = all_prim_labels)
sec_hot = pd.DataFrame(0, index = test_cats, columns = all_sec_labels)

for cat in test_cats:
    for prim_label in cats[cat][0]:
        prim_hot.loc[cat, prim_label] += 1
    for sec_label in cats[cat][1]:
        sec_hot.loc[cat, sec_label] += 1
    
    
    
    

