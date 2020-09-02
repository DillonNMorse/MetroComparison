# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 14:36:03 2020

@author: Dillon Morse
"""

import pandas as pd
from time import time
from unidecode import unidecode
import venue_category_processing_functions as vcpf
import random
import numpy as np


# =============================================================================
# The labels to be applied to the four-square provided categories
# =============================================================================
prim = {0: 'outdoor',
        1: 'services',
        2: 'entertainment',
        3: 'transit',
        4: 'restaurant',
        5: 'school',
        6: 'shopping',
        7: 'bar',
        9: 'exercise',
       10: 'hospitality'
        }


# =============================================================================
# Load all data
# =============================================================================
df = pd.concat( [pd.read_csv('denver_venue_data.csv'),
                 pd.read_csv('denver_venue_data.csv') ]).reset_index(drop = True)


# =============================================================================
# Drop 'Moving Target' category since it can apply to multiple venue types
# =============================================================================
df = df[ df['Category'] != 'Moving Target' ]


# =============================================================================
# Remove accents from categories (or similar)
# =============================================================================
df['Category'] = [unidecode(word) for word in df['Category'] ]


# =============================================================================
# Get all unique category types
# =============================================================================
all_cats = df['Category'].unique()


# =============================================================================
# Label each category with one or more of the above labels
# =============================================================================
cats = {'Trail': [prim[0], prim[9]],
        'Home Service': [prim[1]],
        'Arcade': [prim[2]],
        'Nature Preserve': [prim[0]],
        'Airport Service': [prim[3]],
        'Stables': [prim[0]],
        'Campground': [prim[0], prim[10]],
        'Boat or Ferry': [prim[0]],
        'Bike Trail': [prim[0], prim[9]],
        'Farmers Market': [prim[0], prim[6]],
        'Park': [prim[0]],
        'Salon / Barbershop': [prim[1]],
        'Breakfast Spot': [prim[4]],
        'Pizza Place': [prim[4]],
        'Print Shop': [prim[1], prim[6]],
        'Elementary School': [prim[5]],
        'Business Service': [prim[1]],
        'Burger Joint': [prim[4]],
        'Skate Park': [prim[0], prim[9]],
        'Pool': [prim[0], prim[9]],
        'Lawyer': [prim[1]],
        'Bed & Breakfast': [prim[10]],
        'Farm': [prim[0]],
        'Dog Run': [prim[0]],
        'Spa': [prim[10], prim[2]],
        'Pharmacy': [prim[6], prim[1]],
        'Grocery Store': [prim[6]],
        'Coffee Shop': [prim[4]],
        'Sandwich Place': [prim[4]],
        'Locksmith': [prim[1]],
        'Fast Food Restaurant': [prim[4]],
        'Bank': [prim[1]],
        'Ice Cream Shop': [prim[4]],
        'Liquor Store': [prim[6]],
        'Massage Studio': [prim[10], prim[2]],
        'Gym / Fitness Center': [prim[9]],
        'Health & Beauty Service': [prim[1]],
        'Bar': [prim[7]],
        'Golf Course': [prim[0], prim[2]],
        'Construction & Landscaping': [prim[1]],
        'Marijuana Dispensary': [prim[6]],
        'Playground': [prim[0], prim[2]],
        'Photography Studio': [prim[1]],
        'Food Truck': [prim[4], prim[0]],
        'Art Gallery': [prim[2]],
        'Gym': [prim[9]],
        'Theater': [prim[2]],
        'Bus Stop': [prim[3]],
        'Diner': [prim[4]],
        'Mexican Restaurant': [prim[4]], 
        'Salad Place': [prim[4]], 
        'Sporting Goods Shop': [prim[6]],
        'Dessert Shop': [prim[4]], 
        'Sushi Restaurant': [prim[4]], 
        "Doctor's Office": [prim[1]],
        'Light Rail Station': [prim[3]], 
        'Gift Shop': [prim[6]], 
        'Hotel': [prim[10]], 
        'Optical Shop': [prim[6]],
        'Sculpture Garden': [prim[0], prim[2]], 
        'Science Museum': [prim[2], prim[5]],
        'Residential Building (Apartment / Condo)': [prim[1]],
        'Furniture / Home Store': [prim[6]], 
        'Beach': [prim[0]], 
        'Wings Joint': [prim[4]],
        'Fried Chicken Joint': [prim[4]], 
        'Cafe': [prim[4]], 
        'Convenience Store': [prim[6], prim[1]], 
        'Pub': [prim[7]],
        'Gas Station': [prim[6], prim[1]], 
        'Chinese Restaurant': [prim[4]], 
        'Trade School': [prim[5]],
        'Noodle House': [prim[4]], 
        'Supermarket': [prim[6]],
        'Shipping Store': [prim[6], prim[1]], 
        'ATM': [prim[1]], 
        'Asian Restaurant': [prim[4]], 
        'Japanese Restaurant': [prim[4]],
        'Vietnamese Restaurant': [prim[4]], 
        'Hardware Store': [prim[6]], 
        'Greek Restaurant': [prim[4]],
        'Rental Car Location': [prim[1]], 
        'Thrift / Vintage Store': [prim[6]], 
        'Garden Center': [prim[6], prim[0]],
        'Tennis Court': [prim[0], prim[9]], 
        'Athletics & Sports': [prim[0], prim[2], prim[9]], 
        'Mobile Phone Shop': [prim[6]],
        'BBQ Joint': [prim[4]], 
        'Brewery': [prim[4], prim[7]], 
        'Boxing Gym': [prim[9]], 
        'Automotive Shop': [prim[1]],
        'Clothing Store': [prim[6]], 
        'Shopping Mall': [prim[6]], 
        'Outdoor Supply Store': [prim[6], prim[0]],
        'Tennis Stadium': [prim[2]], 
        'Bookstore': [prim[6]], 
        'Boutique': [prim[6]], 
        'Rock Climbing Spot': [prim[0], prim[9]],
        'River': [prim[0]],
        'Mongolian Restaurant': [prim[4]],
        'Parking': [prim[1]],
        'Fireworks Store': [prim[6]],
        'Cricket Ground': [prim[0], prim[2]],
        'Toy / Game Store': [prim[6]],
        'National Park': [prim[0]],
        'Storage Facility': [prim[1]],
        'Truck Stop': [prim[1], prim[10]],
        'Cafeteria': [prim[4]],
        'Skating Rink': [prim[0], prim[2]],
        'Souvenir Shop': [prim[6]],
        'Dive Shop': [prim[6]],
        'Bowling Alley': [prim[2]],
        'Surf Spot': [prim[0]],
        'Ethiopian Restaurant': [prim[4]],
        'Auto Workshop': [prim[1]],
        'Eastern European Restaurant': [prim[4]],
        'Bakery': [prim[4]],
        'Rest Area': [prim[0], prim[1]],
        'Mattress Store': [prim[6]],
        'Skydiving Drop Zone': [prim[0]],
        'Check Cashing Service': [prim[1]],
        'Food Service': [prim[4]],
        'Martial Arts Dojo': [prim[9]],
        'Spanish Restaurant': [prim[4]],
        'Entertainment Service': [prim[1], prim[2]],
        'Bistro': [prim[4]],
        'Nightclub': [prim[7]],
        'Casino': [prim[2]],
        'IT Services': [prim[1]],
        'Taco Place': [prim[4]],
        'Persian Restaurant': [prim[4]],
        'Bike Shop': [prim[6]],
        'Track': [prim[0], prim[9]],
        'Rugby Pitch': [prim[0], prim[2]],
        'Adult Boutique': [prim[6]],
        'Tanning Salon': [prim[1], prim[10]],
        'Australian Restaurant': [prim[4]],
        'Korean Restaurant': [prim[4]],
        'Canal': [prim[0]],
        'Opera House': [prim[2]],
        'High School': [prim[5]],
        'Bridge': [prim[0]],
        'Concert Hall': [prim[2]],
        'Flower Shop': [prim[6]],
        'Fair': [prim[0], prim[2]],
        'College Arts Building': [prim[5]],
        'Laboratory': [prim[1]],
        'Veterinarian': [prim[1]],
        'Credit Union': [prim[1]],
        'Jazz Club': [prim[7], prim[2]],
        'Airport': [prim[3]],
        'South American Restaurant': [prim[4]],
        'Herbs & Spices Store': [prim[6]],
        'Airport Lounge': [prim[3]],
        "Women's Store": [prim[6]],
        'Border Crossing': [prim[0]],
        'Smoke Shop': [prim[6]],
        'Indian Restaurant': [prim[4]],
        'Bath House': [prim[10]],
        'Paintball Field': [prim[0], prim[2]],
        'Fabric Shop': [prim[6]],
        'Field': [prim[0], prim[2]],
        'Multiplex': [prim[2]],
        'Floating Market': [prim[6]],
        'Outlet Store': [prim[6]],
        'Yoga Studio': [prim[9]],
        'Fondue Restaurant': [prim[4]],
        'College Quad': [prim[5]],
        'Distillery': [prim[7], prim[4]],
        'Ski Area': [prim[0], prim[2], prim[9]],
        'Rafting': [prim[0], prim[2]],
        'Music School': [prim[5]],
        'Tree': [prim[0]],
        'Post Office': [prim[1]],
        'Shoe Store': [prim[6]],
        'Theme Park': [prim[0], prim[2]],
        'Pilates Studio': [prim[9]],
        'Restaurant': [prim[4]],
        'Car Wash': [prim[1]],
        'Neighborhood': [prim[0]],
        'Theme Park Ride / Attraction': [prim[0], prim[2]],
        'Waterfront': [prim[0]],
        'Nail Salon': [prim[1], prim[10]],
        'Miscellaneous Shop': [prim[6]],
        'Hill': [prim[0]],
        'Steakhouse': [prim[4]],
        'Shoe Repair': [prim[1]],
        'Planetarium': [prim[2], prim[5]],
        'Tunnel': [prim[0]],
        'Auto Dealership': [prim[6]],
        'Karaoke Bar': [prim[7], prim[2]]    ,
        'Event Space': [prim[2]],
        'Gastropub': [prim[4], prim[7]],
        'Gluten-free Restaurant': [prim[4]],
        'Baseball Stadium': [prim[0], prim[2]],
        'Animal Shelter': [prim[1]],
        'Disc Golf': [prim[0], prim[2]],
        'Antique Shop': [prim[6]],
        'Beach Bar' : [prim[0], prim[7]],
        'Gym Pool': [prim[9]],
        'Go Kart Track': [prim[0], prim[2]],
        'Baseball Field': [prim[0], prim[2]],
        'Gourmet Shop': [prim[6]],
        'Moroccan Restaurant': [prim[4]],
        'Zoo Exhibit': [prim[0], prim[2]],
        'Airport Food Court': [prim[3], prim[4]],
        'Indie Theater': [prim[2]],
        'Movie Theater': [prim[2]],
        'Hobby Shop': [prim[6]],
        'Music Venue': [prim[2]],
        'Gay Bar': [prim[7]],
        'Dry Cleaner': [prim[1]],
        'Jewelry Store': [prim[6]],
        'Soup Place': [prim[4]],
        'Church': [prim[1]],
        'Insurance Office': [prim[1]],
        'Football Stadium': [prim[0], prim[2]],
        'Used Bookstore': [prim[6]],
        'Botanical Garden': [prim[0], prim[2]],
        'Performing Arts Venue': [prim[2]],
        'Warehouse': [prim[1]],
        'Forest': [prim[0]],
        'Smoothie Shop': [prim[4]],
        'Racetrack': [prim[0], prim[2]],
        'American Restaurant': [prim[4]],
        'Pool Hall': [prim[7], prim[2]],
        'Speakeasy': [prim[7]],
        'Lounge': [prim[7]],
        'RV Park': [prim[0]],
        'Fruit & Vegetable Store': [prim[6]],
        'Summer Camp': [prim[0]],
        'Vape Store': [prim[6]],
        'Motorsports Shop': [prim[6]],
        'University': [prim[5]],
        'Metro Station': [prim[3]],
        'Food Court': [prim[4]],
        'Weight Loss Center': [prim[1], prim[9]],
        'Water Park': [prim[0], prim[2]],
        'Country Dance Club': [prim[2], prim[7]],
        'Comfort Food Restaurant': [prim[4]],
        'Platform': [prim[3]],
        'Seafood Restaurant': [prim[4]],
        'Motorcycle Shop': [prim[6]],
        'Supplement Shop': [prim[6]],
        'Hawaiian Restaurant': [prim[4]],
        'Sports Club': [prim[2], prim[9]],
        'German Restaurant': [prim[4]],
        'Hookah Bar': [prim[7]],
        'Poke Place': [prim[4]],
        }


# =============================================================================
# Build lists that contain all labels and categories used for training
# =============================================================================
all_prim_labels = [prim[k] for k in prim]
random.seed(13)
rand_cats1 = random.sample( range(100, len(all_cats)), 100)
remaining = [k for k in np.arange(100, len(all_cats), 1) if not k in rand_cats1]
random.seed(23)
rand_cats2 = random.sample( remaining, 50)
random.seed(13)
test_cats = ( list(all_cats[:100]) 
              + list(all_cats[ rand_cats1 ])
              + list(all_cats[ rand_cats2 ])
              )


# =============================================================================
# Manually one-hot-encode using the above, each label corresponding to a
#   column and each category to a row. May be more than one label for a
#   category, marked with "1's" in multiple columns.
# =============================================================================
prim_hot = pd.DataFrame(0, index = test_cats, columns = all_prim_labels)

for cat in test_cats:
    for prim_label in cats[cat]:
        prim_hot.loc[cat, prim_label] += 1
        
prim_hot.reset_index(inplace = True)
prim_hot.rename(columns = {'index':'Category'}, inplace = True)     
        
    
# =============================================================================
# Quick function to allow access to the one-hot-encoded labelled categories        
# =============================================================================
def get_labeled_cats():
    return prim_hot


# =============================================================================
# Replaces each four-square-provided cateogry with a string containing the 
#   definition of each word in the category and (optionally) the definitions
#   of the synonyms of these words as well. Also includes in the string the
#   original category. Adding synonyms causes significant slow-downs.
#
# The result of this encoding is printed to a pickle file for easy later usage
#   without the need to re-encode all categories.   
# =============================================================================
def get_encoded_cats(prim_hot, stem = False, incl_syns = False, num_syns = 0):
    
    print('Encoding category defintions...')
    t0 = time()
    
    df = prim_hot.copy()
    
    df['CategoryKeyWords'] = (prim_hot['Category']
                              .apply(lambda x: 
                                     vcpf.get_definition_bow(x,
                                                             stem_bow = stem, 
                                                             incl_synonyms = incl_syns, 
                                                             num_syns = num_syns,
                                                             common = [],
                                                             ))
                              .apply (lambda x: vcpf.list_to_str(x) )
                              )
    t1 = time()
    
    print('Encoding took {:.0f} seconds.'.format(t1-t0))
         
    
            
    df.to_pickle('encoded_category_definitions.pickle')
    
    return pd.read_pickle('encoded_category_definitions.pickle')


# =============================================================================
# Re-encode the categories if and only if this module is run as main.
# =============================================================================
if __name__ == '__main__':
    get_encoded_cats(prim_hot,
                     stem = True,
                     incl_syns = False,
                     num_syns = 2
                     )





# =============================================================================
# 
# =============================================================================
def get_cats():
    return list(all_cats)