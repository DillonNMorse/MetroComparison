# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 14:36:40 2020

@author: Dillon Morse
"""

import requests
import GetCircleCenters as CC
import pandas as pd
import random as rm
import time


CLIENT_ID = 'HS5PXH0KPM2HAJ21JRCXTEJD2ILAP1YHY13TQRKYMWQNWVZE' 
CLIENT_SECRET = 'TJBXRHEDGLSN2ZEHELWTZI2EDPIE150CQZISKEMCIURCAHIR'
VERSION = '20180605'


de_centers = CC.build_de_circle_centers()

num_results = len(de_centers)
all_results = []
for circle_num, coord in enumerate(de_centers):

    lat = coord[0]
    lng = coord[1]
    radius = 707 #Covers everything with minimal overlap
    #radius = 1609.34 # 1 mile search radius for venues (try reducing - near
                     # or at limit of number venues per circle)
    LIMIT = 200
    
    url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT
            )
    

    if circle_num % 15 == 0:
        delay = rm.uniform(0,5)
        time.sleep( delay )
    if circle_num % 6 == 0:
        delay = rm.uniform(0,3)
        time.sleep( delay )
    
    try:
        results = requests.get(url).json()["response"]['groups'][0]['items']
    except:
        results = [{ 'venue': {'id':None,
                              'name':None,
                              'location':{'lat':None, 'lng':None},
                              'categories': [ {'id': None,'name':None}]
                              }
                   }]
    
    if circle_num % 20 == 0:
        print( '{:.1f} percent complete.'.format(circle_num/num_results*100))
    
    for result in results:
        try:
            info = [circle_num,
                    result['venue']['id'],
                    result['venue']['name'],
                    result['venue']['location']['lat'],
                    result['venue']['location']['lng'],
                    result['venue']['categories'][0]['id'],
                    result['venue']['categories'][0]['name']
                    ]
        except:
            info = [None]*7


        all_results.append(info)

df_columns = ['CircleNum',
              'ID',
              'Name',
              'Lat',
              'Lng',
              'CategoryID',
              'Category']
df = pd.DataFrame( all_results, columns = df_columns)
        
df_columns = ['CircleNum',
              'ID',
              'Name',
              'Lat',
              'Lng',
              'CategoryID',
              'Category']
df = pd.DataFrame( all_results, columns = df_columns)


df.to_csv('denver_venue_data.csv')     


