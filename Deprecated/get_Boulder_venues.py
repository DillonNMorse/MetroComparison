# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 13:34:43 2020

@author: Dillon Morse
"""
import requests
import GetCircleCenters as CC
import pandas as pd


CLIENT_ID = 'HS5PXH0KPM2HAJ21JRCXTEJD2ILAP1YHY13TQRKYMWQNWVZE' 
CLIENT_SECRET = 'TJBXRHEDGLSN2ZEHELWTZI2EDPIE150CQZISKEMCIURCAHIR'
VERSION = '20180605'


Boulder_circle_centers = CC.build_Boulder_circle_centers()

all_results = []
for circle_num, coord in enumerate(Boulder_circle_centers):

    lat = coord[0]
    lng = coord[1]
    radius = 1609.34 # 1 mile search radius for venues (try reducing - near
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
    
    results = requests.get(url).json()["response"]['groups'][0]['items']
    
    
    for result in results:
        info = [circle_num,
                result['venue']['id'],
                result['venue']['name'],
                result['venue']['location']['lat'],
                result['venue']['location']['lng'],
                result['venue']['categories'][0]['id'],
                result['venue']['categories'][0]['name']
                ]
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


df.to_csv('Boulder_venue_data.csv')     


