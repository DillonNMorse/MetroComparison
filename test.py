# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:39:29 2020

@author: Dillon Morse
"""
import pandas as pd
import tkinter as tk
import random



df = pd.concat( [pd.read_csv('denver_venue_data.csv'),
                 pd.read_csv('denver_venue_data.csv') ]).reset_index(drop = True)


cats =  list( df['Category'].unique()  )

random.shuffle(cats)

labeled = {}


import geopandas