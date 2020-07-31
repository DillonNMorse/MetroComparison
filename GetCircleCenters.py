# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:14:24 2020

@author: Dillon Morse
"""


def build_CH_circle_centers():
    
    circle_centers = []
    
    bl_lat = 35.873
    bl_lng = -79.10
    
    delta_lng = 0.0111
    delta_lat = 0.009
    
    for j in range(12):
        for k in range(9):
            circle_centers.append( [bl_lat + j*delta_lat, bl_lng + k*delta_lng] )
    
    return circle_centers
    

def build_Boulder_circle_centers():
    
    circle_centers = []
    
    bl_lat = 39.969324
    bl_lng = -105.288731
    
    delta_lng = 0.0111
    delta_lat = 0.009
    
    for j in range(12):
        for k in range(8):
            circle_centers.append( [bl_lat + j*delta_lat, bl_lng + k*delta_lng] )
    
    return circle_centers