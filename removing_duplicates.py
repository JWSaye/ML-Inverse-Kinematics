# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 23:31:01 2019

@author: Stefan Borkovski
"""

# This code is used to clear the dataset from duplicates (points that are very close). This method 
# proved that it is improving the training phase. 

import pandas as pd
import numpy as np
import time

# Load the dataset
df = pd.read_csv(r'.\datasets_6DOF\d6DOF.csv', encoding='utf8')
df = df.drop(['Unnamed: 0'], axis=1)

# Optimization starts here
st = time.time()

# Set decimal precision
decimal = 1

# Create a dictionary to store unique points (rounded)
unique_points = {}
drop_list = []

# Iterate through the dataframe
for idx, row in df.iterrows():
    # Create a tuple of rounded coordinates
    rounded_point = (round(row[0], decimal), round(row[1], decimal), round(row[2], decimal))
    
    # Check if this rounded point has been seen before
    if rounded_point in unique_points:
        # If seen, mark the current index for dropping (duplicate)
        drop_list.append(idx)
    else:
        # If not seen, add it to the dictionary
        unique_points[rounded_point] = idx

# Drop duplicates
df = df.drop(drop_list, axis=0).reset_index(drop=True)

en = time.time()
print(f"Time needed: {en-st} seconds")

df.to_csv(r'.\datasets_6DOF\d6DOF.csv')