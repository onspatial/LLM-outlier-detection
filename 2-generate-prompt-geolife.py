import numpy as np
import pandas as pd
from tqdm import tqdm 
import pickle
import argparse
import os
import math
import matplotlib.pyplot as plt

def haversine_distance(coord1, coord2):
    """
    Calculate the Haversine distance between two points on the earth specified by latitude/longitude.

    Parameters:
    coord1 : tuple of float
        (lat1, lon1)
    coord2 : tuple of float
        (lat2, lon2)

    Returns:
    distance : float
        Distance between the two points in kilometers.
    """
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

    # Compute differences in latitude and longitude
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Calculate distance using Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c

    return distance

data_path = '../data/geolife/outliers-train-test'
df = pd.read_csv(os.path.join(data_path, 'test-20-outliers-69-agents-0.8-normal-portion.tsv'), sep=' ')
df = df.rename(
    columns={
        'Longitude': 'X',
        'Latitude': 'Y',
        'ArrivingTime': 'CheckinTime',
        'LocationType': 'VenueType',
        'AgentID': 'UserId',
    }
)
userid_list = sorted(list(df.UserId.unique()))
weekdayDict = {
    0 : 'Mon', 1 : 'Tue', 2 : 'Wed', 3 : 'Thu', 4 : 'Fri', 5 : 'Sat', 6 : 'Sun',
}
dict_seq = {}
lens = []
for userid in tqdm(userid_list):
    item = df.loc[df['UserId']==userid]
    lens.append(len(item))
for userid in tqdm(userid_list):
    item = df.loc[df['UserId']==userid]
    item['CheckinTime'] = item['CheckinTime'].apply(lambda x: 'T'.join(x.split(',')))
    item['CheckinTime'] = pd.to_datetime(item['CheckinTime'])
    item['dayofweek'] = item.CheckinTime.apply(lambda x: weekdayDict[x.dayofweek])
    sequence = ""
    prev_X, prev_Y = None, None
    for i in range(len(item)):
        cur_item = item.iloc[i]
        CheckinTime, VenueType, dayofweek, X, Y = cur_item['CheckinTime'], cur_item['VenueType'], cur_item['dayofweek'], cur_item['X'], cur_item['Y']
        # add distance
        if i > 0:
            dist = haversine_distance((prev_X, prev_Y), (X, Y)) #np.sqrt((X-prev_X)**2 + (Y-prev_Y)**2)
            sequence += ', {:.1f} km ->'.format(dist)
        prev_X, prev_Y = X, Y
        
        # form trajectory sequnece
        time = ':'.join(str(CheckinTime).split(' ')[1].split(':')[:-1])
        sequence += f"{dayofweek} {str(time)}, {VenueType}"
        if i == int(len(item)*0.8):
             sequence += " ***<deviate_point>*** "
    dict_seq[userid] = sequence

for userid in tqdm(userid_list):
    sequence = dict_seq[userid]
    prompt = f"""
Task: You are a human mobility trajectory behavior anomaly detector. Given a historical human trajectory information, can you analyse the pattern behind the trajectory and give an anomaly score (from 0 to 1, where larger value indicates more abnormal) of this user's behavior?
Hint: The anomaly users would suddenly change their mobility pattern starting from a time point, which means after a certain time, their mobility behavior would significantly deviate from their past behaviors. We would use "***<deviate_point>***" inside each trajectory to denote the time point as hint.

Description of input trajectory data: A temporal sequence of visited place points, each place is consisted of the visited timestamp and its type of location. Then the traveled distance to next location is given.

Here is the sequence of trajector: {sequence}.

Give your analysis and present your esimated anomaly score (from 0 to 1, where larger value indicates more abnormal) inside a pair of square brackets [] : 
"""
    save_path = '../data/prompts/geolife/outliers-train-test'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, f'user_{userid}-with-hint'), 'w') as f:
        print(prompt, file=f)
    

with open(os.path.join(save_path, f'userid_list'), 'w') as f:
    print(userid_list, file=f)

N = len(userid_list)
prompt = f"""
Task: You are a human mobility trajectory behavior anomaly detector. Given a set of {N} users' historical human trajectories information, can you analyse the pattern behind each user's trajectory and give an anomaly score (from 0 to 1, where larger value indicates more abnormal) of users' behavior?
Hint: The anomaly users would suddenly change their mobility pattern starting from a time point, which means after a certain time, their mobility behavior would significantly deviate from their past behaviors. We would use "***<deviate_point>***" inside each trajectory to denote the time point as hint.

Description of input trajectory data: The trajectories of {N} users will be given line by line. Each user's trajectory will be a temporal sequence of visited place points, where each place is consisted of the visited timestamp and its type of location. Then the traveled distance to next location is given.

"""
for userid in tqdm(userid_list):
    sequence = dict_seq[userid]
    prompt += f"Here is the sequence of user {userid} : {sequence}\n"
prompt += "Given your analysis and present your esimated anomaly scores about all users (from 0 to 1, where larger value indicates more abnormal): "
    
with open(os.path.join(save_path, f'combined_prompt-with-hint.txt'), 'w') as f:
    print(prompt, file=f)


print("average len:", np.mean(lens))