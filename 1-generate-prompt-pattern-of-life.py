#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from tqdm import tqdm 
import pickle
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

data_path = '../data/PofL'
df = pd.read_csv(os.path.join(data_path, 'Checkin.tsv'), sep='\t')
groundtruth = pd.read_csv(os.path.join(data_path, 'groundtruth.csv'))['agentId']
full_userid_list = sorted(list(df.UserId.unique()))
userid_list = []
t = 0
for userid in full_userid_list:
    if userid in groundtruth:
        userid_list.append(userid)
    elif t < 913:
        userid_list.append(userid)
        t += 1
    else:
        pass

weekdayDict = {
    0 : 'Mon', 1 : 'Tue', 2 : 'Wed', 3 : 'Thu', 4 : 'Fri', 5 : 'Sat', 6 : 'Sun',
}

lens = []
for userid in tqdm(userid_list):
    item = df.loc[df['UserId']==userid]
    lens.append(len(item))
for userid in tqdm(userid_list[:1]):
    item = df.loc[df['UserId']==userid]
    item['CheckinTime'] = pd.to_datetime(item['CheckinTime'])
    item['dayofweek'] = item.CheckinTime.apply(lambda x: weekdayDict[x.dayofweek])
    sequence = ""
    prev_X, prev_Y = None, None
    flag = True
    for i in range(len(item)):
        cur_item = item.iloc[i]
        CheckinTime, VenueType, dayofweek, X, Y = cur_item['CheckinTime'], cur_item['VenueType'], cur_item['dayofweek'], cur_item['X'], cur_item['Y']
        # add distance
        if i > 0:
            dist = np.sqrt((X-prev_X)**2 + (Y-prev_Y)**2)
            sequence += ', {:.1f} km ->'.format(dist/(10**3))
        prev_X, prev_Y = X, Y
        
        # form trajectory sequnece
        time = ':'.join(str(CheckinTime).split(' ')[1].split(':')[:-1])
        sequence += f"{dayofweek} {str(time)}, {VenueType}"
        if flag and str(CheckinTime).split(' ')[0] == '2019-08-15':
            flag = False
            sequence += " ***<deviate_point>*** "


    prompt = f"""
Task: You are a human mobility trajectory behavior anomaly detector. Given a historical human trajectory information, can you analyse the pattern behind the trajectory and give an anomaly score (from 0 to 1, where larger value indicates more abnormal) of this user's behavior?
Hint: The anomaly users would suddenly change their mobility pattern starting from a time point, which means after a certain time, their mobility behavior would significantly deviate from their past behaviors. We would use "***<deviate_point>***" inside each trajectory to denote the time point as hint.

Description of input trajectory data: A temporal sequence of visited place points, each place is consisted of the visited timestamp and its type of location. Then the traveled distance to next location is given.

Here is the sequence of trajector: {sequence}.

Give your analysis and present your esimated anomaly score (from 0 to 1, where larger value indicates more abnormal) inside a pair of square brackets [] : 
"""

    save_path = '../data/prompts/PofL/outliers-train-test'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, f'user_{userid}-with-hint'), 'w') as f:
        print(prompt, file=f)
with open(os.path.join(save_path, f'userid_list'), 'w') as f:
    print(userid_list, file=f)



print("average len:", np.mean(lens))