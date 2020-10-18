# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 10:16:39 2020

@author: Nils

TODO: 
    - add respective senitment scores from all_files to usa_tweets using the idx variable
"""

#%% Init libraries
import pandas as pd
import re
import glob


#%% merge akk csv files into one for hydrating
'''
path = r'D:\NLP_Project_2020\big_dataset' # use your path
all_files = glob.glob(path + "/*.csv")
tweets = pd.concat((pd.read_csv(f,  header=None) for f in all_files))

sample_tweets = tweets.sample(frac=0.01, random_state=1) 

sample_tweets.to_csv(r'D:\NLP_Project_2020\sample_big_data_tweets.csv', index = False)
'''

#%% Init twarc to hydrate tweets
from twarc import Twarc

consumer_key='jMSOeT4akbgDCH6ZF6XC1hwIO'
consumer_secret='OYoFQdqbQH3OK2o7IfFVOioXoeJrQuUzDdROBmv0y8Ul5pZvwX'
access_token_key='296259116-iZh92oTkQvGL6V93JZ0irquEFkwOb5j9mjg5ILEc'
access_token_secret='mDVC1ED99WOqQHNkgQojx2P5ECedgWzGtr5EDcKbPBqC7'

t = Twarc(consumer_key, consumer_secret, access_token_key, access_token_secret)

#%% Function that specifies what attributes we are saving from the tweet object

def save_data(index, val=False):
    
    if val:
        if tweet["user"]["location"] is not None:
            loc = tweet["user"]["location"] # location from profile
        else:
            loc = 0
    else:        
        loc = tweet["place"]["country"] # place based on the "point" location
    
    if tweet["full_text"] is not None and tweet["coordinates"] is not None:
        text = tweet["full_text"]  
        #longitude, latitude = tweet["coordinates"]["coordinates"]
        date = tweet['created_at']
        language = tweet['lang']
        data.append({'Text':text, 'Location':loc, 'Language':language, 'date':date, "idx": index })

#%% Check for relevant coordinates and save data
# Takes ~32 min
data = []
idx = 0

for tweet in t.hydrate(open(r'D:\NLP_Project_2020\all_tweets.csv')):
     idx += 1
     if tweet["coordinates"]:
         if tweet["place"] is not None and tweet["place"]["country"] == 'Verenigde Staten':
             save_data(idx)

     elif tweet["place"] and tweet["place"]["country"] == 'Verenigde Staten':
         save_data(idx)     
     else:
         #TODO: check the value in "loc_profile" if it is from a country of your interest
         save_data(idx, val = True)

df = pd.DataFrame(data)
df.to_csv(r'D:\NLP_Project_2020\usa_tweets.csv', index = False)