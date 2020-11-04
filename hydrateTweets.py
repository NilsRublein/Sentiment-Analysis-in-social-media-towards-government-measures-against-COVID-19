# -*- coding: utf-8 -*-
"""
Hydrates tweets from the geotagged corona virus tweet data set and filteres the tweets for the location "USA" 
The initial data set for tweeets of the usa can be found here: https://ieee-dataport.org/open-access/coronavirus-covid-19-geo-tagged-tweets-dataset

Warning: This code is handed in just for reference, it's quite messy.
"""

#%% Init libraries
import pandas as pd
import re
import glob


#%% merge all csv files into one for hydrating
'''
path = r'D:\NLP_Project_2020\big_dataset' # use your path
all_files = glob.glob(path + "/*.csv")
tweets = pd.concat((pd.read_csv(f,  header=None) for f in all_files))

sample_tweets = tweets.sample(frac=0.01, random_state=1) 

sample_tweets.to_csv(r'D:\NLP_Project_2020\sample_big_data_tweets.csv', index = False)
'''

#%% Init twarc to hydrate tweets
from twarc import Twarc

# Fill in your keys here
consumer_key='consumer_key'
consumer_secret='consumer_secret'
access_token_key='access_token_key'
access_token_secret='access_token_secret'

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

#%%  add sentiment values 

df1 = pd.read_csv(r'D:\NLP_Project_2020\usa_tweets.csv')
df2 = pd.read_csv(r'D:\NLP_Project_2020\all_tweets.csv')    
sent = [df2['1'][val] for val in df1['idx']]
sent = pd.DataFrame({"Sentiment": sent})
df1 = df1.drop(['idx'], axis=1)
df1.rename(columns={"date": "Date"})
result = pd.concat([df1, sent], axis=1, sort=False)
result.to_csv(r'D:\NLP_Project_2020\usa_tweets_with_sent_scores.csv', index = False)