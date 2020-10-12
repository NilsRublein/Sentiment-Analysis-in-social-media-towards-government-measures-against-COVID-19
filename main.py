# -*- coding: utf-8 -*-
"""
Sentiment analysis of dummy twitter set using VADER.
Data set can be found here: https://www.kaggle.com/kazanova/sentiment140
"""
#%% Import libraries
import os
import re
import pandas as pd
import numpy as np
import glob
import string
import nltk
import math
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


#%% Load Data

'''
sentiment: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)

ids: The id of the tweet ( 2087)

date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)

flag: The query (lyx). If there is no query, then this value is NO_QUERY.

user: the user that tweeted (robotickilldozr)

text: the text of the tweet (Lyx is cool)
'''

colnames=['Sentiment', 'ID', 'Date', 'Flag', 'User', 'Text'] 
data = pd.read_csv(r'C:\Users\Nils\Documents\I-Tech\Courses\NLP\NLP_Project_2020\data.csv', names=colnames, encoding= 'latin-1') # Can also use windows-1252
df  = pd.DataFrame(data)
df = df[:1000] # Take first 1000 entries for testing.
print(df.head())

#%% Preprocessing
# Adapted from https://medium.com/towards-artificial-intelligence/blacklivesmatter-twitter-vader-sentiment-analysis-using-python-8b6e6fc2cd6a

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)        
    return input_txt

def clean_tweets(tweets):
    #remove twitter Return handles (RT @xxx:)
    tweets = np.vectorize(remove_pattern)(tweets, "RT @[\w]*:") 
    
    #remove twitter handles (@xxx)
    tweets = np.vectorize(remove_pattern)(tweets, "@[\w]*")
    
    #remove URL links (httpxxx)
    tweets = np.vectorize(remove_pattern)(tweets, "https?://[A-Za-z0-9./]*")

    #remove special characters, numbers, punctuations (except for #)
    tweets = np.core.defchararray.replace(tweets, "[^a-zA-Z]", " ")
    
    return tweets

df['Text']= clean_tweets(df['Text'])

# ADD Lemmatization, tokenization?

#%% VADER
# Adapted from https://medium.com/towards-artificial-intelligence/blacklivesmatter-twitter-vader-sentiment-analysis-using-python-8b6e6fc2cd6a

analyzer = SentimentIntensityAnalyzer()

#Storing the scores in list of dictionaries
scores = []
# Declare variables for scores
compound_list = []
positive_list = []
negative_list = []
neutral_list = []

for i in range(df['Text'].shape[0]):
#print(analyser.polarity_scores(sentiments_pd['Tweet'][i]))
    compound = analyzer.polarity_scores(df['Text'][i])["compound"]
    pos = analyzer.polarity_scores(df['Text'][i])["pos"]
    neu = analyzer.polarity_scores(df['Text'][i])["neu"]
    neg = analyzer.polarity_scores(df['Text'][i])["neg"]
    
    scores.append({"Compound": compound,
                       "Positive": pos,
                       "Negative": neg,
                       "Neutral": neu
                  })

#Appending the scores into the dataframe for further analysis 
vader_sentiments_score = pd.DataFrame.from_dict(scores)
#df = df.join(sentiments_score)
plt.hist(vader_sentiments_score['Neutral'], bins=20) #Similarly plots for negative, positive and neutral were also made.

