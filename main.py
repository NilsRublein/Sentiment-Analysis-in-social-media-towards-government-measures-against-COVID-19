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
import sklearn
import math
import matplotlib.pyplot as plt
import fasttext

from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

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
data = pd.read_csv(r'C:\Users\Nils\Documents\I-Tech\Courses\NLP\NLP_Project_2020\data.csv', names=colnames, encoding= 'ISO-8859-1') # Can also use windows-1252,latin-1 
df  = pd.DataFrame(data)

# Random sample with fraction of 0.001 of the original data for faster prototyping
df = df.sample(frac=0.001, random_state=1) 
print(df.head())

#%% TO DO: Data balancing

# Check if data is balanced
# If not, up / down sample the data

#%% Cleaning data
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
    
    #remove multiple white spaces
    tweets = np.vectorize(remove_pattern)(tweets, "[\s][\s]+")

    #remove special characters, numbers, punctuations (except for #)
    tweets = np.core.defchararray.replace(tweets, "[^a-zA-Z]", " ")
    
    #remove stopwords
    #must be for every token not tweet
    #tweets =  np.vectorize(remove_pattern)(tweets, f"[{stopwords}]")
    
    
    # Probably not the most elegant solution, but it works
    tweets_ = []
    for row in tweets:
        row = row.lower() # Make text lowercase
        #tokenize
        #lemmatize
        tweets_.append(row)
    
    return tweets_

df['Text'] = clean_tweets(df['Text'])


#%% Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Sentiment'], test_size=0.2, random_state=42)

#%% VADER. TO DO: add evaluation
# Adapted from https://medium.com/towards-artificial-intelligence/blacklivesmatter-twitter-vader-sentiment-analysis-using-python-8b6e6fc2cd6a

analyzer = SentimentIntensityAnalyzer()

#Storing the scores in list of dictionaries
scores = []

# Declare variables for scores
compound_list = []
positive_list = []
negative_list = []
neutral_list  = []

for row in X_train:
    compound = analyzer.polarity_scores(row)["compound"]
    pos = analyzer.polarity_scores(row)["pos"]
    neu = analyzer.polarity_scores(row)["neu"]
    neg = analyzer.polarity_scores(row)["neg"]
    
    scores.append({"Compound": compound,
                       "Positive": pos,
                       "Negative": neg,
                       "Neutral": neu
                  })

#Appending the scores into the dataframe for further analysis 
vader_sentiments_score = pd.DataFrame.from_dict(scores)
#df = df.join(sentiments_score)
plt.hist(vader_sentiments_score['Neutral'], bins=20) 

#%% FastText

# Append __label__0 prefix for fastText, so it recognizes it as a label and not a word
y_train_ = "__label__" + y_train.astype(str) #append_labels(y_train)
y_test_ = "__label__" + y_test.astype(str)

ft_train = pd.concat([X_train, y_train_], axis=1, sort=False)
ft_test = pd.concat([X_test, y_test_], axis=1, sort=False)
ft_train.to_csv('ft_train.txt', sep='\t', index = False, header = False) # FastText needs a .txt file as input
ft_test.to_csv('ft_test.txt', sep='\t', index = False, header = False)

# TO DO: What params are optimal?
hyper_params = {"lr": 0.01,
    "epoch": 20,
    "wordNgrams": 2,
    "dim": 20}     
        
# Train the model
model = fasttext.train_supervised(input='ft_train.txt', **hyper_params)
#print("Model trained with the hyperparameter \n {}".format(hyper_params))

# Evaluate the model     
result = model.test('ft_train.txt')
validation = model.test('ft_test.txt')
        
# Display accuracy
text_line = "\n Hyper paramaters used:\n" + str(hyper_params) + ",\n training accuracy:" + str(result[1])  + ", \n test accuracy:" + str(validation[1]) + '\n' 
print(text_line)


















