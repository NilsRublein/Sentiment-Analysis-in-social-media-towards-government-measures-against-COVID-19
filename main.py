# -*- coding: utf-8 -*-
"""
Sentiment analysis of geotagged tweets with location USA over time.

The training dataset (sentiment values of tweets) can be found here:
https://www.kaggle.com/kazanova/sentiment140

The initial data set for tweeets of the usa can be found here: https://ieee-dataport.org/open-access/coronavirus-covid-19-geo-tagged-tweets-dataset
The tweet IDs in this data set need then to be hydrated (e.g. using twarc) and filtered for the location "USA".

The cells contain the following steps:
    - Load libraries
    - Loads data
    - Prepcesses / clean data
    - Split data into test and train
    - Classify
        - Vader
        - Fast Text
        - Logistic Regression
        - Naive Bayes
    - Plot Bar chart for f1 and acc scores
    - Plot ROC curves
    
TODO:
    - plot sentiment over time for the tweets from usa using the best classifier.
"""
#%% Import libraries
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
import seaborn as sns

from nltk.corpus import stopwords
nltk.download('stopwords')
en_stopwords = set(stopwords.words("english"))

from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, accuracy_score, mean_squared_error
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score

from sklearn.naive_bayes import MultinomialNB, ComplementNB, GaussianNB, BernoulliNB
from pprint import pprint
import json
from datetime import datetime
import pickle

#%% Load Data
colnames=['Sentiment', 'ID', 'Date', 'Flag', 'User', 'Text']
df = pd.read_csv(r'D:\NLP_Project_2020\data.csv', names=colnames, encoding= 'ISO-8859-1') # Can also use windows-1252,latin-1 
df = df.drop(['ID', 'Flag', 'User'], axis=1)
sample_size = 800000 #800000
df = pd.concat([df.query("Sentiment==0").sample(sample_size),df.query("Sentiment==4").sample(sample_size)]) # Better way of sampling, this way have guaranteed balanced data.
df['Sentiment'] = df['Sentiment'].map({0:0,4:1}) # map 4 to 1

# Smaller sample for SVM / LR
sample_size = 40000 #40000
df2 = pd.concat([df.query("Sentiment==0").sample(sample_size),df.query("Sentiment==1").sample(sample_size)]) # Better way of sampling, this way have guaranteed balanced data.

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
df2['Text'] = clean_tweets(df2['Text'])


#%% Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Sentiment'], test_size=0.2, random_state=42)
X_train_, X_test_, y_train_, y_test_ = train_test_split(df2['Text'], df2['Sentiment'], test_size=0.2, random_state=42)

#%% VADER
# Adapted from https://medium.com/towards-artificial-intelligence/blacklivesmatter-twitter-vader-sentiment-analysis-using-python-8b6e6fc2cd6a
analyzer = SentimentIntensityAnalyzer()

# We don't need to train, since this is a pre-trained model!
# Use compound values, those are the normalized values from pos, neg and neutral. Range: [-1,1]
vs_predictions = [analyzer.polarity_scores(row)["compound"] for row in X_test]
vs_predictions = pd.cut(vs_predictions, bins=2,labels=[0, 1]) # Map cont. values from [-1,1] to either 0 or 1.

VS_acc = accuracy_score(y_test,vs_predictions)
VS_f1 = f1_score(y_test,vs_predictions)
VS_fpr, VS_tpr, _ = roc_curve(y_test,vs_predictions)

#%% FastText
# can also try unsupervised version (CBOW, skipgram)

# Append __label__0 prefix for fastText, so it recognizes it as a label and not a word
y_train_ft = "__label__" + y_train.astype(str) #append_labels(y_train)
y_test_ft = "__label__" + y_test.astype(str)

ft_train = pd.concat([X_train, y_train_ft], axis=1, sort=False)
ft_test = pd.concat([X_test, y_test_ft], axis=1, sort=False)
ft_train.to_csv('ft_train.txt', sep='\t', index = False, header = False) # FastText needs a .txt file as input
ft_test.to_csv('ft_test.txt', sep='\t', index = False, header = False)

# TO DO: What params are optimal?
hyper_params = {"lr": 0.01,
    "epoch": 20,
    "wordNgrams": 2,
    "dim": 20}     
        
# Train the model
model = fasttext.train_supervised(input='ft_train.txt', **hyper_params)
# optimization: https://notebook.community/fclesio/learning-space/Python/fasttext-autotune
#model = fasttext.train_supervised(input='ft_train.txt', autotuneValidationFile='ft_test.txt')
#print("Model trained with the hyperparameter \n {}".format(hyper_params))

# Evaluate the model     
result = model.test('ft_train.txt')
validation = model.test('ft_test.txt')

'''        
# Display accuracy. I think it is actually F1
text_line = "\n Hyper paramaters used:\n" + str(hyper_params) + ",\n training accuracy:" + str(result[1])  + ", \n test accuracy:" + str(validation[1]) + '\n' 
print(text_line)


def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

print_results(*result)
print_results(*validation)
'''
#%% Obtain FT F1, acc and roc values
def predict(row):
    return model.predict(row['Text'])
ft_pred = ft_test.apply(predict,axis=1)

ft_pred = pd.DataFrame(ft_pred) #convert from series to df
ft_pred = ft_pred[0].str.get(0) #get first tuple

def strip_FT(input):
    # Removes label from FT data and convert to int
    output = []
    for row in input:
        line =  ''.join(str(x) for x in row)
        line = re.sub('__label__','',line) 
        line = int(line)
        output.append(line)
    return output

ft_pred_stripped = strip_FT(ft_pred)
ft_test_stripped = strip_FT(ft_test['Sentiment'])

FT_acc = accuracy_score(ft_test_stripped,ft_pred_stripped)
FT_f1 = f1_score(ft_test_stripped,ft_pred_stripped) 
FT_fpr, FT_tpr, _ = roc_curve(ft_test_stripped,ft_pred_stripped)



#%% LR 
# Adopted from https://www.kaggle.com/lbronchal/sentiment-analysis-with-svm

def tokenize(text): 
    tknzr = TweetTokenizer()
    return tknzr.tokenize(text)

def stem(doc):
    return (SnowballStemmer.stem(w) for w in analyzer(doc)) 

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    ngram_range=(1, 1),
    stop_words = en_stopwords)

#%% LR: parameter optimization
kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# re-seed the random generator. 
np.random.seed(1)

# Linear kernel since it is a binary problem (pos, neg)
pipeline_LR = make_pipeline(vectorizer, LogisticRegression(max_iter=1000))

# For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
param_grid_ = {
     'logisticregression__penalty' : ['l1', 'l2'],
    'logisticregression__C' : np.logspace(-4, 4, 20),
    'logisticregression__solver' : ['liblinear']} #liblinear; saga often the best choice but takes way more time

grid_LR = GridSearchCV(pipeline_LR,
                    param_grid = param_grid_, 
                    cv = kfolds,
                    scoring="roc_auc",
                    verbose=1,   
                    n_jobs=-1) 

grid_LR.fit(X_train_, y_train_)
grid_LR.score(X_test, y_test)
print('best LR paramater:' + str(grid_LR.best_params_))
print('score: ' + str(grid_LR.best_score_))

#%% LR: Take best parameters 
def report_results(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)        
    
    auc = roc_auc_score(y, pred_proba)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    fpr, tpr, _ = roc_curve(y, pred_proba)
    result = [auc,f1, acc, prec, rec, fpr, tpr]
    return result
    
LR_auc,LR_f1,LR_acc,LR_prec,LR_rec,LR_fpr, LR_tpr = report_results(grid_LR.best_estimator_, X_test, y_test)
print({'auc': LR_auc, 'f1': LR_f1, 'acc': LR_acc, 'precision': LR_prec, 'recall': LR_rec})


#%% NB: MultinomialNB with unigrams and TF-IDF

def NB(clf, vectorizer):
    text_counts = vectorizer.fit_transform(df['Text'])

    X_train, X_test, y_train, y_test = train_test_split(text_counts, df['Sentiment'], test_size=0.2,
                                                        random_state=42)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    acc = accuracy_score(y_test,predicted)
    f1 = f1_score(y_test,predicted)    
    fpr, tpr, _ = roc_curve(y_test,predicted)
    return acc, f1, fpr, tpr

tk = TweetTokenizer()

vectorizer = TfidfVectorizer(ngram_range=(1, 1), tokenizer=tk.tokenize)
NB_acc, NB_f1, NB_fpr, NB_tpr = NB(MultinomialNB(), vectorizer)


#%% Bar chart for f1 and acc scores for all classifiers

labels = ['NB', 'LR', 'VADER', 'FastText']
Accs = [NB_acc, LR_acc, VS_acc, FT_acc]
F1 = [NB_f1, LR_f1, VS_f1, FT_f1]

def round_vals(input):
    output = []
    for num in input:
        num = round(num*100,2)
        output.append(num)
    return output

Accs = round_vals(Accs)
F1 = round_vals(F1)

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, Accs, width, label='Acc', color='#4a1d7a')
rects2 = ax.bar(x + width/2, F1, width, label='F1', color='#ac71ec')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Accuracy and F1 scores per classifier')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='lower right')

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.show()

#%% Plot ROC curves for all classifers in one graph

plt.plot(LR_fpr, LR_tpr, color="red")
plt.plot(VS_fpr, VS_tpr, color="blue")
plt.plot(FT_fpr, FT_tpr, color="pink")
plt.plot(NB_fpr, NB_tpr, color="cyan")
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc curve')
plt.legend(['LR', 'VD', 'FT','NB'])
plt.show()


'''
#%% Bonus: SVM
# https://www.kaggle.com/lbronchal/sentiment-analysis-with-svm

def tokenize(text): 
    tknzr = TweetTokenizer()
    return tknzr.tokenize(text)

def stem(doc):
    return (SnowballStemmer.stem(w) for w in analyzer(doc)) 

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    ngram_range=(1, 1),
    stop_words = en_stopwords)

#%%
# use cross validation and grid search to find good hyperparameters for our SVM model. 
#We need to build a pipeline to don't get features from the validation folds when building each training model.

# 5 Folds
kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# re-seed the random generator. 

np.random.seed(1)

# Linear kernel since it is a binary problem (pos, neg)
pipeline_svm = make_pipeline(vectorizer, 
                            SVC(probability=True, kernel="linear", class_weight="balanced"))

grid_svm = GridSearchCV(pipeline_svm,
                    param_grid = {'svc__C': [0.01, 0.1, 1]}, # what does this mean?
                    cv = kfolds,
                    scoring="roc_auc",
                    verbose=1,   
                    n_jobs=-1) 

grid_svm.fit(X_train_, y_train_)
grid_svm.score(X_test, y_test)

print('best svm paramater:' + str(grid_svm.best_params_))
print('score: ' + str(grid_svm.best_score_))
#%% SVM: use best parameters to train model
# Let's see how the model (with the best hyperparameters) works on the test data:
    
def report_results(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)        
    
    auc = roc_auc_score(y, pred_proba)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    result = {'auc': auc, 'f1': f1, 'acc': acc, 'precision': prec, 'recall': rec}
    return result
    

print(report_results(grid_svm.best_estimator_, X_test, y_test))
#%%
def get_roc_curve(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, pred_proba)
    return fpr, tpr

roc_svm = get_roc_curve(grid_svm.best_estimator_, X_test, y_test)

svm_fpr, svm_tpr = roc_svm
plt.figure(figsize=(14,8))
plt.plot(svm_fpr, svm_tpr, color="red")
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc curve')
plt.show()
'''