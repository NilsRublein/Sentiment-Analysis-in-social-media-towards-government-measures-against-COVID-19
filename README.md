# Sentiment Analysis in social media towards government measures against COVID-19

This project was set out to investigate sentiment in social media towards government measures against COVID19. The rapid spread of the corona virus is holding the world in its grip, creating a strong need for discovering efficient and effective strategies to interpret the flow of information and and the development of mass sentiment in pandemic scenarios. The wide spread use of social media, readily available data, and the exponential growth in advancements corresponding to machine learning and natural language processing make it possible to analyse public sentiment. Understanding people’s thoughts on their government and its policies can also help to find out how such policies should be introduced and communicated.

The file main.py compares the predicted sentiment of tweets relating to COVID-19 by four different classifiers, namely Naive Bayes, Logisitic Regression, VADER
and FastText. Each classifier (where applicable) used an n-gram length of 1, made use of the TfidfVectoriser, used no stopword filtering, and no lemmatisation or
stemming. It has been observed that none of the classifiers performs significantly better than another one with exception of the VADER model, which exerted
lower performance in terms of accuracy and F1 score. In addition to comparing the performance of these various classifiers, the sentiment of corona related tweets originating from the U.S is visualized over time.

You can use !pip install env_requirements.txt to install the necessary environment requirements. 
In addittion, the file hydrateTweets.py obtains tweets filtered for the location of the U.S.. You can adjust this file to filter tweets for a different location.



