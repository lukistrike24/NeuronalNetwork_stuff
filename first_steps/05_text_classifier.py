import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Get a pandas DataFrame object of all the data in the csv file:
df = pd.read_csv('training_data\\tweets.csv')

# Get pandas Series object of the "tweet text" column:
text = df['tweet_text']

# Get pandas Series object of the "emotion" column:
target = df['is_there_an_emotion_directed_at_a_brand_or_product']

# Remove the blank rows from the series:
target = target[pd.notnull(text)]
text = text[pd.notnull(text)]

# Perform feature extraction:
count_vect = CountVectorizer()
count_vect.fit(text)
counts = count_vect.transform(text)

# Train with this data with a Naive Bayes classifier:
nb = MultinomialNB()
nb.fit(counts, target)

#Try the classifier
print(nb.predict(count_vect.transform(['i hate my iphone'])))
print(nb.predict(count_vect.transform(['This iphone is awsome'])))
