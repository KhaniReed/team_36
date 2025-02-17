import string

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Download the stopwords from NLTK
nltk.download('stopwords')

# Load the spam email dataset
df = pd.read_csv('spam_ham_dataset.csv')
df['text'] = df['text'].apply(lambda x: x.replace('\r\n', ' ')) # Replace newline characters in the text column
df.info()

#Initialize the PorterStemmer to reduce words to their root form
#As well as creating a list to store preprocessed text data
stemmer = PorterStemmer()
corpus = []

# Convert stopwords list to a set for faster lookup
stopwords_set = set(stopwords.words('english'))

# Preprocess the text data
for i in range(len(df)):
    text = df['text'].iloc[i].lower() # Convert text to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)).split() # Remove punctuation and split into words
    text = [stemmer.stem(word) for word in text if word not in stopwords_set] # Remove stopwords and stem the words
    text = ' '.join(text) # Join the words back into a single string
    corpus.append(text) # Append the preprocessed text to the corpus list