import string

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

nltk.download('stopwords')

df = pd.read_csv('emails.csv')
df['text'] = df['text'].apply(lambda x: x.replace('\n', ' '))
df.text.iloc[1]