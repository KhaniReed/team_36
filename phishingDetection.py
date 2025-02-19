import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import re
import numpy as np
from urllib.parse import urlparse


# Load datasets
email_df = pd.read_csv("spam_ham_dataset.csv")
url_df = pd.read_csv("dataset_phishing.csv")

# ----- Preprocessing Emails -----
email_df = email_df[['text', 'label_num']]
email_df.rename(columns={'label_num': 'phishing_label'}, inplace=True)

# Convert email text into TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=500)
email_tfidf = tfidf_vectorizer.fit_transform(email_df['text']).toarray()
email_tfidf_df = pd.DataFrame(email_tfidf, columns=[f'tfidf_{i}' for i in range(email_tfidf.shape[1])])
email_tfidf_df['phishing_label'] = email_df['phishing_label']

# ----- Preprocessing URLs -----
url_df['phishing_label'] = (url_df['status'] == 'phishing').astype(int)
url_df['url_length'] = url_df['url'].apply(len)
url_df['num_dots'] = url_df['url'].apply(lambda x: x.count('.'))
url_df['num_hyphens'] = url_df['url'].apply(lambda x: x.count('-'))
url_df['num_slashes'] = url_df['url'].apply(lambda x: x.count('/'))
url_df['has_https'] = url_df['url'].apply(lambda x: int('https' in x))

url_features = url_df[['url_length', 'num_dots', 'num_hyphens', 'num_slashes', 'has_https']]
scaler = StandardScaler()
url_features_scaled = scaler.fit_transform(url_features)
url_features_df = pd.DataFrame(url_features_scaled, columns=url_features.columns)
url_features_df['phishing_label'] = url_df['phishing_label']