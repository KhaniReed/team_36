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

# ----- Model Training (Separate Models) -----
# Email Model
X_email = email_tfidf_df.drop(columns=['phishing_label'])
y_email = email_tfidf_df['phishing_label']
X_email_train, X_email_test, y_email_train, y_email_test = train_test_split(X_email, y_email, test_size=0.2, random_state=42, stratify=y_email)
email_model = RandomForestClassifier(n_estimators=100, random_state=42)
email_model.fit(X_email_train, y_email_train)

# URL Model
X_url = url_features_df.drop(columns=['phishing_label'])
y_url = url_features_df['phishing_label']
X_url_train, X_url_test, y_url_train, y_url_test = train_test_split(X_url, y_url, test_size=0.2, random_state=42, stratify=y_url)
url_model = RandomForestClassifier(n_estimators=100, random_state=42)
url_model.fit(X_url_train, y_url_train)

# ----- Function to Determine Type and Predict -----
def is_url(input_text):
    return bool(re.match(r'^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$', input_text))

def extract_url_features(url):
    parsed_url = urlparse(url)
    url_length = len(url)
    num_dots = url.count('.')
    num_hyphens = url.count('-')
    num_slashes = url.count('/')
    has_https = int(parsed_url.scheme == 'https')
    
    url_data = pd.DataFrame([[url_length, num_dots, num_hyphens, num_slashes, has_https]], 
                            columns=['url_length', 'num_dots', 'num_hyphens', 'num_slashes', 'has_https'])
    
    return scaler.transform(url_data)

def predict_phishing(input_text):
    if is_url(input_text):
        url_data = extract_url_features(input_text)
        url_features_filled = pd.DataFrame(url_data, columns=url_features_df.drop(columns=['phishing_label']).columns)
        prediction = url_model.predict(url_features_filled)
    else:
        email_features = tfidf_vectorizer.transform([input_text]).toarray()
        email_features_filled = pd.DataFrame(email_features, columns=[f'tfidf_{i}' for i in range(email_features.shape[1])])
        prediction = email_model.predict(email_features_filled)
    
    return "Phishing" if prediction[0] == 1 else "Legitimate"

# ----- User Input for Prediction -----
if __name__ == "__main__":
    user_input = input("Enter an email or URL to check: ")
    result = predict_phishing(user_input)
    print("Prediction:", result)