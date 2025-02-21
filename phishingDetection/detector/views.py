from django.shortcuts import render

# Create your views here.
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import re
import numpy as np
from urllib.parse import urlparse

from .forms import MessageForm

# Load datasets
email_df = pd.read_csv("spam_ham_dataset.csv")
url_df = pd.read_csv("dataset_phishing.csv")

# ----- Preprocessing Emails -----
email_df = email_df[['text', 'label_num']]
email_df.rename(columns={'label_num': 'phishing_label'}, inplace=True)

# Convert email text into TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,3))
email_tfidf = tfidf_vectorizer.fit_transform(email_df['text']).toarray()
email_tfidf_df = pd.DataFrame(email_tfidf, columns=[f'tfidf_{i}' for i in range(email_tfidf.shape[1])])
email_tfidf_df['phishing_label'] = email_df['phishing_label']

# ----- Preprocessing URLs -----
def is_suspicious_domain(url):
    suspicious_keywords = ['login', 'secure', 'verify', 'update', 'banking', 'account', 'webscr', 'confirm', 'billing', 'support']
    return int(any(keyword in url for keyword in suspicious_keywords))

def has_multiple_subdomains(url):
    return int(url.count('.') > 2)

def is_long_url(url):
    return int(len(url) > 75)

url_df['phishing_label'] = (url_df['status'] == 'phishing').astype(int)
url_df['url_length'] = url_df['url'].apply(len)
url_df['num_dots'] = url_df['url'].apply(lambda x: x.count('.'))
url_df['num_hyphens'] = url_df['url'].apply(lambda x: x.count('-'))
url_df['num_slashes'] = url_df['url'].apply(lambda x: x.count('/'))
url_df['has_https'] = url_df['url'].apply(lambda x: int('https' in x))
url_df['num_digits'] = url_df['url'].apply(lambda x: sum(c.isdigit() for c in x))
url_df['num_special_chars'] = url_df['url'].apply(lambda x: sum(1 for c in x if c in "!@#$%^&*()"))
url_df['is_shortened'] = url_df['url'].apply(lambda x: int(any(short in x for short in ['bit.ly', 'tinyurl', 'goo.gl'])))
url_df['num_subdomains'] = url_df['url'].apply(lambda x: x.count('.') - 1)
url_df['is_ip'] = url_df['url'].apply(lambda x: int(bool(re.match(r'^(\d{1,3}\.){3}\d{1,3}$', urlparse(x).netloc))))
url_df['suspicious_domain'] = url_df['url'].apply(is_suspicious_domain)
url_df['multiple_subdomains'] = url_df['url'].apply(has_multiple_subdomains)
url_df['long_url'] = url_df['url'].apply(is_long_url)

url_features = url_df[['url_length', 'num_dots', 'num_hyphens', 'num_slashes', 'has_https', 'num_digits', 'num_special_chars', 'is_shortened', 'num_subdomains', 'is_ip', 'suspicious_domain', 'multiple_subdomains', 'long_url']]
scaler = StandardScaler()
url_features_scaled = scaler.fit_transform(url_features)
url_features_df = pd.DataFrame(url_features_scaled, columns=url_features.columns)
url_features_df['phishing_label'] = url_df['phishing_label']

# ----- Handle Imbalanced Data -----
smote = SMOTE(random_state=42)
X_email, y_email = email_tfidf_df.drop(columns=['phishing_label']), email_tfidf_df['phishing_label']
X_email, y_email = smote.fit_resample(X_email, y_email)
X_url, y_url = url_features_df.drop(columns=['phishing_label']), url_features_df['phishing_label']
X_url, y_url = smote.fit_resample(X_url, y_url)

# ----- Model Training (Using XGBClassifier for URLs) -----
X_email_train, X_email_test, y_email_train, y_email_test = train_test_split(X_email, y_email, test_size=0.2, random_state=42, stratify=y_email)
email_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=700, max_depth=12, learning_rate=0.08)
email_model.fit(X_email_train, y_email_train)

X_url_train, X_url_test, y_url_train, y_url_test = train_test_split(X_url, y_url, test_size=0.2, random_state=42, stratify=y_url)
url_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=700, max_depth=18, learning_rate=0.08)
url_model.fit(X_url_train, y_url_train)

# ----- Function to Determine Type and Predict -----
def is_url(input_text):
    return bool(re.match(r'^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$', input_text))

def extract_url_features(url):
    parsed_url = urlparse(url)
    url_data = pd.DataFrame([[
        len(url), url.count('.'), url.count('-'), url.count('/'), int(parsed_url.scheme == 'https'),
        sum(c.isdigit() for c in url), sum(1 for c in url if c in "!@#$%^&*()"),
        int(any(short in url for short in ['bit.ly', 'tinyurl', 'goo.gl'])),
        url.count('.') - 1,
        int(bool(re.match(r'^(\d{1,3}\.){3}\d{1,3}$', parsed_url.netloc))),
        is_suspicious_domain(url), has_multiple_subdomains(url), is_long_url(url)
    ]], columns=url_features.columns)
    return scaler.transform(url_data)

def predict_phishing(input_text):
    if is_url(input_text):
        url_data = extract_url_features(input_text)
        prediction = url_model.predict(url_data)
    else:
        email_features = tfidf_vectorizer.transform([input_text]).toarray()
        prediction = email_model.predict(email_features)
    
    return "Phishing" if prediction[0] == 1 else "Legitimate"

# ----- Validate Model Performance -----
print("Email Model Accuracy:", accuracy_score(y_email_test, email_model.predict(X_email_test)))
print("URL Model Accuracy:", accuracy_score(y_url_test, url_model.predict(X_url_test)))

# ----- User Input for Prediction -----
'''if __name__ == "__main__":
    user_input = input("Enter an email or URL to check: ")
    result = predict_phishing(user_input)
    print("Prediction:", result)'''

def Home(request):
    result = None
    if request.method == 'POST':
        form = MessageForm(request.POST)
        if form.is_valid():
            message = form.cleaned_data['message']
            result = predict_phishing(message)
    else:
        form = MessageForm()

    return render(request, 'home.html', {'form': form, 'result': result})