#suspicious link detector
#free open source library below
import re # regex to detect patterns in link
import pandas as pd # datasets library to use
import numpy as np # library to work with large numbers and datasets
from sklearn.model_selection import train_test_split #splits dataset to train and test
from sklearn.ensemble import RandomForestClassifier # implementing machine learning model to study patterns for future predictions
from sklearn.metrics import accuracy_score # a function to measure performance of AI model
import joblib

def extract_features(url): # checks url links
    return [
        len(url),
        url.count('http'),
        url.count('.'),
        url. count('_'),
        url.count('@'),
        url.startswith("https")
    ]

#load dataset
dataset = pd.read_csv("phising_dataset.csv")


# process of extracting features from dataset
x = np.array([extract_features(url) for url in dataset["url"]])
y = np.array(dataset["label"])

# training model
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

#save model
joblib.dump(model, "phising_detector.pkl")

#function to check a new URL

def check_url(url):
    model = joblib.load("phising_detector.pkl")
    features = np.array([extract_feature(url)])
    prediction = model.predict(features)
    return "Phising" if prediction[0] == 1 else "Safe"

#example usage
url_to_check = "http://example.com"
print("URL:", url_to_check, "is", check_url(url_to_check))
