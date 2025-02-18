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

# Create a CountVectorizer to convert text data into numerical data
vectorizer = CountVectorizer()

x = vectorizer.fit_transform(corpus).toarray() # Convert the corpus into a matrix of token counts
y = df.label_num 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) # Split the data into training and testing sets

# Create a RandomForestClassifier and fit it to the training data
clf = RandomForestClassifier(n_jobs=-1)

clf.fit(x_train, y_train)

# Evaluate the model on the test data
print(clf.score(x_test, y_test))

# Predict the class of a new email from user input
email_to_classify = input() # Select an email to classify from input of user
print(email_to_classify) 
email_text = email_to_classify.lower().translate(str.maketrans('', '', string.punctuation)).split() # Remove punctuation and split into words
email_text = [stemmer.stem(word) for word in email_text if word not in stopwords_set] # Remove stopwords and stem the words
email_text = ' '.join(email_text) # Join the words back into a single string

email_corpus = [email_text] # Create a new corpus with the email text

x_email = vectorizer.transform(email_corpus) # Convert the email corpus into a matrix of token counts

print(clf.predict(x_email)) # Predict the class of the email
