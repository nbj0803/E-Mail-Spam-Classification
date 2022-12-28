import pandas as pd
import numpy as np
import pickle

df = pd.read_csv("spam.csv",encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.rename(columns = {"v1":"label", "v2":"message"}, inplace = True)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})


import nltk
import re
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
ps = PorterStemmer()

for sms_string in list(df.message):
  message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sms_string)
  message = message.lower()
  words = message.split()
  words = [word for word in words if word not in set(stopwords.words('english'))]
  words = [ps.stem(word) for word in words]
  message = ' '.join(words)
  corpus.append(message)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus).toarray()
y = pd.get_dummies(df['label'])
y = y.iloc[:, 1].values

pickle.dump(tfidf, open('tfidf.pkl', 'wb'))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from sklearn.svm import LinearSVC
classifier = LinearSVC()
classifier.fit(X_train, y_train)


pickle.dump(classifier, open('svm.pkl', 'wb'))