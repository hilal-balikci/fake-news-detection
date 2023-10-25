"""
id: unique id for a news article
title: the title of a news article

author: author of the news article

text: the text of the article; could be incomplete

label: a label that marks whether the news article is real or fake:

0: True News
1: Fake news
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix
import re
import nltk

nltk.download('stopwords')

# printing the stopwords in English
print(stopwords.words('english'))

# loading the dataset to a pandas DataFrame
news_data = pd.read_csv('train.csv')
print(news_data.head())

print(news_data.shape)

print(news_data.info())

# counting the number of missing values in the dataset
news_data.isnull().sum()

news_data = news_data.fillna('') #replace the the null values 
news_data['content'] = news_data['author']+' '+news_data['title'] #merging the author name and news title

print(news_data['content'])

# separating the data & label
X = news_data.drop(labels= 'label' , axis=1)
Y = news_data['label']

print(X)
print(Y)

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


news_data['content'] = news_data['content'].apply(stemming)

print(news_data['content'])

#separating the data and label
X = news_data['content'].values
Y = news_data['label'].values

print(X)
print(Y)

print(Y.shape)


vectorizer = TfidfVectorizer() #converting the textual data to numerical data
vectorizer.fit(X)

X = vectorizer.transform(X)

print(X)

X_train, X_test ,Y_train , Y_test = train_test_split(X , Y , test_size=0.2 , random_state=2)

#training the model 
model = LogisticRegression()
model.fit(X_train, Y_train)

# accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the testing data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the testing data : ', testing_data_accuracy)

y_pred = model.predict(X_test)
confusion_matrix = confusion_matrix(Y_test, y_pred.round())
sns.heatmap(confusion_matrix, annot=True, fmt="d", cbar = False)
plt.title("Confusion Matrix")
plt.show()

X_new = X_test[1939]

prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('The news is True')
else:
  print('The news is Fake')
  
print(Y_test[1939])














