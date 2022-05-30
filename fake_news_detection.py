import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import cohen_kappa_score

import re
import nltk

nltk.download('english')

from ntlk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WorldCloud

def getPie(true: list, false: list):
    d = [len(true), len(false)]
    t = ['Real', 'Fake']

    plt.pie(d, labels = t, autopct = '%1.2f%%')
    plt.show()

def getWordCloud(corpus: list):
    text = ' '
    
    for word in corpus:
        text += ' '.join(word.spit(' '))

    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def getFeature(input: list):

    corpus = []

    for i in range(0, 40000):
        arr = re.sub('[^a-zA-Z]', ' ', input[i])
        arr = arr.lower()
        arr = arr.split()

        ps = PorterStemmer()

        arr = [ps.stem(word) for word in arr if not word in set(stopwords.words('english'))] 

        arr = ' '.join(arr)

        corpus.append(arr)
    
    return corpus

def accuracy(title, x):
    print('-' * 50, '\n' + title, 'Accuracy: %.2f%%' % x)

def holdout(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    result = model.score(X_train, Y_train)
    
    accuracy('Holdout Validation Approach', (result * 100))
    
def kfold(X, Y):
    kfold = KFold(n_splits = 5)
    model = LogisticRegression()
    results = cross_val_score(model, X, Y, cv = kfold)

    accuracy('K-fold Cross-Validation', (results.mean() * 100))

def sufflesplit(X, Y):
    kfold2 = ShuffleSplit(n_splits = 10, test_size = 0.20)
    model = LogisticRegression()
    results = cross_val_score(model, X, Y, cv = kfold2)

    accuracy('Repeated Random Test-Train Splits', (results.mean() * 100))

def main():
    false_dataset = pd.read_csv('//content//fake-news-detection//datasets//fake.csv')
    true_dataset = pd.read_csv('//content//fake-news-detection//datasets//true.csv')

    false_dataset['fake'] = 0
    true_dataset['fake'] = 1

    table = pd.DataFrame()
    table = true_dataset.append(false_dataset)
    table = table.drop(columns=['subject','date'])
    
    print('-' * 50)
    print("Table shape: " + table.shape)

    input_arr = np.array(table['title'])

    corpus = getFeature(input_arr);
    countv = CountVectorizer(max_features = 5000)

    X = countv.fit_transform(corpus).toarray()
    Y = table.iloc[0:40000, 2].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    print('-' * 50)
    print('Classification report\n', classification_report(Y_test, Y_pred))
    print('Confusion Matrix\n', confusion_matrix(Y_test, Y_pred))
    print('Precision: ', precision_score(Y_test, Y_pred))
    print('Accuracy: ', accuracy_score(Y_test, Y_pred))
    print('Cohen Kappa: ', cohen_kappa_score(Y_test, Y_pred))

    holdout(X, Y)
    kfold(X, Y)
    sufflesplit(X, Y)

if __name__ == __main__:
    main()