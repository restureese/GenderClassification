from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import os
#save model
from sklearn.externals import joblib

import pandas as pd

# logistic
clf = SGDClassifier(loss="log")

#extraction feature string
tf = TfidfVectorizer()

X, y = [], []      

dataset = pd.read_csv('static/dataset.csv')

for (nama,gender) in zip(dataset['name'],dataset['gender']):
	X.append(str(nama).lower())
	y.append(str(gender))

# ganti label jika laki-laki = 1
y = np.array(y)
y = np.where(y=='m', 1, 0)

X_train = tf.fit_transform(X)

clf.fit(X_train, y)   

#save model
joblib.dump(clf, 'model.pkl')
joblib.dump(tf, 'vektor.pkl')