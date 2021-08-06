# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 10:38:18 2021

@author: HP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(r'news.csv')


# test train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df.label, test_size=0.2,random_state = 7)

from sklearn.feature_extraction.text import TfidfVectorizer
tv=TfidfVectorizer(stop_words='english',max_df=0.7)
tv_train= tv.fit_transform(X_train).toarray()
tv_test= tv.transform(X_test).toarray()

from sklearn.svm import SVC
classifier1=SVC(kernel='rbf',random_state=0)
classifier1.fit(tv_train,y_train)

y_pred1 = classifier1.predict(tv_test)

from sklearn.metrics import confusion_matrix
cm1= confusion_matrix(y_test,y_pred1)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred1)