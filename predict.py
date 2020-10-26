#!/usr/bin/env python3

import pandas as pd
import numpy as np
from naivebayes import NaiveBayes
import sklearn.model_selection

data = pd.read_csv("predict_player_value.csv")
data = data[["overall", "value_eur", "pace", "shooting", "passing", "dribbling", "defending", "physic"]]
#print(data.head())
#print(data.iloc[[1]])


predict = "value_eur"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.01)
clf = NaiveBayes()
print(len(x_train), len(y_train))
clf.fit(x_train, y_train)

#acc = clf.predict(x_test)
#print(x_test[0], acc[0])
#print(clf.predict([[60, 55, 65, 50, 70, 60, 60]]))
#for i in range(len(x_test)):
#    print(x_test[i], acc[i])
