import pandas as pd
import numpy as np
data = pd.read_csv("Iris.csv")
data.drop("Id",axis=1,inplace=True)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

X = data.iloc[:, :-1].values
y = data.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 50)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
import joblib

joblib.dump(classifier, 'trained_model.pkl')