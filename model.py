import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
import pickle
import os

data = pd.read_csv("dataset.csv")

X = data.drop("performance", axis=1)
y = data["performance"]

selector = SelectKBest(score_func=chi2, k=5)
X_new = selector.fit_transform(X, y)

model = RandomForestClassifier()
model.fit(X_new, y)

importance = model.feature_importances_

os.makedirs("saved_models", exist_ok=True)

pickle.dump(model, open("saved_models/model.pkl","wb"))
pickle.dump(selector, open("saved_models/selector.pkl","wb"))
pickle.dump(importance, open("saved_models/importance.pkl","wb"))

print("Model trained successfully!")