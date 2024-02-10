import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")

# df["language"], _ = pd.factorize(df["language"])

x = np.array(df["Text"])
y = np.array(df["language"])

cv = CountVectorizer()
X = cv.fit_transform(x)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=43)

model = MultinomialNB()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print(f"Accuracy: {100 * accuracy_score(Y_pred, Y_test):.2f}")

data = cv.transform(["你好，我叫特隆"]).toarray()
output = model.predict(data)
print(output)