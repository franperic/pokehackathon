import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Data import
battles = pd.read_csv("01_data/raw/Battle_Results.csv", sep="|")

# Prep
cat_vars = battles.select_dtypes(object).columns.values.tolist()
battles = battles.astype({"Legendary_1": int, "Legendary_2": int})

h1 = FeatureHasher(n_features=5, input_type='string')
h2 = FeatureHasher(n_features=5, input_type='string')
d1 = h1.fit_transform(battles["Name_1"])
d2 = h2.fit_transform(battles["Name_2"])

d1 = pd.DataFrame(data=d1.toarray())
d1.columns = ["Name_1_" + str(x) for x in range(5)]
d2 = pd.DataFrame(data=d2.toarray())
d2.columns = ["Name_2_" + str(x) for x in range(5)]

battles = battles.drop(columns=cat_vars[0:2])
battles = pd.concat([battles, d1, d2], axis=1)
battles = pd.get_dummies(battles)

X = battles.drop(labels="BattleResult", axis=1)
y = battles.BattleResult

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Linear regression
# statsmodels for model agnostics
ols = sm.OLS(y_train, X_train)
model = ols.fit()
model.summary()

# CV
lin_mod = LinearRegression()
scores = cross_val_score(lin_mod, X_train, y_train, 
                         cv=5, scoring="neg_mean_squared_error")

lin_mod.fit(X_train, y_train)
pred = lin_mod.predict(X_test)

np.mean((y_test - pred)**2)
y_test.agg(["min", "max"])

# Visual inspection
x = np.linspace(-2000, 2000, 100)
plt.scatter(y_test, pred, s=1, alpha=0.7)
plt.plot(x, x, c="red")
plt.xlabel("Actual")
plt.ylabel("Prediction")
plt.show()

