# Data handling
import pandas as pd
import numpy as np

# Modelling
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Agnostic
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt




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

model_df = battles.copy()

X = model_df.drop(labels="BattleResult", axis=1)
y = model_df.BattleResult

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

pick = np.random.randint(len(model_df), size=5000)
illust = model_df.iloc[pick]
sns.pairplot(illust[["BattleResult", "Level_1", "Level_2"]])
plt.show()

train_stats = X_train.describe().transpose()

def norm(x):
    return (x - train_stats["mean"])/train_stats["std"]

norm_X_train = norm(X_train)
norm_X_test = norm(X_test)

def build_model():
    model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(X_train.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model

model = build_model()
model.summary()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

epochs = 1000
early_history = model.fit(norm_X_train, y_train, 
                    epochs=epochs, validation_split = 0.2, verbose=0, 
                    callbacks=[early_stop, tfdocs.modeling.EpochDots()])

hist = pd.DataFrame(early_history.history)
hist["epoch"] = early_history.epoch

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

plotter.plot({'Early Stopping': early_history}, metric = "mse")
plt.ylabel('MSE [BattleResults]')
plt.show()


loss, mae, mse = model.evaluate(norm_X_test, y_test, verbose=2)

test_predictions = model.predict(norm_X_test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_test, test_predictions)
plt.xlabel('Actuals [BattleResult]')
plt.ylabel('Predictions [BattleResult]')
lims = [-2500,2500]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims, c="orange")
plt.show()

error = test_predictions - y_test
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()