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




battles_raw = pd.read_csv("01_data/processed/battle_results_diff_ratio.csv", sep=",")
battles = battles_raw.copy()

# Prep
columns = battles.columns.tolist()
battles = battles.astype({"Legendary_1": int, "Legendary_2": int})



battles = pd.get_dummies(battles, columns=["WeatherAndTime"])

neglect = ["WeatherAndTime_a", "WeatherAndTime_b", 
           "WeatherAndTime_c", "WeatherAndTime_d", "WeatherAndTime_e",
           "Name_1", "Name_2", "Type_1_1", "Type_1_2", "Type_2_1", "Type_2_2",
           "ID_1", "ID_2"]

battles = battles.drop(columns=neglect)


model_df = battles.copy()

X = model_df.drop(labels="BattleResult", axis=1)
y = model_df.BattleResult

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

pick = np.random.randint(len(model_df), size=5000)
illust = model_df.iloc[pick]
sns.pairplot(illust[["BattleResult", "Level_1", "Level_2"]])
plt.show()

train_stats = X_train.describe().transpose()
target_stats = y_train.describe()

def norm(x):
    return (x - train_stats["mean"])/train_stats["std"]

def target_norm(x):
    return (x - target_stats["mean"])/target_stats["std"]

norm_X_train = norm(X_train)
norm_X_test = norm(X_test)
norm_y_train = target_norm(y_train)
norm_y_test = target_norm(y_test)

def build_model():
    model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(X_train.keys())]),
    layers.Dense(64, activation='relu'),
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
early_history = model.fit(norm_X_train, norm_y_train, 
                    epochs=epochs, validation_split = 0.2, verbose=0, 
                    callbacks=[early_stop, tfdocs.modeling.EpochDots()])

hist = pd.DataFrame(early_history.history)
hist["epoch"] = early_history.epoch

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

plotter.plot({'Early Stopping': early_history}, metric = "mse")
plt.ylabel('MSE [BattleResults]')
plt.show()


loss, mae, mse = model.evaluate(norm_X_test, norm_y_test, verbose=2)

test_predictions = model.predict(norm_X_test).flatten()
test_predictions = test_predictions * target_stats["std"] - target_stats["mean"]

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

inspect = np.where(error == error.max())

test_predictions[inspect]
y_test.loc[inspect]
norm_X_test.loc[inspect]

def denorm(x):
    return x * train_stats["std"] + train_stats["mean"]

denorm(norm_X_test.loc[inspect]).loc[:, ["Level_1", "Level_2", "Attack_1", "Attack_2"]]


rslt = y_test.loc[inspect].values[0]
battles.loc[(battles["BattleResult"] == rslt) &
             (battles["Level_1"] == 63) &
             (battles["Level_2"] == 29) &
             (battles["Attack_1"] == 379) &
             (battles["Attack_2"] == 176)].transpose()


model.save("03_model/20200530/")
check = keras.models.load_model("03_model/20200530")
check.predict(norm_X_test)

