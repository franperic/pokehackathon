import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# Data overview
files = sorted(glob.glob("01_data/raw/*"))
pokemon_index = pd.read_csv(files[0], sep="|")
pokemon = pd.read_csv(files[1], sep="|")
battle = pd.read_csv(files[2], sep="|")
submission = pd.read_csv(files[3], sep="|")
weakness = pd.read_csv(files[4], sep="|")

# Battle
plt.boxplot(battle.BattleResult)
plt.show()

# Abra subset
abra = battle.loc[battle["Name_1"] == "Abra"]
plt.scatter(abra.Level_1, abra.BattleResult)
plt.show()