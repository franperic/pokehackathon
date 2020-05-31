import pandas as pd
import numpy as np
import glob

# Data import
files = sorted(glob.glob("01_data/raw/*"))
pokemon = pd.read_csv(files[1], sep="|")
submission = pd.read_csv(files[3], sep="|")


available = pokemon.loc[pokemon["Price_1"] < 3500]

primary_key = np.arange(len(available))
primary_key = primary_key.repeat(6)

data_list = []
for i in np.arange(len(available)):
    for row in np.arange(len(submission)):
        step = submission.iloc[row].to_dict()
        data_list.append(step)

possible_battles = pd.DataFrame(data_list)
possible_battles.drop("SelectedPokemonID", axis=1, inplace=True)
possible_battles["primary_key"] = primary_key
available["primary_key"] = np.arange(len(available))

possible_battles = available.merge(possible_battles, on="primary_key", how="left")