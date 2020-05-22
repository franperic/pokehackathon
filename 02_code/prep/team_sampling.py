import pandas as pd
import numpy as np
import glob
import re
import sys
import matplotlib.pyplot as plt

if "./02_code/modularization" not in sys.path:
    sys.path.append("./02_code/modularization")
    
import pokesampling

files = glob.glob("01_data/raw/*")
available = pd.read_csv(files[3], sep="|")

relevant_columns = ["Name_1", "Level_1", "Price_1"]
available = available.loc[available["Price_1"] < 3500, relevant_columns]



sim = 10000
overview_teams = []
budget = []
for i in range(sim):
    # Progress
    if i % 500 == 0:
        print(i)
    
    available_set = available.copy()
    team_size = 6
    team = []
    
    for team_position in range(team_size):
        
        # Pokemon Draft
        pick = np.random.randint(len(available_set), size=1)
        pokemon_pick = available_set.iloc[pick].reset_index(drop=True).to_dict("r")
        available_set = pokesampling.update_available(
            available_set, pokemon_pick[0]["Name_1"]
        )

        team.append(pokemon_pick[0])
    
    team = pd.DataFrame(team)
    team["sim"] = i
    budget.append(team.Price_1.sum())
    for k in range(len(team)):
        row = team.loc[k].to_dict()
        overview_teams.append(row)

budget = np.array(budget)

plt.hist(budget)
plt.show()

plt.boxplot(budget)
plt.show()