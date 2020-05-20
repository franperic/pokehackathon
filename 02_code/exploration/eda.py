import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import plotly.graph_objects as go


# Data import
files = sorted(glob.glob("01_data/raw/*"))
# pokemon_index = pd.read_csv(files[0], sep="|")
# pokemon = pd.read_csv(files[1], sep="|")
battle = pd.read_csv(files[2], sep="|")
# submission = pd.read_csv(files[3], sep="|")
# weakness = pd.read_csv(files[4], sep="|")

# Battle
# Target variable
plt.boxplot(battle.BattleResult)
plt.show()

plt.hist(battle.BattleResult, bins=30)
plt.show()

# Create heatmap - pokemon vs pokemon
target = "BattleResult"
grouping_var = ["Name_1", "Name_2"]
mean_rslts = battle.groupby(grouping_var, as_index=False)[target].mean()

battle_pokemons = mean_rslts.Name_1.unique()
add = pd.DataFrame(data=dict(Name_1=battle_pokemons,
                             Name_2=battle_pokemons,
                             BattleResult=np.nan))
mean_rslts = pd.concat([mean_rslts, add], axis=0)
mean_rslts.sort_values(["Name_1", "Name_2"], inplace=True)

rslts = mean_rslts.BattleResult.values.copy()
rslts.resize([144, 144])

# Normalization
rslts_min, rslts_max = np.abs(np.nanmin(rslts)),np.abs(np.nanmax(rslts))
rslts_norm = (rslts + rslts_min) / (rslts_min + rslts_max)
rslts_norm = np.nan_to_num(rslts_norm, nan=0)

# Plot
fig = go.Figure(data=go.Heatmap(
        z=rslts_norm,
        x=mean_rslts.Name_1.unique(),
        y=mean_rslts.Name_2.unique(),
        colorscale='Viridis'))

fig.update_layout(
    title='Mean Battle Results',
    xaxis_nticks=50)

fig.show()