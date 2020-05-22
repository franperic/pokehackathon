import pandas as pd
import numpy as np


def update_available(data, recent_pick):
    """
    Reduce available pokemon set by most recent pick.
    
    Parameter
    ---------
    
    data: DataFrame with available Pokemons
    
    recent_pick: Picked Pokemon name
    
    """
    df = data.copy()
    
    return df.loc[df["Name_1"] != recent_pick]