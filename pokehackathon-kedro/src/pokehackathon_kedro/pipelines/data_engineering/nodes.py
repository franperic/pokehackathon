import pandas as pd
from sklearn.feature_extraction import FeatureHasher


def preprocess_available_pokemons(available_pokemons: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data for available_pokemons.
       Set Legendary to integer.

        Args:
            available_pokemons: Source data.
        Returns:
            Preprocessed data.

    """
    available_pokemons = available_pokemons.astype({"Legendary_1": int})


    return available_pokemons


def preprocess_battle_results(battle_results: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data for battle_results.
       Set Legendary to integer.

        Args:
            battle_results: Source data.
        Returns:
            Preprocessed data.

    """
    battle_results = battle_results.astype({"Legendary_1": int, "Legendary_2": int})
    

    return battle_results
    
    
def create_battle_results_with_types(
    all_pokemons: pd.DataFrame, battle_results: pd.DataFrame) -> pd.DataFrame:
    """Integrates Type_1 and Type_2 from 'all_pokemons' into battle_results.
       Gets projected onto Type_1_1 and Type_2_1 for Pokemon1 and Type_1_2 and Type_2_2 for Pokemon2.
       'NaN' is replaced with '0'
       Also integrates IDs for Pokemons => ID_1, ID_2
       P1, P2 used for 'Pokemon 1' and 'Pokemon 2'

        Args:
            all_pokemons: Preprocessed data for All Pokemons.
            battle_results: Preprocessed data for Battle Results.
        Returns:
            battle_results_with_types

    """
    #Avoid Nidorana a second time (IDs 29 and 32) in all_pokemons
    all_pokemons = all_pokemons[all_pokemons['ID'] != 32] 
    
    ## Pokemon1
    # Merge battle_results with all_pokemons based on the Name of P1
    battle_results_with_types_P1 = battle_results.merge(all_pokemons, left_on="Name_1", right_on="Name")
    # Rename Type_1 and Type_2 to Type_1_1 and Type_2_1
    battle_results_with_types_P1.rename(columns={'Type_1': 'Type_1_1', 'Type_2': 'Type_2_1', 'ID':'ID_1'}, inplace=True)
    
    #Drop 'Name' as it is already saved in 'Name_1'
    battle_results_with_types_P1.drop(["Name"], axis=1, inplace=True)
    
    
    ## Adding Pokemon2
    # Merge battle_results_with_types_P1 with all_pokemons based on the Name of P2
    battle_results_with_types = battle_results_with_types_P1.merge(all_pokemons, left_on="Name_2", right_on="Name")
    
    # Rename Type_1 and Type_2 to Type_1_1 and Type_2_1
    battle_results_with_types.rename(columns={'Type_1': 'Type_1_2', 'Type_2': 'Type_2_2', 'ID':'ID_2'}, inplace=True)
    
    #Drop 'Name' as it is already saved in 'Name_2'
    battle_results_with_types.drop(["Name"], axis=1, inplace=True)
    
    battle_results_with_types = battle_results_with_types.fillna('')
    return battle_results_with_types


def create_battle_results_with_hashed_types(
    battle_results: pd.DataFrame) -> pd.DataFrame:
    """Feature hashes 'Type_1' and 'Type_2' and 'WeatherAndTime' from 'battle_results_with_types'.
       Example for labeling: Type_1_1 => Type_1_1_a, Type_1_1_b, Type_1_1_c, Type_1_1_d, Type_1_1_e
       number of features: 5 (labeled with a,b,c,d,e)
       For transparency, original Types/WeatherandTime are not dropped in DataFrame

        Args:
            battle_results: Preprocessed data for Battle Results with types
        Returns:
            battle_results_with_hashed_types_weather
    """
    
    ##Feature Hasher for Type
    h_type = FeatureHasher(n_features=5, input_type='string')
    
    #Transform Types of Pokemon1: Type1_1, Type2_1 and of Pokemon2: Type1_2, Type2_2
    dtype1_1 = h_type.fit_transform(battle_results["Type_1_1"])
    dtype2_1 = h_type.fit_transform(battle_results["Type_2_1"])
    dtype1_2 = h_type.fit_transform(battle_results["Type_1_2"])
    dtype2_2 = h_type.fit_transform(battle_results["Type_2_2"])
    
    #Transform into Pandas DataFrame and rename labels
    dtype1_1 = pd.DataFrame(data=dtype1_1.toarray())
    dtype1_1.rename(columns={0: 'Type_1_1_a', 1: 'Type_1_1_b', 2: 'Type_1_1_c', 3: 'Type_1_1_d', 4: 'Type_1_1_e'}, inplace=True)
    dtype2_1 = pd.DataFrame(data=dtype2_1.toarray())
    dtype2_1.rename(columns={0: 'Type_2_1_a', 1: 'Type_2_1_b', 2: 'Type_2_1_c', 3: 'Type_2_1_d', 4: 'Type_2_1_e'}, inplace=True)
    dtype1_2 = pd.DataFrame(data=dtype1_2.toarray())
    dtype1_2.rename(columns={0: 'Type_1_2_a', 1: 'Type_1_2_b', 2: 'Type_1_2_c', 3: 'Type_1_2_d', 4: 'Type_1_2_e'}, inplace=True)
    dtype2_2 = pd.DataFrame(data=dtype2_2.toarray())
    dtype2_2.rename(columns={0: 'Type_2_2_a', 1: 'Type_2_2_b', 2: 'Type_2_2_c', 3: 'Type_2_2_d', 4: 'Type_2_2_e'}, inplace=True)

    
    ##Feature Hasher for WeatherandTime
    h_wt = FeatureHasher(n_features=5, input_type='string')
    dwt = h_wt.fit_transform(battle_results["WeatherAndTime"])
    dwt = pd.DataFrame(data=dwt.toarray())
    dwt.rename(columns={0: 'WeatherAndTime_a', 1: 'WeatherAndTime_b', 2: 'WeatherAndTime_c', 3: 'WeatherAndTime_d', 4: 'WeatherAndTime_e'}, inplace=True)
    
    
    #(No) Drop of Type_1_1, ... and WeatherandTime column.
    #battle_results = battle_results.drop(columns = ["WeatherAndTime", "Type_1_1", "Type_2_1", "Type_1_2", "Type_2_2"])
    
    #Concatenate battle_results with hashed features
    battle_results_hashed = pd.concat([battle_results, dtype1_1, dtype1_2, dtype2_1, dtype2_2, dwt], axis=1)
    
    return battle_results_hashed


def create_battle_results_AD(
    battle_results: pd.DataFrame) -> pd.DataFrame:
    """Creates additional features 'Attack1/Defense2', 'Attack2/Defense1'. Same for Special Attack/Special Defense

        Args:
            battle_results: Preprocessed data for Battle Results
        Returns:
            battle_results: Battle Results with A/D
    """
    
    battle_results['A1/D2'] = battle_results['Attack_1']/battle_results['Defense_2']
    battle_results['A2/D1'] = battle_results['Attack_2']/battle_results['Defense_1']
    battle_results['Sp_A1/Sp_D2'] = battle_results['Sp_Atk_1']/battle_results['Sp_Def_2']
    battle_results['Sp_A2/Sp_D1'] = battle_results['Sp_Atk_2']/battle_results['Sp_Def_1']

    return battle_results


def create_battle_results_with_weakness(
    battle_results: pd.DataFrame, weakness_pokemons: pd.DataFrame) -> pd.DataFrame:
    """Adds addtitional features for 'Effectiveness of Attacks' ('Eff_11', 'Eff_12', 'Eff_21', Eff_22'). 
       These are based on weakness ratio between the Pokemon's types. The numbers refer only to the types
       as the Pokemon can be omitted (ratio is 1/Eff) for the other => there is no new information.
       
       Notation: 11 => Type_1_1 : Type_1_2 (this means Type1 of Pokemon1 divided by Type1 of Pokemon2)
       
       
       When there is no Type 2 => Set Eff_12 (or Eff_21) to 1 as it should be neutral.
       
       Final Effectiveness is Eff = Eff_11 * Eff_12 * Eff_21 * Eff_22
       
       
        Args:
            battle_results: Preprocessed data for Battle Results
            weakness_pokemons: Preprocessed data for Weakness Pokemons.
        Returns:
            battle_results: Battle Results with Effectiveness

    """
    #Unpivot pokemon_weakness to Table with Type1, Type2 and value as columns
    weakness_long = weakness_pokemons.melt(id_vars="Types")
    weakness_long.columns = ["Type_P1", "Type_P2", "value"]

    #Eff_11
    battle_results_with_weakness = battle_results.merge(weakness_long, left_on=["Type_1_1","Type_1_2"], right_on=["Type_P1","Type_P2"])
    battle_results_with_weakness.rename(columns={'value': 'value_1_1ab'}, inplace=True)
    battle_results_with_weakness.drop(['Type_P1', 'Type_P2'], axis=1, inplace=True)

    battle_results_with_weakness = battle_results_with_weakness.merge(weakness_long, left_on=["Type_1_2","Type_1_1"], right_on=["Type_P1","Type_P2"])
    battle_results_with_weakness.rename(columns={'value': 'value_1_1ba'}, inplace=True)
    battle_results_with_weakness.drop(['Type_P1', 'Type_P2'], axis=1, inplace=True)

    battle_results_with_weakness = battle_results_with_weakness.fillna(1)

    battle_results_with_weakness['Eff_11'] = battle_results_with_weakness['value_1_1ab']/battle_results_with_weakness['value_1_1ba']

    
    #Eff_12
    battle_results_with_weakness_2 = battle_results_with_weakness.merge(weakness_long, left_on=["Type_1_1","Type_2_2"], right_on=["Type_P1","Type_P2"], how='left')
    battle_results_with_weakness_2.rename(columns={'value': 'value_1_2ab'}, inplace=True)
    battle_results_with_weakness_2.drop(['Type_P1', 'Type_P2'], axis=1, inplace=True)

    battle_results_with_weakness_2 = battle_results_with_weakness_2.merge(weakness_long, left_on=["Type_2_2","Type_1_1"], right_on=["Type_P1","Type_P2"], how='left')
    battle_results_with_weakness_2.rename(columns={'value': 'value_2_1ba'}, inplace=True)
    battle_results_with_weakness_2.drop(['Type_P1', 'Type_P2'], axis=1, inplace=True)

    battle_results_with_weakness_2 = battle_results_with_weakness_2.fillna(1)

    battle_results_with_weakness_2['Eff_12'] = battle_results_with_weakness_2['value_1_2ab']/battle_results_with_weakness_2['value_2_1ba']

    
    #Eff_21
    battle_results_with_weakness_3 = battle_results_with_weakness_2.merge(weakness_long, left_on=["Type_2_1","Type_1_2"], right_on=["Type_P1","Type_P2"], how='left')
    battle_results_with_weakness_3.rename(columns={'value': 'value_2_1ab'}, inplace=True)
    battle_results_with_weakness_3.drop(['Type_P1', 'Type_P2'], axis=1, inplace=True)

    battle_results_with_weakness_3 = battle_results_with_weakness_3.merge(weakness_long, left_on=["Type_1_2","Type_2_1"], right_on=["Type_P1","Type_P2"], how='left')
    battle_results_with_weakness_3.rename(columns={'value': 'value_1_2ba'}, inplace=True)
    battle_results_with_weakness_3.drop(['Type_P1', 'Type_P2'], axis=1, inplace=True)

    battle_results_with_weakness_3 = battle_results_with_weakness_3.fillna(1)

    battle_results_with_weakness_3['Eff_21'] = battle_results_with_weakness_3['value_2_1ab']/battle_results_with_weakness_3['value_1_2ba']

    
    #Eff_22
    battle_results_with_weakness_4 = battle_results_with_weakness_3.merge(weakness_long, left_on=["Type_2_1","Type_2_2"], right_on=["Type_P1","Type_P2"], how='left')
    battle_results_with_weakness_4.rename(columns={'value': 'value_2_2ab'}, inplace=True)
    battle_results_with_weakness_4.drop(['Type_P1', 'Type_P2'], axis=1, inplace=True)

    battle_results_with_weakness_4 = battle_results_with_weakness_4.merge(weakness_long, left_on=["Type_2_2","Type_2_1"], right_on=["Type_P1","Type_P2"], how='left')
    battle_results_with_weakness_4.rename(columns={'value': 'value_2_2ba'}, inplace=True)
    battle_results_with_weakness_4.drop(['Type_P1', 'Type_P2'], axis=1, inplace=True)

    battle_results_with_weakness_4 = battle_results_with_weakness_4.fillna(1)

    battle_results_with_weakness_4['Eff_22'] = battle_results_with_weakness_4['value_2_2ab']/battle_results_with_weakness_4['value_2_2ba']
    
    
    #Final Effectiveness 'Eff'
    
    battle_results_with_weakness_4['Eff'] = battle_results_with_weakness_4['Eff_11'] * battle_results_with_weakness_4['Eff_12'] * battle_results_with_weakness_4['Eff_21'] * battle_results_with_weakness_4['Eff_22']
    
    
    return battle_results_with_weakness_4 #master_table