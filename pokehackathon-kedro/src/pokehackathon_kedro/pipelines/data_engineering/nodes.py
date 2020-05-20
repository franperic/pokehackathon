import pandas as pd


def _is_true(x):
    return x == "True"


def preprocess_available_pokemons(available_pokemons: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data for available_pokemons.

        Args:
            available_pokemons: Source data.
        Returns:
            Preprocessed data.

    """

    available_pokemons["Legendary_1"] = available_pokemons["Legendary_1"].apply(_is_true)


    return available_pokemons


def preprocess_battle_results(battle_results: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data for battle_results.

        Args:
            battle_results: Source data.
        Returns:
            Preprocessed data.

    """
    battle_results["Legendary_1"] = battle_results["Legendary_1"].apply(_is_true)

    battle_results["Legendary_2"] = battle_results["Legendary_2"].apply(
        _is_true
    )

    return battle_results
    

