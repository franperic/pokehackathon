from kedro.pipeline import node, Pipeline
from pokehackathon_kedro.pipelines.data_engineering.nodes import (
    preprocess_available_pokemons,
    preprocess_battle_results,
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess_available_pokemons,
                inputs="available_pokemons",
                outputs="preprocessed_available_pokemons",
                name="preprocessing_available_pokemons",
            ),
            node(
                func=preprocess_battle_results,
                inputs="battle_results",
                outputs="preprocessed_battle_results",
                name="preprocessing_battle_results",
            ),
        ]
    )
