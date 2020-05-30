from kedro.pipeline import node, Pipeline
from pokehackathon_kedro.pipelines.data_engineering.nodes import (
    preprocess_available_pokemons,
    preprocess_battle_results,
    create_battle_results_with_types,
    create_battle_results_with_hashed_types,
    create_battle_results_AD,
    create_battle_results_with_weakness
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
            node(
                func=create_battle_results_with_types,
                inputs=["all_pokemons","preprocessed_battle_results"],
                outputs="battle_results_with_types",
                name="creating_battle_results_with_types",
            ),
            node(
                func=create_battle_results_with_hashed_types,
                inputs="battle_results_with_types",
                outputs="battle_results_with_hashed_types",
                name="creating_battle_results_with_hashed_types",
            ),
            node(
                func=create_battle_results_AD,
                inputs="battle_results_with_hashed_types",
                outputs="battle_results_AD",
                name="creating_battle_results_AD",
            ),
            node(
                func=create_battle_results_with_weakness,
                inputs=["battle_results_AD","weakness_pokemons"],
                outputs="battle_results_with_weakness",
                name="creating_battle_results_with_weakness",
            ),
            
        ]
    )

