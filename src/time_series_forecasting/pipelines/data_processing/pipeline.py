"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import feature_engineer_weather, preprocess_weather, split_and_normalize


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_weather,
                inputs="weather",
                outputs="primary_weather",
                name="preprocess_weather",
            ),
            node(
                func=feature_engineer_weather,
                inputs="primary_weather",
                outputs="feature_weather",
                name="feature_engineer_weather",
            ),
            node(
                func=split_and_normalize,
                inputs=["feature_weather", "params:data_splits"],
                outputs=["train", "val", "test"],
                name="split_and_normalize",
            ),
        ]
    )
