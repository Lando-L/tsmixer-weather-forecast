"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train,
                inputs=[
                    "train",
                    "val",
                    "params:data_config",
                    "params:model_config",
                    "params:train_config",
                ],
                outputs=None,
                name="train",
            )
        ]
    )
