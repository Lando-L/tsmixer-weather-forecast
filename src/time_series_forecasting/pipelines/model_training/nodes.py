"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.18.13
"""

import logging
from typing import Any, Dict, List, Mapping, Tuple

import jax
import optax
import pandas as pd
import tensorflow as tf
from flax.training import orbax_utils, train_state
from jax import numpy as jnp
from jax import random, tree_util
from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointer,
)

from time_series_forecasting.data.window import Window, make_dataset
from time_series_forecasting.models.tsmixer import TSMixer

Scalars = Mapping[str, jax.Array]


class TrainState(train_state.TrainState):
    batch_stats: Any


@jax.jit
def train_step(
    state: TrainState, window: Window, key: random.PRNGKey
) -> Tuple[TrainState, Scalars]:
    def loss_fn(params):
        logits, updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            window.inputs,
            train=True,
            mutable=["batch_stats"],
            rngs={"dropout": key},
        )
        loss = jnp.mean(jnp.square(jnp.subtract(window.labels, logits)))
        return loss, (logits, updates)

    (loss, (logits, updates)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params
    )
    return (
        state.apply_gradients(grads=grads).replace(batch_stats=updates["batch_stats"]),
        {"loss": loss, "mae": jnp.mean(jnp.abs(jnp.subtract(window.labels, logits)))},
    )


@jax.jit
def eval_step(state: TrainState, window: Window) -> Tuple[TrainState, Scalars]:
    logits = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        window.inputs,
        train=False,
    )
    loss = jnp.mean(jnp.square(jnp.subtract(window.labels, logits)))
    return (
        state,
        {"loss": loss, "mae": jnp.mean(jnp.abs(jnp.subtract(window.labels, logits)))},
    )


def collect(metrics: List[Dict[str, jax.Array]]) -> Dict[str, jax.Array]:
    return tree_util.map(lambda *args: jnp.stack(args), *metrics)


def train(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    data_config: Dict,
    model_config: Dict,
    train_config: Dict,
) -> None:
    logger = logging.getLogger(__name__)

    ds_train = make_dataset(df_train, **data_config)
    ds_val = make_dataset(df_val, **data_config)

    model = TSMixer(**model_config)

    init_key, train_key = random.split(random.key(train_config["seed"]))
    batch = next(ds_train.as_numpy_iterator())
    variables = model.init(init_key, batch.inputs, train=False)

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        batch_stats=variables["batch_stats"],
        tx=optax.adam(train_config["learning_rate"]),
    )

    train_summary_writer = tf.summary.create_file_writer(train_config["train_log_dir"])
    val_summary_writer = tf.summary.create_file_writer(train_config["val_log_dir"])
    checkpoint_manager = CheckpointManager(
        train_config["checkpoint_dir"],
        PyTreeCheckpointer(),
        CheckpointManagerOptions(max_to_keep=2, create=True),
    )

    for epoch in range(1, train_config["num_epochs"] + 1):
        logger.info("Epoch {}/{}".format(epoch, train_config["num_epochs"]))

        train_metrics = []

        for batch in ds_train.as_numpy_iterator():
            state, train_metric = train_step(
                state, batch, random.fold_in(key=train_key, data=state.step)
            )
            train_metrics.append(train_metric)

        with train_summary_writer.as_default():
            for name, metrics in collect(train_metrics).items():
                tf.summary.scalar(name, jnp.mean(metrics), step=epoch)

        val_metrics = []

        for batch in ds_val.as_numpy_iterator():
            _, val_metric = eval_step(state, batch)
            val_metrics.append(val_metric)

        with val_summary_writer.as_default():
            for name, metrics in collect(val_metrics).items():
                tf.summary.scalar(name, jnp.mean(metrics), step=epoch)

        if epoch % train_config["checkpoint_freq"] == 0:
            checkpoint_manager.save(
                epoch,
                {"state": state},
                save_kwargs={
                    "save_args": orbax_utils.save_args_from_target({"state": state})
                },
            )
