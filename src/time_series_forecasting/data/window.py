from typing import NamedTuple

import numpy as np
import pandas as pd
import tensorflow as tf


class Window(NamedTuple):
    inputs: np.ndarray
    labels: np.ndarray


def make_dataset(
    df: pd.DataFrame, input_size: int, output_size: int, batch_size: int, seed: int
) -> tf.data.Dataset:
    def split_window(x: tf.Tensor) -> Window:
        input_slice = slice(0, input_size)
        label_slice = slice(input_size, None)

        inputs = x[:, input_slice, :]
        labels = x[:, label_slice, :]

        inputs.set_shape([None, input_size, None])
        labels.set_shape([None, output_size, None])

        return Window(inputs, labels)

    return tf.keras.utils.timeseries_dataset_from_array(
        data=np.array(df, dtype=np.float32),
        targets=None,
        sequence_length=input_size + output_size,
        sequence_stride=1,
        shuffle=True,
        seed=seed,
        batch_size=batch_size,
    ).map(split_window)
