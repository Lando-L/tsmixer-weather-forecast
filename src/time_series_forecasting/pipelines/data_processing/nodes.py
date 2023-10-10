"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.13
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def preprocess_weather(weather: pd.DataFrame) -> pd.DataFrame:
    weather["Date Time"] = pd.to_datetime(
        weather["Date Time"], format="%d.%m.%Y %H:%M:%S"
    )
    weather["wv (m/s)"] = weather["wv (m/s)"].where(0 <= weather["wv (m/s)"], other=0.0)
    weather["max. wv (m/s)"] = weather["max. wv (m/s)"].where(
        0 <= weather["max. wv (m/s)"], other=0.0
    )

    return weather


def feature_engineer_weather(weather: pd.DataFrame) -> pd.DataFrame:
    wv = weather.pop("wv (m/s)")
    max_wv = weather.pop("max. wv (m/s)")
    wd_rad = weather.pop("wd (deg)") * np.pi / 180

    weather["Wx"] = wv * np.cos(wd_rad)
    weather["Wy"] = wv * np.sin(wd_rad)

    weather["max Wx"] = max_wv * np.cos(wd_rad)
    weather["max Wy"] = max_wv * np.sin(wd_rad)

    timestamps = weather.pop("Date Time").map(pd.Timestamp.timestamp)

    day = 24 * 60 * 60
    year = 365.2425 * day

    weather["Day sin"] = np.sin(timestamps * (2 * np.pi / day))
    weather["Day cos"] = np.cos(timestamps * (2 * np.pi / day))
    weather["Year sin"] = np.sin(timestamps * (2 * np.pi / year))
    weather["Year cos"] = np.cos(timestamps * (2 * np.pi / year))

    return weather


def split_and_normalize(
    weather: pd.DataFrame, splits: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(weather)
    test_start = n - int(n * splits["test_split"])
    val_start = n - int(n * (splits["val_split"] + splits["test_split"]))

    train = weather[:val_start]
    val = weather[val_start:test_start]
    test = weather[test_start:]

    train_mean, train_std = train.mean(), train.std()

    return (
        (train - train_mean) / train_std,
        (val - train_mean) / train_std,
        (test - train_mean) / train_std,
    )
