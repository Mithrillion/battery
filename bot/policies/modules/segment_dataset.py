from typing import List, Optional, Tuple
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, PowerTransformer
from torch.utils.data import Dataset, ConcatDataset
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import pandas as pd
from .battery import get_peak_indicator


class SegmentDataset(Dataset):
    def __init__(self, df, duration=21, skip_step=3):
        self.interval_len = duration * 60 * (60 // 5)  # convert days to 5-min intervals
        self.data = sliding_window_view(df, self.interval_len, 0)[::skip_step, :].copy()
        self.duration = duration
        self.skip_step = skip_step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].T


class SegmentAndEnvDataset(Dataset):
    def __init__(
        self, df, price, pv_power, interval_len=14 * 60 * (60 // 5), skip_step=(60 // 5)
    ):
        self.interval_len = interval_len
        self.data = sliding_window_view(df, self.interval_len, 0)[::skip_step, :].copy()
        self.skip_step = skip_step
        self.price = sliding_window_view(price, self.interval_len, 0)[
            ::skip_step, :
        ].copy()
        self.pv_power = sliding_window_view(pv_power, self.interval_len, 0)[
            ::skip_step, :
        ].copy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].T, self.pv_power[idx].T, self.price[idx].T


class SegmentAndEnvDatasetWithPeaks(Dataset):
    def __init__(
        self,
        data,
        price,
        pv_power,
        peak_indicator,
        interval_len=14 * 60 * (60 // 5),
        skip_step=(60 // 5),
    ):
        self.interval_len = interval_len
        self.data = sliding_window_view(data, self.interval_len, 0)[
            ::skip_step, :
        ].copy()
        self.skip_step = skip_step
        self.price = sliding_window_view(price, self.interval_len, 0)[
            ::skip_step, :
        ].copy()
        self.pv_power = sliding_window_view(pv_power, self.interval_len, 0)[
            ::skip_step, :
        ].copy()
        self.peak_indicator = sliding_window_view(peak_indicator, self.interval_len, 0)[
            ::skip_step, :
        ].copy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx].T,
            self.pv_power[idx].T,
            self.price[idx].T,
            self.peak_indicator[idx].T,
        )


active_columns = [
    "timestamp",
    "price",
    "demand",
    # "temp_air",
    "pv_power",
    "pv_power_forecast_1h",
    "pv_power_forecast_2h",
    "pv_power_forecast_24h",
    "pv_power_basic",
]


def drop_no_solar(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(
        subset=[
            "pv_power",
            "pv_power_forecast_1h",
            "pv_power_forecast_2h",
            "pv_power_forecast_24h",
            "pv_power_basic",
        ]
    )


# custom price and solar data transformer
class PriceSolarTransformer(TransformerMixin, BaseEstimator):
    """
    Transformer class for preprocessing price and solar power data.

    Parameters:
    -----------
    scaler : BaseEstimator, optional
        Scaler object for scaling the data, by default None
    active_columns : List, optional
        List of active columns to be used in the dataset, by default active_columns
    include_pos_code : bool, optional
        Flag indicating whether to include position code features, by default False
    include_peak_indicator : bool, optional
        Flag indicating whether to include peak indicator feature, by default False
    include_price_augmentations : bool, optional
        Flag indicating whether to include price augmentation features, by default False
    quantile_random_state : int, optional
        Random state for quantile transformer, by default 7777
    interpolate_missing : bool, optional
        Flag indicating whether to interpolate missing values, by default False
    """

    def __init__(
        self,
        scaler: BaseEstimator = None,  # must be a pre-fitted scaler object if not None!
        active_columns: List = active_columns,
        include_pos_code: bool = False,
        include_peak_indicator: bool = False,
        include_price_augmentations: bool = False,
        quantile_random_state: int = 7777,
        interpolate_missing: bool = False,
        price_transform_patch: bool = True,  # set to False for the old version!
        # include_ema: bool = False,
        # ema_windows: Tuple[int, int, int] = (32, 244, 105),
    ):
        self.scaler = scaler
        self.active_columns = active_columns
        self.include_pos_code = include_pos_code
        self.include_peak_indicator = include_peak_indicator
        self.include_price_augmentations = include_price_augmentations
        self.quantile_random_state = quantile_random_state
        self.interpolate_missing = interpolate_missing
        self.price_transform_patch = price_transform_patch

    def fit(self, X, y=None):
        """
        Fit the transformer on the input data.

        Parameters:
        -----------
        X : DataFrame
            Input data to fit the transformer on
        y : None, optional
            Target variable (unused), by default None

        Returns:
        --------
        self : PriceSolarTransformer
            Returns the fitted transformer object
        """
        # X needs to be a dataframe
        dat = X[active_columns].copy()
        if self.interpolate_missing:
            col_ind_without_time = [x for x in active_columns if x != "timestamp"]
            interp_values = dat[col_ind_without_time].interpolate(
                axis=0, limit_area="inside", limit=1
            )
            dat.loc[:, col_ind_without_time] = interp_values
        # here we pre-compute some stats on the training data to be used in the transform method
        if self.include_price_augmentations:
            self.price_params = {
                "mean": dat["price"].mean(),
                "std": dat["price"].std(),
            }
            self.price_quantile = QuantileTransformer(
                random_state=self.quantile_random_state
            ).fit(dat["price"].values.reshape(-1, 1))
        # this is to make the price data symmetric and appear more normal
        self.price_transformer = PowerTransformer().fit(
            dat["price"].values.reshape(-1, 1)
        )
        # NOTE: changing the price values here! Do not compute price features after this step!
        if self.price_transform_patch:
            dat["price"] = self.price_transformer.transform(
                dat["price"].values.reshape(-1, 1)
            )
        if self.scaler is None:
            # StandardScaler might also work, but having the minimum value always be -1 is useful
            # for my augmentation workflows elsewhere
            self.scaler = MinMaxScaler((-1, 1))
            self.scaler.fit(dat[[x for x in active_columns if x not in ["timestamp"]]])
        return self

    def transform(self, X):
        """
        Transform the input data using the fitted transformer.

        Parameters:
        -----------
        X : DataFrame
            Input data to transform

        Returns:
        --------
        res : ndarray
            Transformed data
        pv_power : Series
            Solar power data
        price : Series
            Price data
        peak_indicator : ndarray
            Peak indicator data
        """
        dat = X[active_columns].copy()
        if self.interpolate_missing:
            col_ind_without_time = [x for x in active_columns if x != "timestamp"]
            interp_values = dat[col_ind_without_time].interpolate(
                axis=0, limit_area="inside", limit=1
            )
            dat.loc[:, col_ind_without_time] = interp_values

        price = dat["price"].copy()
        pv_power = dat["pv_power"].copy()
        # make symmetric
        dat["price"] = self.price_transformer.transform(
            dat["price"].values.reshape(-1, 1)
        )
        # scale data
        res = self.scaler.transform(
            dat[[x for x in active_columns if x not in ["timestamp"]]]
        )
        # combine features
        if self.include_price_augmentations:
            price_is_spike = (
                price - self.price_params["mean"]
            ) > 3 * self.price_params["std"]
            price_is_negative_spike = (
                price - self.price_params["mean"]
            ) < -3 * self.price_params["std"]
            price_is_positive = price > 0
            price_quantiles = self.price_quantile.transform(price.values.reshape(-1, 1))
            res = np.concatenate(
                [
                    res,
                    price_is_spike.values.reshape(-1, 1),
                    price_is_negative_spike.values.reshape(-1, 1),
                    price_is_positive.values.reshape(-1, 1),
                    price_quantiles,
                ],
                axis=1,
            )

        # positional encoding to indicate time of day and day of week
        if self.include_pos_code:
            hour = dat["timestamp"].dt.hour
            weekday = dat["timestamp"].dt.weekday
            fraction_of_day = hour + dat["timestamp"].dt.minute / 60
            fraction_of_week = weekday + fraction_of_day / 24
            pos_code_day = np.stack(
                [
                    np.sin(2 * np.pi * fraction_of_day),
                    np.cos(2 * np.pi * fraction_of_day),
                ],
                axis=1,
            )
            pos_code_week = np.stack(
                [
                    np.sin(2 * np.pi * fraction_of_week),
                    np.cos(2 * np.pi * fraction_of_week),
                ],
                axis=1,
            )
            res = np.concatenate([res, pos_code_day, pos_code_week], axis=1)
        # peak hour binary encoding
        if self.include_peak_indicator:
            peak_indicator = get_peak_indicator(dat["timestamp"]).numpy()
            res = np.concatenate([res, peak_indicator[:, None]], axis=1)
        return res, pv_power, price, peak_indicator


def split_segment_for_validation(
    data: np.ndarray,
    pv_power: np.ndarray,
    price: np.ndarray,
    peak_ind: np.ndarray,
    segment: Tuple[int, int],
    interval_len: int,
    validation_len: Optional[int] = None,
    train_skip_step: int = 1,
    validation_skip_step: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    brfore_data, valid_data, after_data = np.split(data, segment, axis=0)
    before_pv_power, valid_pv_power, after_pv_power = np.split(
        pv_power, segment, axis=0
    )
    before_price, valid_price, after_price = np.split(price, segment, axis=0)
    before_peak_ind, valid_peak_ind, after_peak_ind = np.split(
        peak_ind, segment, axis=0
    )
    before_dataset = SegmentAndEnvDatasetWithPeaks(
        brfore_data,
        before_price,
        before_pv_power,
        before_peak_ind,
        interval_len,
        train_skip_step,
    )

    validation_len = (
        len(valid_data)
        if validation_len is None
        else min(validation_len, len(valid_data))
    )
    validation_skip_step = (
        validation_len if validation_skip_step is None else validation_skip_step
    )

    valid_dataset = SegmentAndEnvDatasetWithPeaks(
        valid_data,
        valid_price,
        valid_pv_power,
        valid_peak_ind,
        validation_len,
        validation_skip_step,
    )
    after_dataset = SegmentAndEnvDatasetWithPeaks(
        after_data,
        after_price,
        after_pv_power,
        after_peak_ind,
        interval_len,
        train_skip_step,
    )
    train_dataset = ConcatDataset([before_dataset, after_dataset])
    return train_dataset, valid_dataset
