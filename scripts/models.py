from datetime import datetime, timedelta
import polars as pl
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from loguru import logger
from .process_data import OrderBookProcessor, split_df_into_time_frames


class CIModel:
    """
    Estimate the log return for a given stock symbol based on the cross-impact terms during a time window (30 minutes) .
    Expects a dataframe with the BPM-10 schema.
    """

    def __init__(self, train_df: pl.DataFrame, symbol: str, alpha: float = 0.1):
        self.train_df = train_df.sort("ts_event")
        # Get unique symbols. Sort to ensure reproducibility
        self.symbols = self.train_df["symbol"].unique().sort().to_list()
        if symbol not in self.symbols:
            raise ValueError("Symbol not found in the dataframe")
        self.symbol = symbol
        self.model = Lasso(alpha=alpha)
        self.scaler = StandardScaler()

    def prepare_data(self, time_window_df: pl.DataFrame) -> tuple:
        """
        Prepare the data for training the model or making predictions.
        """
        # Make sure the symbols are the same as the training dataframe
        if self.symbols != time_window_df["symbol"].unique().sort().to_list():
            raise ValueError("Symbols do not match")

        symbols = time_window_df["symbol"].unique().sort().to_list()
        # Split the df into time frames of 1 minute intervals
        time_frame_dfs = split_df_into_time_frames(time_window_df, timedelta(minutes=1))

        # Calculate log returns and integrated OFIs
        y = np.zeros(len(time_frame_dfs))
        X = np.zeros((len(time_frame_dfs), len(symbols)))
        for sample_idx, sample_df in enumerate(time_frame_dfs):
            if self.symbol not in sample_df["symbol"].to_list():
                # Set the log return and order flow imbalance to 0 if the symbol is not found in the dataframe
                y[sample_idx] = 0
                X[sample_idx, :] = 0
            else:
                # logger.debug(f"Processing time frame {sample_df['ts_event'][0]}")
                # logger.debug(f"Time frame size {sample_df.height}")
                obp = OrderBookProcessor(sample_df)
                y[sample_idx] = obp.calculate_log_return(self.symbol)
                # Calculate self-impact and cross-impact terms
                # If the other symbol is not found in the dataframe, the order flow imbalance will be considered as 0
                X[sample_idx, :] = [
                    (
                        0
                        if symbol not in obp.symbols
                        else obp.calculate_integrated_ofi(symbol)
                    )
                    for symbol in symbols
                ]

        logger.debug(f"X.shape {X.shape} y.shape {y.shape}")

        # Standardize features
        X = self.scaler.fit_transform(X)
        return X, y

    def fit(self) -> dict:
        X, y = self.prepare_data(self.train_df)
        self.model.fit(X, y)

        coef_dict = {
            symbol: coef for symbol, coef in zip(self.symbols, self.model.coef_)
        }
        coef_dict["intercept"] = self.model.intercept_
        logger.debug(f"Model coefficients: {coef_dict.keys()}")
        return coef_dict

    def evaluate(self, df: pl.DataFrame):
        if self.symbol not in df["symbol"].to_list():
            raise ValueError("Symbol not found in the dataframe")
        X, y = self.prepare_data(df)
        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)
        return r2, X, y, y_pred


class PIModel:
    """
    Estimate the log return for a given stock symbol based on self-impact terms during a time window (30 minutes).
    Expects a dataframe with the BPM-10 schema.
    """

    def __init__(self, train_df: pl.DataFrame, symbol: str):
        # Get unique symbols. Sort to ensure reproducibility
        self.symbols = train_df["symbol"].unique().sort().to_list()
        if symbol not in self.symbols:
            raise ValueError("Symbol not found in the dataframe")
        self.symbol = symbol
        self.train_df = train_df.sort("ts_event")
        self.model = LinearRegression()
        self.scaler = StandardScaler()

    def prepare_data(self, time_window_df: pl.DataFrame) -> tuple:
        """
        Prepare the data for training the model or making predictions.
        """
        # Make sure the symbols are the same as the training dataframe
        if self.symbols != time_window_df["symbol"].unique().sort().to_list():
            raise ValueError("Symbols do not match")
        # Split the df into time frames of 1 minute intervals
        time_frame_dfs = split_df_into_time_frames(
            time_window_df, every=timedelta(minutes=1)
        )

        # Calculate log returns and integrated OFIs
        y = np.zeros(len(time_frame_dfs))
        X = np.zeros((len(time_frame_dfs), 1))
        for sample_idx, sample_df in enumerate(time_frame_dfs):
            if self.symbol not in sample_df["symbol"].to_list():
                # Set the log return and order flow imbalance to 0 if the symbol is not found in the dataframe
                y[sample_idx] = 0
                X[sample_idx, 0] = 0
            else:
                # logger.debug(f"Processing time frame {sample_df['ts_event'][0]}")
                # logger.debug(f"Time frame size {sample_df.height}")
                obp = OrderBookProcessor(sample_df)
                y[sample_idx] = obp.calculate_log_return(self.symbol)
                # Calculate self-impact term
                X[sample_idx, 0] = obp.calculate_integrated_ofi(self.symbol)

        logger.debug(f"X.shape {X.shape} y.shape {y.shape}")

        # Standardize features
        X = self.scaler.fit_transform(X)
        return X, y

    def fit(self) -> dict:
        X, y = self.prepare_data(self.train_df)
        self.model.fit(X, y)

        # Prepare coefficients dictionary
        coef_dict = {
            "self": self.model.coef_[0],
            "intercept": self.model.intercept_,
        }
        logger.debug(f"Model coefficients: {coef_dict.keys()}")
        return coef_dict

    def evaluate(self, df: pl.DataFrame):
        if self.symbol not in df["symbol"].to_list():
            raise ValueError("Symbol not found in the dataframe")
        X, y = self.prepare_data(df)
        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)
        return r2, X, y, y_pred


# The paper is not very clear about how the model is train. My interpretation is as follows:
#     - Given a time stamp, you use the last 30 minutes from this time stamp to calculate the lagged OFIs, and use the next 1 minute from this time stamp to calculate the log return.
#     - You move the time stamp 1 minute forward and do the same thing again. Repeat 30 times to get 30 data points.
#     - The "previous 30 minutes" mentioned in section 4.2. of the paper is interpreted as the 30 time stamps used to calculate 30 data points.
#     - We have non-overlapping time windows of 30 minutes to train separate models.


class FPIModel:
    """
    Predict the 1-minute future log returns of a given stock symbol using the self-impact terms during a time window (30 minutes).
    Expects a dataframe with the BPM-10 schema.
    """

    def __init__(
        self,
        time_window_df: pl.DataFrame,
        whole_df: pl.DataFrame,
        symbol: str,
        lags: list[int] = [1, 2, 3, 5, 10, 20, 30],
    ):
        # Get unique symbols. Sort to ensure reproducibility
        self.symbols = time_window_df["symbol"].unique().sort().to_list()
        if symbol not in self.symbols:
            raise ValueError("Symbol not found in the dataframe")
        self.train_df = time_window_df.sort("ts_event")
        self.whole_df = whole_df.sort("ts_event")
        self.symbol = symbol
        self.lags = lags
        self.model = LinearRegression()
        self.scaler = StandardScaler()

    def _prepare_time_frame_dfs(self, time_stamp: datetime) -> tuple:
        """
        Prepare the lagged dataframes and future dataframe with the given time stamp.
        """
        lagged_dfs = []
        for lag in self.lags:
            time_lag = timedelta(minutes=lag)
            lagged_df = self.whole_df.filter(
                pl.col("ts_event").is_between(time_stamp - time_lag, time_stamp)
            )
            lagged_dfs.append(lagged_df)
        future_df = self.whole_df.filter(
            pl.col("ts_event").is_between(time_stamp, time_stamp + timedelta(minutes=1))
        )
        return lagged_dfs, future_df

    def prepare_data(self, time_window_df: pl.DataFrame) -> tuple:
        """
        Prepare the data for training the model or making predictions.
        """
        # Make sure the symbols are the same as the training dataframe
        if self.symbols != time_window_df["symbol"].unique().sort().to_list():
            raise ValueError("Symbols do not match")
        # Get the time stamps for the time window
        time_stamps = (
            time_window_df.group_by_dynamic("ts_event", every="1m", closed="left")
            .agg()["ts_event"]
            .to_list()
        )

        # Calculate log returns and integrated OFIs
        y = np.zeros(len(time_stamps))
        X = np.zeros((len(time_stamps), len(self.lags)))
        for sample_idx, time_stamp in enumerate(time_stamps):
            lagged_dfs, future_df = self._prepare_time_frame_dfs(time_stamp)
            # If the symbol is not found in the future dataframe, the log return will be considered as 0
            y[sample_idx] = (
                OrderBookProcessor(future_df).calculate_log_return(self.symbol)
                if self.symbol in future_df["symbol"].to_list()
                else 0
            )
            # For any lagged dataframe where the symbol is not found, the order flow imbalance will be considered as 0
            X[sample_idx, :] = [
                (
                    OrderBookProcessor(lagged_df).calculate_integrated_ofi(self.symbol)
                    if self.symbol in lagged_df["symbol"].to_list()
                    else 0
                )
                for lagged_df in lagged_dfs
            ]

        logger.debug(f"X.shape {X.shape} y.shape {y.shape}")

        # Standardize features
        X = self.scaler.fit_transform(X)
        return X, y

    def fit(self) -> dict:
        X, y = self.prepare_data(self.train_df)
        self.model.fit(X, y)

        # Prepare coefficients dictionary
        coef_dict = {
            f"lag_{lag}": coef for lag, coef in zip(self.lags, self.model.coef_)
        }
        coef_dict["intercept"] = self.model.intercept_
        logger.debug(f"Model coefficients: {coef_dict.keys()}")
        return coef_dict

    def evaluate(self, df: pl.DataFrame):
        if self.symbol not in df["symbol"].to_list():
            raise ValueError("Symbol not found in the dataframe")
        X, y = self.prepare_data(df)
        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)
        return r2, X, y, y_pred


class FCIModel:
    """
    Predict the 1-minute future log returns of a given stock symbol using the cross-impact terms during a time window (30 minutes).
    Expects a dataframe with the BPM-10 schema.
    """

    def __init__(
        self,
        time_window_df: pl.DataFrame,
        whole_df: pl.DataFrame,
        symbol: str,
        lags: list[int] = [1, 2, 3, 5, 10, 20, 30],
        alpha: float = 0.1,
    ):
        # Get unique symbols. Sort to ensure reproducibility
        self.symbols = time_window_df["symbol"].unique().sort().to_list()
        if symbol not in self.symbols:
            raise ValueError("Symbol not found in the dataframe")
        self.train_df = time_window_df.sort("ts_event")
        self.whole_df = whole_df.sort("ts_event")
        self.symbol = symbol
        self.lags = lags
        self.model = Lasso(alpha=alpha)
        self.scaler = StandardScaler()

    def _prepare_time_frame_dfs(self, time_stamp: datetime) -> tuple:
        """
        Prepare the lagged dataframes and future dataframe with the given time stamp.
        """
        lagged_dfs = []
        for lag in self.lags:
            time_lag = timedelta(minutes=lag)
            lagged_df = self.whole_df.filter(
                pl.col("ts_event").is_between(time_stamp - time_lag, time_stamp)
            )
            lagged_dfs.append(lagged_df)
        future_df = self.whole_df.filter(
            pl.col("ts_event").is_between(time_stamp, time_stamp + timedelta(minutes=1))
        )
        return lagged_dfs, future_df

    def prepare_data(self, time_window_df: pl.DataFrame) -> tuple:
        """
        Prepare the data for training the model or making predictions.
        """
        # Make sure the symbols are the same as the training dataframe
        if self.symbols != time_window_df["symbol"].unique().sort().to_list():
            raise ValueError("Symbols do not match")
        # Get the time stamps for the time window
        time_stamps = (
            time_window_df.group_by_dynamic("ts_event", every="1m", closed="left")
            .agg()["ts_event"]
            .to_list()
        )

        # Calculate log returns and integrated OFIs
        y = np.zeros(len(time_stamps))
        X = np.zeros((len(time_stamps), len(self.lags) * len(self.symbols)))
        for sample_idx, time_stamp in enumerate(time_stamps):
            lagged_dfs, future_df = self._prepare_time_frame_dfs(time_stamp)
            # If the symbol is not found in the future dataframe, the log return will be considered as 0
            y[sample_idx] = (
                OrderBookProcessor(future_df).calculate_log_return(self.symbol)
                if self.symbol in future_df["symbol"].to_list()
                else 0
            )
            # For any lagged dataframe where a symbol is not found, the order flow imbalance will be considered as 0
            for symbol_idx, symbol in enumerate(self.symbols):
                X[
                    sample_idx,
                    symbol_idx * len(self.lags) : (symbol_idx + 1) * len(self.lags),
                ] = [
                    (
                        OrderBookProcessor(lagged_df).calculate_integrated_ofi(symbol)
                        if symbol in lagged_df["symbol"].to_list()
                        else 0
                    )
                    for lagged_df in lagged_dfs
                ]

        logger.debug(f"X.shape {X.shape} y.shape {y.shape}")

        # Standardize features
        X = self.scaler.fit_transform(X)

        return X, y

    def fit(self) -> dict:
        X, y = self.prepare_data(self.train_df)
        self.model.fit(X, y)

        # Prepare coefficients dictionary
        coef_dict = {
            f"lag_{lag}_{symbol}": coef
            for lag in self.lags
            for symbol, coef in zip(self.symbols, self.model.coef_)
        }
        coef_dict["intercept"] = self.model.intercept_
        logger.debug(f"Model coefficients: {coef_dict.keys()}")
        return coef_dict

    def evaluate(self, df: pl.DataFrame):
        if self.symbol not in df["symbol"].to_list():
            raise ValueError("Symbol not found in the dataframe")
        X, y = self.prepare_data(df)
        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)
        return r2, X, y, y_pred
