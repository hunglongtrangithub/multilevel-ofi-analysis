from datetime import timedelta
import polars as pl
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from loguru import logger
from process_data import OrderBookProcessor, split_df_into_time_frames


class CrossImpactEstimator:
    """
    Estimate the cross-impact terms for a given stock symbol during a time window.
    Expects a dataframe with the BPM-10 schema.
    """

    def __init__(self, time_window_df: pl.DataFrame, symbol: str, alpha: float = 1):
        self.train_df = time_window_df
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
        logger.debug(f"Number of time frames {len(time_frame_dfs)}")

        # Calculate log returns and integrated OFIs
        y = np.zeros(len(time_frame_dfs))
        X = np.zeros((len(time_frame_dfs), len(symbols)))
        for sample_idx, sample_df in enumerate(time_frame_dfs):
            if self.symbol not in sample_df["symbol"].to_list():
                # Set the log return and order flow imbalance to 0 if the symbol is not found in the dataframe
                logger.warning(
                    f"Symbol {self.symbol} not found in time frame. Setting to 0"
                )
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
        return coef_dict

    def evaluate(self, df: pl.DataFrame):
        if self.symbols != df["symbol"].unique().sort().to_list():
            raise ValueError("Symbols do not match")
        X, y = self.prepare_data(df)
        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)
        return r2


class SelfImpactEstimator:
    """
    Estimate the self-impact terms for a given stock symbol during a time window.
    Expects a dataframe with the BPM-10 schema.
    """

    def __init__(self, time_window_df: pl.DataFrame, symbol: str):
        # Get unique symbols. Sort to ensure reproducibility
        self.symbols = time_window_df["symbol"].unique().sort().to_list()
        if symbol not in self.symbols:
            raise ValueError("Symbol not found in the dataframe")
        self.symbol = symbol
        self.train_df = time_window_df
        self.model = LinearRegression()
        self.scaler = StandardScaler()

    def prepare_data(self, time_window_df: pl.DataFrame) -> tuple:
        # Split the df into time frames of 1 minute intervals
        time_frame_dfs = split_df_into_time_frames(
            time_window_df, every=timedelta(minutes=1)
        )
        # logger.debug(f"Number of time frames {len(time_frame_dfs)}")

        # Calculate log returns and integrated OFIs
        y = np.zeros(len(time_frame_dfs))
        X = np.zeros((len(time_frame_dfs), 1))
        for sample_idx, sample_df in enumerate(time_frame_dfs):
            if self.symbol not in sample_df["symbol"].to_list():
                # Set the log return and order flow imbalance to 0 if the symbol is not found in the dataframe
                logger.warning(
                    f"Symbol {self.symbol} not found in time frame. Setting to 0"
                )
                y[sample_idx] = 0
                X[sample_idx, 0] = 0
            else:
                # logger.debug(f"Processing time frame {sample_df['ts_event'][0]}")
                # logger.debug(f"Time frame size {sample_df.height}")
                obp = OrderBookProcessor(sample_df)
                y[sample_idx] = obp.calculate_log_return(self.symbol)
                # Calculate self-impact term
                X[sample_idx, 0] = obp.calculate_integrated_ofi(self.symbol)

        # logger.debug(f"X.shape {X.shape} y.shape {y.shape}")

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

        return coef_dict

    def evaluate(self, df: pl.DataFrame):
        if self.symbol not in df["symbol"].to_list():
            raise ValueError("Symbol not found in the dataframe")
        X, y = self.prepare_data(df)
        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)
        return r2, X, y, y_pred


class SelfImpactPredictiveModel:
    def __init__(self, train_df: pl.DataFrame, symbol: str):
        pass
