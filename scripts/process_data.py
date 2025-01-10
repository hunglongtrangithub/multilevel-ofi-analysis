from pathlib import Path
from datetime import timedelta
import databento as db
import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from loguru import logger
import matplotlib.pyplot as plt

SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMGN", "GILD", "TSLA", "PEP", "JPM", "V", "XOM"]


def convert_to_parquet(file_path: str):
    data_path = Path(file_path)

    for file_path in data_path.rglob("*.dbn.zst"):
        logger.info(f"Processing {file_path}")
        data = db.DBNStore.from_file(file_path)
        data.to_parquet(file_path.with_suffix(".parquet"))
        logger.info("Converted to Parquet")


class OrderBookProcessor:
    """
    OrderBookProcessor class to calculate the integrated OFI vector for a given stock symbol.
    The dataframe is expected to have the BPM-10 schema.
    """

    def __init__(self, df: pl.DataFrame, num_levels: int = 10):
        if num_levels < 1 or num_levels > 10:
            raise ValueError("Number of levels must be between 1 and 10")
        self.num_levels = num_levels
        self.df = df
        self.symbols = df["symbol"].unique().sort().to_list()

    def calculate_log_return(self, symbol: str) -> float:
        if symbol not in self.symbols:
            raise ValueError("Symbol not found in the dataframe")
        # Filter for the stock symbol and then sort by event timestamp to ensure chronological order
        symbol_df = self.df.filter(pl.col("symbol") == symbol).sort("ts_event")

        # Handle null values
        # In one order, if either the best bid or best ask price is null, use the non-null price as the mid price
        # If both are null, set mid price to 0

        mid_prices = (
            symbol_df.select("bid_px_00", "ask_px_00").mean_horizontal().fill_null(0)
        ).to_list()

        # logger.info(f"mid_prices length {len(mid_prices)}")
        earliest_mid_price = mid_prices[0]
        latest_mid_price = mid_prices[-1]
        # logger.info(
        #     f"earliest_mid_price {earliest_mid_price} latest_mid_price {latest_mid_price}"
        # )

        # Add a small value to avoid division by zero
        log_return = np.log(latest_mid_price / earliest_mid_price + 1e-8)
        return log_return

    def calculate_integrated_ofi(self, symbol: str) -> float:
        if symbol not in self.symbols:
            raise ValueError("Symbol not found in the dataframe")
        ofi_vector = self.calculate_ofi_vector(symbol)
        weight_vector = self.calculate_normalized_pca_first_component()
        integrated_ofi = np.dot(ofi_vector, weight_vector)
        return integrated_ofi

    def calculate_normalized_pca_first_component(self) -> np.ndarray:
        # Get unique symbols. Sort to ensure reproducibility
        ofi_vectors = [self.calculate_ofi_vector(symbol) for symbol in self.symbols]
        ofi_matrix = np.vstack(ofi_vectors)
        # logger.debug(f"ofi_matrix shape {ofi_matrix.shape}")

        # Perform PCA on the OFI matrix
        scaler = StandardScaler()
        ofi_matrix = scaler.fit_transform(ofi_matrix)
        pca = PCA(n_components=1)
        pca.fit(ofi_matrix)

        # Normalize the first component
        first_component = pca.components_[0]
        normalized_first_component = first_component / np.linalg.norm(
            first_component, ord=1
        )
        return normalized_first_component

    def calculate_ofi_vector(self, symbol: str) -> np.ndarray:
        if symbol not in self.symbols:
            raise ValueError("Symbol not found in the dataframe")
        # Filter for the stock symbol and then sort by event timestamp to ensure chronological order
        symbol_df = self.df.filter(pl.col("symbol") == symbol).sort("ts_event")

        # Calculate order flows
        order_flow_df = self._calculate_order_flows(symbol_df)

        # Calculate order flow imbalances
        ofi_df = self._calculate_order_flow_imbalances(order_flow_df)

        # Get OFI values as a list
        ofi_values = [ofi_df[f"ofi_{i:02d}"][0] for i in range(self.num_levels)]

        ofi_vector = np.array(ofi_values)

        return ofi_vector

    def _calculate_order_flows(self, df: pl.DataFrame) -> pl.DataFrame:
        # Create expressions for each level (0-9)
        of_expressions = []

        for level in range(self.num_levels):
            # Define column names for this level
            bid_px_col = f"bid_px_{level:02d}"
            ask_px_col = f"ask_px_{level:02d}"
            bid_sz_col = f"bid_sz_{level:02d}"
            ask_sz_col = f"ask_sz_{level:02d}"

            # Cast size columns to Int64 before calculations
            bid_sz_i64 = pl.col(bid_sz_col).cast(pl.Int64)
            ask_sz_i64 = pl.col(ask_sz_col).cast(pl.Int64)

            bid_px = pl.col(bid_px_col)
            ask_px = pl.col(ask_px_col)

            # Handling null values:
            # 1. If current price is null and previous price is not null, set order flow to 0
            # 2. If current price is null and previous price is null, set order flow to 0
            # 3. If current price is not null and previous price is null, set order flow to the current size
            # To implement this, we just simply fill the null values with 0, and the above logic will be applied according to the formula in the paper.

            # Calculate bid order flow
            bid_flow = (
                pl.when(bid_px.fill_null(1) > bid_px.shift(2, fill_value=0))
                .then(bid_sz_i64)
                .when(bid_px.fill_null(0) == bid_px.shift(1, fill_value=0))
                .then(bid_sz_i64 - bid_sz_i64.shift(1, fill_value=0))
                .otherwise(-bid_sz_i64)
            ).alias(f"bid_flow_{level:02d}")

            # Calculate ask order flow
            ask_flow = (
                pl.when(ask_px.fill_null(0) > ask_px.shift(1, fill_value=0))
                .then(-ask_sz_i64)
                .when(ask_px.fill_null(0) == ask_px.shift(1, fill_value=0))
                .then(ask_sz_i64 - ask_sz_i64.shift(1, fill_value=0))
                .otherwise(ask_sz_i64)
            ).alias(f"ask_flow_{level:02d}")

            of_expressions.extend([bid_flow, ask_flow])

        # Add all flow calculations to the dataframe
        result_df = df.with_columns(of_expressions)

        return result_df

    def _calculate_order_flow_imbalances(self, df: pl.DataFrame) -> pl.DataFrame:
        # Calculate q_scales for all levels at once
        q_scale_exprs = [
            (pl.col(f"bid_sz_{i:02d}") + pl.col(f"ask_sz_{i:02d}")).sum()
            / (2 * df.height)
            for i in range(self.num_levels)
        ]

        # Get q_scales as a list of values
        q_scales = df.select(q_scale_exprs).row(0)
        total_scale = sum(q_scales)
        normalized_q_scales = [scale / total_scale for scale in q_scales]

        # Calculate OFIs with zero handling
        # If the normalized_q_scales is 0, the OFI will be 0 as well as there are no bid and ask orders at that level
        ofi_exprs = [
            (
                pl.when(normalized_q_scales[i] > 0)
                .then(
                    (pl.col(f"bid_flow_{i:02d}") - pl.col(f"ask_flow_{i:02d}")).sum()
                    / normalized_q_scales[i]
                )
                .otherwise(0)
            ).alias(f"ofi_{i:02d}")
            for i in range(self.num_levels)
        ]

        # Add OFI columns to dataframe
        return df.with_columns(ofi_exprs)


def split_df_into_time_frames(
    df: pl.DataFrame, every: str | timedelta
) -> list[pl.DataFrame]:
    """
    Split the dataframe into time frames of 1 minute intervals.
    Expects a dataframe with the BPM-10 schema.
    """
    df_schema = df.schema
    grouped_df = (
        df.sort("ts_event")
        .with_columns(pl.col("ts_event").alias("ts_event_group"))
        .group_by_dynamic("ts_event_group", every=every, closed="left")
        .agg(pl.all())
    )
    dfs = []
    for i in range(grouped_df.height):
        sub_df_dict = grouped_df[i].to_dicts()[0]
        sub_df_dict.pop("ts_event_group")
        sub_df = pl.DataFrame(data=sub_df_dict, schema=df_schema)
        sub_df = sub_df.sort("ts_event")
        dfs.append(sub_df)
    return dfs
