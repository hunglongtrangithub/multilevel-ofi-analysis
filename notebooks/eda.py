import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import databento as db
    from dotenv import load_dotenv

    load_dotenv()

    def get_data():
        if Path("./data/test/").exists():
            print(f"{"./data/test/"} already exists")
            return
        client = db.Historical()

        data = client.timeseries.get_range(
            dataset="XNAS.ITCH",
            start="2024-11-01",
            limit=1000,
            symbols=["AAPL", "MSFT", "NVDA", "AMGN", "GILD", "TSLA", "PEP", "JPM", "V", "XOM"],
            schema="mbp-10",
        )

        data.to_file("./data/test/test.dbn.zst")
        data.to_csv("./data/test/test.csv")
        data.to_json("./data/test/test.json")
        data.to_parquet("./data/test/test.parquet")

    # get_data()
    print(Path.cwd())
    return Path, db, get_data, load_dotenv


@app.cell
def _():
    import polars as pl
    df = pl.read_parquet("./data/test/test.dbn.parquet")
    print(df.select(pl.col("depth").value_counts()))
    print(df.select(pl.col("action").value_counts()))
    df
    return df, pl


@app.cell
def _(df, pl):
    def check_uniqueness():
        for symbol in df["symbol"].unique():
            test_df = df.filter(pl.col("symbol") == symbol).select(
                pl.col("symbol").value_counts(),
                pl.col("instrument_id").value_counts(), 
                pl.col("publisher_id").value_counts()
            )
            print(test_df)
    return (check_uniqueness,)


@app.cell
def _(Path, pl):
    def check_parquet_files():
        for file in Path("data").rglob("*.dbn.parquet"):
            df = pl.read_parquet(_file)
            print(df.select(pl.col("action").value_counts()))
    return (check_parquet_files,)


@app.cell
def _(df, pl):
    print(df.select(pl.col("action").value_counts()))
    print(df.filter(pl.col("action") == "T").group_by(pl.col("symbol")).agg(pl.len()))

    def write_sample_ndjson():
        df = pl.read_ndjson("./data/test/test.dbn.json")
        test_df = df.filter(pl.col("symbol") == "TSLA").sort("ts_recv")
        test_df.write_ndjson("./data/download_TSLA.json")

    write_sample_ndjson()
    return (write_sample_ndjson,)


@app.cell
def _(Path, pl):
    def check_rtype_in_dataset():
        for file_path in Path("data").rglob("*.dbn.parquet"):
            df = pl.read_parquet(file_path)
            print(file_path)
            print(df.select(pl.col("rtype").value_counts()))
            print(df.select(pl.col("action").value_counts()))

    # check_rtype_in_dataset()
    _file_path = "data/XNAS-20250105-S6R97734QU/xnas-itch-20241205.mbp-10.dbn.parquet"
    _df = pl.read_parquet(_file_path)
    print(_df.filter(pl.col("action") == "F").group_by(pl.col("symbol")).agg(pl.len()))
    _index = _df.filter(pl.col("symbol") == "MSFT").sort("ts_recv").with_row_index().filter(pl.col("action") == "F")[0]["index"]
    print(_index)
    _df.filter(pl.col("symbol") == "MSFT").sort("ts_recv").with_row_index().filter(pl.col("index").is_between(_index - 2, _index + 2))
    return (check_rtype_in_dataset,)


@app.cell
def _(df, pl):
    print(df.filter(pl.col("action") == "F", pl.col("symbol") == "A").height)
    print(df.filter(pl.col("action") == "T", pl.col("symbol") == "B").height)
    return


@app.cell
def _(pl):
    import time
    _file_path = "data/XNAS-20250105-S6R97734QU/xnas-itch-20241205.mbp-10.dbn.parquet"
    _df = pl.read_parquet(_file_path)

    # sort by ts_event
    start = time.time()
    _df = _df.sort("ts_event")
    print(f"sort by ts_event: {time.time() - start}")

    # sort again
    start = time.time()
    _df = _df.sort("ts_event")
    print(f"sort again: {time.time() - start}")
    return start, time


@app.cell
def _(pl):
    from datetime import datetime, timezone

    def create_test_data(num_levels=10):
        """
        Create a small test dataframe with BMP-10 schema including edge cases.
        """
        # Create base data with different scenarios
        data = {
            # Standard datetime format for ts_event and ts_recv
            "ts_event": [
                datetime(2024, 1, 1, 9, 30, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 9, 30, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 9, 30, 2, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 9, 30, 3, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 9, 30, 4, tzinfo=timezone.utc),
            ],
            "ts_recv": [
                datetime(2024, 1, 1, 9, 30, 0, 100000, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 9, 30, 1, 100000, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 9, 30, 2, 100000, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 9, 30, 3, 100000, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 9, 30, 4, 100000, tzinfo=timezone.utc),
            ],
            "rtype": [1, 1, 1, 1, 1],
            "publisher_id": [1, 1, 1, 1, 1],
            "instrument_id": [1, 1, 1, 1, 1],
            "action": ["insert", "update", "update", "update", "delete"],
            "side": ["buy", "sell", "buy", "sell", "buy"],
            "depth": [0, 0, 0, 0, 0],
            "price": [100.0, 100.1, 100.2, 100.1, None],
            "size": [100, 200, 150, 175, None],
            "flags": [0, 0, 0, 0, 0],
            "ts_in_delta": [0, 1000, 2000, 3000, 4000],
            "sequence": [1, 2, 3, 4, 5],
            "symbol": ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL"]
        }

        # Add order book levels (0-9) with different scenarios
        for i in range(num_levels):
            # Bid price levels
            data[f"bid_px_{i:02d}"] = [100.0, 100.0, 99.9, 99.9, None] if i == 0 else [None] * 5
            data[f"bid_sz_{i:02d}"] = [1000, 1000, 800, 800, 0] if i == 0 else [0] * 5
            data[f"bid_ct_{i:02d}"] = [5, 5, 4, 4, 0] if i == 0 else [0] * 5

            # Ask price levels
            data[f"ask_px_{i:02d}"] = [100.1, 100.2, 100.2, 100.3, None] if i == 0 else [None] * 5
            data[f"ask_sz_{i:02d}"] = [800, 900, 900, 1000, 0] if i == 0 else [0] * 5
            data[f"ask_ct_{i:02d}"] = [4, 5, 5, 6, 0] if i == 0 else [0] * 5

        # Create complete schema
        schema = {
            "ts_event": pl.Datetime(time_unit="ns", time_zone="UTC"),
            "ts_recv": pl.Datetime(time_unit="ns", time_zone="UTC"),
            "rtype": pl.UInt8,
            "publisher_id": pl.UInt16,
            "instrument_id": pl.UInt32,
            "action": pl.String,
            "side": pl.String,
            "depth": pl.UInt8,
            "price": pl.Float64,
            "size": pl.UInt32,
            "flags": pl.UInt8,
            "ts_in_delta": pl.Int32,
            "sequence": pl.UInt32,
            "symbol": pl.String
        }

        # Add order book columns to schema
        for i in range(num_levels):
            schema.update({
                f"bid_px_{i:02d}": pl.Float64,
                f"ask_px_{i:02d}": pl.Float64,
                f"bid_sz_{i:02d}": pl.UInt32,
                f"ask_sz_{i:02d}": pl.UInt32,
                f"bid_ct_{i:02d}": pl.UInt32,
                f"ask_ct_{i:02d}": pl.UInt32,
            })

        # Create DataFrame with complete schema
        df = pl.DataFrame(data, schema=schema)
        return df
    return create_test_data, datetime, timezone


@app.cell
def _(pl):
    def calculate_order_flows(df: pl.DataFrame, symbol: str, num_levels: int = 10) -> pl.DataFrame:
        """
        Calculate bid and ask order flows for each level (0-9) of the order book.

        The calculation follows these rules for each level m:

        Bid Order Flow:
        - If P_bid[n] > P_bid[n-1]: OF = q_bid[n]
        - If P_bid[n] = P_bid[n-1]: OF = q_bid[n] - q_bid[n-1]
        - If P_bid[n] < P_bid[n-1]: OF = -q_bid[n]

        Ask Order Flow:
        - If P_ask[n] > P_ask[n-1]: OF = -q_ask[n]
        - If P_ask[n] = P_ask[n-1]: OF = q_ask[n] - q_ask[n-1]
        - If P_ask[n] < P_ask[n-1]: OF = q_ask[n

        Parameters:
        -----------
        df : pl.DataFrame
            Input dataframe with BMP-10 schema
        symbol : str
            Stock symbol to filter for

        Returns:
        --------
        pl.DataFrame
            Dataframe with additional bid_flow_XX and ask_flow_XX columns
        """
        if num_levels > 10 or num_levels < 1:
            raise ValueError("Number of levels must be between 1 and 10")

        # Filter for the specific stock and sort by event timestamp
        df = (
            df.filter(pl.col("symbol") == symbol)
            .sort("ts_event")
        )

        # Create expressions for each level (0-9)
        of_expressions = []

        for level in range(num_levels):
            # Define column names for this level
            bid_px_col = f'bid_px_{level:02d}'
            ask_px_col = f'ask_px_{level:02d}'
            bid_sz_col = f'bid_sz_{level:02d}'
            ask_sz_col = f'ask_sz_{level:02d}'

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
                pl.when(bid_px.fill_null(0) > bid_px.shift(1, fill_value=0))
                .then(bid_sz_i64)
                .when(bid_px.fill_null(0) == bid_px.shift(1, fill_value=0))
                .then(bid_sz_i64 - bid_sz_i64.shift(1, fill_value=0))
                .otherwise(-bid_sz_i64)
            ).alias(f'bid_flow_{level:02d}')

            # Calculate ask order flow
            ask_flow = (
                pl.when(ask_px.fill_null(0) > ask_px.shift(1, fill_value=0))
                .then(-ask_sz_i64)
                .when(ask_px.fill_null(0) == ask_px.shift(1, fill_value=0))
                .then(ask_sz_i64 - ask_sz_i64.shift(1, fill_value=0))
                .otherwise(ask_sz_i64)
            ).alias(f'ask_flow_{level:02d}')

            of_expressions.extend([bid_flow, ask_flow])

        # Add all flow calculations to the dataframe
        result_df = df.with_columns(of_expressions)

        return result_df

    def calculate_order_flow_imbalances(df: pl.DataFrame, num_levels: int = 10) -> pl.DataFrame:
        """
        Calculate order flow imbalances across multiple price levels.

        Parameters:
        -----------
        df : pl.DataFrame
            Input dataframe with bid_flow_XX and ask_flow_XX columns

        Returns:
        --------
        pl.DataFrame
            Original dataframe with added scaled OFI columns
        """
        # Calculate q_scales for all levels at once
        q_scale_exprs = [
            (pl.col(f"bid_sz_{i:02d}") + pl.col(f"ask_sz_{i:02d}")).sum() / (2 * df.height)
            for i in range(num_levels)
        ]

        # Get q_scales as a list of values
        q_scales = df.select(q_scale_exprs).row(0)
        total_scale = sum(q_scales)
        normalized_q_scales = [scale / total_scale for scale in q_scales]
        print(normalized_q_scales)

        # Calculate OFIs with zero handling
        # If the normalized_q_scales is 0, the OFI will be 0 as well as there are no bid and ask orders at that level
        ofi_exprs = [
            (
                pl.when(normalized_q_scales[i] > 0)
                .then((pl.col(f"bid_flow_{i:02d}") - pl.col(f"ask_flow_{i:02d}")).sum() / normalized_q_scales[i])
                .otherwise(0)
            ).alias(f"ofi_{i:02d}")
            for i in range(num_levels)
        ]

        # Add OFI columns to dataframe
        return df.with_columns(ofi_exprs)
    return calculate_order_flow_imbalances, calculate_order_flows


@app.cell
def _(
    calculate_order_flow_imbalances,
    calculate_order_flows,
    create_test_data,
):
    # Create and test the data
    _test_df = create_test_data(num_levels=1)
    _result_df = calculate_order_flows(_test_df, "AAPL", num_levels=1)
    _result_df = calculate_order_flow_imbalances(_result_df, num_levels=1)
    _result_df.select([
        "bid_sz_00",
        "ask_sz_00",
        "bid_flow_00",
        "ask_flow_00",
        "ofi_00"
    ])
    return


@app.cell
def _(df, pl):
    df.filter(pl.col("bid_px_00").is_not_null(), pl.col("ask_px_00").is_not_null())[0].select("bid_px_00", "ask_px_00", pl.mean_horizontal("bid_px_00", "ask_px_00").alias("mean"))
    return


@app.cell
def _(datetime, pl):
    def test_date_time():
        # Create a DataFrame with a datetime column
        df = pl.DataFrame({
            "timestamp": [
                datetime(2023, 1, 1, 12, 0),
                datetime(2023, 1, 2, 15, 30),
                datetime(2023, 1, 1, 9, 45),
            ],
            "value": [10, 20, 15]
        })

        print(df)

        # Sort the DataFrame by the 'timestamp' column in ascending order
        sorted_df = df.sort("timestamp")
        print(sorted_df)

        # Find the minimum timestamp
        min_timestamp = df["timestamp"].min()
        print("Minimum timestamp:", min_timestamp)

        # Find the maximum timestamp
        max_timestamp = df["timestamp"].max()
        print("Maximum timestamp:", max_timestamp)
    test_date_time()
    return (test_date_time,)


@app.cell
def _(datetime, pl):
    _df = pl.DataFrame({"foo": [
            datetime(2023, 1, 1, 12, 0),
            datetime(2023, 1, 2, 15, 30),
            datetime(2023, 1, 1, 9, 45),
            datetime(2023, 1, 2, 10, 30),
        ]})
    # _df = _df.with_columns(
    #     pl.col("foo").qcut(
    #         2, 
    #         labels=["a", "b"]
    #     )
    #     .alias("qcut")
    # )
    _df.select(pl.col("foo").is_between(datetime(2023, 1, 1, 9, 45), datetime(2023, 1, 2, 10, 30)))
    return


@app.cell
def _(datetime, pl):
    def split_df_into_time_frames(df: pl.DataFrame) -> list[pl.DataFrame]:
        """
        Split the dataframe into time frames of 1 minute intervals.
        Expects a dataframe with the BPM-10 schema.
        """
        df_schema = df.schema
        grouped_df = (
            df.sort("ts_event")
            .with_columns(pl.col("ts_event").alias("ts_event_group"))
            # Group by 1 minute intervals
            .group_by_dynamic("ts_event_group", every="1h", closed="left")
            .agg(pl.all())
        )
        dfs = []
        for i in range(grouped_df.height):
            sub_df_dict = grouped_df[i].to_dicts()[0]
            sub_df_dict.pop("ts_event_group")
            sub_df = pl.DataFrame(data=sub_df_dict, schema=df_schema)
            # sub_df = sub_df.sort("ts_event")
            dfs.append(sub_df)
        return dfs

    _df = pl.DataFrame(
        {
            "time": pl.datetime_range(
                start=datetime(2021, 12, 16),
                end=datetime(2021, 12, 16, 3),
                interval="1h",
                eager=True,
            ),
        }
    )

    # print(_df)
    # _time_frames = split_df_into_time_frames(_df)
    # for _sub_df in _time_frames:
    #     print(_sub_df.shape)

    print(_df)
    _grouped_df = _df.with_columns(pl.col("time").alias("time_group")).group_by_dynamic("time_group", every="2h", closed="left").agg()
    print(_grouped_df)
    # for i in range(_grouped_df.height):
    #     _row = _grouped_df[i].to_dicts()[0]
    #     _row.pop("time_group")
    #     _sub_df = pl.DataFrame(_row)
    #     print(_sub_df)
    return (split_df_into_time_frames,)


@app.cell
def _(np):
    _X = np.zeros((2, 9))
    for _i in range(3):
        _X[0, _i * 3 : (_i + 1) * 3] = [1,2,3]
    _X
    return


@app.cell
def _(pl):
    import numpy as np
    _a = pl.Series("a", [1,2,3,4])
    _log_returns = np.log(_a / _a.shift(1))
    _log_returns
    return (np,)


@app.cell
def _():
    import json
    import random

    # Set a seed for reproducibility
    random.seed(42)

    # Define the symbols and lag structure
    SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMGN", "GILD", "TSLA", "PEP", "JPM", "V", "XOM"]
    LAGS = [1, 2, 3, 5, 10, 20, 30]

    def generate_coefficients(symbols=SYMBOLS, lags=None, cross_impact=False):
        coefficients = {}
        if cross_impact and lags:
            for symbol in symbols:
                for lag in lags:
                    coefficients[f"lag_{lag}_{symbol}"] = round(random.uniform(-0.1, 0.1), 4)
        elif not cross_impact and lags:
            for lag in lags:
                coefficients[f"lag_{lag}"] = round(random.uniform(-0.1, 0.1), 4)
        elif cross_impact and not lags:
            for symbol in symbols:
                coefficients[f"{symbol}"] = round(random.uniform(-0.1, 0.1), 4)
        else:
            coefficients["self"] = round(random.uniform(-0.1, 0.1), 4)
        coefficients["intercept"] = round(random.uniform(-0.05, 0.05), 4)
        return coefficients

    def generate_r2():
        return round(random.uniform(0.6, 0.8), 4)

    def generate_example_results(num_samples):
        results = {}
        for symbol in SYMBOLS:
            results[symbol] = {
            "pi_coef": [generate_coefficients()] * num_samples,
            "is_pi_r2": [generate_r2()] * num_samples,
            "os_pi_r2": [generate_r2()] * num_samples,
            "ci_coef": [generate_coefficients(cross_impact=True)] * num_samples,
            "is_ci_r2": [generate_r2()] * num_samples,
            "os_ci_r2": [generate_r2()] * num_samples,
            "fpi_coef": [generate_coefficients(lags=LAGS)] * num_samples,
            "is_fpi_r2": [generate_r2()] * num_samples,
            "os_fpi_r2": [generate_r2()] * num_samples,
            "fci_coef": [generate_coefficients(lags=LAGS, cross_impact=True)] * num_samples,
            "is_fci_r2": [generate_r2()] * num_samples,
            "os_fci_r2": [generate_r2()] * num_samples,
        }
        return results

    results = generate_example_results(9 * 5)
    def save_to_file():
        # Save the results to a JSON file
        with open("example_results.json", "w") as f:
            json.dump(results, f, indent=4)

        print("Randomized results saved to example_results.json")
    return (
        LAGS,
        SYMBOLS,
        generate_coefficients,
        generate_example_results,
        generate_r2,
        json,
        random,
        results,
        save_to_file,
    )


@app.cell
def _():
    print(1e-64 != 0)
    print(int(True))
    print("lag_1_APPL".rsplit("_", 1))
    return


@app.cell
def _(pl):
    _df = pl.DataFrame({
        "Stock": ["NVDA", "NVDA", "APPL", "TSLA"],
        "Lag": [10, 20, 30, 40],
        "Coefficient": [1,2,3,4],
    }).pivot(index="Stock", on="Lag", values="Coefficient")
    _df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
