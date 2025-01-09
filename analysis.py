from pathlib import Path
from datetime import datetime, timedelta
import polars as pl
from loguru import logger
import matplotlib.pyplot as plt
from process_data import split_df_into_time_frames
from models import SelfImpactContemporaneousModel

SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMGN", "GILD", "TSLA", "PEP", "JPM", "V", "XOM"]


def plot_data(X, y, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, label="Data points")
    plt.plot(X, y_pred, color="red", label="Predicted values")
    plt.xlabel("Integrated OFI")
    plt.ylabel("Log Return")
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.show()


def process(symbol: str, data_dir: str = "./data/XNAS-20250105-S6R97734QU", plot=False):
    coef_list = []
    r2_list = []
    for df_path in Path(data_dir).glob("*.dbn.parquet"):
        logger.info(f"Reading {df_path}")
        df = pl.read_parquet(df_path)
        logger.info(f"Number of rows {df.height}")
        # Find the day in the dataframe
        date = df["ts_event"].cast(pl.Datetime).dt.date()[0]
        logger.info(f"Filtering for {date}")
        # Filter the df between 10:00 am and 3:00 pm
        df = df.filter(
            pl.col("ts_event")
            .cast(pl.Datetime)
            .is_between(
                datetime(date.year, date.month, date.day, 10, 0, 0),
                datetime(date.year, date.month, date.day, 15, 0, 0),
            )
        )
        logger.debug(f"Number of rows after filtering {df.height}")

        if symbol not in df["symbol"].unique().to_list():
            logger.warning(f"Symbol {symbol} not found in the dataframe. Skipping...")
            continue

        symbol_counts = df.group_by("symbol").len()
        logger.info(symbol_counts)

        time_window_dfs = split_df_into_time_frames(df, timedelta(minutes=30))
        logger.info(f"Number of time windows {len(time_window_dfs)}")

        for time_window_df in time_window_dfs:
            if symbol not in time_window_df["symbol"].unique().to_list():
                # Skip this time window if the symbol is not found
                logger.warning(f"Symbol {symbol} not found in time window. Skipping...")
                continue
            logger.info(f"Processing time window {time_window_df['ts_event'][0]}")

            # clf = CrossImpactEstimator(time_window_df, symbol)
            clf = SelfImpactContemporaneousModel(time_window_df, symbol)

            coef_dict = clf.fit()
            logger.info(f"Coefficients: {coef_dict}")
            # Do in-sample evaluation
            r2, X, y, y_pred = clf.evaluate(time_window_df)

            logger.info(f"Coefficients: {coef_dict}")
            logger.info(f"R2: {r2}")

            # Plot data
            if plot:
                plot_data(X, y, y_pred, f"Integrated OFI vs Log Return for {symbol}")

            coef_list.append(coef_dict)
            r2_list.append(r2)

    coefs_df = pl.from_dicts(coef_list)
    r2_sr = pl.Series("r2", r2_list)

    logger.info(f"Coefficients dataframe shape {coefs_df.shape}")
    logger.info(f"R2 series length {r2_sr.len()}")

    avg_coefs = coefs_df.mean()
    avg_r2 = r2_sr.mean()

    logger.info(f"Average coefficients:\n{avg_coefs}")
    logger.info(f"Average R2:\n{avg_r2}")

    return avg_coefs, avg_r2


def main():
    avg_coef_list = []
    avg_r2_list = []
    logger.info(f"Processing data for all symbols {SYMBOLS}")
    for symbol in SYMBOLS:
        logger.info(f"Processing {symbol}")
        avg_coefs, avg_r2 = process(symbol)
        avg_coef_list.append(avg_coefs)
        avg_r2_list.append(avg_r2)

    logger.info("Processing complete")

    coefs_df = pl.concat(avg_coef_list)
    r2_sr = pl.concat(avg_r2_list)

    logger.info(f"Coefficients dataframe shape {coefs_df.shape}")
    logger.info(f"R2 series length {r2_sr.len()}")

    avg_coefs = coefs_df.mean()
    avg_r2 = r2_sr.mean()

    logger.info(f"Average coefficients:\n{avg_coefs}")
    logger.info(f"Average R2:\n{avg_r2}")


if __name__ == "__main__":
    main()
