import json
from pathlib import Path
from datetime import datetime, timedelta
import polars as pl
from loguru import logger
import matplotlib.pyplot as plt
from process_data import split_df_into_time_frames
from models import PIModel, CIModel, FPIModel, FCIModel

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


def train_models(symbol: str, data_dir: str):
    results = {
        "pi_coef": [],
        "is_pi_r2": [],
        "os_pi_r2": [],
        "ci_coef": [],
        "is_ci_r2": [],
        "os_ci_r2": [],
        "fpi_coef": [],
        "is_fpi_r2": [],
        "os_fpi_r2": [],
        "fci_coef": [],
        "is_fci_r2": [],
        "os_fci_r2": [],
    }

    for df_path in Path(data_dir).glob("*.dbn.parquet"):
        logger.info(f"Reading {df_path}")
        df = pl.read_parquet(df_path)
        logger.info(f"Number of rows {df.height}")
        # Find the day in the dataframe
        date = df["ts_event"].cast(pl.Datetime).dt.date()[0]
        logger.info(f"Filtering for {date}")
        # Filter the df between 10:00 am and 3:30 pm
        # Need the first 30 minutes of the day for predictive models, and the last 30 minutes for doing out-of-sample evaluation
        filtered_df = df.filter(
            pl.col("ts_event")
            .cast(pl.Datetime)
            .is_between(
                datetime(date.year, date.month, date.day, 10, 0, 0),
                datetime(date.year, date.month, date.day, 15, 30, 0),
            )
        )
        logger.debug(f"Number of rows after filtering {filtered_df.height}")

        if symbol not in filtered_df["symbol"].unique().to_list():
            logger.warning(f"Symbol {symbol} not found in the dataframe. Skipping...")
            continue

        symbol_counts = filtered_df.group_by("symbol").len()
        logger.info(symbol_counts)

        time_window_dfs = split_df_into_time_frames(filtered_df, timedelta(minutes=30))
        logger.info(f"Total number of time windows collected: {len(time_window_dfs)}")

        # Loop through the collected time windows to do the training and evaluation
        for window_idx in range(1, len(time_window_dfs) - 1):
            time_window_df = time_window_dfs[window_idx]
            logger.info(
                f"Processing time window {time_window_df['ts_event'][0]}. "
                "Number of rows in time window {time_window_df.height}"
            )
            if symbol not in time_window_df["symbol"].unique().to_list():
                # Skip this time window if the symbol is not found
                logger.warning(f"Symbol {symbol} not found in time window. Skipping...")
                continue

            logger.info("Initializing the models")
            ci_model = CIModel(time_window_df, symbol)
            pi_model = PIModel(time_window_df, symbol)

            # Combine the previous time window with the current time window to train the predictive models
            prev_time_window_df = time_window_dfs[window_idx - 1]
            whole_df = pl.concat([prev_time_window_df, time_window_df])

            fpi_model = FPIModel(time_window_df, whole_df, symbol)
            fci_model = FCIModel(time_window_df, whole_df, symbol)

            # Train the models
            logger.info("Training the models")
            ci_coef_dict = ci_model.fit()
            pi_coef_dict = pi_model.fit()
            fpi_coef_dict = fpi_model.fit()
            fci_coef_dict = fci_model.fit()

            # Save the coefficients
            results["ci_coef"].append(ci_coef_dict)
            results["pi_coef"].append(pi_coef_dict)
            results["fpi_coef"].append(fpi_coef_dict)
            results["fci_coef"].append(fci_coef_dict)

            # Evaluate the models
            # 1. In-sample evaluation
            logger.info("Doing in-sample evaluation")
            is_ci_r2 = ci_model.evaluate(time_window_df)
            is_pi_r2 = pi_model.evaluate(time_window_df)
            is_fpi_r2 = fpi_model.evaluate(time_window_df)
            is_fci_r2 = fci_model.evaluate(time_window_df)

            results["is_ci_r2"].append(is_ci_r2)
            results["is_pi_r2"].append(is_pi_r2)
            results["is_fpi_r2"].append(is_fpi_r2)
            results["is_fci_r2"].append(is_fci_r2)

            # 2. Out-of-sample evaluation
            next_time_window_df = time_window_dfs[window_idx + 1]
            logger.info("Doing out-of-sample evaluation")
            os_ci_r2 = ci_model.evaluate(next_time_window_df)
            os_pi_r2 = pi_model.evaluate(next_time_window_df)
            os_fpi_r2 = fpi_model.evaluate(next_time_window_df)
            os_fci_r2 = fci_model.evaluate(next_time_window_df)

            results["os_ci_r2"].append(os_ci_r2)
            results["os_pi_r2"].append(os_pi_r2)
            results["os_fpi_r2"].append(os_fpi_r2)
            results["os_fci_r2"].append(os_fci_r2)

    return results


def main():
    logger.info(f"Processing data for all symbols {SYMBOLS}")
    results = {}
    for symbol in SYMBOLS:
        logger.info(f"Processing {symbol}")
        symbol_results = train_models(symbol, "./data/XNAS-20250105-S6R97734QU")
        results[symbol] = symbol_results
    # Save to a json file
    logger.info("Saving results to results.json file")
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Analysis complete")


if __name__ == "__main__":
    logger.add("analysis.log", rotation="10 MB")
    main()
