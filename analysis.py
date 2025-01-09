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


def get_coef_and_r2(model, train_df, test_df=None, plot=False):
    coef_dict = model.fit()
    logger.info(f"Coefficients: {coef_dict}")
    if test_df is None:
        # Do in-sample evaluation
        r2, X, y, y_pred = model.evaluate(train_df)
    else:
        # Do out-of-sample evaluation
        r2, X, y, y_pred = model.evaluate(test_df)
    logger.info(f"Coefficients: {coef_dict}")
    logger.info(f"R2: {r2}")
    if plot:
        plot_data(X, y, y_pred, model.__class__.__name__)
    return coef_dict, r2


def train_models(
    symbol: str,
    data_dir: str,
    out_sample=False,  # If True, use out-of-sample evaluation
    plot=False,
):
    pi_coef_list = []
    pi_r2_list = []
    ci_coef_list = []
    ci_r2_list = []
    fpi_coef_list = []
    fpi_r2_list = []
    fci_coef_list = []
    fci_r2_list = []
    time_window_dfs = []

    # Collect all time windows from all parquet files
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

        time_window_dfs.extend(
            split_df_into_time_frames(filtered_df, timedelta(minutes=30))
        )
    logger.info(f"Total number of time windows collected: {len(time_window_dfs)}")

    # Loop through the collected time windows to do the training and evaluation
    for window_idx in range(1, len(time_window_dfs) - 1):
        time_window_df = time_window_dfs[window_idx]
        if symbol not in time_window_df["symbol"].unique().to_list():
            # Skip this time window if the symbol is not found
            logger.warning(f"Symbol {symbol} not found in time window. Skipping...")
            continue
        logger.info(f"Processing time window {time_window_df['ts_event'][0]}")

        ci_model = CIModel(time_window_df, symbol)
        pi_model = PIModel(time_window_df, symbol)

        prev_time_window_df = time_window_dfs[window_idx - 1]
        # Combine the previous time window with the current time window to train the predictive models
        whole_df = pl.concat([prev_time_window_df, time_window_df])

        fpi_model = FPIModel(time_window_df, whole_df, symbol)
        fci_model = FCIModel(time_window_df, whole_df, symbol)

        eval_df = None
        if out_sample:
            # Find the next time window that includes the same symbol
            next_window_idx = window_idx + 1
            while next_window_idx < len(time_window_dfs):
                next_time_window_df = time_window_dfs[next_window_idx]
                if symbol in next_time_window_df["symbol"].unique().to_list():
                    break
                next_window_idx += 1
            if next_window_idx == len(time_window_dfs):
                logger.warning("No next time window found. Using in-sample evaluation")
            else:
                eval_df = next_time_window_df

        ci_coef_dict, ci_r2 = get_coef_and_r2(
            ci_model, time_window_df, eval_df, plot=plot
        )
        pi_coef_dict, pi_r2 = get_coef_and_r2(
            pi_model, time_window_df, eval_df, plot=plot
        )
        fpi_coef_dict, fpi_r2 = get_coef_and_r2(
            fpi_model, time_window_df, eval_df, plot=plot
        )
        fci_coef_dict, fci_r2 = get_coef_and_r2(
            fci_model, time_window_df, eval_df, plot=plot
        )

        pi_coef_list.append(pi_coef_dict)
        pi_r2_list.append(pi_r2)
        ci_coef_list.append(ci_coef_dict)
        ci_r2_list.append(ci_r2)
        fpi_coef_list.append(fpi_coef_dict)
        fpi_r2_list.append(fpi_r2)
        fci_coef_list.append(fci_coef_dict)
        fci_r2_list.append(fci_r2)

    pi_coefs_df = pl.from_dicts(pi_coef_list)
    pi_r2_sr = pl.Series("r2", pi_r2_list)
    ci_coefs_df = pl.from_dicts(ci_coef_list)
    ci_r2_sr = pl.Series("r2", ci_r2_list)
    fpi_coefs_df = pl.from_dicts(fpi_coef_list)
    fpi_r2_sr = pl.Series("r2", fpi_r2_list)
    fci_coefs_df = pl.from_dicts(fci_coef_list)
    fci_r2_sr = pl.Series("r2", fci_r2_list)

    logger.info(
        f"Self-impact contemporaneous model's coefficients and R2 scores shape:\n{pi_coefs_df.shape, pi_r2_sr.shape}"
    )
    logger.info(
        f"Cross-impact contemporaneous model's coefficients and R2 scores shape:\n{ci_coefs_df.shape, ci_r2_sr.shape}"
    )
    logger.info(
        f"Self-impact predictive model's coefficients and R2 scores shape:\n{fpi_coefs_df.shape, fpi_r2_sr.shape}"
    )
    logger.info(
        f"Cross-impact predictive model's coefficients and R2 scores shape:\n{fci_coefs_df.shape, fci_r2_sr.shape}"
    )

    pi_avg_coefs = pi_coefs_df.mean()
    pi_avg_r2 = pi_r2_sr.mean()
    ci_avg_coefs = ci_coefs_df.mean()
    ci_avg_r2 = ci_r2_sr.mean()
    fpi_avg_coefs = fpi_coefs_df.mean()
    fpi_avg_r2 = fpi_r2_sr.mean()
    fci_avg_coefs = fci_coefs_df.mean()
    fci_avg_r2 = fci_r2_sr.mean()

    logger.info(
        f"Self-impact contemporaneous model's average coefficients: {pi_avg_coefs}, average R2: {pi_avg_r2}"
    )
    logger.info(
        f"Cross-impact contemporaneous model's average coefficients: {ci_avg_coefs}, average R2: {ci_avg_r2}"
    )
    logger.info(
        f"Self-impact predictive model's average coefficients: {fpi_avg_coefs}, average R2: {fpi_avg_r2}"
    )
    logger.info(
        f"Cross-impact predictive model's average coefficients: {fci_avg_coefs}, average R2: {fci_avg_r2}"
    )

    return (
        pi_avg_coefs,
        pi_avg_r2,
        ci_avg_coefs,
        ci_avg_r2,
        fpi_avg_coefs,
        fpi_avg_r2,
        fci_avg_coefs,
        fci_avg_r2,
    )


def main():
    avg_coef_list = []
    logger.info(f"Processing data for all symbols {SYMBOLS}")
    for symbol in SYMBOLS:
        logger.info(f"Processing {symbol}")
        (
            pi_avg_coefs,
            pi_avg_r2,
            ci_avg_coefs,
            ci_avg_r2,
            fpi_avg_coefs,
            fpi_avg_r2,
            fci_avg_coefs,
            fci_avg_r2,
        ) = train_models(symbol, "./data/XNAS-20250105-S6R97734QU")
        avg_coef_list.append(
            {
                "symbol": symbol,
                "pi_avg_coefs": pi_avg_coefs,
                "pi_avg_r2": pi_avg_r2,
                "ci_avg_coefs": ci_avg_coefs,
                "ci_avg_r2": ci_avg_r2,
                "fpi_avg_coefs": fpi_avg_coefs,
                "fpi_avg_r2": fpi_avg_r2,
                "fci_avg_coefs": fci_avg_coefs,
                "fci_avg_r2": fci_avg_r2,
            }
        )

    logger.info("Processing complete")

    results_df = pl.DataFrame(avg_coef_list)
    results_df.write_csv("model_results.csv")

    logger.info(f"Results saved to model_results.csv")


if __name__ == "__main__":
    main()
