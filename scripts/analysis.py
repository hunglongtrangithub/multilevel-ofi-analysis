import json
from pathlib import Path
from datetime import datetime, timedelta
import polars as pl
from loguru import logger
from process_data import split_df_into_time_frames, OrderBookProcessor
from models import PIModel, CIModel, FPIModel, FCIModel
from config import SYMBOLS


def collect_coefficients_and_r2_scores(input_dir: str, symbols: list, output_dir: str):
    """
    Analyze market data for multiple symbols across multiple files, training and evaluating
    various models for each time window.

    Args:
        data_dir (str): Directory containing the parquet files
        symbols (list): List of symbols to analyze
        output_file (str): Path to save the results JSON file

    Returns:
        dict: Dictionary containing all results for all symbols
    """
    logger.info(f"Starting analysis for symbols: {symbols}")

    # Initialize results structure for all symbols
    results = {
        symbol: {
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
        for symbol in symbols
    }

    # Process each data file
    for df_path in Path(input_dir).glob("*.dbn.parquet"):
        logger.info(f"Reading {df_path}")
        df = pl.read_parquet(df_path)
        date = df["ts_event"].cast(pl.Datetime).dt.date()[0]
        logger.info(f"Processing date: {date}")

        # Filter time window once for all symbols
        filtered_df = df.filter(
            pl.col("ts_event")
            .cast(pl.Datetime)
            .is_between(
                datetime(date.year, date.month, date.day, 10, 0, 0),
                datetime(date.year, date.month, date.day, 15, 30, 0),
            )
        )
        logger.debug(f"Filtered rows: {filtered_df.height}")

        # Split into time windows once for all symbols
        time_window_dfs = split_df_into_time_frames(filtered_df, timedelta(minutes=30))
        logger.info(f"Created {len(time_window_dfs)} time windows")

        # Process each symbol
        for symbol in symbols:
            if symbol not in filtered_df["symbol"].unique().to_list():
                logger.warning(f"Symbol {symbol} not found in {df_path}. Skipping...")
                continue

            logger.info(f"Processing symbol: {symbol}")

            # Process each time window for this symbol
            for window_idx in range(1, len(time_window_dfs) - 1):
                time_window_df = time_window_dfs[window_idx]
                if symbol not in time_window_df["symbol"].unique().to_list():
                    continue

                logger.debug(
                    f"Processing window {window_idx} for {symbol}: "
                    f"{time_window_df['ts_event'][0]} to {time_window_df['ts_event'][-1]}"
                )

                # Prepare data for predictive models
                prev_time_window_df = time_window_dfs[window_idx - 1]
                whole_df = pl.concat([prev_time_window_df, time_window_df])

                try:
                    # Initialize models
                    logger.debug("Initializing models")
                    ci_model = CIModel(time_window_df, symbol)
                    pi_model = PIModel(time_window_df, symbol)
                    fpi_model = FPIModel(time_window_df, whole_df, symbol)
                    fci_model = FCIModel(time_window_df, whole_df, symbol)

                    # Train models and collect results
                    logger.debug("Training models")
                    results[symbol]["ci_coef"].append(ci_model.fit())
                    results[symbol]["pi_coef"].append(pi_model.fit())
                    results[symbol]["fpi_coef"].append(fpi_model.fit())
                    results[symbol]["fci_coef"].append(fci_model.fit())

                    # In-sample evaluation
                    logger.debug("Performing in-sample evaluation")
                    results[symbol]["is_ci_r2"].append(
                        ci_model.evaluate(time_window_df)
                    )
                    results[symbol]["is_pi_r2"].append(
                        pi_model.evaluate(time_window_df)
                    )
                    results[symbol]["is_fpi_r2"].append(
                        fpi_model.evaluate(time_window_df)
                    )
                    results[symbol]["is_fci_r2"].append(
                        fci_model.evaluate(time_window_df)
                    )

                    # Out-of-sample evaluation
                    logger.debug("Performing out-of-sample evaluation")
                    next_time_window_df = time_window_dfs[window_idx + 1]
                    results[symbol]["os_ci_r2"].append(
                        ci_model.evaluate(next_time_window_df)
                    )
                    results[symbol]["os_pi_r2"].append(
                        pi_model.evaluate(next_time_window_df)
                    )
                    results[symbol]["os_fpi_r2"].append(
                        fpi_model.evaluate(next_time_window_df)
                    )
                    results[symbol]["os_fci_r2"].append(
                        fci_model.evaluate(next_time_window_df)
                    )

                except Exception as e:
                    logger.error(
                        f"Error processing window {window_idx} for {symbol}: {str(e)}"
                    )
                    continue

        # Save intermediate results after each file
        output_file = Path(output_dir) / "models_results.json"
        logger.info(f"Saving intermediate results to {output_file}")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

    logger.info("Analysis complete")
    return results


def collect_ofis(input_dir: str, symbols: list, output_dir: str):
    """
    Collect Order Flow Imbalance (OFI) data for multiple symbols across multiple files.
    Processes data in a file-first manner to minimize I/O operations.

    Args:
        data_dir (str): Directory containing the parquet files
        symbols (list): List of symbols to process
        output_file (str): Path to save the results parquet file

    Returns:
        pl.DataFrame: DataFrame containing OFI results
    """
    results = []
    for df_path in Path(input_dir).glob("*.dbn.parquet"):
        logger.info(f"Reading {df_path}")
        df = pl.read_parquet(df_path)
        logger.info(f"Number of rows {df.height}")
        # Find the day in the dataframe
        date = df["ts_event"].cast(pl.Datetime).dt.date()[0]
        logger.info(f"Filtering for {date}")

        # Filter the df between 10:00 am and 3:30 pm
        filtered_df = df.filter(
            pl.col("ts_event")
            .cast(pl.Datetime)
            .is_between(
                datetime(date.year, date.month, date.day, 10, 0, 0),
                datetime(date.year, date.month, date.day, 15, 30, 0),
            )
        )
        logger.debug(f"Number of rows after filtering {filtered_df.height}")

        time_window_dfs = split_df_into_time_frames(filtered_df, timedelta(minutes=1))
        logger.info(
            f"Total number of 1-minute intervals collected: {len(time_window_dfs)}"
        )

        obp = OrderBookProcessor(time_window_df)
        for symbol in symbols:
            logger.info(f"Processing symbol {symbol}")
            if symbol not in filtered_df["symbol"].unique().to_list():
                logger.warning(
                    f"Symbol {symbol} not found in the dataframe. Skipping..."
                )
                continue

            for window_idx, time_window_df in enumerate(time_window_dfs):
                if symbol not in time_window_df["symbol"].unique().to_list():
                    continue

                try:
                    # Calculate OFIs for this symbol
                    multi_level_ofis = obp.calculate_ofi_vector(symbol)
                    integrated_ofi = obp.calculate_integrated_ofi(symbol)

                    # Create result dictionary
                    result = {
                        "timestamp": time_window_df["ts_event"][0],
                        "symbol": symbol,
                        "integrated_ofi": integrated_ofi,
                    }

                    # Add individual OFI levels
                    for i, ofi in enumerate(multi_level_ofis):
                        result[f"ofi_level_{i:02d}"] = ofi

                    results.append(result)

                except Exception as e:
                    logger.error(
                        f"Error processing {symbol} in window {window_idx}: {str(e)}"
                    )
                    continue

    results_df = pl.DataFrame(results)

    ofis_file = Path(output_dir) / "ofis_results.parquet"
    results_df.write_parquet(ofis_file)
    logger.info(f"Saved OFIs to {ofis_file}")
    return results_df


if __name__ == "__main__":
    data_dir = Path(__file__).parents[1] / "data"
    # logger.add("analysis.log", rotation="10 MB")
    collect_coefficients_and_r2_scores(
        data_dir / "XNAS-20250105-S6R97734QU", SYMBOLS, data_dir
    )
    # collect_ofis(data_dir / "XNAS-20250105-S6R97734QU", SYMBOLS, data_dir)
