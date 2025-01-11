import json
from pathlib import Path
from datetime import datetime, timedelta
import polars as pl
from loguru import logger
import multiprocessing as mp
from functools import partial
import shutil
from process_data import split_df_into_time_frames, OrderBookProcessor
from models import PIModel, CIModel, FPIModel, FCIModel
from config import SYMBOLS


def process_symbol(
    symbol: str,
    date_str: str,
    time_window_dfs: list,
    checkpoint_dir: Path,
    output_dir: Path,
):
    """
    Process a single symbol across all time windows.
    This function will be run in parallel for different symbols.
    """
    symbol_results = {
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

    def save_checkpoint(window_idx, data):
        checkpoint_file = checkpoint_dir / f"{symbol}_{date_str}_{window_idx}.json"
        with open(checkpoint_file, "w") as f:
            json.dump(data, f)

    def load_checkpoint(window_idx):
        checkpoint_file = checkpoint_dir / f"{symbol}_{date_str}_{window_idx}.json"
        if checkpoint_file.exists():
            with open(checkpoint_file) as f:
                return json.load(f)
        return None

    def save_symbol_results():
        symbol_file = output_dir / f"{symbol}_{date_str}_results.json"
        with open(symbol_file, "w") as f:
            json.dump(symbol_results, f, indent=4)

    for window_idx in range(1, len(time_window_dfs) - 1):
        # Check existing checkpoint
        checkpoint = load_checkpoint(window_idx)
        if checkpoint is not None:
            for key, value in checkpoint.items():
                symbol_results[key].append(value)
            continue

        time_window_df = time_window_dfs[window_idx]
        if symbol not in time_window_df["symbol"].unique().to_list():
            continue

        try:
            # Prepare data
            prev_time_window_df = time_window_dfs[window_idx - 1]
            whole_df = pl.concat([prev_time_window_df, time_window_df])

            # Process window
            window_results = {}

            # Modle training & in-sample evaluation & out-of-sample evaluation
            next_time_window_df = time_window_dfs[window_idx + 1]

            ci_model = CIModel(time_window_df, symbol)
            window_results["ci_coef"] = ci_model.fit()
            window_results["is_ci_r2"] = ci_model.evaluate(time_window_df)[0]
            window_results["os_ci_r2"] = ci_model.evaluate(next_time_window_df)[0]

            pi_model = PIModel(time_window_df, symbol)
            window_results["pi_coef"] = pi_model.fit()
            window_results["is_pi_r2"] = pi_model.evaluate(time_window_df)[0]
            window_results["os_pi_r2"] = pi_model.evaluate(next_time_window_df)[0]

            fpi_model = FPIModel(time_window_df, whole_df, symbol)
            window_results["fpi_coef"] = fpi_model.fit()
            window_results["is_fpi_r2"] = fpi_model.evaluate(time_window_df)[0]
            window_results["os_fpi_r2"] = fpi_model.evaluate(next_time_window_df)[0]

            fci_model = FCIModel(time_window_df, whole_df, symbol)
            window_results["fci_coef"] = fci_model.fit()
            window_results["is_fci_r2"] = fci_model.evaluate(time_window_df)[0]
            window_results["os_fci_r2"] = fci_model.evaluate(next_time_window_df)[0]

            # Save checkpoint and update results
            save_checkpoint(window_idx, window_results)
            for key, value in window_results.items():
                symbol_results[key].append(value)

            # Save intermediate results periodically
            if window_idx % 5 == 0:
                save_symbol_results()

        except Exception as e:
            logger.error(f"Error processing window {window_idx} for {symbol}: {str(e)}")
            continue

    # Final save for this symbol
    save_symbol_results()
    return symbol, symbol_results


def collect_coefficients_and_r2_scores(
    input_dir: str | Path,
    symbols: list,
    output_dir: str | Path,
    n_processes: int | None = None,
):
    """
    Parallel processing of market data analysis with checkpointing.

    Args:
        input_dir (str): Directory containing the parquet files
        symbols (list): List of symbols to analyze
        output_dir (str): Directory to save results and checkpoints
        n_processes (int): Number of parallel processes to use. Defaults to CPU count - 1
    """
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    results = {}

    # Process each data file
    for df_path in Path(input_dir).glob("*.dbn.parquet"):
        logger.info(f"Reading {df_path}")
        df = pl.read_parquet(df_path)
        date = df["ts_event"].cast(pl.Datetime).dt.date()[0]
        date_str = date.strftime("%Y%m%d")

        filtered_df = df.filter(
            pl.col("ts_event")
            .cast(pl.Datetime)
            .is_between(
                datetime(date.year, date.month, date.day, 10, 0, 0),
                datetime(date.year, date.month, date.day, 15, 30, 0),
            )
        )

        time_window_dfs = split_df_into_time_frames(filtered_df, timedelta(minutes=30))

        # Filter symbols that exist in this file
        available_symbols = [
            s for s in symbols if s in filtered_df["symbol"].unique().to_list()
        ]

        if not available_symbols:
            logger.warning("No requested symbols found in this file")
            continue

        logger.info(
            f"Processing {len(available_symbols)} symbols using {n_processes} processes"
        )

        # Create partial function with fixed arguments
        process_func = partial(
            process_symbol,
            date_str=date_str,
            time_window_dfs=time_window_dfs,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
        )

        # Process symbols in parallel
        with mp.Pool(n_processes) as pool:
            for symbol, symbol_results in pool.imap_unordered(
                process_func, available_symbols
            ):
                results[symbol] = symbol_results
                logger.info(f"Completed processing {symbol}")

    # Combine all results into final file
    final_output = output_dir / "models_results.json"
    with open(final_output, "w") as f:
        json.dump(results, f, indent=4)

    # Clean up checkpoints
    shutil.rmtree(checkpoint_dir)

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
                    obp = OrderBookProcessor(time_window_df)
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
