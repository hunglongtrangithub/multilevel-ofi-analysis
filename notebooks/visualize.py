import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def _(__file__):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1]))
    print(sys.path)
    return Path, sys


@app.cell
def _(Path, __file__):
    from scripts.config import SYMBOLS, LAGS
    import json
    from tabulate import tabulate
    import polars as pl
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math
    import copy

    data_path = Path(__file__).parents[1] / "data"
    file_path = data_path / "models_results.json"
    with open(file_path, "r") as f:
        model_data = json.load(f)

    return (
        LAGS,
        SYMBOLS,
        copy,
        data_path,
        f,
        file_path,
        json,
        math,
        model_data,
        np,
        nx,
        pl,
        plt,
        sns,
        tabulate,
    )


@app.cell
def _(LAGS, SYMBOLS, random):
    # Generate Example Data (for testing purposes)
    def generate_coefficients(symbols=SYMBOLS, lags=None, cross_impact=False):
        coefficients = {}
        if cross_impact and lags:
            for symbol in symbols:
                for lag in lags:
                    coefficients[f"lag_{lag}_{symbol}"] = round(
                        random.uniform(-0.1, 0.1), 4
                    )
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
                "fci_coef": [generate_coefficients(lags=LAGS, cross_impact=True)]
                * num_samples,
                "is_fci_r2": [generate_r2()] * num_samples,
                "os_fci_r2": [generate_r2()] * num_samples,
            }
        return results
    return generate_coefficients, generate_example_results, generate_r2


@app.cell
def _(copy, model_data, pl):
    def summarize_is_contemp_r2(data):
        rows = []
        for stock, stock_data in data.items():
            for model in ["PI", "CI"]:
                r2_values = stock_data.get(f"is_{model.lower()}_r2", [])
                rows.extend({"Stock": stock, "Model": model, "R^2": r2} for r2 in r2_values)

        df = pl.DataFrame(rows)
        summary = df.group_by("Model").agg(
            [pl.col("R^2").mean().alias("Mean_R2"), pl.col("R^2").std().alias("Std_R2")]
        )
        return summary
    is_contemp_r2_summary = summarize_is_contemp_r2(copy.deepcopy(model_data))
    print(
        "In-sample R2 score for contemporaneous models (self-impact and cross-impact):"
    )
    print(is_contemp_r2_summary)
    print("PI: self-impact, CI: cross-impact")
    return is_contemp_r2_summary, summarize_is_contemp_r2


@app.cell
def _(copy, model_data, pl):
    def summarize_os_contemp_r2(data):
        rows = []
        for stock, stock_data in data.items():
            for model in ["PI", "CI"]:
                r2_values = stock_data.get(f"os_{model.lower()}_r2", [])
                rows.extend({"Stock": stock, "Model": model, "R^2": r2} for r2 in r2_values)

        df = pl.DataFrame(rows)
        summary = df.group_by("Model").agg(
            [pl.col("R^2").mean().alias("Mean_R2"), pl.col("R^2").std().alias("Std_R2")]
        )
        return summary
    os_contemp_r2_summary = summarize_os_contemp_r2(copy.deepcopy(model_data))
    print(
        "Out-of-sample R2 score for contemporaneous models (self-impact and cross-impact):"
    )
    print(os_contemp_r2_summary)
    print("PI: self-impact, CI: cross-impact")
    return os_contemp_r2_summary, summarize_os_contemp_r2


@app.cell
def _(copy, model_data, pl):
    def summarize_os_predictive_r2(data):
        rows = []
        for stock, stock_data in data.items():
            for model in ["FPI", "FCI"]:
                r2_values = stock_data.get(f"os_{model.lower()}_r2", [])
                rows.extend({"Stock": stock, "Model": model, "R^2": r2} for r2 in r2_values)

        df = pl.DataFrame(rows)
        summary = df.group_by("Model").agg(
            [pl.col("R^2").mean().alias("Mean_R2"), pl.col("R^2").std().alias("Std_R2")]
        )
        return summary
    os_predictive_r2_summary = summarize_os_predictive_r2(copy.deepcopy(model_data))
    print(
        "Out-of-sample R2 score for predictive models (self-impact and cross-impact):"
    )
    print(os_predictive_r2_summary)
    print("FPI: self-impact, FCI: cross-impact")
    return os_predictive_r2_summary, summarize_os_predictive_r2


@app.cell
def _(copy, math, model_data, pl, tabulate):
    def summarize_ci_coefficients(data):
        rows = []
        for stock, stock_data in data.items():
            for model in ["CI"]:
                coef_samples = stock_data.get(f"{model.lower()}_coef", [])
                for coefs in coef_samples:
                    # Get rid of the intercept
                    coefs.pop("intercept", None)
                    self_coef = coefs.get(stock, 0)
                    cross_coefs = [v for k, v in coefs.items() if k != stock]

                    rows.append(
                        {
                            "Stock": stock,
                            "Model": model,
                            "Self Frequency": int(not math.isclose(self_coef, 0)),
                            "Self Magnitude": abs(self_coef),
                            "Cross Frequency": (
                                sum([int(not math.isclose(c, 0)) for c in cross_coefs])
                                / len(cross_coefs)
                                if cross_coefs
                                else 0
                            ),
                            "Cross Magnitude": (
                                sum(abs(v) for v in cross_coefs) / len(cross_coefs)
                                if cross_coefs
                                else 0
                            ),
                        }
                    )

        df = pl.DataFrame(rows)
        summary = df.group_by("Model").agg(
            [
                pl.col("Self Frequency").mean().alias("Mean_Self_Frequency"),
                pl.col("Self Frequency").std().alias("Std_Self_Frequency"),
                pl.col("Self Magnitude").mean().alias("Mean_Self_Magnitude"),
                pl.col("Self Magnitude").std().alias("Std_Self_Magnitude"),
                pl.col("Cross Frequency").mean().alias("Mean_Cross_Frequency"),
                pl.col("Cross Frequency").std().alias("Std_Cross_Frequency"),
                pl.col("Cross Magnitude").mean().alias("Mean_Cross_Magnitude"),
                pl.col("Cross Magnitude").std().alias("Std_Cross_Magnitude"),
            ]
        )
        return summary
        
    def format_summary_table_ci(summary):
        # Prepare data for the table
        data = []
        for row in summary:
            data.append([
                row["Model"],
                "Self",
                row["Mean_Self_Frequency"],
                row["Std_Self_Frequency"],
                row["Mean_Self_Magnitude"],
                row["Std_Self_Magnitude"],
            ])
            data.append([
                row["Model"],
                "Cross",
                row["Mean_Cross_Frequency"],
                row["Std_Cross_Frequency"],
                row["Mean_Cross_Magnitude"],
                row["Std_Cross_Magnitude"],
            ])

        # Define table headers
        headers = [
            "Model",
            "Type",
            "Mean Frequency",
            "Std Frequency",
            "Mean Magnitude",
            "Std Magnitude",
        ]

        # Print the table
        print(tabulate(data, headers=headers, tablefmt="simple_grid"))
    ci_coefficients_summary = summarize_ci_coefficients(copy.deepcopy(model_data)).to_dicts()
    format_summary_table_ci(ci_coefficients_summary)
    return (
        ci_coefficients_summary,
        format_summary_table_ci,
        summarize_ci_coefficients,
    )


app._unparsable_cell(
    r"""
    def generate_ofi_summary(
        df: pl.DataFrame, save_file_dir: Path | None = None
    ) -> pl.DataFrame:
        \"\"\"
        Generate summary statistics for OFI data.
        \"\"\"

        # Function to calculate kurtosis (using Fisher definition to match paper)
        def kurt(x):
            return (
                stats.kurtosis(x, fisher=True) + 3
            )  # Adding 3 to get regular kurtosis instead of excess kurtosis

        # Get all OFI level columns
        ofi_columns = [col for col in df.columns if col.startswith(\"ofi_level_\")]
        ofi_columns.append(\"integrated_ofi\")

        # Initialize results list
        results = []

        # Calculate statistics for each OFI level
        for col in ofi_columns:
            values = df[col].to_numpy()
            # Compute statistics
            mean = np.mean(values)
            std = np.std(values)
            skewness = (np.mean((values - mean) ** 3)) / (std**3) if std != 0 else 0
            kurtosis = (np.mean((values - mean) ** 4)) / (std**4) if std != 0 else 0
            percentiles = np.percentile(values, [10, 25, 50, 75, 90])
            stats_dict = {
                \"variable\": col,
                \"Mean\": mean,
                \"Std\": std,
                \"Skewness\": skewness,
                \"Kurtosis\": kurtosis,
                \"10%\": percentiles[0],
                \"25%\": percentiles[1],
                \"50%\": percentiles[2],
                \"75%\": percentiles[3],
                \"90%\": percentiles[4],
            }
            results.append(stats_dict)

        # Convert to DataFrame
        summary_df = pl.DataFrame(results)

        # Round all numeric columns to 2 decimal places
        numeric_cols = [
            \"Mean\",
            \"Std\",
            \"Skewness\",
            \"Kurtosis\",
            \"10%\",
            \"25%\",
            \"50%\",
            \"75%\",
            \"90%\",
        ]

        for col in numeric_cols:
            summary_df = summary_df.with_columns(pl.col(col).round(2))

        # Optionally save to file
        if save_file_dir:
            summary_df.write_csv(save_file_dir / \"ofi_summary.csv\")

        return summary_df

    def format_summary_table(df: pl.DataFrame) -> None:
        def format_variable_name(var_name):
            if var_name.startswith(\"ofi_level_\"):
                level = int(var_name.split(\"_\")[-1])
                return f\"ofi{level},(1m)\"
            elif var_name == \"integrated_ofi\":
                return \"ofiᴵ,(1m)\"
            return var_name

        df = df.with_columns(
            pl.col(\"variable\")
            .map_elements(format_variable_name, return_dtype=pl.String)
            .alias(\"variable\")
        )

        data = df.to_dicts()

        headers = {
            \"variable\": \"Variable\",
            \"Mean\": \"Mean\",
            \"Std\": \"Std\",
            \"Skewness\": \"Skewness\",
            \"Kurtosis\": \"Kurtosis\",
            \"10%\": \"10%\",
            \"25%\": \"25%\",
            \"50%\": \"50%\",
            \"75%\": \"75%\",
            \"90%\": \"90%\",
        }

        print(\"\nTable 1. Summary statistics of OFIs and returns.\")
        print(tabulate(data, headers=headers, tablefmt=\"rst\"))
        print(
            \"\nNote: These statistics are computed at the minute level across each stock and the full sample period.\"
        )

    ofi_data = pl.read_parquet(data_path / \"ofis_results.parquet\")
    results_dir = Path(__file__).parents[1] / \"results\")
    ofi_summary = generate_ofi_summary(ofi_data, results_dir)
    format_summary_table(ofi_summary)
    """,
    name="_"
)


@app.cell
def _(model_data, nx, plt):
    def normalize_coefficients(coefficients):
        max_value = max(abs(v) for coefs in coefficients.values() for v in coefs.values())
        if max_value == 0:
            return coefficients  # Avoid division by zero
        return {stock: {k: v / max_value for k, v in coefs.items()} for stock, coefs in coefficients.items()}

    def average_ci_coefficients(data):
        averaged_coefs = {}
        for stock, stock_data in data.items():
            for model in ["CI"]:
                coef_samples = stock_data.get(f"{model.lower()}_coef", [])
                # Initialize cumulative coefficients
                cumulative_coefs = {}
                total_samples = len(coef_samples)
                for coefs in coef_samples:
                    for k, v in coefs.items():
                        cumulative_coefs[k] = cumulative_coefs.get(k, 0) + v

                # Average coefficients
                if total_samples > 0:
                    averaged_coefs[stock] = {
                        k: cumulative_coefs[k] / total_samples for k in cumulative_coefs
                    }
                else:
                    averaged_coefs[stock] = {}

        return averaged_coefs


    def average_fci_coefficients(data):
        averaged_coefs = {}
        for stock, stock_data in data.items():
            for model in ["FCI"]:
                coef_samples = stock_data.get(f"{model.lower()}_coef", [])
                # Initialize cumulative coefficients per target stock
                cumulative_coefs = {}
                total_samples = len(coef_samples)

                for coefs in coef_samples:
                    for k, v in coefs.items():
                        lag_averaged = {}
                        lag_counts = {}
                        # First averages over all lag values for a stock
                        for k, v in coefs.items():
                            target_stock = k.split("_", 2)[
                                -1
                            ]  # Extract target stock (e.g., AAPL from lag_1_AAPL)
                            lag_averaged[target_stock] = (
                                lag_averaged.get(target_stock, 0) + v
                            )
                            lag_counts[target_stock] = lag_counts.get(target_stock, 0) + 1

                        for target_stock in lag_averaged:
                            lag_averaged[target_stock] /= lag_counts[
                                target_stock
                            ]  # Average over lags

                        for target_stock, avg_value in lag_averaged.items():
                            cumulative_coefs[target_stock] = (
                                cumulative_coefs.get(target_stock, 0) + avg_value
                            )

                # Average coefficients over all lags
                if total_samples > 0:
                    averaged_coefs[stock] = {
                        k: cumulative_coefs[k] / total_samples for k in cumulative_coefs
                    }
                else:
                    averaged_coefs[stock] = {}

        return averaged_coefs

    # Filter coefficients based on percentile
    def filter_coefficients(data, percentile, model="CI"):
        threshold = percentile / 100
        filtered = {}
        averaged_data = (
            average_ci_coefficients(data)
            if model == "CI"
            else average_fci_coefficients(data)
        )
        averaged_data = normalize_coefficients(averaged_data)
        
        for stock, coefs in averaged_data.items():
            abs_values = sorted(abs(v) for v in coefs.values())
            cutoff = abs_values[int(len(abs_values) * (1 - threshold))] if abs_values else 0
            filtered[stock] = {k: v for k, v in coefs.items() if abs(v) >= cutoff}

        return filtered


    # Generate network from coefficients
    def generate_network(data, threshold, top_k, model="CI"):
        filtered_data = filter_coefficients(data, threshold, model)
        G = nx.DiGraph()

        for stock, coefs in filtered_data.items():
            for target, weight in coefs.items():
                G.add_edge(stock, target, weight=abs(weight))
                print(abs(weight))

        edges = sorted(G.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)[
            :top_k
        ]
        G_top = nx.DiGraph()
        G_top.add_edges_from(edges)

        return G_top

    def plot_network(G, title):
        pos = nx.circular_layout(G)
        plt.figure(figsize=(12, 12))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=500,
            font_size=10,
            edge_color=[d["weight"] for _, _, d in G.edges(data=True)],
            edge_cmap=plt.cm.Blues,
            edge_vmin=0,
            edge_vmax=1,
        )
        plt.title(title)
        plt.show()
        
    # Generate and plot CI network
    threshold = 100 # Percentile
    top_k = 50  # Top edges by weight
    G_ci = generate_network(model_data, threshold, top_k, model="CI")
    plot_network(G_ci, title="Averaged Cross-Impact Network (CI)")
    return (
        G_ci,
        average_ci_coefficients,
        average_fci_coefficients,
        filter_coefficients,
        generate_network,
        normalize_coefficients,
        plot_network,
        threshold,
        top_k,
    )


@app.cell
def _(average_ci_coefficients, copy, model_data, pl, sns):
    averaged_ci_coefs = average_ci_coefficients(copy.deepcopy(model_data))
    averaged_ci_coefs_df = pl.DataFrame(averaged_ci_coefs)

    def convert(data):
        # Convert nested dict to list of records
        records = []
        for outer_key, inner_dict in data.items():
            for inner_key, value in inner_dict.items():
                records.append({
                    "symbol1": outer_key,
                    "symbol2": inner_key,
                    "value": value
                })
        
        # Create polars dataframe
        df = pl.DataFrame(records)
        # Pivot the data to create a matrix format
        matrix_df = df.pivot(
            values="value",
            index="symbol1",
            columns="symbol2"
        ).drop("symbol1", "intercept")
        return matrix_df

    matrix_df = convert(averaged_ci_coefs)
    sns.heatmap(matrix_df, annot=True, cmap="coolwarm")
    return averaged_ci_coefs, averaged_ci_coefs_df, convert, matrix_df


@app.cell
def _(copy, generate_network, model_data, plot_network, threshold, top_k):
    # Generate and plot FCI network
    G_fci = generate_network(copy.deepcopy(model_data), threshold, top_k, model="FCI")
    plot_network(G_fci, title="Averaged Forward-Looking Cross-Impact Network (FCI)")
    return (G_fci,)


@app.cell
def _(copy, model_data, np, plt):
    def plot_r2_scores(data):
        stocks = list(data.keys())
        in_sample = [np.mean(data[stock]["is_ci_r2"]) for stock in stocks]
        out_sample = [np.mean(data[stock]["os_ci_r2"]) for stock in stocks]
        for stock, is_r2, os_r2 in zip(stocks, in_sample, out_sample):
            print(f"{stock}: In-Sample R² = {is_r2:.4f}, Out-of-Sample R² = {os_r2:.4f}")
        x = np.arange(len(stocks))
        width = 0.35

        plt.figure(figsize=(10, 6))
        plt.bar(x - width / 2, in_sample, width, label="In-Sample R²")
        plt.bar(x + width / 2, out_sample, width, label="Out-of-Sample R²")
        plt.xticks(x, stocks, rotation=45)
        plt.ylabel("R² Score")
        plt.title("In-Sample and Out-of-Sample R² Scores for CI Model")
        plt.legend()
        plt.tight_layout()
        plt.show()
    plot_r2_scores(copy.deepcopy(model_data))
    return (plot_r2_scores,)


@app.cell
def _(copy, model_data, pl, plt, sns):
    # Plot Lag Heatmap using Polars
    def plot_lag_heatmap(data):
        lag_data = []

        for stock, stock_data in data.items():
            ci_coefs = stock_data["fci_coef"]
            for coefs in ci_coefs:
                for key, value in coefs.items():
                    if "lag_" in key:
                        lag = int(key.split("_")[1])
                        lag_data.append(
                            {"Stock": stock, "Lag": lag, "Coefficient": abs(value)}
                        )

        df = pl.from_dicts(
            lag_data, schema={"Stock": str, "Lag": int, "Coefficient": float}
        )
        heatmap_data = (
            df.group_by(["Stock", "Lag"])
            .mean()
            .pivot(index="Stock", on="Lag", values="Coefficient")
        ).drop("Stock")
        # .to_pandas()
        # use Stock column as index
        # heatmap_data.set_index("Stock", inplace=True)

        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title("Average Magnitude of Cross-Impact Coefficients by Lag")
        plt.ylabel("Stock")
        plt.xlabel("Lag")
        plt.tight_layout()
        plt.show()
    plot_lag_heatmap(copy.deepcopy(model_data))
    return (plot_lag_heatmap,)


@app.cell
def _(ofi_data, pl, plt, sns):
    def plot_ofi_histograms(ofi_df: pl.DataFrame) -> None:
        ofi_df.group_by("symbol").agg(pl.all().mean())
        # Get all OFI level columns
        ofi_columns = [col for col in ofi_df.columns if col.startswith("ofi_level_")]
        ofi_columns.append("integrated_ofi")

        for col in ofi_columns:
            data = ofi_df[col].to_numpy()
            print(data.min(), data.max())
            plt.figure(figsize=(10, 6))
            sns.histplot(data, bins=1000, kde=True)
            plt.title(f"Histogram of {col}")
            plt.xlabel("OFI")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()

    plot_ofi_histograms(ofi_data)
    return (plot_ofi_histograms,)


@app.cell
def _(ofi_data, pl, plt, sns):
    def plot_ofi_heatmap(ofi_df: pl.DataFrame) -> None:
        # Average over all stocks
        ofi_df = (
            ofi_df.group_by("symbol").agg(pl.all().mean()).drop(["symbol", "timestamp"])
        ).drop("integrated_ofi")
        print(ofi_df)
        plt.figure(figsize=(10, 8))
        sns.heatmap(ofi_df.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap of OFI Levels (Average Across Stocks)")
        plt.tight_layout()
        plt.show()

    plot_ofi_heatmap(ofi_data)
    return (plot_ofi_heatmap,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
