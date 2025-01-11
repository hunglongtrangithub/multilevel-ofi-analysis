import polars as pl
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import seaborn as sns
from scipy import stats
from config import SYMBOLS, LAGS


# Generate Example Data
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


def summarize_ci_coefficients(data):
    rows = []
    for stock, stock_data in data.items():
        for model in ["CI"]:
            coef_samples = stock_data.get(f"{model.lower()}_coef", [])
            for coefs in coef_samples:
                self_coef = coefs.get(stock, 0)
                cross_coefs = [v for k, v in coefs.items() if k != stock]

                rows.append(
                    {
                        "Stock": stock,
                        "Model": model,
                        "Self Frequency": int(self_coef != 0),
                        "Self Magnitude": abs(self_coef),
                        "Cross Frequency": (
                            sum([1 if c != 0 else 0 for c in cross_coefs])
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


def plot_r2_scores(data):
    stocks = list(data.keys())
    in_sample = [np.mean(data[stock]["is_ci_r2"]) for stock in stocks]
    out_sample = [np.mean(data[stock]["os_ci_r2"]) for stock in stocks]

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
    ).to_pandas()
    # use Stock column as index
    heatmap_data.set_index("Stock", inplace=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Average Magnitude of Cross-Impact Coefficients by Lag")
    plt.ylabel("Stock")
    plt.xlabel("Lag")
    plt.tight_layout()
    plt.show()


def calculate_summary_statistics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate summary statistics for OFI levels and returns.
    All values are in basis points (bp).
    """

    # Function to calculate kurtosis (using Fisher definition to match paper)
    def kurt(x):
        return (
            stats.kurtosis(x, fisher=True) + 3
        )  # Adding 3 to get regular kurtosis instead of excess kurtosis

    # Get all OFI level columns
    ofi_columns = [col for col in df.columns if col.startswith("ofi_level_")]
    ofi_columns.append("integrated_ofi")

    # Initialize results list
    results = []

    # Calculate statistics for each OFI level
    for col in ofi_columns:
        values = df[col].to_numpy() * 10000  # Convert to basis points

        stats_dict = {
            "variable": col,
            "Mean (bp)": np.mean(values),
            "Std (bp)": np.std(values),
            "Skewness": stats.skew(values),
            "Kurtosis": kurt(values),
            "10% (bp)": np.percentile(values, 10),
            "25% (bp)": np.percentile(values, 25),
            "50% (bp)": np.percentile(values, 50),
            "75% (bp)": np.percentile(values, 75),
            "90% (bp)": np.percentile(values, 90),
        }
        results.append(stats_dict)

    # Convert to DataFrame
    summary_df = pl.DataFrame(results)

    # Round all numeric columns to 2 decimal places
    numeric_cols = [
        "Mean (bp)",
        "Std (bp)",
        "Skewness",
        "Kurtosis",
        "10% (bp)",
        "25% (bp)",
        "50% (bp)",
        "75% (bp)",
        "90% (bp)",
    ]

    for col in numeric_cols:
        summary_df = summary_df.with_columns(pl.col(col).round(2))

    return summary_df


def format_summary_table(df: pl.DataFrame) -> None:
    """
    Print the summary statistics in a format similar to the paper's table.
    """
    # Create header
    print("\nTable 1. Summary statistics of OFIs and returns.")
    print("-" * 120)
    print(
        f"{'Variable':<12} {'Mean (bp)':>10} {'Std (bp)':>10} {'Skewness':>10} {'Kurtosis':>10} "
        f"{'10% (bp)':>10} {'25% (bp)':>10} {'50% (bp)':>10} {'75% (bp)':>10} {'90% (bp)':>10}"
    )
    print("-" * 120)

    # Print each row
    for row in df.iter_rows(named=True):
        var_name = row["variable"]
        # Format variable name to match paper style (e.g., "ofi_level_01" -> "ofi¹")
        if var_name.startswith("ofi_level_"):
            level = int(var_name.split("_")[-1])
            var_name = f"ofi{level},(1m)"
        elif var_name == "integrated_ofi":
            var_name = "ofiᴵ,(1m)"

        print(
            f"{var_name:<12} "
            f"{row['Mean (bp)']:>10.2f} {row['Std (bp)']:>10.2f} {row['Skewness']:>10.2f} "
            f"{row['Kurtosis']:>10.2f} {row['10% (bp)']:>10.2f} {row['25% (bp)']:>10.2f} "
            f"{row['50% (bp)']:>10.2f} {row['75% (bp)']:>10.2f} {row['90% (bp)']:>10.2f}"
        )

    print("-" * 120)
    print(
        "Note: These statistics are computed at the minute level across each stock and the full sample period. 1bp = 0.0001 = 0.01%."
    )


# Usage example:
def generate_ofi_summary(results_df: pl.DataFrame) -> None:
    """
    Generate and display summary statistics for OFI data.
    """
    # Calculate summary statistics
    summary_stats = calculate_summary_statistics(results_df)

    # Display formatted table
    format_summary_table(summary_stats)

    # Optionally save to file
    summary_stats.write_csv("ofi_summary_statistics.csv")


# Example usage
def main():
    import json
    from pathlib import Path

    # Load data (replace with actual JSON file path)
    file_path = Path(__file__).parents[1] / "data" / "models_results.json"
    with open(file_path, "r") as f:
        data = json.load(f)

    is_contemp_r2_summary = summarize_is_contemp_r2(data)
    print(
        "In-sample R2 score for contemporaneous models (self-impact and cross-impact):"
    )
    print(is_contemp_r2_summary)

    os_contemp_r2_summary = summarize_os_contemp_r2(data)
    print(
        "Out-of-sample R2 score for contemporaneous models (self-impact and cross-impact):"
    )
    print(os_contemp_r2_summary)

    os_predictive_r2_summary = summarize_os_predictive_r2(data)
    print(
        "Out-of-sample R2 score for predictive models (self-impact and cross-impact):"
    )
    print(os_predictive_r2_summary)

    # Generate and plot CI network
    threshold = 95  # Percentile
    top_k = 50  # Top edges by weight
    G_ci = generate_network(data, threshold, top_k, model="CI")
    plot_network(G_ci, title="Averaged Cross-Impact Network (CI)")

    # Generate and plot FCI network
    G_fci = generate_network(data, threshold, top_k, model="FCI")
    plot_network(G_fci, title="Averaged Forward-Looking Cross-Impact Network (FCI)")

    plot_r2_scores(data)

    plot_lag_heatmap(data)


if __name__ == "__main__":
    main()
    # glue = sns.load_dataset("glue").pivot(index="Model", columns="Task", values="Score")
    # print(glue)
