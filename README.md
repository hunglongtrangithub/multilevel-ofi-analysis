# Cross-Impact Analysis of Order Flow Imbalance in Equity Markets

## Overview

This project investigates the cross-impact relationships of Order Flow Imbalance (OFI) within equity markets. Using data from ten major Nasdaq 100 stocks, we evaluate self-impact and cross-impact effects on price dynamics over short horizons. The analysis involves calculating multi-level OFIs, applying Principal Component Analysis (PCA) for dimensionality reduction, and employing regression models (linear and Lasso) to study contemporaneous and predictive impacts.

### Key Findings

- **Self-Impact Models:** Show modest predictive power in contemporaneous settings, with mean in-sample R\(^2\) values of 0.12.
- **Cross-Impact Models:** Displayed no significant predictive ability, with coefficients and out-of-sample R\(^2\) values near zero.
- **OFI Spread:** Variations across OFI levels highlight the importance of robust data preprocessing.
- **Predictive Models:** Lagged OFIs fail to provide meaningful predictions for future price changes.

## Installation

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

   This project uses uv as a package manager. If uv is not installed, follow the instructions in this [GitHub repository](https://github.com/astral-sh/uv).

2. Install required Python packages:
   ```bash
   uv sync
   ```

## Repository Contents:

- **`data/`**: Contains raw and processed data used in the project.
- **`scripts/`**:
  - `fetch_data.py`: Script for data acquisition.
  - `process_data.py`: Functions and classes for processing limit order book data and calculating OFI.
  - `models.py`: Implementation of regression models for self- and cross-impact analysis.
  - `analysis.py`: Scripts for training and evaluating model performance.
- **`notebooks/`**:
  - `eda.py`: Exploratory Data Analysis (EDA) of the dataset.
  - `visualize.py`: Code to generate visualizations, including heatmaps and performance plots.
- **`results/`**: Stores results, visualizations, and summary statistics.
- **`README.md`**: Project description and summary of findings (this file).

## Usage

1. Fetch the data:
   ```bash
   python scripts/fetch_data.py
   ```
2. Train and evaluate models:
   ```bash
   python analysis.py
   ```
3. This project uses [marimo](https://marimo.io/) for running notebooks. To run the local marimo server, execute the following command:
   ```bash
   marimo edit
   ```

## Contact

For any questions, feel free to reach out to the project maintainer.
