import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def _(__file__):
    from scripts.visualize import summarize_is_contemp_r2, summarize_os_contemp_r2, summarize_os_predictive_r2, summarize_ci_coefficients
    from pathlib import Path
    import json

    data_path = Path(__file__).parents[1] / "data"
    file_path = data_path / "models_results.json"
    with open(file_path, "r") as f:
        data = json.load(f)

    summarize_is_contemp_r2(data)
    return (
        Path,
        data,
        data_path,
        f,
        file_path,
        json,
        summarize_ci_coefficients,
        summarize_is_contemp_r2,
        summarize_os_contemp_r2,
        summarize_os_predictive_r2,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
