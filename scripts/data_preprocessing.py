"""
This script preprocesses macroeconomic data for analysis.
"""

import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter


def preprocess_data():
    def load_data(file_path):
        """
        Load data from an Excel file and return a DataFrame.

        @param file_path: Path to the Excel file.
        @return: DataFrame containing the data.
        """
        df = pd.read_excel(file_path)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df

    def monthly_to_quarterly(df):
        """
        Convert monthly data to quarterly data by taking the mean of each quarter.

        @param df: DataFrame containing the monthly data.
        @return: DataFrame with quarterly data.
        """
        quarterly_data = df.resample("QS").mean()
        return quarterly_data

    def apply_hp_filter(df, lamb=1600):
        """
        Apply the Hodrick-Prescott filter to the data.

        @param df: DataFrame containing the data to be filtered.
        @param lamb: Smoothing parameter for the HP filter.
        @return: DataFrame with the GDP gap.
        """
        cycle, trend = hpfilter(df, lamb=lamb)

        # GDP gap in percent = (Actual - Potential) / Potential * 100
        gdp_gap_percent = (df["GDP"] - trend) / trend * 100
        df["Potential_GDP"] = trend
        df["GDP_Gap"] = gdp_gap_percent
        return df

    def combine_data(dfs):
        """
        Combine multiple DataFrames into one.

        @param dfs: List of DataFrames to combine.
        @return: Combined DataFrame.
        """
        combined_df = pd.concat(dfs, axis=1)
        return combined_df

    # Load data from multiple files
    file_paths = [
        "../data/GDP.xlsx",
        "../data/EFFR.xlsx",
        "../data/PCE.xlsx",
    ]

    dataframes = {
        "GDP": load_data(file_paths[0]),
        "EFFR": load_data(file_paths[1]),
        "Inflation": load_data(file_paths[2]),
    }

    # Convert monthly data to quarterly data
    quarterlyFedFunds = monthly_to_quarterly(dataframes["EFFR"])

    # Apply HP filter to real GDP data
    filtered_realGDP = apply_hp_filter(dataframes["GDP"])

    # Percent change in Inflation
    dataframes["Inflation"]["PCE"] = dataframes["Inflation"]["PCE"].pct_change() * 100

    # Combine all data into one DataFrame
    combined_data = combine_data(
        [filtered_realGDP["GDP_Gap"], quarterlyFedFunds, dataframes["Inflation"]]
    )
    combined_data.columns = ["gdp_gap", "interest", "inflation"]
    combined_data.dropna(inplace=True)

    # Save the combined data to a new Excel file
    combined_data.to_excel("../data/us_macro_data.xlsx", index=True)


if __name__ == "__main__":
    preprocess_data()
    print("Data preprocessing completed and saved to data/us_macro_data.xlsx")
