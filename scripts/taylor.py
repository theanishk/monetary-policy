# Using Taylor (1993) to calculate nominal interest rates
# Nominal Interest Rate = r* + π + φπ(π - π*) + φy(y - y*)

import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")


class TaylorRuleModel:
    def __init__(
        self,
        data,
        inflation_col="inflation",
        gdp_gap_col="gdp_gap",
        actual_rate_col="interest",
        r_star=2.0,
        pi_star=2.0,
        phi_pi=0.5,
        phi_y=0.5,
    ):
        """
        Initializes the Taylor Rule model with the dataset and parameters.

        @param data: DataFrame containing the economic data
        @param inflation_col: Column name for inflation rate
        @param gdp_gap_col: Column name for GDP gap
        @param actual_rate_col: Column name for actual Fed rate
        @param r_star: Neutral real interest rate (default is 2.0%)
        @param pi_star: Target inflation rate (default is 2.0%)
        @param phi_pi: Coefficient for inflation (default is 0.5)
        @param phi_y: Coefficient for GDP gap (default is 0.5)
        """
        self.df = data.copy()
        self.inflation_col = inflation_col
        self.gdp_gap_col = gdp_gap_col
        self.actual_rate_col = actual_rate_col
        self.r_star = r_star
        self.pi_star = pi_star
        self.phi_pi = phi_pi
        self.phi_y = phi_y

    def compute_taylor_rate(self):
        """Computes the Taylor Rule nominal interest rate and adds it to the DataFrame."""
        self.df["Taylor_Rule_Rate"] = (
            self.r_star
            + self.df[self.inflation_col]
            + self.phi_pi * (self.df[self.inflation_col] - self.pi_star)
            + self.phi_y * self.df[self.gdp_gap_col]
        )
        return self.df

    def compute_policy_gap(self):
        """Computes the policy gap: actual Fed rate minus Taylor Rule rate."""
        if "Taylor_Rule_Rate" not in self.df.columns:
            self.compute_taylor_rate()
        self.df["Policy_Gap"] = (
            self.df[self.actual_rate_col] - self.df["Taylor_Rule_Rate"]
        )
        return self.df

    def plot_rates(self, save_path=None):
        """Plots actual vs Taylor Rule interest rates over time."""
        if "Taylor_Rule_Rate" not in self.df.columns:
            self.compute_taylor_rate()

        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot using date as x-axis
        ax.plot(
            self.df["date"],
            self.df[self.actual_rate_col],
            label="Actual_Rate",
            color="blue",
        )
        ax.plot(self.df["date"], self.df["Taylor_Rule_Rate"], label="Taylor_Rule_Rate")

        ax.set_title("Actual vs Taylor Rule Interest Rate")
        ax.set_ylabel("Interest Rate (%)")
        ax.grid(True)
        ax.legend()

        # Format x-axis for dates if needed
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")

        plt.show()

    def get_results(self, save_path=None):
        """Returns the internal DataFrame with all computed columns."""
        if save_path:
            self.df.to_excel(save_path, index=False)
            print(f"Results saved to {save_path}")

        return self.df


if __name__ == "__main__":
    df = pd.read_excel(r"../data/us_macro_data.xlsx")
    model = TaylorRuleModel(df, phi_pi=0.5, phi_y=0.5)

    model.compute_taylor_rate()
    model.compute_policy_gap()

    model.plot_rates(save_path="../figures/taylor_rule_plot.png")

    # Final dataset with all computed columns
    final_df = model.get_results(save_path="../data/taylor_rule_results.xlsx")
