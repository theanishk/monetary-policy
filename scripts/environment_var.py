# Environment Estimation using VAR
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
import pandas as pd


class EnvironmentVAR:
    def __init__(self, df, variables, lags=2):
        self.df = df[variables].dropna().copy()
        self.variables = variables
        self.lags = lags
        self.model_fitted = None

    def fit_model(self):
        model = VAR(self.df)
        self.model_fitted = model.fit(self.lags)
        print(self.model_fitted.summary())
        return self.model_fitted

    def forecast(self, steps=5):
        if self.model_fitted is None:
            raise ValueError("Model not fitted yet.")
        forecast_input = self.df.values[-self.lags :]
        forecast = self.model_fitted.forecast(y=forecast_input, steps=steps)
        forecast_df = pd.DataFrame(forecast, columns=self.variables)
        return forecast_df

    def plot_forecast(self, forecast_df):
        plt.figure(figsize=(12, 6))
        for var in self.variables:
            plt.plot(self.df[var].values, label=f"Actual {var}")
            plt.plot(
                range(len(self.df), len(self.df) + len(forecast_df)),
                forecast_df[var],
                label=f"Forecast {var}",
                linestyle="--",
            )
        plt.title("VAR Model Forecast")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    data = pd.read_excel("../data/us_macro_data.xlsx")
    variables = ["gdp_gap", "inflation", "interest"]
    env_var = EnvironmentVAR(df=data, variables=variables, lags=4)
    env_var.fit_model()

    env_var.plot_forecast(env_var.forecast(steps=5))
