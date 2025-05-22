"""
Enhanced evaluation script for DDPG monetary policy agent.
Provides comprehensive analysis including:
- Historical interest rate comparison
- Taylor rule benchmarking
- Period-specific performance (recession vs normal times)
- Policy response surface visualization
- Counterfactual simulations
- Inflation and GDP targeting performance
- Multiple agent comparison
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from datetime import datetime
from matplotlib import cm
import seaborn as sns
from actor_critic import Actor
from environment_ann import EnvironmentANN
import json


def load_actor(
    state_dim, path="../models/trained_actor.pth", hidden_dims=(256, 128, 64)
):
    """Load a trained actor model"""
    model = Actor(input_dim=state_dim, hidden_dims=hidden_dims)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def taylor_rule(inflation, gdp_gap, r_neutral=2.0, phi_pi=1.5, phi_y=0.5):
    """Standard Taylor rule for monetary policy"""
    return max(0, r_neutral + phi_pi * (inflation - 2.0) + phi_y * gdp_gap)


def create_output_directory():
    """Create timestamped directory for evaluation outputs"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"../evaluation_results/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {output_dir}")
    return output_dir


def partial_dependence_surface(
    actor, num_lags, output_dir, compare_taylor=True, include_heatmap=True
):
    """
    Create 3D surface plots showing how interest rates respond to
    inflation and output gap.

    Args:
        actor: Trained actor model
        num_lags: Number of lags in state representation
        output_dir: Directory to save plots
        compare_taylor: Whether to plot Taylor rule for comparison
        include_heatmap: Whether to include 2D heatmap view
    """
    infl_range = np.linspace(0, 6, 50)  # Expanded range
    gdp_range = np.linspace(-6, 6, 50)  # Expanded range
    grid_x, grid_y = np.meshgrid(infl_range, gdp_range)

    # Calculate DDPG agent rates
    agent_rates = np.zeros_like(grid_x)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            state = []
            state += [grid_y[i, j]] + [0.0] * (num_lags - 1)  # gdp_gap lags
            state += [grid_x[i, j]] + [2.0] * (num_lags - 1)  # inflation lags
            state += [2.0] * num_lags  # interest lags (neutral rate)
            x = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                agent_rates[i, j] = actor(x).item()

    # Calculate Taylor rule rates for comparison
    if compare_taylor:
        taylor_rates = np.zeros_like(grid_x)
        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                taylor_rates[i, j] = taylor_rule(grid_x[i, j], grid_y[i, j])

    # 3D Surface plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        grid_x, grid_y, agent_rates, cmap="viridis", alpha=0.8, edgecolor="none"
    )

    if compare_taylor:
        ax.plot_wireframe(
            grid_x, grid_y, taylor_rates, color="red", alpha=0.5, label="Taylor Rule"
        )

    ax.set_xlabel("Inflation (%)")
    ax.set_ylabel("Output Gap (%)")
    ax.set_zlabel("Interest Rate (%)")
    ax.set_title("Interest Rate Response Surface")

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    if compare_taylor:
        ax.legend()

    # Save 3D surface
    plt.savefig(f"{output_dir}/policy_surface_3d.png", dpi=300)
    plt.close()

    # Create 2D heatmap if requested
    if include_heatmap:
        plt.figure(figsize=(10, 8))

        # Agent policy heatmap
        plt.subplot(1, 2 if compare_taylor else 1, 1)
        im = plt.pcolormesh(grid_x, grid_y, agent_rates, cmap="viridis", shading="auto")
        plt.colorbar(im, label="Interest Rate (%)")
        plt.xlabel("Inflation (%)")
        plt.ylabel("Output Gap (%)")
        plt.title("DDPG Agent Policy")

        # Overlay contour lines
        contours = plt.contour(
            grid_x, grid_y, agent_rates, colors="white", alpha=0.5, levels=5
        )
        plt.clabel(contours, inline=True, fontsize=8)

        # Add Taylor rule comparison if requested
        if compare_taylor:
            plt.subplot(1, 2, 2)
            im2 = plt.pcolormesh(
                grid_x, grid_y, taylor_rates, cmap="viridis", shading="auto"
            )
            plt.colorbar(im2, label="Interest Rate (%)")
            plt.xlabel("Inflation (%)")
            plt.ylabel("Output Gap (%)")
            plt.title("Taylor Rule Policy")

            # Overlay contour lines
            contours2 = plt.contour(
                grid_x, grid_y, taylor_rates, colors="white", alpha=0.5, levels=5
            )
            plt.clabel(contours2, inline=True, fontsize=8)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/policy_heatmap.png", dpi=300)

        # Also create a difference heatmap
        if compare_taylor:
            plt.figure(figsize=(8, 6))
            diff = agent_rates - taylor_rates
            im3 = plt.pcolormesh(
                grid_x, grid_y, diff, cmap="RdBu_r", shading="auto", vmin=-2, vmax=2
            )
            plt.colorbar(im3, label="Difference in Interest Rate (%)")
            plt.xlabel("Inflation (%)")
            plt.ylabel("Output Gap (%)")
            plt.title("Policy Difference: DDPG minus Taylor Rule")

            # Overlay contour at zero
            zero_contour = plt.contour(
                grid_x, grid_y, diff, colors="black", levels=[0], linestyles="dashed"
            )
            plt.clabel(zero_contour, inline=True, fontsize=8)

            plt.tight_layout()
            plt.savefig(f"{output_dir}/policy_difference.png", dpi=300)

        plt.close()


def compare_historical_performance(
    df,
    actor,
    num_lags,
    output_dir,
    include_taylor=True,
    r_neutral=2.0,
    phi_pi=1.5,
    phi_y=0.5,
):
    """
    Compare the agent's decisions with historical data and Taylor rule.

    Args:
        df: DataFrame with historical data
        actor: Trained actor model
        num_lags: Number of lags in state
        output_dir: Directory to save results
        include_taylor: Whether to include Taylor rule comparison
        r_neutral: Neutral rate for Taylor rule
        phi_pi: Inflation coefficient for Taylor rule
        phi_y: Output gap coefficient for Taylor rule
    """
    variables = ["gdp_gap", "inflation", "interest"]
    state_dim = num_lags * len(variables)

    # Prepare for storing results
    dates = (
        df.index[num_lags:] if hasattr(df.index, "date") else range(num_lags, len(df))
    )
    agent_rates = []
    true_rates = []
    taylor_rates = [] if include_taylor else None

    # Also collect economic conditions
    inflation_series = []
    gdp_gap_series = []

    # Generate decisions
    for t in range(num_lags, len(df)):
        window = df.iloc[t - num_lags : t]

        # Current economic conditions (most recent values)
        current_inflation = window["inflation"].iloc[-1]
        current_gdp_gap = window["gdp_gap"].iloc[-1]

        # Store economic conditions
        inflation_series.append(current_inflation)
        gdp_gap_series.append(current_gdp_gap)

        # Prepare state for agent
        state = []
        for var in variables:
            state += list(window[var].values[::-1])

        # Get agent's decision
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            rate = actor(state_tensor).item()

        agent_rates.append(rate)
        true_rates.append(df.iloc[t]["interest"])

        # Calculate Taylor rule rate if requested
        if include_taylor:
            taylor_rate = taylor_rule(
                current_inflation,
                current_gdp_gap,
                r_neutral=r_neutral,
                phi_pi=phi_pi,
                phi_y=phi_y,
            )
            taylor_rates.append(taylor_rate)

    # Create results for analysis
    results_df = pd.DataFrame(
        {
            "Date": dates,
            "Actual_Rate": true_rates,
            "Agent_Rate": agent_rates,
            "Inflation": inflation_series,
            "GDP_Gap": gdp_gap_series,
        }
    )

    if include_taylor:
        results_df["Taylor_Rate"] = taylor_rates

    # Calculate error metrics
    results_df["Agent_Error"] = results_df["Agent_Rate"] - results_df["Actual_Rate"]
    if include_taylor:
        results_df["Taylor_Error"] = (
            results_df["Taylor_Rate"] - results_df["Actual_Rate"]
        )

    # Save results to CSV
    results_df.to_csv(f"{output_dir}/historical_comparison.csv", index=False)

    # Plot historical comparison
    plt.figure(figsize=(14, 6))
    plt.plot(
        results_df["Actual_Rate"],
        label="Actual Interest Rate",
        linestyle="-",
        color="black",
        linewidth=2,
    )
    plt.plot(
        results_df["Agent_Rate"], label="DDPG Interest Rate", color="blue", linewidth=2
    )

    if include_taylor:
        plt.plot(
            results_df["Taylor_Rate"], label="Taylor Rule", linestyle="--", color="red"
        )

    plt.title("Actual vs. Predicted Interest Rates")
    plt.xlabel("Time Step")
    plt.ylabel("Interest Rate (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/interest_rate_comparison.png", dpi=300)
    plt.close()

    # Create a second plot showing economic conditions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top panel: Interest rates
    ax1.plot(
        results_df["Actual_Rate"], label="Actual Interest", color="black", linewidth=2
    )
    ax1.plot(results_df["Agent_Rate"], label="DDPG Policy", color="blue", linewidth=2)
    if include_taylor:
        ax1.plot(
            results_df["Taylor_Rate"], label="Taylor Rule", linestyle="--", color="red"
        )
    ax1.set_ylabel("Interest Rate (%)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Economic conditions
    ax2.plot(results_df["Inflation"], label="Inflation", color="orange", linewidth=2)
    ax2.plot(results_df["GDP_Gap"], label="Output Gap", color="green", linewidth=2)
    ax2.axhline(y=2.0, color="red", linestyle="--", alpha=0.5, label="Inflation Target")
    ax2.axhline(
        y=0.0, color="green", linestyle="--", alpha=0.5, label="Zero Output Gap"
    )
    ax2.set_ylabel("Percent (%)")
    ax2.set_xlabel("Time Step")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/economic_conditions.png", dpi=300)
    plt.close()

    # Calculate and return performance metrics
    metrics = {
        "agent_mae": np.mean(np.abs(results_df["Agent_Error"])),
        "agent_rmse": np.sqrt(np.mean(np.square(results_df["Agent_Error"]))),
    }

    if include_taylor:
        metrics.update(
            {
                "taylor_mae": np.mean(np.abs(results_df["Taylor_Error"])),
                "taylor_rmse": np.sqrt(np.mean(np.square(results_df["Taylor_Error"]))),
            }
        )

    # Save metrics to file
    with open(f"{output_dir}/historical_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return results_df, metrics


def segment_analysis(df, actor, num_lags, output_dir):
    """
    Analyze agent performance in different economic regimes.

    Args:
        df: DataFrame with historical data
        actor: Trained actor model
        num_lags: Number of lags in state
        output_dir: Directory to save results
    """
    # First run the general historical comparison
    results_df, _ = compare_historical_performance(
        df, actor, num_lags, output_dir, include_taylor=True
    )

    # Now create segmentation criteria
    results_df["Recession"] = results_df["GDP_Gap"] < -1.0
    results_df["High_Inflation"] = results_df["Inflation"] > 3.0
    results_df["Low_Inflation"] = results_df["Inflation"] < 1.0
    results_df["Normal"] = (
        ~results_df["Recession"]
        & ~results_df["High_Inflation"]
        & ~results_df["Low_Inflation"]
    )

    # Calculate metrics for each segment
    segments = ["Recession", "High_Inflation", "Low_Inflation", "Normal"]
    segment_metrics = {}

    for segment in segments:
        segment_data = results_df[results_df[segment]]
        if len(segment_data) == 0:
            segment_metrics[segment] = {"count": 0}
            continue

        segment_metrics[segment] = {
            "count": len(segment_data),
            "agent_mae": np.mean(np.abs(segment_data["Agent_Error"])),
            "agent_rmse": np.sqrt(np.mean(np.square(segment_data["Agent_Error"]))),
            "taylor_mae": np.mean(np.abs(segment_data["Taylor_Error"])),
            "taylor_rmse": np.sqrt(np.mean(np.square(segment_data["Taylor_Error"]))),
            "avg_actual_rate": segment_data["Actual_Rate"].mean(),
            "avg_agent_rate": segment_data["Agent_Rate"].mean(),
            "avg_taylor_rate": segment_data["Taylor_Rate"].mean(),
        }

    # Save segment metrics
    with open(f"{output_dir}/segment_metrics.json", "w") as f:
        json.dump(segment_metrics, f, indent=2)

    # Create a bar chart comparing agent vs Taylor performance in each segment
    segments_with_data = [s for s in segments if segment_metrics[s].get("count", 0) > 0]

    if len(segments_with_data) > 0:
        x = np.arange(len(segments_with_data))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        agent_mae = [segment_metrics[s]["agent_mae"] for s in segments_with_data]
        taylor_mae = [segment_metrics[s]["taylor_mae"] for s in segments_with_data]

        rects1 = ax.bar(x - width / 2, agent_mae, width, label="DDPG Agent")
        rects2 = ax.bar(x + width / 2, taylor_mae, width, label="Taylor Rule")

        ax.set_ylabel("Mean Absolute Error")
        ax.set_title("Performance by Economic Regime")
        ax.set_xticks(x)
        ax.set_xticklabels(segments_with_data)
        ax.legend()

        # Add count labels
        for i, s in enumerate(segments_with_data):
            ax.annotate(
                f"n={segment_metrics[s]['count']}",
                xy=(i, max(agent_mae[i], taylor_mae[i]) + 0.1),
                ha="center",
            )

        plt.tight_layout()
        plt.savefig(f"{output_dir}/segment_performance.png", dpi=300)
        plt.close()

    return segment_metrics


def impulse_response_analysis(actor, inflation_model, gdp_model, num_lags, output_dir):
    """
    Generate impulse response functions showing how the agent responds to
    economic shocks and how these propagate through the economy.

    Args:
        actor: Trained actor model
        inflation_model: Model to predict inflation
        gdp_model: Model to predict GDP gap
        num_lags: Number of lags in state
        output_dir: Directory to save results
    """
    variables = ["gdp_gap", "inflation", "interest"]
    steps = 20  # Simulate 20 steps ahead

    # Set up neutral baseline state
    baseline_state = []
    baseline_state += [0.0] * num_lags  # gdp_gap at equilibrium
    baseline_state += [2.0] * num_lags  # inflation at target
    baseline_state += [2.0] * num_lags  # interest at neutral rate

    # Run baseline simulation
    baseline_results = run_simulation(
        actor, inflation_model, gdp_model, baseline_state, steps, variables, num_lags
    )

    # Inflation shock (+2%)
    inflation_shock_state = baseline_state.copy()
    inflation_shock_state[num_lags] = 4.0  # Set current inflation to 4%

    inflation_shock_results = run_simulation(
        actor,
        inflation_model,
        gdp_model,
        inflation_shock_state,
        steps,
        variables,
        num_lags,
    )

    # Negative GDP shock (-2%)
    gdp_shock_state = baseline_state.copy()
    gdp_shock_state[0] = -2.0  # Set current GDP gap to -2%

    gdp_shock_results = run_simulation(
        actor, inflation_model, gdp_model, gdp_shock_state, steps, variables, num_lags
    )

    # Plot results
    # First inflation shock
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(
        inflation_shock_results["inflation"],
        label="Inflation Shock",
        color="red",
        linewidth=2,
    )
    plt.plot(
        baseline_results["inflation"], label="Baseline", color="blue", linestyle="--"
    )
    plt.axhline(y=2.0, color="black", linestyle=":", label="Target")
    plt.ylabel("Inflation (%)")
    plt.title("Inflation Shock: Inflation")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(
        inflation_shock_results["gdp_gap"],
        label="Inflation Shock",
        color="red",
        linewidth=2,
    )
    plt.plot(
        baseline_results["gdp_gap"], label="Baseline", color="blue", linestyle="--"
    )
    plt.axhline(y=0.0, color="black", linestyle=":", label="Equilibrium")
    plt.ylabel("GDP Gap (%)")
    plt.title("Inflation Shock: GDP Gap")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(
        inflation_shock_results["interest"],
        label="Inflation Shock",
        color="red",
        linewidth=2,
    )
    plt.plot(
        baseline_results["interest"], label="Baseline", color="blue", linestyle="--"
    )
    plt.ylabel("Interest Rate (%)")
    plt.xlabel("Time Step")
    plt.title("Inflation Shock: Interest Rate Response")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/inflation_shock_response.png", dpi=300)
    plt.close()

    # GDP shock
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(
        gdp_shock_results["inflation"], label="GDP Shock", color="green", linewidth=2
    )
    plt.plot(
        baseline_results["inflation"], label="Baseline", color="blue", linestyle="--"
    )
    plt.axhline(y=2.0, color="black", linestyle=":", label="Target")
    plt.ylabel("Inflation (%)")
    plt.title("Negative GDP Shock: Inflation")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(
        gdp_shock_results["gdp_gap"], label="GDP Shock", color="green", linewidth=2
    )
    plt.plot(
        baseline_results["gdp_gap"], label="Baseline", color="blue", linestyle="--"
    )
    plt.axhline(y=0.0, color="black", linestyle=":", label="Equilibrium")
    plt.ylabel("GDP Gap (%)")
    plt.title("Negative GDP Shock: GDP Gap")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(
        gdp_shock_results["interest"], label="GDP Shock", color="green", linewidth=2
    )
    plt.plot(
        baseline_results["interest"], label="Baseline", color="blue", linestyle="--"
    )
    plt.ylabel("Interest Rate (%)")
    plt.xlabel("Time Step")
    plt.title("Negative GDP Shock: Interest Rate Response")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/gdp_shock_response.png", dpi=300)
    plt.close()

    return {
        "baseline": baseline_results,
        "inflation_shock": inflation_shock_results,
        "gdp_shock": gdp_shock_results,
    }


def run_simulation(
    actor, inflation_model, gdp_model, initial_state, steps, variables, num_lags
):
    """
    Run a simulation of the economy under the agent's policy.

    Args:
        actor: Trained actor model
        inflation_model: Model to predict inflation
        gdp_model: Model to predict GDP gap
        initial_state: Initial state of the economy
        steps: Number of steps to simulate
        variables: List of state variables
        num_lags: Number of lags in state

    Returns:
        Dictionary with simulation results
    """
    state = initial_state.copy()

    # Initialize results
    results = {"gdp_gap": [], "inflation": [], "interest": []}

    for _ in range(steps):
        # Get agent's policy decision
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            interest_rate = actor(state_tensor).item()

        # Predict next economic state
        next_input = np.array(state).reshape(1, -1)
        next_inflation = inflation_model.predict(next_input)[0]
        next_gdp_gap = gdp_model.predict(next_input)[0]

        # Store results
        results["gdp_gap"].append(next_gdp_gap)
        results["inflation"].append(next_inflation)
        results["interest"].append(interest_rate)

        # Update state for next step
        next_state = []
        # Update GDP gap lags
        next_state += [next_gdp_gap] + state[: num_lags - 1]
        # Update inflation lags
        next_state += [next_inflation] + state[num_lags : 2 * num_lags - 1]
        # Update interest rate lags
        next_state += [interest_rate] + state[2 * num_lags : 3 * num_lags - 1]

        state = next_state

    return results


def compare_multiple_agents(df, model_paths, model_names, num_lags, output_dir):
    """
    Compare multiple trained agents against each other.

    Args:
        df: DataFrame with historical data
        model_paths: List of paths to trained models
        model_names: List of names for the models
        num_lags: Number of lags in state
        output_dir: Directory to save results
    """
    if len(model_paths) != len(model_names):
        raise ValueError("Number of model paths must match number of model names")

    variables = ["gdp_gap", "inflation", "interest"]
    state_dim = num_lags * len(variables)

    # Load all models
    actors = []
    for path in model_paths:
        actor = load_actor(state_dim=state_dim, path=path)
        actors.append(actor)

    # Prepare data structures
    results = {}
    for name in model_names:
        results[name] = []

    # Historical data
    results["Actual"] = []
    results["Taylor"] = []

    # Economic conditions
    conditions = {"Inflation": [], "GDP_Gap": []}

    # Generate predictions for each agent
    for t in range(num_lags, len(df)):
        window = df.iloc[t - num_lags : t]

        # Current economic conditions
        current_inflation = window["inflation"].iloc[-1]
        current_gdp_gap = window["gdp_gap"].iloc[-1]

        conditions["Inflation"].append(current_inflation)
        conditions["GDP_Gap"].append(current_gdp_gap)

        # Prepare state
        state = []
        for var in variables:
            state += list(window[var].values[::-1])

        # Get each agent's decision
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        for i, actor in enumerate(actors):
            with torch.no_grad():
                rate = actor(state_tensor).item()
            results[model_names[i]].append(rate)

        # Actual rate
        results["Actual"].append(df.iloc[t]["interest"])

        # Taylor rule
        results["Taylor"].append(taylor_rule(current_inflation, current_gdp_gap))

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df["Inflation"] = conditions["Inflation"]
    results_df["GDP_Gap"] = conditions["GDP_Gap"]

    # Save to CSV
    results_df.to_csv(f"{output_dir}/multiple_agents_comparison.csv", index=False)

    # Plot comparison
    plt.figure(figsize=(14, 7))

    # Plot historical rate
    plt.plot(results_df["Actual"], label="Historical", color="black", linewidth=2)

    # Plot each agent
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    for i, name in enumerate(model_names):
        plt.plot(results_df[name], label=name, color=colors[i], linewidth=1.5)

    # Plot Taylor rule
    plt.plot(
        results_df["Taylor"],
        label="Taylor Rule",
        color="red",
        linestyle="--",
        alpha=0.7,
    )

    plt.title("Comparison of Multiple Policy Models")
    plt.xlabel("Time Step")
    plt.ylabel("Interest Rate (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/multiple_agents_comparison.png", dpi=300)
    plt.close()

    # Calculate metrics for each model
    metrics = {}
    for name in model_names + ["Taylor"]:
        metrics[name] = {
            "mae": np.mean(np.abs(results_df[name] - results_df["Actual"])),
            "rmse": np.sqrt(
                np.mean(np.square(results_df[name] - results_df["Actual"]))
            ),
        }

    # Save metrics
    with open(f"{output_dir}/multi_agent_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return results_df, metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate DDPG monetary policy agent")
    parser.add_argument(
        "--data", type=str, default="../data/us_macro_data.xlsx", help="Path to dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="../models/trained_actor.pth",
        help="Path to trained model",
    )
    parser.add_argument("--lags", type=int, default=4, help="Number of lags in state")
    parser.add_argument(
        "--compare_taylor", action="store_true", help="Compare with Taylor rule"
    )
    parser.add_argument(
        "--run_all", action="store_true", help="Run all evaluation types"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = create_output_directory()

    # Load data
    print(f"Loading data from {args.data}")
    df = pd.read_excel(args.data)

    # Compute state dimension
    variables = ["gdp_gap", "inflation", "interest"]
    state_dim = len(variables) * args.lags

    # Load actor model
    print(f"Loading model from {args.model}")
    actor = load_actor(state_dim=state_dim, path=args.model)

    # Save settings
    with open(f"{output_dir}/settings.json", "w") as f:
        json.dump(
            {
                "data_path": args.data,
                "model_path": args.model,
                "num_lags": args.lags,
                "state_dim": state_dim,
                "compare_taylor": args.compare_taylor,
                "run_all": args.run_all,
            },
            f,
            indent=2,
        )

    # Run policy surface analysis
    print("Generating policy response surface...")
    partial_dependence_surface(
        actor, args.lags, output_dir, compare_taylor=args.compare_taylor or args.run_all
    )

    # Run historical comparison
    print("Comparing against historical data...")
    compare_historical_performance(
        df,
        actor,
        args.lags,
        output_dir,
        include_taylor=args.compare_taylor or args.run_all,
    )

    # Run segment analysis
    if args.run_all:
        print("Analyzing performance across different economic regimes...")
        segment_analysis(df, actor, args.lags, output_dir)

    # Run impulse response analysis if environment models are available
    if args.run_all:
        try:
            print("Loading environment models for impulse response analysis...")
            # Try to load environment models
            inflation_model = EnvironmentANN(df, "inflation", variables, lags=args.lags)
            gdp_model = EnvironmentANN(df, "gdp_gap", variables, lags=args.lags)

            # Train models with default parameters if needed
            inflation_model.train()
            gdp_model.train()

            print("Running impulse response analysis...")
            impulse_response_analysis(
                actor, inflation_model, gdp_model, args.lags, output_dir
            )
        except Exception as e:
            print(f"Could not run impulse response analysis: {e}")

    # Compare multiple agents if multiple models are available
    if args.run_all:
        # Check if there are other trained models
        model_dir = os.path.dirname(args.model)
        other_models = [
            os.path.join(model_dir, f)
            for f in os.listdir(model_dir)
            if f.endswith(".pth") and f != os.path.basename(args.model)
        ]

        if other_models:
            all_models = [args.model] + other_models
            model_names = ["Primary"] + [
                f"Model_{i + 1}" for i in range(len(other_models))
            ]

            print(f"Comparing {len(all_models)} different models...")
            compare_multiple_agents(df, all_models, model_names, args.lags, output_dir)

    print(f"Evaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
