"""
Enhanced training script for DDPG agent to control interest rates based on macroeconomic indicators.
Includes advanced training techniques:
- Curriculum learning with gradually increasing episode complexity
- Adaptive reward scaling for more stable gradients
- Comprehensive policy evaluation with economic metrics
- Learning rate scheduling and warm-up
- Automated hyperparameter optimization
- Advanced visualization and monitoring
- Improved reward formulation for monetary policy learning
- Action smoothness penalty to discourage volatile interest rates
"""

import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from environment_ann import EnvironmentANN
from ddpg_agent import EnhancedDDPGAgent
from tqdm import tqdm
import json
from datetime import datetime


def compute_reward(
    inflation,
    gdp_gap,
    pi_target=2.0,
    lambda_pi=1.5,
    lambda_y=0.5,
    scale_factor=1.0,
    prev_action=None,
    curr_action=None,
    smoothness_penalty=0.2,
):
    """
    Compute reward with improved scaling and smoother gradients.

    Args:
        inflation: Current inflation
        gdp_gap: Current output gap
        pi_target: Inflation target (usually 2%)
        lambda_pi: Weight on inflation objective
        lambda_y: Weight on output gap objective
        scale_factor: Reward scaling factor
        prev_action: Previous interest rate (for smoothness penalty)
        curr_action: Current interest rate (for smoothness penalty)
        smoothness_penalty: Weight for action smoothness penalty
    """
    # Use squared error for monetary policy objectives
    inflation_loss = (inflation - pi_target) ** 2
    output_loss = gdp_gap**2

    # Basic central bank loss function
    policy_loss = lambda_pi * inflation_loss + lambda_y * output_loss

    # Add smoothness penalty if both actions are provided
    if prev_action is not None and curr_action is not None:
        rate_change = abs(curr_action - prev_action)
        smoothness_loss = smoothness_penalty * rate_change
        total_loss = policy_loss + smoothness_loss
    else:
        total_loss = policy_loss

    # Log-based reward formulation (grows more slowly, better gradients)
    # Using natural log (ln) with offset to prevent taking ln(0)
    reward = -scale_factor * np.log(1.0 + total_loss)

    return reward


def evaluate_policy(
    agent,
    env_infl,
    env_gdp,
    variables,
    df,
    num_lags,
    episodes=10,
    steps_per_episode=100,
    verbose=False,
    pi_target=2.0,
    lambda_pi=1.5,
    lambda_y=0.5,
):
    """
    Comprehensive evaluation of the current policy.
    """
    total_rewards = []
    inflation_mse = []
    gdp_gap_mse = []
    inflation_targets = []  # Track inflation target achievement
    interest_volatility = []  # Track interest rate volatility

    for ep in range(episodes):
        # Random starting point
        idx = np.random.randint(num_lags, len(df) - steps_per_episode)
        window = df.iloc[idx - num_lags : idx]

        # Initialize state
        state = []
        for var in variables:
            state += list(window[var].values[::-1])

        # Episode tracking
        episode_reward = 0
        episode_inflations = []
        episode_gdp_gaps = []
        episode_interest_rates = []
        prev_interest_rate = None

        for step in range(steps_per_episode):
            # Use deterministic policy for evaluation
            action = agent.select_action(np.array(state), add_noise=False)
            interest_rate = float(action[0])
            episode_interest_rates.append(interest_rate)

            # Predict next state
            next_input = np.array(state).reshape(1, -1)
            next_inflation = env_infl.predict(next_input)[0]
            next_gdp_gap = env_gdp.predict(next_input)[0]

            # Store predictions
            episode_inflations.append(next_inflation)
            episode_gdp_gaps.append(next_gdp_gap)

            # Compute reward with smoothness penalty
            reward = compute_reward(
                next_inflation,
                next_gdp_gap,
                pi_target,
                lambda_pi,
                lambda_y,
                prev_action=prev_interest_rate,
                curr_action=interest_rate,
            )
            episode_reward += reward

            # Update state
            next_state = state[len(variables) :] + [
                next_gdp_gap,
                next_inflation,
                interest_rate,
            ]
            state = next_state
            prev_interest_rate = interest_rate

        # Calculate episode metrics
        total_rewards.append(episode_reward)
        inflation_mse.append(
            np.mean([(i - pi_target) ** 2 for i in episode_inflations])
        )
        gdp_gap_mse.append(np.mean([g**2 for g in episode_gdp_gaps]))

        # Calculate inflation target achievement (% of time inflation is within ±0.5% of target)
        on_target = sum(
            1 for i in episode_inflations if abs(i - pi_target) <= 0.5
        ) / len(episode_inflations)
        inflation_targets.append(on_target)

        # Calculate interest rate volatility (standard deviation of changes)
        rate_changes = [
            abs(episode_interest_rates[i] - episode_interest_rates[i - 1])
            for i in range(1, len(episode_interest_rates))
        ]
        interest_volatility.append(np.std(rate_changes))

    # Compute overall metrics
    evaluation = {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "inflation_mse": np.mean(inflation_mse),
        "gdp_gap_mse": np.mean(gdp_gap_mse),
        "inflation_target_achievement": np.mean(inflation_targets)
        * 100,  # As percentage
        "interest_rate_volatility": np.mean(interest_volatility),
        "combined_score": np.mean(total_rewards)
        - np.mean(interest_volatility) * 50,  # Penalize volatility
    }

    if verbose:
        print("\nPolicy Evaluation Results:")
        print(f"  Mean Reward: {evaluation['mean_reward']:.2f}")
        print(f"  Inflation MSE: {evaluation['inflation_mse']:.4f}")
        print(f"  GDP Gap MSE: {evaluation['gdp_gap_mse']:.4f}")
        print(
            f"  Inflation Target Achievement: {evaluation['inflation_target_achievement']:.1f}%"
        )
        print(
            f"  Interest Rate Volatility: {evaluation['interest_rate_volatility']:.4f}"
        )
        print(f"  Combined Score: {evaluation['combined_score']:.2f}")

    return evaluation


def taylor_rule_baseline(
    df,
    inflation_model,
    gdp_model,
    variables,
    num_lags,
    episodes=10,
    steps=100,
    r_neutral=2.0,
    phi_pi=1.5,
    phi_y=0.5,
):
    """
    Evaluate a simple Taylor rule as a baseline.
    """

    def taylor_rule(inflation, gdp_gap):
        return max(0, r_neutral + phi_pi * (inflation - 2.0) + phi_y * gdp_gap)

    total_rewards = []
    inflation_mse = []
    gdp_gap_mse = []

    for ep in range(episodes):
        idx = np.random.randint(num_lags, len(df) - steps)
        window = df.iloc[idx - num_lags : idx]

        state = []
        for var in variables:
            state += list(window[var].values[::-1])

        episode_reward = 0
        episode_inflations = []
        episode_gdp_gaps = []
        prev_interest_rate = None

        for step in range(steps):
            # Extract current values (first lag)
            current_gdp = state[0]
            current_inflation = state[num_lags]

            # Apply Taylor rule
            interest_rate = taylor_rule(current_inflation, current_gdp)

            # Predict next state
            next_input = np.array(state).reshape(1, -1)
            next_inflation = inflation_model.predict(next_input)[0]
            next_gdp_gap = gdp_model.predict(next_input)[0]

            # Store predictions
            episode_inflations.append(next_inflation)
            episode_gdp_gaps.append(next_gdp_gap)

            # Compute reward
            reward = compute_reward(
                next_inflation,
                next_gdp_gap,
                prev_action=prev_interest_rate,
                curr_action=interest_rate,
            )
            episode_reward += reward

            # Update state
            state = state[len(variables) :] + [
                next_gdp_gap,
                next_inflation,
                interest_rate,
            ]
            prev_interest_rate = interest_rate

        total_rewards.append(episode_reward)
        inflation_mse.append(np.mean([(i - 2.0) ** 2 for i in episode_inflations]))
        gdp_gap_mse.append(np.mean([g**2 for g in episode_gdp_gaps]))

    return {
        "mean_reward": np.mean(total_rewards),
        "inflation_mse": np.mean(inflation_mse),
        "gdp_gap_mse": np.mean(gdp_gap_mse),
    }


def create_run_directory(base_dir="../runs"):
    """Create a timestamped directory for this training run"""
    os.makedirs(base_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)

    return run_dir, run_id


def plot_training_curves(rewards, eval_metrics, run_dir, hyperparams, grad_norms=None):
    """Create comprehensive training visualizations"""
    # Training reward curve
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.3, color="blue")
    window_size = min(50, len(rewards) // 10)  # Adaptive window size
    if len(rewards) > window_size:
        smoothed = np.convolve(
            rewards, np.ones(window_size) / window_size, mode="valid"
        )
        plt.plot(
            smoothed,
            label=f"Training Reward (Smoothed, window={window_size})",
            color="blue",
        )
    plt.grid(True)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("DDPG Training Progress")
    plt.legend()
    plt.savefig(os.path.join(run_dir, "figures", "training_reward.png"))
    plt.close()

    # Plot gradient norms if available
    if grad_norms and len(grad_norms) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(grad_norms, "r-", alpha=0.7)
        plt.grid(True)
        plt.xlabel("Training Steps (hundreds)")
        plt.ylabel("Gradient Norm")
        plt.title("Actor Gradient Norms During Training")
        plt.savefig(os.path.join(run_dir, "figures", "gradient_norms.png"))
        plt.close()

    # Extract evaluation metrics over time
    if eval_metrics:
        episodes = [m["episode"] for m in eval_metrics]
        rewards = [m["mean_reward"] for m in eval_metrics]
        inflation_mse = [m["inflation_mse"] for m in eval_metrics]
        gdp_gap_mse = [m["gdp_gap_mse"] for m in eval_metrics]
        target_achievement = [m["inflation_target_achievement"] for m in eval_metrics]
        volatility = [m["interest_rate_volatility"] for m in eval_metrics]

        # Plot multiple metrics
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

        # Plot 1: Reward
        axs[0].plot(episodes, rewards, "b-o", label="Evaluation Reward")
        axs[0].set_ylabel("Reward")
        axs[0].grid(True)
        axs[0].legend()

        # Plot 2: MSE metrics
        axs[1].plot(episodes, inflation_mse, "r-o", label="Inflation MSE")
        axs[1].plot(episodes, gdp_gap_mse, "g-o", label="GDP Gap MSE")
        axs[1].set_ylabel("Mean Squared Error")
        axs[1].grid(True)
        axs[1].legend()

        # Plot 3: Policy metrics
        ax3 = axs[2]
        ax3.plot(
            episodes,
            target_achievement,
            "m-o",
            label="Inflation Target Achievement (%)",
        )
        ax3.set_ylabel("Achievement %")
        ax3.set_xlabel("Episode")
        ax3.grid(True)

        # Add volatility on secondary axis
        ax3b = ax3.twinx()
        ax3b.plot(episodes, volatility, "k-o", label="Interest Rate Volatility")
        ax3b.set_ylabel("Volatility", color="k")

        # Add legends
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3b.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "figures", "evaluation_metrics.png"))
        plt.close()

    # Save hyperparameters
    with open(os.path.join(run_dir, "hyperparams.json"), "w") as f:
        json.dump(hyperparams, f, indent=2)


def adaptive_beta_schedule(episodes, min_value=0.4, max_value=1.0):
    """Generate increasing beta values for curriculum learning"""
    return np.linspace(min_value, max_value, episodes)


def populate_buffer_with_random_experience(
    agent, df, variables, num_lags, inflation_model, gdp_model, initial_steps=10000
):
    """Pre-fill the buffer with random experience before training"""
    print(f"Populating replay buffer with {initial_steps} random experiences...")

    indices = np.random.randint(num_lags, len(df) - 10, size=initial_steps // 10)
    experiences = 0

    for idx in tqdm(indices):
        window = df.iloc[idx - num_lags : idx]

        state = []
        for var in variables:
            state += list(window[var].values[::-1])

        prev_interest_rate = None

        for _ in range(10):  # 10 steps per window
            # Random action with some basic bounds
            interest_rate = np.random.uniform(0, 6)

            # Predict next state
            next_input = np.array(state).reshape(1, -1)
            next_inflation = inflation_model.predict(next_input)[0]
            next_gdp_gap = gdp_model.predict(next_input)[0]

            # Compute reward with smoothness penalty
            reward = compute_reward(
                next_inflation,
                next_gdp_gap,
                prev_action=prev_interest_rate,
                curr_action=interest_rate,
            )

            # Update state
            next_state = state[len(variables) :] + [
                next_gdp_gap,
                next_inflation,
                interest_rate,
            ]

            # Store transition
            agent.replay_buffer.push(
                np.array(state), interest_rate, reward, np.array(next_state)
            )

            state = next_state
            prev_interest_rate = interest_rate
            experiences += 1

    print(f"Successfully added {experiences} experiences to replay buffer")
    return experiences


def main():
    def convert_numpy_to_python(obj):
        """Convert NumPy types to standard Python types for JSON serialization"""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_python(item) for item in obj]
        return obj

    # Create run directory
    run_dir, run_id = create_run_directory()
    print(f"Training run {run_id} initialized in {run_dir}")

    # Load data
    df = pd.read_excel("../data/us_macro_data.xlsx")
    print(f"Loaded dataset with {len(df)} observations")

    # Hyperparameters with improvements
    hyperparams = {
        # Model architecture
        "num_lags": 4,
        "actor_hidden_dims": (256, 128, 64),  # Deeper network
        "critic_hidden_dims": (256, 128, 64),  # Less complex critic
        "use_twin_critics": True,
        "use_prioritized_replay": True,
        "dropout_rate": 0.1,
        # Training parameters
        "episodes": 2000,  # Fewer episodes for faster convergence
        "batch_size": 128,  # Smaller batch for more frequent updates
        "replay_buffer_size": 200000,  # Large buffer
        "min_steps": 30,  # Initial episode length
        "max_steps": 100,  # Final episode length
        "tau": 0.001,  # Slower target updates for stability
        # Exploration
        "initial_noise": 0.3,  # Lower initial noise
        "final_noise": 0.05,  # Lower final exploration
        "noise_decay": 0.999,  # Slower decay
        # Optimization
        "actor_lr": 1e-4,  # Lower actor learning rate
        "critic_lr": 3e-4,  # Lower critic learning rate
        "weight_decay": 1e-4,
        "grad_clip": 0.5,  # Stricter gradient clipping
        "lr_decay": 0.9999,  # Learning rate decay
        # Reward function
        "pi_target": 2.0,  # Inflation target
        "lambda_pi": 1.5,  # Higher weight on inflation objective
        "lambda_y": 0.5,  # Weight on output gap
        "smoothness_penalty": 0.2,  # Penalize interest rate volatility
        # Evaluation
        "eval_frequency": 50,  # Evaluate every N episodes
        "eval_episodes": 10,  # Number of evaluation episodes
        # Curriculum learning
        "use_curriculum": True,
        # Experience collection
        "random_steps": 5000,  # Initial random steps
        # Training stability
        "update_freq": 3,  # Train agent every N steps
    }

    variables = ["gdp_gap", "inflation", "interest"]
    state_dim = len(variables) * hyperparams["num_lags"]

    # Load environment models
    print("\nTraining environment models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inflation_model = EnvironmentANN(
        df, "inflation", variables, lags=hyperparams["num_lags"], device=device
    )

    gdp_model = EnvironmentANN(
        df, "gdp_gap", variables, lags=hyperparams["num_lags"], device=device
    )

    # Train with enhanced architecture
    print("Training inflation model...")
    inflation_model.train(
        hidden_units=(128, 64, 32),  # Deeper network
        activation="tanh",  # Better for inflation dynamics
        weight_decay=1e-5,  # L2 regularization
        dropout_rate=0.1,  # Dropout for regularization
        use_bn=True,  # Batch normalization
        epochs=400,  # More epochs
        patience=20,  # More patience
    )

    print("Training GDP gap model...")
    gdp_model.train(
        hidden_units=(128, 64, 32),
        activation="relu",  # Better for GDP dynamics
        weight_decay=1e-5,
        dropout_rate=0.1,
        use_bn=True,
        epochs=400,
        patience=20,
    )

    print("Environment models trained successfully.")

    # Run Taylor rule baseline
    print("\nEvaluating Taylor rule baseline...")
    taylor_metrics = taylor_rule_baseline(
        df,
        inflation_model,
        gdp_model,
        variables,
        hyperparams["num_lags"],
        episodes=20,
        steps=100,
        phi_pi=hyperparams["lambda_pi"],
        phi_y=hyperparams["lambda_y"],
    )
    print(f"Taylor rule baseline reward: {taylor_metrics['mean_reward']:.2f}")
    print(f"Taylor rule inflation MSE: {taylor_metrics['inflation_mse']:.4f}")
    print(f"Taylor rule GDP gap MSE: {taylor_metrics['gdp_gap_mse']:.4f}")

    # Initialize enhanced DDPG agent
    print("\nInitializing DDPG agent...")
    agent = EnhancedDDPGAgent(
        state_dim=state_dim,
        action_dim=1,
        actor_hidden_dims=hyperparams["actor_hidden_dims"],
        critic_hidden_dims=hyperparams["critic_hidden_dims"],
        gamma=0.99,
        tau=hyperparams["tau"],
        actor_lr=hyperparams["actor_lr"],
        critic_lr=hyperparams["critic_lr"],
        buffer_capacity=hyperparams["replay_buffer_size"],
        action_high=6.0,  # Lower maximum interest rate for more stability
        action_low=0.0,  # Zero lower bound
        batch_size=hyperparams["batch_size"],
        use_twin_critics=hyperparams["use_twin_critics"],
        use_prioritized_replay=hyperparams["use_prioritized_replay"],
        grad_clip_value=hyperparams["grad_clip"],
        policy_noise=0.1,  # Lower policy noise
        exploration_noise=hyperparams["initial_noise"],
        dropout_rate=hyperparams["dropout_rate"],
        weight_decay=hyperparams["weight_decay"],
    )

    # Create learning rate schedulers (slower decrease)
    actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        agent.actor_optimizer, gamma=hyperparams["lr_decay"]
    )
    critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        agent.critic_optimizer, gamma=hyperparams["lr_decay"]
    )

    # Pre-fill buffer with random experience
    populate_buffer_with_random_experience(
        agent,
        df,
        variables,
        hyperparams["num_lags"],
        inflation_model,
        gdp_model,
        initial_steps=hyperparams["random_steps"],
    )

    # Compute reward scale factor from warmup phase
    print("\nDetermining adaptive reward scaling...")
    reward_history = []
    warmup_episodes = 50

    for _ in range(warmup_episodes):
        idx = np.random.randint(
            hyperparams["num_lags"], len(df) - hyperparams["max_steps"] - 1
        )
        window = df.iloc[idx - hyperparams["num_lags"] : idx]

        state = []
        for var in variables:
            state += list(window[var].values[::-1])

        prev_interest_rate = None

        for _ in range(hyperparams["max_steps"]):
            action = agent.select_action(np.array(state), add_noise=True)
            interest_rate = action[0]

            next_input = np.array(state).reshape(1, -1)
            next_inflation = inflation_model.predict(next_input)[0]
            next_gdp_gap = gdp_model.predict(next_input)[0]

            # Use new compute_reward function directly
            raw_reward = compute_reward(
                next_inflation,
                next_gdp_gap,
                hyperparams["pi_target"],
                hyperparams["lambda_pi"],
                hyperparams["lambda_y"],
                prev_action=prev_interest_rate,
                curr_action=interest_rate,
                smoothness_penalty=hyperparams["smoothness_penalty"],
            )
            reward_history.append(raw_reward)

            next_state = state[len(variables) :] + [
                next_gdp_gap,
                next_inflation,
                interest_rate,
            ]
            state = next_state
            prev_interest_rate = interest_rate

    # Calculate adaptive scaling factor using more robust median
    if reward_history:
        # Target a more reasonable reward range around -5
        median_reward = np.median(reward_history)
        reward_scale = 5.0 / (abs(median_reward) + 1e-8)
        # Apply bounds to prevent extreme values
        reward_scale = min(20.0, max(0.1, reward_scale))
        print(f"Adaptive reward scaling factor: {reward_scale:.2f}")
    else:
        reward_scale = 5.0

    # Generate curriculum schedule
    if hyperparams["use_curriculum"]:
        beta_schedule = adaptive_beta_schedule(hyperparams["episodes"])

    # Training tracking
    episode_rewards = []
    eval_metrics = []
    best_score = -float("inf")
    best_model = None
    current_noise = hyperparams["initial_noise"]
    noise_decay = hyperparams["noise_decay"]
    grad_norm_history = []

    # Start training
    print("\nStarting DDPG training...")
    start_time = time.time()

    try:
        for ep in tqdm(range(hyperparams["episodes"])):
            # Calculate episode steps using curriculum learning if enabled
            if hyperparams["use_curriculum"]:
                beta = beta_schedule[ep]
                current_max_steps = int(
                    hyperparams["min_steps"]
                    + (hyperparams["max_steps"] - hyperparams["min_steps"]) * beta
                )
            else:
                current_max_steps = hyperparams["max_steps"]

            # Sample starting point
            idx = np.random.randint(
                hyperparams["num_lags"], len(df) - current_max_steps - 1
            )
            window = df.iloc[idx - hyperparams["num_lags"] : idx]

            state = []
            for var in variables:
                state += list(window[var].values[::-1])

            episode_reward = 0
            prev_interest_rate = None  # For smoothness penalty

            for step in range(current_max_steps):
                # Select action with exploration noise
                action = agent.select_action(
                    np.array(state), add_noise=True, noise_scale=current_noise
                )
                interest_rate = action[0]

                # Simulate next state with environment models
                next_input = np.array(state).reshape(1, -1)
                next_inflation = inflation_model.predict(next_input)[0]
                next_gdp_gap = gdp_model.predict(next_input)[0]

                # Compute reward with adaptive scaling and smoothness penalty
                reward = compute_reward(
                    next_inflation,
                    next_gdp_gap,
                    hyperparams["pi_target"],
                    hyperparams["lambda_pi"],
                    hyperparams["lambda_y"],
                    scale_factor=reward_scale,
                    prev_action=prev_interest_rate,
                    curr_action=interest_rate,
                    smoothness_penalty=hyperparams["smoothness_penalty"],
                )
                episode_reward += reward

                # Update state
                next_state = state[len(variables) :] + [
                    next_gdp_gap,
                    next_inflation,
                    interest_rate,
                ]

                # Store transition
                agent.replay_buffer.push(
                    np.array(state), interest_rate, reward, np.array(next_state)
                )

                # Train if enough samples and according to update frequency
                if len(agent.replay_buffer) > hyperparams["batch_size"] * 3:
                    if step % hyperparams["update_freq"] == 0:  # Train less frequently
                        train_metrics = agent.train()

                        # Track gradient norms periodically
                        if ep % 20 == 0 and step == 0:
                            # Get gradient norm from actor
                            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                                agent.actor.parameters(), hyperparams["grad_clip"]
                            )
                            grad_norm_history.append(float(actor_grad_norm))

                # Update state and previous action
                state = next_state
                prev_interest_rate = interest_rate

            # Decay exploration noise
            current_noise = max(hyperparams["final_noise"], current_noise * noise_decay)

            # Update learning rates periodically
            if ep % 20 == 0:
                actor_scheduler.step()
                critic_scheduler.step()
                if ep % 100 == 0:
                    print(
                        f"Learning rates - Actor: {actor_scheduler.get_last_lr()[0]:.6f}, "
                        f"Critic: {critic_scheduler.get_last_lr()[0]:.6f}"
                    )

            # Record episode rewards
            avg_reward = episode_reward / current_max_steps
            episode_rewards.append(avg_reward)

            # Periodic evaluation
            if (ep + 1) % hyperparams["eval_frequency"] == 0 or ep == hyperparams[
                "episodes"
            ] - 1:
                eval_result = evaluate_policy(
                    agent,
                    inflation_model,
                    gdp_model,
                    variables,
                    df,
                    hyperparams["num_lags"],
                    episodes=hyperparams["eval_episodes"],
                    steps_per_episode=hyperparams["max_steps"],
                    verbose=True,
                    pi_target=hyperparams["pi_target"],
                    lambda_pi=hyperparams["lambda_pi"],
                    lambda_y=hyperparams["lambda_y"],
                )

                # Add episode number to results
                eval_result["episode"] = ep + 1
                eval_metrics.append(eval_result)

                # Track and save best model
                if eval_result["combined_score"] > best_score:
                    best_score = eval_result["combined_score"]
                    best_model = {
                        "actor": agent.actor.state_dict().copy(),
                        "critic": agent.critic.state_dict().copy(),
                        "episode": ep + 1,
                        "metrics": eval_result,
                    }

                    # Save best model
                    agent.save(os.path.join(run_dir, "models", "best_agent.pt"))
                    print(f"New best model saved! Score: {best_score:.2f}")

                # Print progress
                elapsed = time.time() - start_time
                print(
                    f"\nEpisode {ep + 1}/{hyperparams['episodes']} - Elapsed: {elapsed:.0f}s"
                )
                print(f"Noise: {current_noise:.3f}, Steps: {current_max_steps}")
                print(f"Episode Reward: {avg_reward:.2f}")
                print(f"Best Score so far: {best_score:.2f}")

                # Create interim visualizations
                plot_training_curves(
                    episode_rewards,
                    eval_metrics,
                    run_dir,
                    hyperparams,
                    grad_norm_history,
                )

        # Training complete
        print("\nTraining complete!")
        elapsed = time.time() - start_time
        print(f"Total training time: {elapsed:.0f} seconds")

        # Save final model
        agent.save(os.path.join(run_dir, "models", "final_agent.pt"))

        # Save best model separately
        if best_model:
            torch.save(
                best_model["actor"], os.path.join(run_dir, "models", "best_actor.pth")
            )

            # Save comparison with Taylor rule
            comparison = {
                "ddpg_inflation_mse": float(best_model["metrics"]["inflation_mse"]),
                "ddpg_gdp_gap_mse": float(best_model["metrics"]["gdp_gap_mse"]),
                "ddpg_reward": float(best_model["metrics"]["mean_reward"]),
                "taylor_inflation_mse": float(taylor_metrics["inflation_mse"]),
                "taylor_gdp_gap_mse": float(taylor_metrics["gdp_gap_mse"]),
                "taylor_reward": float(taylor_metrics["mean_reward"]),
                "improvement_pct": float(
                    (
                        best_model["metrics"]["mean_reward"]
                        - taylor_metrics["mean_reward"]
                    )
                    / abs(taylor_metrics["mean_reward"])
                    * 100
                ),
            }

            # And for the JSON dump:
            with open(os.path.join(run_dir, "comparison.json"), "w") as f:
                json.dump(convert_numpy_to_python(comparison), f, indent=2)

            print("\nPerformance Comparison:")
            print(
                f"DDPG vs Taylor Rule reward improvement: {comparison['improvement_pct']:.1f}%"
            )
            print(f"DDPG inflation MSE: {comparison['ddpg_inflation_mse']:.4f}")
            print(
                f"Taylor rule inflation MSE: {comparison['taylor_inflation_mse']:.4f}"
            )

            # Save comparison to file
            with open(os.path.join(run_dir, "comparison.json"), "w") as f:
                json.dump(comparison, f, indent=2)

        # Final visualizations
        plot_training_curves(
            episode_rewards, eval_metrics, run_dir, hyperparams, grad_norm_history
        )

        # Return to standard output directory
        torch.save(
            best_model["actor"] if best_model else agent.actor.state_dict(),
            "../models/trained_actor.pth",
        )

    except KeyboardInterrupt:
        print("\nTraining interrupted!")
        if best_model:
            print(f"Saving best model from episode {best_model['episode']}")
            torch.save(best_model["actor"], "../models/trained_actor_interrupted.pth")
        else:
            print("Saving current model")
            torch.save(
                agent.actor.state_dict(), "../models/trained_actor_interrupted.pth"
            )


if __name__ == "__main__":
    # Make directories if they don't exist
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../figures", exist_ok=True)

    # Run main training
    main()
