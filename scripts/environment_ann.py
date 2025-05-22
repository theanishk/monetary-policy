"""
This script implements a feedforward artificial neural network (ANN) for time series prediction
with enhanced training, evaluation, and visualization capabilities.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore")

plt.style.use("ggplot")


class FeedforwardANN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_units=(64, 64),
        activation="tanh",
        use_bn=True,
        dropout_rate=0.1,
    ):
        """
        Enhanced feedforward ANN with batch normalization, dropout, and flexible architecture.

        Args:
            input_dim: Input feature dimension
            hidden_units: Tuple of hidden layer sizes
            activation: Activation function ('tanh' or 'relu')
            use_bn: Whether to use batch normalization
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()

        # Select activation function
        if activation == "tanh":
            act_fn = nn.Tanh()
        elif activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU(0.1)
        else:
            act_fn = nn.ReLU()

        layers = []
        last_dim = input_dim

        # Build hidden layers
        for h in hidden_units:
            # Linear layer
            layers.append(nn.Linear(last_dim, h))

            # Optional batch normalization
            if use_bn:
                layers.append(nn.BatchNorm1d(h))

            # Activation
            layers.append(act_fn)

            # Dropout for regularization
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            last_dim = h

        # Output layer
        layers.append(nn.Linear(last_dim, 1))

        # Create sequential model
        self.model = nn.Sequential(*layers)

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize model weights for better convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight,
                    nonlinearity="relu"
                    if isinstance(self.model[2], nn.ReLU)
                    else "tanh",
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)


class EnvironmentANN:
    def __init__(
        self,
        df,
        target_var,
        lagged_vars,
        lags=4,
        val_split=0.2,
        device="cpu",
        batch_size=64,
    ):
        """
        Initialize the environment ANN model.

        Args:
            df: DataFrame with time series data
            target_var: Target variable to predict
            lagged_vars: List of variables to use as features with lags
            lags: Number of lagged values to include
            val_split: Proportion of data for validation
            device: Computing device ('cpu' or 'cuda')
            batch_size: Training batch size
        """
        self.df = df.copy()
        self.target_var = target_var
        self.lagged_vars = lagged_vars
        self.lags = lags
        self.val_split = val_split
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.model = None
        self._prepare_data()

        print(
            f"Model initialized for {target_var} prediction using {len(lagged_vars)} variables with {lags} lags."
        )
        print(f"Using device: {self.device}")

    def _prepare_data(self):
        """
        Prepare lagged features and create data loaders.
        """
        print("Preparing data with lags...")
        df = self.df.copy()

        # Create lagged features
        for var in self.lagged_vars:
            for l in range(1, self.lags + 1):
                df[f"{var}_lag{l}"] = df[var].shift(l)

        # Create target (next value of target variable)
        df[f"{self.target_var}_target"] = df[self.target_var].shift(-1)

        # Remove rows with NaN values
        df.dropna(inplace=True)

        # Extract feature variables
        feature_cols = [
            f"{var}_lag{l}" for var in self.lagged_vars for l in range(1, self.lags + 1)
        ]

        # Store feature column names for later use
        self.feature_cols = feature_cols

        # Extract features and target
        X = df[feature_cols].values
        y = df[f"{self.target_var}_target"].values.reshape(-1, 1)

        # Save original feature values for plotting
        self.X_orig = X.copy()
        self.y_orig = y.copy()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split into training and validation sets
        val_size = int(len(X_scaled) * self.val_split)
        self.X_train, self.X_val = X_scaled[:-val_size], X_scaled[-val_size:]
        self.y_train, self.y_val = y[:-val_size], y[-val_size:]

        print(f"Training data: {self.X_train.shape[0]} samples")
        print(f"Validation data: {self.X_val.shape[0]} samples")
        print(f"Feature dimension: {self.X_train.shape[1]}")

        # Create PyTorch data loaders
        self.train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(self.X_train), torch.FloatTensor(self.y_train)
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # Convert validation data to tensors
        self.X_val_torch = torch.FloatTensor(self.X_val).to(self.device)
        self.y_val_torch = torch.FloatTensor(self.y_val).to(self.device)

    def train(
        self,
        hidden_units=(64, 64),
        activation="tanh",
        lr=1e-3,
        epochs=300,
        patience=15,
        weight_decay=1e-5,
        dropout_rate=0.1,
        use_bn=True,
        verbose=True,
    ):
        """
        Train the neural network model with enhanced techniques.

        Args:
            hidden_units: Tuple of hidden layer sizes
            activation: Activation function ('tanh' or 'relu')
            lr: Learning rate
            epochs: Maximum number of training epochs
            patience: Early stopping patience
            weight_decay: L2 regularization strength
            dropout_rate: Dropout rate for regularization
            use_bn: Whether to use batch normalization
            verbose: Whether to print training progress
        """
        if verbose:
            print(f"\nTraining model with configuration:")
            print(f"  Hidden units: {hidden_units}")
            print(f"  Activation: {activation}")
            print(f"  Learning rate: {lr}")
            print(f"  Weight decay: {weight_decay}")
            print(f"  Dropout rate: {dropout_rate}")
            print(f"  Batch norm: {use_bn}")

        # Initialize model
        input_dim = self.X_train.shape[1]
        model = FeedforwardANN(
            input_dim,
            hidden_units,
            activation,
            use_bn=use_bn,
            dropout_rate=dropout_rate,
        ).to(self.device)

        # Initialize optimizer with weight decay (L2 regularization)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=patience // 3, verbose=verbose
        )

        # Loss function
        criterion = nn.MSELoss()

        # Training history
        train_losses = []
        val_losses = []
        val_loss_history = []  # For early stopping

        best_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        lr_history = []

        # Training loop
        for epoch in range(epochs):
            # Set model to training mode
            model.train()
            epoch_losses = []

            # Process mini-batches
            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                # Forward pass
                pred = model(xb)
                loss = criterion(pred, yb)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Update weights
                optimizer.step()

                # Track batch loss
                epoch_losses.append(loss.item())

            # Calculate average training loss
            avg_train_loss = np.mean(epoch_losses)
            train_losses.append(avg_train_loss)

            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                val_pred = model(self.X_val_torch)
                val_loss = criterion(val_pred, self.y_val_torch).item()
                val_losses.append(val_loss)

            # Update learning rate scheduler
            scheduler.step(val_loss)
            lr_history.append(optimizer.param_groups[0]["lr"])

            # Early stopping logic
            val_loss_history.append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(
                    f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
                )

            # Check early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            if verbose:
                print(f"Restored best model with validation loss: {best_loss:.6f}")

        # Save the model
        self.model = model
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.lr_history = lr_history

        # Compute additional validation metrics
        self.compute_metrics(self.X_val, self.y_val, "Validation")

        return best_loss

    def compute_metrics(self, X, y, set_name="Validation"):
        """
        Compute comprehensive metrics for model evaluation.

        Args:
            X: Feature matrix
            y: Target values
            set_name: Name of the data set (for printing)
        """
        # Get predictions
        pred = self.predict(X)
        y_flat = y.flatten()

        # Compute metrics
        mse = mean_squared_error(y_flat, pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_flat, pred)
        r2 = r2_score(y_flat, pred)

        # Compute mean absolute percentage error (MAPE) with handling for zero values
        non_zero_mask = np.abs(y_flat) > 1e-8
        mape = (
            np.mean(
                np.abs(
                    (y_flat[non_zero_mask] - pred[non_zero_mask])
                    / y_flat[non_zero_mask]
                )
            )
            * 100
        )

        print(f"\n{set_name} Metrics:")
        print(f"  MSE: {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²: {r2:.6f}")

        return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape, "r2": r2}

    def predict(self, X=None):
        """
        Make predictions with the trained model.

        Args:
            X: Feature matrix (uses validation set if None)

        Returns:
            Numpy array of predictions
        """
        if self.model is None:
            raise ValueError("Model hasn't been trained yet. Call train() first.")

        self.model.eval()
        with torch.no_grad():
            if X is None:
                X = self.X_val

            # Scale features if they're not already scaled
            if X is self.X_val:
                X_scaled = self.X_val
            else:
                X_scaled = self.scaler.transform(X)

            # Convert to tensor
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            # Make predictions
            pred = self.model(X_tensor).cpu().numpy().flatten()

        return pred

    def plot_losses(self, save_path=None):
        """
        Plot training and validation losses.

        Args:
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Training Loss", alpha=0.7)
        plt.plot(self.val_losses, label="Validation Loss", alpha=0.7)
        plt.title(f"{self.target_var} Model Training")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.yscale("log")
        plt.grid(True)
        plt.legend()

        # Add learning rate on secondary axis
        ax2 = plt.gca().twinx()
        ax2.plot(self.lr_history, "g--", alpha=0.5, label="Learning Rate")
        ax2.set_ylabel("Learning Rate")
        ax2.tick_params(axis="y", labelcolor="g")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Loss plot saved to {save_path}")

        plt.show()

    def plot_predictions(self, data_subset="all", save_path=None):
        """
        Plot predictions against actual values.

        Args:
            data_subset: Which data to plot ('train', 'val', or 'all')
            save_path: Optional path to save the plot
        """
        if self.model is None:
            raise ValueError("Model hasn't been trained yet. Call train() first.")

        plt.figure(figsize=(14, 6))

        if data_subset == "train" or data_subset == "all":
            # Get training predictions
            train_pred = self.predict(self.X_train)
            plt.plot(
                range(len(self.y_train)),
                self.y_train.flatten(),
                "b-",
                label="Training Actual",
            )
            plt.plot(
                range(len(train_pred)),
                train_pred,
                "r--",
                alpha=0.7,
                label="Training Predicted",
            )

        if data_subset == "val" or data_subset == "all":
            # Get validation predictions
            val_pred = self.predict(self.X_val)
            # Offset validation data on the x-axis
            if data_subset == "all":
                offset = len(self.y_train)
                plt.plot(
                    range(offset, offset + len(self.y_val)),
                    self.y_val.flatten(),
                    "g-",
                    label="Validation Actual",
                )
                plt.plot(
                    range(offset, offset + len(val_pred)),
                    val_pred,
                    "m--",
                    alpha=0.7,
                    label="Validation Predicted",
                )
                # Add a vertical line to separate training and validation
                plt.axvline(x=offset, color="k", linestyle="--", alpha=0.3)
            else:
                plt.plot(
                    range(len(self.y_val)),
                    self.y_val.flatten(),
                    "g-",
                    label="Validation Actual",
                )
                plt.plot(
                    range(len(val_pred)),
                    val_pred,
                    "m--",
                    alpha=0.7,
                    label="Validation Predicted",
                )

        plt.title(f"{self.target_var} Prediction")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Prediction plot saved to {save_path}")

        plt.show()

        # Also plot the prediction error distribution
        self._plot_error_distribution(data_subset, save_path)

    def _plot_error_distribution(self, data_subset="all", save_path=None):
        """
        Plot the distribution of prediction errors.

        Args:
            data_subset: Which data to use ('train', 'val', or 'all')
            save_path: Optional path to save the plot
        """
        errors = []

        if data_subset == "train" or data_subset == "all":
            train_pred = self.predict(self.X_train)
            train_errors = self.y_train.flatten() - train_pred
            errors.extend(train_errors)

        if data_subset == "val" or data_subset == "all":
            val_pred = self.predict(self.X_val)
            val_errors = self.y_val.flatten() - val_pred
            errors.extend(val_errors)

        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=30, alpha=0.7)
        plt.axvline(x=0, color="r", linestyle="--")
        plt.title(f"{self.target_var} Prediction Error Distribution")
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            base, ext = os.path.splitext(save_path)
            error_path = f"{base}_error_dist{ext}"
            plt.savefig(error_path)
            print(f"Error distribution plot saved to {error_path}")

        plt.show()

    def plot_feature_importance(self, save_path=None):
        """
        Approximate feature importance by perturbing each feature.

        Args:
            save_path: Optional path to save the plot
        """
        if self.model is None:
            raise ValueError("Model hasn't been trained yet. Call train() first.")

        # Get baseline prediction error
        baseline_pred = self.predict(self.X_val)
        baseline_mse = mean_squared_error(self.y_val.flatten(), baseline_pred)

        # Perturb each feature and measure impact
        importance = []
        for i in range(self.X_val.shape[1]):
            # Clone the data
            X_perturbed = self.X_val.copy()

            # Perturb the feature by shuffling it
            np.random.shuffle(X_perturbed[:, i])

            # Measure new performance
            perturbed_pred = self.predict(X_perturbed)
            perturbed_mse = mean_squared_error(self.y_val.flatten(), perturbed_pred)

            # Importance is the relative increase in error
            importance.append((perturbed_mse - baseline_mse) / baseline_mse)

        # Sort features by importance
        feature_importance = list(zip(self.feature_cols, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        # Plot importance
        plt.figure(figsize=(10, 8))
        feature_names = [x[0] for x in feature_importance]
        importances = [x[1] for x in feature_importance]

        plt.barh(feature_names, importances)
        plt.title(f"Feature Importance for {self.target_var} Prediction")
        plt.xlabel("Relative Importance")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Feature importance plot saved to {save_path}")

        plt.show()

    def grid_search(
        self,
        activations=["tanh", "relu", "leaky_relu"],
        layer_sizes=[(32,), (64,), (64, 64), (128, 64), (128, 128, 64)],
        learning_rates=[1e-2, 1e-3, 5e-4],
        weight_decays=[0, 1e-5, 1e-4],
        dropout_rates=[0, 0.1, 0.2],
        trials=3,
        epochs=300,
        patience=15,
        verbose=False,
    ):
        """
        Enhanced grid search with more hyperparameters and better tracking.

        Args:
            activations: List of activation functions to try
            layer_sizes: List of hidden layer configurations
            learning_rates: List of learning rates to try
            weight_decays: List of weight decay values for L2 regularization
            dropout_rates: List of dropout rates to try
            trials: Number of trials for each configuration
            epochs: Maximum training epochs
            patience: Early stopping patience
            verbose: Whether to print detailed progress

        Returns:
            DataFrame with grid search results
        """
        results = []
        best_model = None
        best_loss = float("inf")
        best_config = None

        # Create timestamp for unique run ID
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        total_configs = (
            len(activations)
            * len(layer_sizes)
            * len(learning_rates)
            * len(weight_decays)
            * len(dropout_rates)
        )
        print(
            f"Starting grid search with {total_configs} configurations, {trials} trials each ({total_configs * trials} total runs)"
        )
        print(f"Run ID: {run_id}")

        # Track progress
        config_count = 0

        for activation in activations:
            for hidden_units in layer_sizes:
                for lr in learning_rates:
                    for weight_decay in weight_decays:
                        for dropout_rate in dropout_rates:
                            config_count += 1
                            print(f"\nConfiguration {config_count}/{total_configs}:")
                            print(f"  Activation: {activation}")
                            print(f"  Hidden units: {hidden_units}")
                            print(f"  Learning rate: {lr}")
                            print(f"  Weight decay: {weight_decay}")
                            print(f"  Dropout rate: {dropout_rate}")

                            val_losses = []
                            train_times = []

                            for trial in range(trials):
                                print(f"  Trial {trial + 1}/{trials}")

                                # Time the training
                                start_time = datetime.now()

                                # Train with current configuration
                                val_loss = self.train(
                                    hidden_units=hidden_units,
                                    activation=activation,
                                    lr=lr,
                                    weight_decay=weight_decay,
                                    dropout_rate=dropout_rate,
                                    epochs=epochs,
                                    patience=patience,
                                    verbose=verbose,
                                )

                                # Record training time
                                train_time = (
                                    datetime.now() - start_time
                                ).total_seconds()
                                train_times.append(train_time)

                                # Record validation loss
                                val_losses.append(val_loss)

                                print(
                                    f"    Trial {trial + 1} completed. Val Loss: {val_loss:.6f}, Time: {train_time:.2f}s"
                                )

                            # Calculate average metrics
                            avg_loss = np.mean(val_losses)
                            std_loss = np.std(val_losses)
                            avg_time = np.mean(train_times)

                            # Record results
                            config_result = {
                                "activation": activation,
                                "hidden_units": str(hidden_units),
                                "learning_rate": lr,
                                "weight_decay": weight_decay,
                                "dropout_rate": dropout_rate,
                                "avg_val_mse": avg_loss,
                                "std_val_mse": std_loss,
                                "avg_time": avg_time,
                                "run_id": run_id,
                            }

                            results.append(config_result)

                            # Check if this is the best configuration
                            if avg_loss < best_loss:
                                best_loss = avg_loss
                                best_model = self.model.state_dict()
                                best_config = {
                                    "activation": activation,
                                    "hidden_units": hidden_units,
                                    "learning_rate": lr,
                                    "weight_decay": weight_decay,
                                    "dropout_rate": dropout_rate,
                                }

                                print(
                                    f"  New best configuration! Avg Val MSE: {avg_loss:.6f}"
                                )

        # Convert results to DataFrame
        results_df = pd.DataFrame(results).sort_values(by="avg_val_mse")
        self.results_df = results_df

        # Rebuild best model
        if best_model is not None and best_config is not None:
            input_dim = self.X_train.shape[1]
            self.model = FeedforwardANN(
                input_dim,
                best_config["hidden_units"],
                best_config["activation"],
                dropout_rate=best_config["dropout_rate"],
            ).to(self.device)
            self.model.load_state_dict(best_model)

            print("\nBest configuration:")
            for k, v in best_config.items():
                print(f"  {k}: {v}")
            print(f"Best validation MSE: {best_loss:.6f}")

            # Compute final metrics
            self.compute_metrics(self.X_val, self.y_val, "Best Model Validation")

        return results_df

    def save_model(self, path):
        """
        Save the trained model to disk.

        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No trained model to save. Call train() first.")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save model
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
                "target_var": self.target_var,
            },
            path,
        )

        print(f"Model saved to {path}")

    def load_model(self, path, hidden_units=(64, 64), activation="tanh"):
        """
        Load a trained model from disk.

        Args:
            path: Path to the saved model
            hidden_units: Hidden layer sizes (needed to reconstruct the model)
            activation: Activation function (needed to reconstruct the model)
        """
        # Load saved model
        checkpoint = torch.load(path, map_location=self.device)

        # Rebuild model architecture
        input_dim = self.X_train.shape[1]
        self.model = FeedforwardANN(input_dim, hidden_units, activation).to(self.device)

        # Load model weights
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load scaler if available
        if "scaler" in checkpoint:
            self.scaler = checkpoint["scaler"]

        print(f"Model loaded from {path}")


if __name__ == "__main__":
    # Create directories for outputs
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../figures", exist_ok=True)

    # Load data
    df = pd.read_excel("../data/us_macro_data.xlsx")
    print(f"Loaded dataset with {len(df)} rows")

    # Configure models
    inflation_model = EnvironmentANN(
        df=df,
        target_var="inflation",
        lagged_vars=["gdp_gap", "inflation", "interest"],
        lags=4,  # Increase lags for better temporal context
        val_split=0.2,
        device="cuda",  # Will fall back to CPU if CUDA not available
        batch_size=64,
    )

    gdp_model = EnvironmentANN(
        df=df,
        target_var="gdp_gap",
        lagged_vars=["gdp_gap", "inflation", "interest"],
        lags=4,
        val_split=0.2,
        device="cuda",
        batch_size=64,
    )

    # Grid search for inflation model
    print("\n=== Inflation Model Grid Search ===")
    inflation_results = inflation_model.grid_search(
        activations=["tanh", "relu"],
        layer_sizes=[(64,), (64, 32), (128, 64)],
        learning_rates=[1e-3, 5e-4],
        weight_decays=[1e-5],
        dropout_rates=[0.1],
        trials=2,  # Reduced trials for demonstration
        verbose=False,
    )

    # Save grid search results
    inflation_results.to_csv("../results/inflation_grid_search.csv", index=False)

    # Plot training progress and predictions
    inflation_model.plot_losses(save_path="../figures/inflation_training_loss.png")
    inflation_model.plot_predictions(save_path="../figures/inflation_predictions.png")
    inflation_model.plot_feature_importance(
        save_path="../figures/inflation_feature_importance.png"
    )

    # Save the trained model
    inflation_model.save_model("../models/inflation_model.pt")

    # Grid search for GDP gap model
    print("\n=== GDP Gap Model Grid Search ===")
    gdp_results = gdp_model.grid_search(
        activations=["tanh", "relu"],
        layer_sizes=[(64,), (64, 32), (128, 64)],
        learning_rates=[1e-3, 5e-4],
        weight_decays=[1e-5],
        dropout_rates=[0.1],
        trials=2,  # Reduced trials for demonstration
        verbose=False,
    )

    # Save grid search results
    gdp_results.to_csv("../results/gdp_gap_grid_search.csv", index=False)

    # Plot training progress and predictions
    gdp_model.plot_losses(save_path="../figures/gdp_gap_training_loss.png")
    gdp_model.plot_predictions(save_path="../figures/gdp_gap_predictions.png")
    gdp_model.plot_feature_importance(
        save_path="../figures/gdp_gap_feature_importance.png"
    )

    # Save the trained model
    gdp_model.save_model("../models/gdp_gap_model.pt")

    print("\nAll models trained and saved successfully!")
