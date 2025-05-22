"""
This script implements enhanced ANN models for the Actor-Critic architecture.
It includes an Actor network to output interest rates and a Critic network to estimate Q-values.
The Actor network outputs a scalar interest rate based on the state input,
while the Critic network estimates the Q-value of a state-action pair.
The networks are designed to work with macroeconomic data, specifically for interest rate control.

Improvements include:
- Layer normalization for better gradient flow
- Advanced initialization for faster convergence
- Regularization via dropout
- Tuned activation functions
- Support for deeper architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#########################################
# Actor network to output interest rate #
#########################################


class Actor(nn.Module):
    def __init__(
        self, input_dim, hidden_dims=(128, 64), dropout_rate=0.1, use_layer_norm=True
    ):
        """
        Enhanced Actor network to output a scalar interest rate.

        Args:
            input_dim: Number of features (e.g., lags * 3 variables)
            hidden_dims: Tuple of hidden layer sizes
            dropout_rate: Dropout probability for regularization
            use_layer_norm: Whether to use layer normalization
        """
        super(Actor, self).__init__()

        # Input layer with layer normalization
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.input_norm = (
            nn.LayerNorm(hidden_dims[0]) if use_layer_norm else nn.Identity()
        )

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            layer_block = nn.ModuleList(
                [
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.LayerNorm(hidden_dims[i + 1])
                    if use_layer_norm
                    else nn.Identity(),
                    nn.Tanh(),
                    nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                ]
            )
            self.hidden_layers.extend(layer_block)

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self.output_activation = nn.ReLU()  # Enforce ZLB constraint

        # Better initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="tanh")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state):
        """
        Forward pass through the Actor network.

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            Action (interest rate) tensor [batch_size, 1]
        """
        x = self.input_layer(state)
        x = self.input_norm(x)
        x = F.tanh(x)

        for layer in self.hidden_layers:
            x = layer(x)

        x = self.output_layer(x)
        return self.output_activation(x)  # Enforce non-negative interest rates (ZLB)


######################################
# Critic network to estimate Q-value #
######################################


class Critic(nn.Module):
    def __init__(
        self,
        input_dim,
        action_dim=1,
        hidden_dims=(128, 64),
        dropout_rate=0.1,
        use_layer_norm=True,
    ):
        """
        Enhanced Critic network to estimate Q-value of a state-action pair.

        Args:
            input_dim: Number of state features
            action_dim: Dimension of action input (usually 1)
            hidden_dims: Tuple of hidden layer sizes
            dropout_rate: Dropout probability for regularization
            use_layer_norm: Whether to use layer normalization
        """
        super(Critic, self).__init__()

        # Input processing
        self.input_dim = input_dim
        self.action_dim = action_dim

        # Input layer with layer normalization
        self.input_layer = nn.Linear(input_dim + action_dim, hidden_dims[0])
        self.input_norm = (
            nn.LayerNorm(hidden_dims[0]) if use_layer_norm else nn.Identity()
        )

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            layer_block = nn.ModuleList(
                [
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.LayerNorm(hidden_dims[i + 1])
                    if use_layer_norm
                    else nn.Identity(),
                    nn.LeakyReLU(0.1),  # LeakyReLU for critic for better gradient flow
                    nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                ]
            )
            self.hidden_layers.extend(layer_block)

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        # Better initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        """
        Forward pass through the Critic network.

        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]

        Returns:
            Q-value tensor [batch_size, 1]
        """
        x = torch.cat([state, action], dim=1)
        x = self.input_layer(x)
        x = self.input_norm(x)
        x = F.leaky_relu(x, 0.1)

        for layer in self.hidden_layers:
            x = layer(x)

        return self.output_layer(x)


##############################################
# Twin Critic for improved learning stability #
##############################################


class TwinCritic(nn.Module):
    """
    Implementation of twin critics for more stable learning, similar to TD3.
    Uses two identical but independently initialized critic networks.
    """

    def __init__(
        self,
        input_dim,
        action_dim=1,
        hidden_dims=(128, 64),
        dropout_rate=0.1,
        use_layer_norm=True,
    ):
        super(TwinCritic, self).__init__()

        # Create two identical but separate critics
        self.critic1 = Critic(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
        )

        self.critic2 = Critic(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, state, action):
        """
        Forward pass through both critics.

        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]

        Returns:
            Tuple of Q-values from both critics
        """
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        return q1, q2
