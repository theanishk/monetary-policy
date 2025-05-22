"""
Enhanced implementation of the DDPG (Deep Deterministic Policy Gradient) algorithm.
This version includes:
- Prioritized experience replay for more efficient learning
- Gradient clipping to prevent exploding gradients
- Parameter noise for better exploration
- Twin critics for more stable Q-learning (inspired by TD3)
- Better noise management and action regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from actor_critic import Actor, Critic, TwinCritic


class AdaptiveNoise:
    """
    Enhanced noise process with adaptive parameters based on training progress.
    Combines Ornstein-Uhlenbeck dynamics with adaptive exploration.
    """

    def __init__(
        self, size, mu=0.0, theta=0.15, sigma=0.2, decay=0.995, min_sigma=0.05
    ):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.decay = decay
        self.min_sigma = min_sigma
        self.reset()

    def reset(self):
        """Reset noise state"""
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        """Generate noise sample with current parameters"""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(
            self.size
        )
        self.state += dx
        return self.state

    def adapt(self, score=None, decay=True):
        """Adapt noise parameters based on performance"""
        if decay:
            self.sigma = max(self.min_sigma, self.sigma * self.decay)


class PrioritizedReplayBuffer:
    """
    Enhanced replay buffer with prioritized experience replay for more
    efficient learning from important transitions.
    """

    def __init__(
        self, capacity=100000, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-5
    ):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance sampling exponent
        self.beta_increment = beta_increment  # Annealing rate for beta
        self.epsilon = epsilon  # Small constant to avoid zero priority
        self.max_priority = 1.0  # Initial max priority

    def push(self, state, action, reward, next_state):
        """Add new experience with max priority"""
        self.buffer.append((state, action, reward, next_state))
        # Ensure priority is stored as a scalar value
        self.priorities.append(float(self.max_priority))

    def sample(self, batch_size):
        """Sample batch based on priorities"""
        if len(self.buffer) < batch_size:
            return None

        # Convert priorities to sampling probabilities - ensure all values are scalar
        priorities = np.array([float(p) for p in self.priorities])
        probs = priorities**self.alpha
        probs /= probs.sum() + self.epsilon

        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Calculate importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)  # Anneal beta
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights

        # Extract batch data
        states, actions, rewards, next_states = map(np.stack, zip(*samples))

        return states, actions, rewards, next_states, indices, weights

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, td_errors):
            # Convert error to scalar if it's an array-like object
            if hasattr(error, "__iter__"):
                error = float(error[0])  # Take first element if sequence
            else:
                error = float(error)  # Ensure it's a float

            self.priorities[idx] = error + self.epsilon
            self.max_priority = max(self.max_priority, self.priorities[idx])

    def __len__(self):
        """Return current buffer size"""
        return len(self.buffer)


class EnhancedDDPGAgent:
    def __init__(
        self,
        state_dim,
        action_dim=1,
        actor_hidden_dims=(128, 64),
        critic_hidden_dims=(128, 64),
        gamma=0.99,
        tau=0.005,
        actor_lr=1e-4,
        critic_lr=1e-3,
        buffer_capacity=100000,
        action_high=10.0,
        action_low=0.0,
        batch_size=64,
        use_twin_critics=True,
        use_prioritized_replay=True,
        grad_clip_value=1.0,
        policy_noise=0.2,
        noise_clip=0.5,
        exploration_noise=0.3,
        target_policy_noise=0.2,
        dropout_rate=0.1,
        weight_decay=1e-4,
    ):
        """
        Initialize enhanced DDPG agent with multiple improvements.

        Args:
            state_dim: State dimensionality
            action_dim: Action dimensionality (usually 1 for interest rate)
            actor_hidden_dims: Actor network hidden layer sizes
            critic_hidden_dims: Critic network hidden layer sizes
            gamma: Discount factor
            tau: Target network update rate
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            buffer_capacity: Replay buffer capacity
            action_high: Maximum action value
            action_low: Minimum action value (Zero Lower Bound)
            batch_size: Training batch size
            use_twin_critics: Whether to use twin critics (TD3-style)
            use_prioritized_replay: Whether to use prioritized replay
            grad_clip_value: Gradient clipping threshold
            policy_noise: Noise added to target policy
            noise_clip: Clipping for target policy noise
            exploration_noise: Initial exploration noise
            target_policy_noise: Noise added to target actions
            dropout_rate: Dropout rate for network regularization
            weight_decay: L2 regularization strength
        """
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Store parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.action_high = action_high
        self.action_low = action_low
        self.batch_size = batch_size
        self.use_twin_critics = use_twin_critics
        self.use_prioritized_replay = use_prioritized_replay
        self.grad_clip_value = grad_clip_value
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.target_policy_noise = target_policy_noise

        # Initialize Actor networks
        self.actor = Actor(state_dim, actor_hidden_dims, dropout_rate).to(self.device)
        self.actor_target = Actor(state_dim, actor_hidden_dims, dropout_rate).to(
            self.device
        )
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Initialize Critic networks (either single or twin)
        if use_twin_critics:
            self.critic = TwinCritic(
                state_dim, action_dim, critic_hidden_dims, dropout_rate
            ).to(self.device)
            self.critic_target = TwinCritic(
                state_dim, action_dim, critic_hidden_dims, dropout_rate
            ).to(self.device)
        else:
            self.critic = Critic(
                state_dim, action_dim, critic_hidden_dims, dropout_rate
            ).to(self.device)
            self.critic_target = Critic(
                state_dim, action_dim, critic_hidden_dims, dropout_rate
            ).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # Initialize optimizers with weight decay
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=actor_lr, weight_decay=weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=critic_lr, weight_decay=weight_decay
        )

        # Initialize replay buffer (standard or prioritized)
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity)
        else:
            self.replay_buffer = self._create_standard_buffer(buffer_capacity)

        # Initialize noise process
        self.noise = AdaptiveNoise(action_dim, sigma=exploration_noise)

        # Initialize training tracking
        self.train_step = 0
        self.update_every = 2  # Update policy every N steps (like in TD3)

    def _create_standard_buffer(self, capacity):
        """Create a standard (non-prioritized) replay buffer"""
        buffer = deque(maxlen=capacity)

        # Add methods to match the API
        def push(state, action, reward, next_state):
            buffer.append((state, action, reward, next_state))

        def sample(batch_size):
            if len(buffer) < batch_size:
                return None
            batch = random.sample(buffer, batch_size)
            states, actions, rewards, next_states = map(np.stack, zip(*batch))
            return states, actions, rewards, next_states, None, None

        class Buffer:
            def __init__(self):
                self.push = push
                self.sample = sample
                self.__len__ = lambda: len(buffer)

        return Buffer()

    def select_action(self, state, add_noise=True, noise_scale=1.0):
        """
        Select action based on current policy.

        Args:
            state: Current state
            add_noise: Whether to add exploration noise
            noise_scale: Scale factor for noise (decreased during training)

        Returns:
            Selected action
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Get action from policy
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()

        # Add exploration noise if requested
        if add_noise:
            noise_sample = self.noise.sample() * noise_scale
            action += noise_sample

        # Clip action to valid range
        action = np.clip(action, self.action_low, self.action_high)

        return action

    def train(self):
        """
        Train the agent using a batch from replay buffer

        Returns:
            Dictionary with training metrics
        """
        # Increment train step counter
        self.train_step += 1

        # Check if enough samples in buffer
        if len(self.replay_buffer) < self.batch_size:
            return {"status": "buffer_too_small"}

        # Sample from replay buffer
        sample_result = self.replay_buffer.sample(self.batch_size)
        if sample_result is None:
            return {"status": "sampling_failed"}

        if self.use_prioritized_replay:
            states, actions, rewards, next_states, indices, weights = sample_result
            weights_tensor = torch.FloatTensor(weights).to(self.device).unsqueeze(1)
        else:
            states, actions, rewards, next_states = sample_result[:4]
            indices, weights_tensor = None, None

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)

        # ===== Critic Update =====
        with torch.no_grad():
            # Select next actions from target policy with noise
            noise = torch.randn_like(actions) * self.target_policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)

            next_actions = self.actor_target(next_states) + noise
            next_actions = torch.clamp(next_actions, self.action_low, self.action_high)

            # Compute target Q values
            if self.use_twin_critics:
                # Use twin critics (TD3 style)
                q1_next, q2_next = self.critic_target(next_states, next_actions)
                # Take the minimum of the two Q values (reduces overestimation)
                next_Q = torch.min(q1_next, q2_next)
            else:
                # Standard single critic
                next_Q = self.critic_target(next_states, next_actions)

            # Compute target using Bellman equation
            target_Q = rewards + self.gamma * next_Q

        # Compute current Q estimates
        if self.use_twin_critics:
            current_Q1, current_Q2 = self.critic(states, actions)

            # Compute critic losses with prioritized weights if applicable
            if weights_tensor is not None:
                critic1_loss = (
                    weights_tensor * F.mse_loss(current_Q1, target_Q, reduction="none")
                ).mean()
                critic2_loss = (
                    weights_tensor * F.mse_loss(current_Q2, target_Q, reduction="none")
                ).mean()
            else:
                critic1_loss = F.mse_loss(current_Q1, target_Q)
                critic2_loss = F.mse_loss(current_Q2, target_Q)

            critic_loss = critic1_loss + critic2_loss

            # Calculate TD errors for prioritized replay
            if self.use_prioritized_replay and indices is not None:
                # Use scalar TD errors
                td_errors = (
                    torch.abs(current_Q1 - target_Q).detach().cpu().numpy().flatten()
                )
                self.replay_buffer.update_priorities(indices, td_errors)
        else:
            current_Q = self.critic(states, actions)

            # Compute critic loss with prioritized weights if applicable
            if weights_tensor is not None:
                critic_loss = (
                    weights_tensor * F.mse_loss(current_Q, target_Q, reduction="none")
                ).mean()
            else:
                critic_loss = F.mse_loss(current_Q, target_Q)

            # Calculate TD errors for prioritized replay
            if self.use_prioritized_replay and indices is not None:
                # Use scalar TD errors
                td_errors = (
                    torch.abs(current_Q - target_Q).detach().cpu().numpy().flatten()
                )
                self.replay_buffer.update_priorities(indices, td_errors)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_value)

        self.critic_optimizer.step()

        # ===== Actor Update =====
        # Delayed policy updates (like in TD3)
        actor_loss = None
        if self.train_step % self.update_every == 0:
            # Compute actor loss
            if self.use_twin_critics:
                # Use only first critic for policy gradients
                actor_loss = -self.critic.critic1(states, self.actor(states)).mean()
            else:
                actor_loss = -self.critic(states, self.actor(states)).mean()

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.grad_clip_value
            )

            self.actor_optimizer.step()

            # ===== Target Networks Update =====
            self._update_target_networks()

        # Return training metrics
        metrics = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item() if actor_loss is not None else None,
            "train_step": self.train_step,
        }

        return metrics

    def _update_target_networks(self):
        """Soft-update target networks"""
        # Update critic target
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        # Update actor target
        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, path):
        """Save agent state"""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "train_step": self.train_step,
                "config": {
                    "state_dim": self.state_dim,
                    "action_dim": self.action_dim,
                    "gamma": self.gamma,
                    "tau": self.tau,
                    "action_high": self.action_high,
                    "action_low": self.action_low,
                    "use_twin_critics": self.use_twin_critics,
                    "use_prioritized_replay": self.use_prioritized_replay,
                    "grad_clip_value": self.grad_clip_value,
                    "policy_noise": self.policy_noise,
                    "noise_clip": self.noise_clip,
                    "target_policy_noise": self.target_policy_noise,
                },
            },
            path,
        )
        print(f"Agent saved to {path}")

    def load(self, path):
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.train_step = checkpoint["train_step"]

        print(f"Agent loaded from {path} (training step: {self.train_step})")
        return checkpoint["config"]
