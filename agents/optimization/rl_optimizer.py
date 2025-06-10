# optimization/rl_optimizer.py
# Purpose: Implements a reinforcement learning optimizer using a policy network to
# dynamically adjust trading strategy parameters, optimizing for options trading
# performance metrics like Sharpe ratio and drawdown.

import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


class RLPolicyNetwork(nn.Module):
    """Neural network for the reinforcement learning policy."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Initialize the policy network.

        Args:
            input_dim (int): Dimension of the state space.
            hidden_dim (int): Hidden layer dimension.
            output_dim (int): Number of possible actions.
        """
        super(RLPolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Action probabilities.
        """
        return self.network(x)


class RLOptimizer:
    """Uses reinforcement learning to optimize agent parameters dynamically."""

    def __init__(
        self, agent_configs: List[Dict], input_dim: int = 3, hidden_dim: int = 64
    ):
        """Initialize the RLOptimizer with a policy network.

        Args:
            agent_configs (List[Dict]): List of possible agent configurations.
            input_dim (int): Dimension of the state space (e.g., Sharpe, drawdown, return).
            hidden_dim (int): Hidden layer dimension for the policy network.
        """
        self.agent_configs = agent_configs
        output_dim = len(agent_configs)
        self.policy = RLPolicyNetwork(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
        logger.info({"message": "RLOptimizer initialized"})

    def select_action(self, state: np.ndarray) -> int:
        """Select an action (configuration) based on the current state.

        Args:
            state (np.ndarray): State vector (e.g., [sharpe_ratio, drawdown, total_return]).

        Returns:
            int: Selected action (index of the configuration).
        """
        logger.debug({"message": "Selecting action"})
        try:
            state_tensor = torch.FloatTensor(state)
            probs = self.policy(state_tensor)
            action = torch.multinomial(probs, 1).item()
            logger.debug({"message": f"Selected action: {action}"})
            return action
        except Exception as e:
            logger.error({"message": f"Failed to select action: {str(e)}"})
            return 0

    def update_policy(self, state: np.ndarray, action: int, reward: float):
        """Update the policy network based on the observed reward.

        Args:
            state (np.ndarray): State vector.
            action (int): Action taken.
            reward (float): Reward received.
        """
        logger.info({"message": f"Updating RL policy with reward {reward}"})
        try:
            state_tensor = torch.FloatTensor(state)
            action_tensor = torch.tensor(action)
            reward_tensor = torch.tensor(reward)

            probs = self.policy(state_tensor)
            log_prob = torch.log(probs[action_tensor])
            loss = -log_prob * reward_tensor

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            logger.debug({"message": "Policy updated successfully"})
        except Exception as e:
            logger.error({"message": f"Failed to update RL policy: {str(e)}"})

    def optimize(self, performance_metrics: Dict) -> Dict:
        """Optimize agent parameters using reinforcement learning.

        Args:
            performance_metrics (Dict): Metrics to evaluate performance (e.g., {'sharpe_ratio': 1.5}).

        Returns:
            Dict: Selected configuration for options trading strategy.
        """
        logger.info({"message": "Optimizing agent parameters with RL"})
        try:
            # State: [sharpe_ratio, drawdown, total_return]
            state = np.array(
                [
                    performance_metrics.get("sharpe_ratio", 0.0),
                    performance_metrics.get("max_drawdown", 0.0),
                    performance_metrics.get("total_return", 0.0),
                ]
            )
            action = self.select_action(state)
            # Calculate reward (prioritize Sharpe ratio, penalize drawdown)
            reward = performance_metrics.get("sharpe_ratio", 0.0) - 0.5 * abs(
                performance_metrics.get("max_drawdown", 0.0)
            )
            self.update_policy(state, action, reward)
            selected_config = self.agent_configs[action]
            logger.info({"message": f"Selected agent config: {selected_config}"})
            return selected_config
        except Exception as e:
            logger.error({"message": f"Failed to optimize with RL: {str(e)}"})
            return self.agent_configs[0] if self.agent_configs else {}
