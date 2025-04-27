import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
import os
from typing import Tuple, List, Dict, Any
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class CNN_Extractor(nn.Module):
    """
    CNN feature extractor for processing visual observations.
    Based on architecture from stable-baselines3 for consistency.
    """
    def __init__(self, observation_space):
        super().__init__()
        n_input_channels = 12
        
        # Initial convolution
        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Add residual connections
        self.res1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        
        self.flatten = nn.Flatten()
        
        # Calculate output dimension
        with torch.no_grad():
            sample_input = torch.zeros(1, n_input_channels, 96, 96)
            x = F.relu(self.conv1(sample_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x + self.res1(x)  # Residual connection
            x = self.flatten(x)
            self.feature_dim = x.shape[1]
    
    def forward(self, x):
        if x.shape[-1] == 12 and len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2)
        
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x + self.res1(x)  # Residual connection
        return self.flatten(x)

class ActorCriticModel(nn.Module):
    """
    Combined actor-critic network for A2C algorithm.
    Actor outputs mean of action distribution (assuming Gaussian).
    Critic outputs state value estimate.
    """
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.feature_extractor = CNN_Extractor(observation_space)
        feature_dim = self.feature_extractor.feature_dim
        
        # Actor head (policy network)
        self.actor_mean = nn.Linear(feature_dim, action_space.shape[0])
        
        # Critic head (value network)
        self.critic = nn.Linear(feature_dim, 1)
        
        # Log standard deviation of action distribution (learnable parameter)
        self.log_std = nn.Parameter(torch.zeros(action_space.shape[0]))
        
        # Initialize weights
        self.apply(self._weights_init)
        
    @staticmethod
    def _weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, value
    
    def get_action_and_value(self, x, deterministic=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.distributions.Normal]:
        action_mean, value = self(x)
        
        # Create normal distribution with learned mean and std
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)
        
        if deterministic:
            action = action_mean
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Clip actions to valid range [-1, 1]
        action = torch.clamp(action, -1.0, 1.0)
        
        return action, value, log_prob, dist

class A2C:
    """
    Advantage Actor-Critic (A2C) implementation.
    Based on the algorithm as described in:
    "Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al., 2016)
    https://arxiv.org/abs/1602.01783
    """
    def __init__(self, 
                env,
                lr=3e-4,
                gamma=0.99,
                value_coef=0.5,
                entropy_coef=0.01,
                max_grad_norm=0.5,
                device="auto"):
        
        # Set device (GPU/CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else "cpu")
        print(f"Using device: {self.device}")
        
        # Environment and hyperparameters
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Create actor-critic model
        self.model = ActorCriticModel(self.observation_space, self.action_space).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Initialize logs
        self.rewards_history = []
        self.episode_rewards = []
        
    def _preprocess_obs(self, obs):
        """Convert observations to PyTorch tensors and send to device"""
        if isinstance(obs, np.ndarray):
            # Ensure observation is the right shape for the CNN
            # Expected shape is (batch, channels, height, width)
            if len(obs.shape) == 3:
                obs = np.expand_dims(obs, 0)  # Add batch dimension
            return torch.FloatTensor(obs).to(self.device)
        return obs  # Already a tensor

    def train(self, num_steps=1000000, update_freq=2048, eval_freq=10000, save_path="models"):
        """
        Train the A2C agent.
        Args:
            num_steps: Total environment steps to train for
            update_freq: Collect this many steps before each parameter update
            eval_freq: How often to evaluate the agent
            save_path: Where to save the model
        """
        os.makedirs(save_path, exist_ok=True)
        best_mean_reward = float('-inf')
        step_count = 0
        episode_count = 0
        update_count = 0
        
        # For logging
        episode_rewards = []
        episode_lengths = []
        
        obs = self.env.reset()[0]  # Reset environment at start of training
        episode_reward = 0
        episode_length = 0
        
        print("Starting A2C training...")
        while step_count < num_steps:
            # Storage for the current update cycle
            collected_obs = []
            collected_actions = []
            rewards = []
            masks = []  # For tracking episode endings
            
            # Collect experience for one update
            for _ in range(update_freq):
                obs_tensor = self._preprocess_obs(obs)
                
                with torch.no_grad():
                    action, _, _, dist = self.model.get_action_and_value(obs_tensor)
                
                # Execute action in environment
                action_np = action.cpu().numpy()  # Get numpy array
                if len(action_np.shape) == 2 and action_np.shape[0] == 1:
                    # If this is a batch with size 1, we're passing this through a vectorized environment
                    # The vectorized env will handle the [0] indexing
                    step_result = self.env.step(action_np)
                    if len(step_result) == 5:
                        next_obs, reward, terminated, truncated, _ = step_result
                    else:
                        # Handle the case when the environment returns 4 values (older format)
                        next_obs, reward, done, _ = step_result
                        terminated = done
                        truncated = False
                else:
                    # If not a batch with proper shape, reshape it to have proper format for vectorized env
                    action_reshaped = action_np.reshape(1, -1)  # Reshape to [1, action_dim]
                    step_result = self.env.step(action_reshaped)
                    if len(step_result) == 5:
                        next_obs, reward, terminated, truncated, _ = step_result
                    else:
                        # Handle the case when the environment returns 4 values (older format)
                        next_obs, reward, done, _ = step_result
                        terminated = done
                        truncated = False
                
                done = terminated or truncated
                
                # Store transition
                if isinstance(reward, np.ndarray):
                    reward_value = reward.item() if reward.size == 1 else reward.flatten()[0]
                else:
                    reward_value = reward
                rewards.append(torch.tensor([reward_value]).to(self.device))
                collected_obs.append(obs)
                collected_actions.append(action)
                masks.append(torch.tensor([1.0 - float(bool(done))]).to(self.device))
                
                # Update episode statistics
                episode_reward += reward
                episode_length += 1
                step_count += 1
                
                # Handle episode end
                if done:
                    if isinstance(episode_reward, np.ndarray):
                        episode_reward_value = episode_reward.item() if episode_reward.size == 1 else episode_reward.flatten()[0]
                    else:
                        episode_reward_value = episode_reward

                    if isinstance(episode_length, np.ndarray):
                        episode_length_value = episode_length.item() if episode_length.size == 1 else episode_length.flatten()[0]
                    else:
                        episode_length_value = episode_length

                    print(f"Episode {episode_count+1} finished: Reward={episode_reward_value:.2f}, Length={episode_length_value}")
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    
                    # Reset for next episode
                    obs = self.env.reset()[0]
                    episode_count += 1
                    episode_reward = 0
                    episode_length = 0
                else:
                    obs = next_obs
                
                # Break if we reach the total step limit
                if step_count >= num_steps:
                    break
            
            # Calculate returns and advantages
            values = []
            log_probs = []
            entropies = []

            # Recompute with gradient tracking for all collected observations
            for i in range(len(collected_obs)):
                obs_tensor = self._preprocess_obs(collected_obs[i])
                _, value, _, dist = self.model.get_action_and_value(obs_tensor)
                action = collected_actions[i]
                log_prob = dist.log_prob(action).sum(dim=-1)
                entropy = dist.entropy().mean()
                
                values.append(value)
                log_probs.append(log_prob)
                entropies.append(entropy)
            
            values = torch.cat(values)
            log_probs = torch.cat(log_probs)
            rewards = torch.cat(rewards)
            masks = torch.cat(masks)
            entropies = torch.stack(entropies)
            
            # Compute returns and advantages
            if done:
                last_value = torch.zeros(1, 1).to(self.device)
            else:
                with torch.no_grad():
                    last_value = self.model.get_action_and_value(self._preprocess_obs(obs))[1]
            
            returns = self.compute_returns(rewards, masks, last_value, n_steps=5)
            
            # Normalize returns for stability (optional)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Compute advantages (returns - values)
            advantages = returns - values
            
            # Update policy and value function
            values = values.flatten()
            log_probs = log_probs.flatten()
            
            # Calculate losses
            policy_loss = -(log_probs * advantages.detach()).mean()
            # Ensure both have the same shape for MSE loss
            value_loss = F.mse_loss(values, returns.flatten())
            entropy_loss = -entropies.mean()  # Maximize entropy (exploration)
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            update_count += 1
            
            # Log training information
            if update_count % 10 == 0:
                recent_rewards = episode_rewards[-10:] if episode_rewards else [0]
                print(f"Step: {step_count}/{num_steps}, Updates: {update_count}, "
                      f"Mean reward (last 10): {np.mean(recent_rewards):.2f}, "
                      f"Value loss: {value_loss.item():.4f}, "
                      f"Policy loss: {policy_loss.item():.4f}, "
                      f"Entropy: {-entropy_loss.item():.4f}")
            
            # Evaluate and save the best model
            if step_count % eval_freq < update_freq:
                mean_reward = self.evaluate()
                print(f"Evaluation at step {step_count}: Mean reward = {mean_reward:.2f}")
                
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    self.save(os.path.join(save_path, "a2c_car_racing_best"))
                    print(f"New best model saved with mean reward: {best_mean_reward:.2f}")
                
                # Also save periodic checkpoints
                # self.save(os.path.join(save_path, f"a2c_car_racing_step_{step_count}"))
        
        # Save final model
        self.save(os.path.join(save_path, "a2c_car_racing_final"))
        print("Training complete!")
        return self.rewards_history
    
    def evaluate(self, n_episodes=3):
        """Evaluate the current policy"""
        total_rewards = []
        
        for _ in range(n_episodes):
            obs = self.env.reset()[0]
            done = False
            episode_reward = 0
            
            while not done:
                obs_tensor = self._preprocess_obs(obs)
                with torch.no_grad():
                    action, _, _, _ = self.model.get_action_and_value(obs_tensor, deterministic=True)
                
                action_np = action.cpu().numpy()
                if len(action_np.shape) == 2 and action_np.shape[0] == 1:
                    step_result = self.env.step(action_np)
                    if len(step_result) == 5:
                        obs, reward, terminated, truncated, _ = step_result
                    else:
                        obs, reward, done, _ = step_result
                        terminated = done
                        truncated = False
                else:
                    action_reshaped = action_np.reshape(1, -1)  # Reshape to [1, action_dim]
                    step_result = self.env.step(action_reshaped)
                    if len(step_result) == 5:
                        obs, reward, terminated, truncated, _ = step_result
                    else:
                        obs, reward, done, _ = step_result
                        terminated = done
                        truncated = False
                done = terminated or truncated
                episode_reward += reward
            
            total_rewards.append(episode_reward)
        
        mean_reward = np.mean(total_rewards)
        self.rewards_history.append(mean_reward)
        return mean_reward
    
    def predict(self, observation, deterministic=True):
        """Get action from policy for a single observation"""
        obs_tensor = self._preprocess_obs(observation)
        with torch.no_grad():
            action, _, _, _ = self.model.get_action_and_value(obs_tensor, deterministic=deterministic)
        
        # Return in the format expected by interface.py
        return action.cpu().numpy().flatten(), None
    
    def save(self, path):
        """Save the model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, f"{path}.pt")
    
    def load(self, path):
        """Load the model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return self
    
    # Modify the advantage calculation in A2C class
    def compute_returns(self, rewards, masks, next_value, n_steps=5):
        """Compute returns with n-step bootstrapping"""
        returns = torch.zeros_like(rewards)
        future_return = next_value
        
        # Standard GAE computation
        for t in reversed(range(len(rewards))):
            if t + n_steps < len(rewards):
                # Use n-step return
                n_step_return = 0
                for i in range(n_steps):
                    n_step_return += (self.gamma**i) * rewards[t+i]
                n_step_return += (self.gamma**n_steps) * future_return
                returns[t] = n_step_return
            else:
                # For the last steps, use standard 1-step return
                returns[t] = rewards[t] + self.gamma * future_return * masks[t]
            
            future_return = returns[t]
        
        return returns

# Add a wrapper to reshape rewards
class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_pos = None
        self.prev_velocity = None
        
    def reset(self, **kwargs):
        self.prev_pos = None
        self.prev_velocity = None
        return self.env.reset(**kwargs)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract car state
        if hasattr(self.env.unwrapped, 'car'):
            pos = self.env.unwrapped.car.hull.position
            velocity = self.env.unwrapped.car.hull.linearVelocity
            
            # Reward for velocity in the right direction
            speed_reward = 0
            if self.prev_pos is not None and self.prev_velocity is not None:
                # Calculate progress on track
                speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
                speed_reward = min(speed * 0.1, 1.0)  # Cap the speed reward
                
                # Penalize going off-track
                grass_penalty = 0
                if hasattr(obs, 'shape') and obs.shape[-1] >= 3:
                    # Check green channel for grass
                    green_intensity = np.mean(obs[..., 1])
                    if green_intensity > 150:
                        grass_penalty = -0.2
                
                shaped_reward = reward + speed_reward + grass_penalty
            else:
                shaped_reward = reward
            
            self.prev_pos = pos
            self.prev_velocity = velocity
            return obs, shaped_reward, terminated, truncated, info
        
        return obs, reward, terminated, truncated, info

# Add observation normalization
class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x):
        # Reshape x to match the shape of self.mean if needed
        if isinstance(x, np.ndarray) and x.shape != self.mean.shape:
            if self.mean.size == x.size:  # They have the same number of elements
                # If this is our first serious update, update the shape based on the input
                if np.all(self.mean == 0) and np.all(self.var == 1) and self.count <= 1:
                    # Reset with the proper shape
                    orig_shape = x.shape
                    if len(x.shape) > 1 and x.shape[0] == 1:
                        # If x has a batch dimension of 1, remove it
                        orig_shape = x.shape[1:]
                    self.mean = np.zeros(orig_shape, dtype=np.float64)
                    self.var = np.ones(orig_shape, dtype=np.float64)
                else:
                    # Otherwise, reshape x to match current stats shape
                    x = x.reshape(self.mean.shape)
        
        # Compute batch statistics
        batch_mean = np.mean(x, axis=0) if x.shape[0] > 1 else x[0]
        batch_var = np.var(x, axis=0) if x.shape[0] > 1 else np.zeros_like(x[0])
        batch_count = x.shape[0]
        
        # Update running statistics
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        # Update mean and variance using Welford's algorithm
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count

class NormalizeObservation(gym.Wrapper):
    def __init__(self, env, epsilon=1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        # We'll initialize the running stats after we see the first observation
        self.obs_rms = None
        # Whether we've already initialized the running stats
        self.initialized = False
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        normalized_obs = self._normalize_observation(obs)
        return normalized_obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._normalize_observation(obs)
    
    def _normalize_observation(self, obs):
        # Handle tuple observations (from SB3 reset)
        if isinstance(obs, tuple) and len(obs) == 2:
            obs_array = obs[0]
            info = obs[1]
        else:
            obs_array = obs
            info = None

        # Skip normalization for non-numpy arrays
        if not isinstance(obs_array, np.ndarray):
            return obs

        # Initialize running statistics if this is the first observation
        if self.obs_rms is None:
            self.obs_rms = RunningMeanStd(shape=obs_array.shape)
            # Make a copy to avoid modifying the original
            obs_for_stats = obs_array.copy().reshape(1, *obs_array.shape)
            self.obs_rms.update(obs_for_stats)
            
        # Update statistics with the current observation
        obs_for_stats = obs_array.copy().reshape(1, *obs_array.shape)
        self.obs_rms.update(obs_for_stats)
            
        # Normalize the observation
        normalized_obs = (obs_array - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        
        # Return in the same format as received
        if isinstance(obs, tuple):
            return (normalized_obs, info)
        return normalized_obs

# Training script at the bottom for when the file is run directly
if __name__ == "__main__":
    import os
    from stable_baselines3.common.vec_env import VecTransposeImage
    
    # Create directories for saving models and logs
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    def make_env():
        """Create and wrap the CarRacing environment with improvements"""
        def _init():
            env = gym.make("CarRacing-v3", continuous=True)
            env = Monitor(env, "logs/car_racing_a2c_improved")
            env = RewardShapingWrapper(env)
            env = NormalizeObservation(env)
            return env
        return _init
    
    # Create and wrap the training environment
    env = DummyVecEnv([make_env()])
    env = VecFrameStack(env, n_stack=4)  # Stack 4 frames to capture motion information
    env = VecTransposeImage(env)  # Convert from (H,W,C) to (C,H,W)
    
    # Create the A2C agent
    agent = A2C(
        env=env,
        lr=1e-4,  # Lower learning rate for more stability
        gamma=0.99,
        value_coef=0.5,
        entropy_coef=0.05,  # Higher entropy coefficient for better exploration
        max_grad_norm=0.5,
        device="auto"
    )
    
    # Train the agent
    agent.train(
        num_steps=2000000,  # Double the training steps
        update_freq=128,    # More frequent updates
        eval_freq=10000,
        save_path="models"
    )