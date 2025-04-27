import gc
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
import random
from collections import deque

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

torch.set_default_dtype(torch.float32)  # Use proper modern syntax
USE_HALF_PRECISION = True  # Control flag

class ReplayBuffer:
    """Experience replay buffer to store and sample transitions"""
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device
        self.obs_shape = None  # Track observation shape for consistency
        
    def push(self, state, action, reward, next_state, done):
        """Add transition to buffer with memory optimization"""
        # Convert to tensors with memory optimization
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
            if USE_HALF_PRECISION and self.device.type == 'cuda':
                state = state.half()
                
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action)
            if USE_HALF_PRECISION and self.device.type == 'cuda':
                action = action.half()
                
        if isinstance(reward, (np.ndarray, np.number)):
            reward = torch.FloatTensor([float(reward)])
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state)
            if USE_HALF_PRECISION and self.device.type == 'cuda':
                next_state = next_state.half()
                
        if isinstance(done, (np.ndarray, np.number, bool)):
            done = torch.FloatTensor([float(done)])
        
        # Ensure consistent tensor shapes for observations
        if state.dim() == 4 and state.shape[0] == 1:  # If shape is [1, channels, H, W]
            state = state.squeeze(0)  # Make it [channels, H, W]
            
        if next_state.dim() == 4 and next_state.shape[0] == 1:  # If shape is [1, channels, H, W]
            next_state = next_state.squeeze(0)  # Make it [channels, H, W]
            
        # Store the transition
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Memory-efficient sampling"""
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        # Process in smaller chunks to reduce peak memory usage
        chunk_size = min(16, batch_size)
        states_list, actions_list = [], []
        rewards_list, next_states_list, dones_list = [], [], []
        
        for i in range(0, batch_size, chunk_size):
            chunk_indices = indices[i:i+chunk_size]
            chunk = [self.buffer[idx] for idx in chunk_indices]
            
            states_chunk = [self._process_state(b[0]) for b in chunk]
            next_states_chunk = [self._process_state(b[3]) for b in chunk]
            
            states_list.append(torch.cat(states_chunk))
            actions_list.append(torch.stack([b[1] for b in chunk]))
            rewards_list.append(torch.stack([b[2] for b in chunk]))
            next_states_list.append(torch.cat(next_states_chunk))
            dones_list.append(torch.stack([b[4] for b in chunk]))
        
        states = torch.cat(states_list).to(self.device)
        actions = torch.cat(actions_list).to(self.device)
        rewards = torch.cat(rewards_list).to(self.device)
        next_states = torch.cat(next_states_list).to(self.device)
        dones = torch.cat(dones_list).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def _process_state(self, state):
        """Process state for consistent shape before stacking"""
        if state.dim() == 3:  # [C, H, W]
            state = state.unsqueeze(0)  # [1, C, H, W]
        return state
    
    def __len__(self):
        return len(self.buffer)

class EfficientCNN_Extractor(nn.Module):
    def __init__(self, observation_space, reduced_size=64):
        super().__init__()
        n_input_channels = 12  # 4 stacked frames with 3 channels each
        
        # Much smaller CNN
        self.conv1 = nn.Conv2d(n_input_channels, 8, kernel_size=8, stride=4)   # 16→8
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2)                 # 32→16
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1)                # 32→16
        
        self.flatten = nn.Flatten()
        
        # Calculate output dimension based on reduced image size
        with torch.no_grad():
            sample_input = torch.zeros(1, n_input_channels, reduced_size, reduced_size)
            x = F.relu(self.conv1(sample_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = self.flatten(x)
            self.feature_dim = x.shape[1]
    
    def forward(self, x):
        # Handle different input formats
        if len(x.shape) == 5:
            x = x.squeeze(1)
        elif x.shape[-1] == 12 and len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2)
        
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.flatten(x)

class QNetwork(nn.Module):
    """Critic network for SAC"""
    def __init__(self, observation_space, action_space, hidden_dim=256):
        super().__init__()
        
        self.feature_extractor = EfficientCNN_Extractor(observation_space)
        feature_dim = self.feature_extractor.feature_dim
        action_dim = action_space.shape[0]
        
        # Q1 architecture
        self.q1_linear1 = nn.Linear(feature_dim + action_dim, hidden_dim)
        self.q1_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_linear3 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture (for reducing overestimation bias)
        self.q2_linear1 = nn.Linear(feature_dim + action_dim, hidden_dim)
        self.q2_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_linear3 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._weights_init)
        
    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, state, action):
        features = self.feature_extractor(state)
        
        # Concatenate state and action
        x = torch.cat([features, action], 1)
        
        # Q1 network
        q1 = F.relu(self.q1_linear1(x))
        q1 = F.relu(self.q1_linear2(q1))
        q1 = self.q1_linear3(q1)
        
        # Q2 network
        q2 = F.relu(self.q2_linear1(x))
        q2 = F.relu(self.q2_linear2(q2))
        q2 = self.q2_linear3(q2)
        
        return q1, q2

class GaussianPolicy(nn.Module):
    """Actor network for SAC that outputs a Gaussian policy"""
    def __init__(self, observation_space, action_space, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        
        self.feature_extractor = EfficientCNN_Extractor(observation_space)
        feature_dim = self.feature_extractor.feature_dim
        self.action_dim = action_space.shape[0]
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Policy network
        self.linear1 = nn.Linear(feature_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Mean and log_std outputs
        self.mean_linear = nn.Linear(hidden_dim, self.action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, self.action_dim)
        
        # Initialize weights
        self.apply(self._weights_init)
        
    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        features = self.feature_extractor(state)
        
        x = F.relu(self.linear1(features))
        x = F.relu(self.linear2(x))
        
        # Output mean and log_std
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state):
        """Sample an action from the policy"""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # Calculate log probability with correction for tanh squashing
        log_prob = normal.log_prob(x_t)
        # Apply tanh squashing correction: log_prob -= log(1 - tanh(x)^2)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # For mean action (used in evaluation)
        mean_action = torch.tanh(mean)
        
        return action, log_prob, mean_action

class SAC:
    """
    Soft Actor-Critic (SAC) algorithm implementation
    Based on the paper "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
    """
    def __init__(self, 
                env,
                lr_actor=3e-4,
                lr_critic=3e-4,
                lr_alpha=3e-4,
                gamma=0.99,
                tau=0.005,
                alpha=0.2,
                auto_entropy_tuning=True,
                buffer_size=1000000,
                batch_size=256,
                initial_random_steps=10000,
                update_every=1,
                device="auto"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else "cpu")
        print(f"Using device: {self.device}")
        
        # Environment and hyperparameters
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.initial_random_steps = initial_random_steps
        self.update_every = update_every
        
        # Initialize actor (policy) network
        self.actor = GaussianPolicy(self.observation_space, self.action_space).to(self.device)
        
        # Initialize critic (Q) networks and target networks
        self.critic = QNetwork(self.observation_space, self.action_space).to(self.device)
        self.critic_target = QNetwork(self.observation_space, self.action_space).to(self.device)
        
        # Copy weights from critic to critic_target
        self.hard_update(self.critic_target, self.critic)
        
        # Set up optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize entropy coefficient (alpha)
        self.auto_entropy_tuning = auto_entropy_tuning
        if auto_entropy_tuning:
            # Target entropy is -|A|
            self.target_entropy = -torch.prod(torch.tensor(self.action_space.shape)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor([alpha], device=self.device)
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, self.device)
        
        # For tracking progress
        self.rewards_history = []
        self.total_steps = 0
        
    def _preprocess_obs(self, obs):
        """Convert observations to PyTorch tensors and send to device"""
        if isinstance(obs, np.ndarray):
            # Ensure observation is the right shape for the CNN
            if len(obs.shape) == 3:  # [H, W, C] or [C, H, W]
                # If last dimension is 12 (stacked frames), it's likely [H, W, C]
                if obs.shape[-1] == 12:
                    # Convert to [C, H, W] for PyTorch
                    obs = np.transpose(obs, (2, 0, 1))
                obs = np.expand_dims(obs, 0)  # Add batch dimension -> [1, C, H, W]
            elif len(obs.shape) == 4:
                # If it's already a batch with shape [B, H, W, C]
                if obs.shape[-1] == 12:
                    # Convert to [B, C, H, W] for PyTorch
                    obs = np.transpose(obs, (0, 3, 1, 2))
        
            return torch.FloatTensor(obs).to(self.device)
        
        # If it's already a tensor, ensure it's on the right device
        if isinstance(obs, torch.Tensor):
            return obs.to(self.device)
        
        return obs  # Return as is if not numpy or tensor
    
    def select_action(self, obs, evaluate=False):
        """Select action using the policy"""
        with torch.no_grad():
            obs_tensor = self._preprocess_obs(obs)
            
            if not evaluate:  # Training mode
                if self.total_steps < self.initial_random_steps:
                    # Random exploration in the beginning
                    action = torch.FloatTensor(self.action_space.sample()).to(self.device)
                else:
                    # Sample from policy
                    action, _, _ = self.actor.sample(obs_tensor)
                    action = action.squeeze(0)
            else:  # Evaluation mode - use mean action
                _, _, action = self.actor.sample(obs_tensor)
                action = action.squeeze(0)
                
            return action.cpu().numpy()
    
    def update_parameters(self):
        """Update SAC parameters using a batch from replay buffer"""
        # Skip if not enough transitions in buffer
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample a batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Update critic (Q-functions)
        self.update_critic(states, actions, rewards, next_states, dones)
        
        # Update actor (policy) and alpha (temperature)
        self.update_actor_and_alpha(states)
        
        # Soft update target networks
        self.soft_update(self.critic_target, self.critic)
    
    def update_critic(self, states, actions, rewards, next_states, dones):
        """Update the critic networks"""
        with torch.no_grad():
            # Sample actions from current policy for next states
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            
            # Get target Q-values
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + (1 - dones) * self.gamma * (next_q - self.alpha * next_log_probs)
        
        # Get current Q-values
        current_q1, current_q2 = self.critic(states, actions)
        
        # Calculate critic loss
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = q1_loss + q2_loss
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
    
    def update_actor_and_alpha(self, states):
        """Update the actor (policy) network and entropy coefficient (alpha)"""
        # Get actions and log probs from current policy
        actions, log_probs, _ = self.actor.sample(states)
        
        # Get Q-values for actions from current policy
        q1, q2 = self.critic(states, actions)
        min_q = torch.min(q1, q2)
        
        # Calculate actor loss
        actor_loss = (self.alpha * log_probs - min_q).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha (temperature) if auto-tuning is enabled
        if self.auto_entropy_tuning:
            # Calculate alpha loss
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            # Update alpha
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # Update alpha value
            self.alpha = self.log_alpha.exp()
    
    def soft_update(self, target, source):
        """Soft update model parameters: θ_target = τ*θ_source + (1-τ)*θ_target"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def hard_update(self, target, source):
        """Hard update model parameters: θ_target = θ_source"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)
    
    def train(self, num_steps=1000000, eval_freq=10000, save_path="models"):
        """
        Train the SAC agent.
        Args:
            num_steps: Total environment steps to train for
            eval_freq: How often to evaluate the agent
            save_path: Where to save the model
        """
        os.makedirs(save_path, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        best_mean_reward = float('-inf')
        episode_count = 0
        episode_reward = 0
        episode_length = 0
        
        # For logging
        episode_rewards = []
        episode_lengths = []
        
        # Reset environment
        obs = self.env.reset()[0]
        
        print("Starting SAC training...")
        for step in range(1, num_steps + 1):
            # Select action
            action = self.select_action(obs)
            
            # Execute action in environment
            action_np = action.reshape(1, -1)  # Reshape for vectorized env
            step_result = self.env.step(action_np)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, _ = step_result
            else:
                # Handle the case when the environment returns 4 values (older format)
                next_obs, reward, done, _ = step_result
                terminated = done
                truncated = False
            
            done = terminated or truncated
            
            # Add transition to replay buffer - with error checking
            try:
                # Ensure observation is properly preprocessed
                if isinstance(obs, np.ndarray):
                    obs_tensor = torch.FloatTensor(obs)
                else:
                    obs_tensor = obs
                
                if isinstance(next_obs, np.ndarray):
                    next_obs_tensor = torch.FloatTensor(next_obs)
                else:
                    next_obs_tensor = next_obs
                    
                # Get scalar values
                reward_value = float(reward) if not isinstance(reward, (list, tuple, np.ndarray)) else float(reward.item() if hasattr(reward, 'item') else reward[0])
                done_value = float(done) if not isinstance(done, (list, tuple, np.ndarray)) else float(done.item() if hasattr(done, 'item') else done[0])
                
                self.replay_buffer.push(
                    obs_tensor, 
                    torch.FloatTensor(action), 
                    torch.FloatTensor([reward_value]), 
                    next_obs_tensor, 
                    torch.FloatTensor([done_value])
                )
            except Exception as e:
                print(f"Error adding to replay buffer: {e}")
                print(f"obs type: {type(obs)}, shape: {np.shape(obs) if isinstance(obs, np.ndarray) else None}")
                print(f"action type: {type(action)}, shape: {np.shape(action) if isinstance(action, np.ndarray) else None}")
                print(f"reward type: {type(reward)}, value: {reward}")
                print(f"next_obs type: {type(next_obs)}, shape: {np.shape(next_obs) if isinstance(next_obs, np.ndarray) else None}")
                print(f"done type: {type(done)}, value: {done}")
            
            # Extract scalar values from numpy arrays
            if isinstance(reward, np.ndarray):
                reward_value = reward.item() if reward.size == 1 else reward.flatten()[0]
            else:
                reward_value = reward
                
            # Update observations and stats
            obs = next_obs
            episode_reward += reward_value
            episode_length += 1
            self.total_steps += 1
            
            # Update parameters if it's time
            if step >= self.initial_random_steps and step % self.update_every == 0:
                self.update_parameters()
            
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
                episode_rewards.append(episode_reward_value)
                episode_lengths.append(episode_length_value)
                
                # Reset for next episode
                obs = self.env.reset()[0]
                episode_count += 1
                episode_reward = 0
                episode_length = 0
                
                # Log every few episodes
                if episode_count % 10 == 0:
                    recent_rewards = episode_rewards[-10:]
                    print(f"Step: {step}/{num_steps}, Episodes: {episode_count}, "
                          f"Mean reward (last 10): {np.mean(recent_rewards):.2f}, "
                          f"Buffer size: {len(self.replay_buffer)}")
            
            # Evaluate and save the best model
            if step % eval_freq == 0:
                # Free up memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Keep only a single checkpoint to save disk space
                if step > eval_freq:
                    for old_step in range(eval_freq, step, eval_freq):
                        old_checkpoint = os.path.join(save_path, f"sac_car_racing_step_{old_step}.pt")
                        if os.path.exists(old_checkpoint):
                            os.remove(old_checkpoint)
                
                mean_reward = self.evaluate()
                print(f"Evaluation at step {step}: Mean reward = {mean_reward:.2f}")
                self.rewards_history.append(mean_reward)
                
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    self.save(os.path.join(save_path, "sac_car_racing_best"))
                    print(f"New best model saved with mean reward: {best_mean_reward:.2f}")
                    
                # Also save periodic checkpoints
                self.save(os.path.join(save_path, f"sac_car_racing_step_{step}"))
        
        # Save final model
        self.save(os.path.join(save_path, "sac_car_racing_final"))
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
                action = self.select_action(obs, evaluate=True)
                action_np = action.reshape(1, -1)
                step_result = self.env.step(action_np)
                
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, _ = step_result
                else:
                    obs, reward, done, _ = step_result
                    terminated = done
                    truncated = False
                
                done = terminated or truncated
                episode_reward += reward
            
            if isinstance(episode_reward, np.ndarray):
                episode_reward = episode_reward.item() if episode_reward.size == 1 else episode_reward.flatten()[0]
            
            total_rewards.append(episode_reward)
        
        mean_reward = np.mean(total_rewards)
        return mean_reward
    
    def predict(self, observation, deterministic=True):
        """Get action from policy for a single observation (interface compatibility)"""
        with torch.no_grad():
            obs_tensor = self._preprocess_obs(observation)
            _, _, action_mean = self.actor.sample(obs_tensor)
            action = action_mean if deterministic else self.select_action(observation)
            
        # Return in the format expected by interface.py
        return action.cpu().numpy().flatten(), None
    
    def save(self, path):
        """Save the model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.auto_entropy_tuning else None,
            'total_steps': self.total_steps,
            'rewards_history': self.rewards_history
        }, f"{path}.pt")
    
    def load(self, path):
        """Load the model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if self.auto_entropy_tuning:
            if checkpoint['log_alpha'] is not None:
                self.log_alpha = checkpoint['log_alpha']
                self.alpha = self.log_alpha.exp()
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        self.total_steps = checkpoint['total_steps']
        if 'rewards_history' in checkpoint:
            self.rewards_history = checkpoint['rewards_history']
            
        return self

# Add this class after your imports
class ResizeObservationWrapper(gym.ObservationWrapper):
    """Resize observations to reduce memory usage."""
    def __init__(self, env, target_size=(64, 64)):
        super().__init__(env)
        self.target_size = target_size
        
        # Update observation space to match new size
        if isinstance(env.observation_space, gym.spaces.Box):
            old_shape = env.observation_space.shape
            if len(old_shape) == 3 and old_shape[0:2] == (96, 96):
                low = env.observation_space.low.min()
                high = env.observation_space.high.max()
                self.observation_space = gym.spaces.Box(
                    low=low, 
                    high=high, 
                    shape=(target_size[0], target_size[1], old_shape[2]),
                    dtype=env.observation_space.dtype
                )
    
    def observation(self, obs):
        """Resize the observation."""
        return self._resize_observation(obs)
    
    def _resize_observation(self, obs):
        """Resize observations using NumPy only"""
        if isinstance(obs, np.ndarray):
            if len(obs.shape) == 3 and obs.shape[:2] == (96, 96):
                resized = np.zeros((self.target_size[0], self.target_size[1], obs.shape[2]), dtype=obs.dtype)
                h_ratio = obs.shape[0] / self.target_size[0]
                w_ratio = obs.shape[1] / self.target_size[1]
                
                for y in range(self.target_size[0]):
                    for x in range(self.target_size[1]):
                        src_y = min(int(y * h_ratio), obs.shape[0] - 1)
                        src_x = min(int(x * w_ratio), obs.shape[1] - 1)
                        resized[y, x] = obs[src_y, src_x]
                return resized
            elif len(obs.shape) == 4 and obs.shape[1:3] == (96, 96):  # Batched observation
                resized = np.zeros((obs.shape[0], self.target_size[0], self.target_size[1], obs.shape[3]), dtype=obs.dtype)
                for i in range(obs.shape[0]):
                    h_ratio = obs.shape[1] / self.target_size[0]
                    w_ratio = obs.shape[2] / self.target_size[1]
                    
                    for y in range(self.target_size[0]):
                        for x in range(self.target_size[1]):
                            src_y = min(int(y * h_ratio), obs.shape[1] - 1)
                            src_x = min(int(x * w_ratio), obs.shape[2] - 1)
                            resized[i, y, x] = obs[i, src_y, src_x]
                            
                return resized
        return obs

# Training script at the bottom for when the file is run directly
if __name__ == "__main__":
    import os
    from stable_baselines3.common.vec_env import VecTransposeImage
    
    # Create directories for saving models and logs
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    def preprocess_observation(obs, target_size=(64, 64)):
        """Resize observations using NumPy only"""
        if isinstance(obs, np.ndarray):
            if len(obs.shape) == 3 and obs.shape[:2] == (96, 96):  # Single observation
                resized = np.zeros((target_size[0], target_size[1], obs.shape[2]), dtype=obs.dtype)
                h_ratio = obs.shape[0] / target_size[0]
                w_ratio = obs.shape[1] / target_size[1]
                
                for y in range(target_size[0]):
                    for x in range(target_size[1]):
                        src_y = min(int(y * h_ratio), obs.shape[0] - 1)
                        src_x = min(int(x * w_ratio), obs.shape[1] - 1)
                        resized[y, x] = obs[src_y, src_x]
                        
                return resized
                
            elif len(obs.shape) == 4 and obs.shape[1:3] == (96, 96):  # Batched observation
                resized = np.zeros((obs.shape[0], target_size[0], target_size[1], obs.shape[3]), dtype=obs.dtype)
                for i in range(obs.shape[0]):
                    h_ratio = obs.shape[1] / target_size[0]
                    w_ratio = obs.shape[2] / target_size[1]
                    
                    for y in range(target_size[0]):
                        for x in range(target_size[1]):
                            src_y = min(int(y * h_ratio), obs.shape[1] - 1)
                            src_x = min(int(x * w_ratio), obs.shape[2] - 1)
                            resized[i, y, x] = obs[i, src_y, src_x]
                            
                return resized
        return obs
    
    def make_env():
        """Create and wrap the CarRacing environment"""
        def _init():
            env = gym.make("CarRacing-v3", continuous=True)
            env = Monitor(env, "logs/car_racing_sac")
            # Use our custom wrapper instead of LambdaObservation
            env = ResizeObservationWrapper(env, target_size=(64, 64))
            return env
        return _init
    
    # Create and wrap the training environment
    env = DummyVecEnv([make_env()])
    env = VecFrameStack(env, n_stack=4)  # Stack 4 frames to capture motion information
    env = VecTransposeImage(env)  # Convert from (H,W,C) to (C,H,W)
    
    # Create the SAC agent
    agent = SAC(
        env=env,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_entropy_tuning=True,
        buffer_size=50000,  # Further reduced from 100,000
        batch_size=32,      # Further reduced from 64
        initial_random_steps=500,  # Further reduced from 1,000
        update_every=2,     # Update less frequently
        device="auto"
    )
    
    # Train the agent
    agent.train(
        num_steps=1500000,  # Total steps
        eval_freq=10000,    # Evaluation frequency
        save_path="models"  # Where to save models
    )