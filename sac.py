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
import psutil
import traceback

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

torch.set_default_dtype(torch.float32)  # Use proper modern syntax
USE_HALF_PRECISION = True  # Control flag

# Memory optimization for sac.py

# 1. More memory-efficient buffer with dynamic growth and lower precision
class ReplayBuffer:
    """Memory-efficient experience replay buffer with gradual growth"""
    def __init__(self, capacity, observation_shape, action_dim, device):
        self.device = device
        self.max_capacity = capacity
        self.pos = 0
        self.full = False
        
        # Start with a smaller buffer size and grow as needed
        self.current_capacity = min(50000, capacity)
        
        # Use float16 for observations to reduce memory footprint by 50%
        self.states = np.zeros((self.current_capacity, *observation_shape), dtype=np.float16)
        self.actions = np.zeros((self.current_capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.current_capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((self.current_capacity, *observation_shape), dtype=np.float16)
        self.dones = np.zeros((self.current_capacity, 1), dtype=np.float32)
        
        # Track actual buffer usage
        self.size = 0
        
    def _maybe_grow_buffer(self):
        """Grow the buffer if needed and possible"""
        # If buffer is 90% full and we haven't reached max capacity
        if self.size > 0.9 * self.current_capacity and self.current_capacity < self.max_capacity:
            new_capacity = min(self.current_capacity * 2, self.max_capacity)
            print(f"Growing buffer from {self.current_capacity} to {new_capacity}")
            
            # Create new arrays with increased capacity
            new_states = np.zeros((new_capacity, *self.states.shape[1:]), dtype=self.states.dtype)
            new_actions = np.zeros((new_capacity, self.actions.shape[1]), dtype=self.actions.dtype)
            new_rewards = np.zeros((new_capacity, 1), dtype=self.rewards.dtype)
            new_next_states = np.zeros((new_capacity, *self.next_states.shape[1:]), dtype=self.next_states.dtype)
            new_dones = np.zeros((new_capacity, 1), dtype=self.dones.dtype)
            
            # Copy existing data
            if self.full:
                # Data wraps around the buffer
                new_states[:self.current_capacity-self.pos] = self.states[self.pos:]
                new_states[self.current_capacity-self.pos:self.current_capacity] = self.states[:self.pos]
                
                new_actions[:self.current_capacity-self.pos] = self.actions[self.pos:]
                new_actions[self.current_capacity-self.pos:self.current_capacity] = self.actions[:self.pos]
                
                new_rewards[:self.current_capacity-self.pos] = self.rewards[self.pos:]
                new_rewards[self.current_capacity-self.pos:self.current_capacity] = self.rewards[:self.pos]
                
                new_next_states[:self.current_capacity-self.pos] = self.next_states[self.pos:]
                new_next_states[self.current_capacity-self.pos:self.current_capacity] = self.next_states[:self.pos]
                
                new_dones[:self.current_capacity-self.pos] = self.dones[self.pos:]
                new_dones[self.current_capacity-self.pos:self.current_capacity] = self.dones[:self.pos]
                
                self.pos = self.current_capacity
            else:
                # Data is contiguous
                new_states[:self.pos] = self.states[:self.pos]
                new_actions[:self.pos] = self.actions[:self.pos]
                new_rewards[:self.pos] = self.rewards[:self.pos]
                new_next_states[:self.pos] = self.next_states[:self.pos]
                new_dones[:self.pos] = self.dones[:self.pos]
            
            # Replace with new arrays
            self.states = new_states
            self.actions = new_actions
            self.rewards = new_rewards
            self.next_states = new_next_states
            self.dones = new_dones
            
            # Update capacity
            self.current_capacity = new_capacity
            
            # Force garbage collection
            gc.collect()
        
    def push(self, state, action, reward, next_state, done):
        """Store transition in buffer"""
        # Convert to numpy if needed
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if isinstance(reward, torch.Tensor):
            reward = reward.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        if isinstance(done, torch.Tensor):
            done = done.cpu().numpy()
            
        # Handle scalar values
        if np.isscalar(reward):
            reward = np.array([reward], dtype=np.float32)
        if np.isscalar(done) or (isinstance(done, np.ndarray) and done.ndim > 0):
            # Extract single element from array to avoid deprecation warning
            if isinstance(done, np.ndarray) and done.size == 1:
                done = done.item()
            done = np.array([float(done)], dtype=np.float32)
        
        # Store data in buffer
        self.states[self.pos] = state.astype(np.float16)  # Convert to float16 for storage
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state.astype(np.float16)
        self.dones[self.pos] = done
        
        # Update position
        self.pos = (self.pos + 1) % self.current_capacity
        if self.pos == 0:
            self.full = True
        
        # Update size
        self.size = self.current_capacity if self.full else self.pos
        
        # Check if we need to grow the buffer
        self._maybe_grow_buffer()
    
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        # Adjust batch size if buffer doesn't have enough samples
        actual_batch_size = min(batch_size, self.size)
        
        # Sample indices
        indices = np.random.randint(0, self.size, actual_batch_size)
        if self.full:
            # Adjust indices for wrapped buffer
            indices = (indices + self.pos) % self.current_capacity
        
        # Get batch data
        state_batch = torch.FloatTensor(self.states[indices].astype(np.float32)).to(self.device)
        action_batch = torch.FloatTensor(self.actions[indices]).to(self.device)
        reward_batch = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_state_batch = torch.FloatTensor(self.next_states[indices].astype(np.float32)).to(self.device)
        done_batch = torch.FloatTensor(self.dones[indices]).to(self.device)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
    
    def __len__(self):
        return self.size

class EfficientCNN_Extractor(nn.Module):
    def __init__(self, observation_space):
        super().__init__()
        n_input_channels = 12  # 4 stacked frames with 3 channels each
        
        # CNN for 96x96 images
        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.flatten = nn.Flatten()
        
        # Calculate output dimension
        with torch.no_grad():
            sample_input = torch.zeros(1, n_input_channels, 96, 96)
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
                buffer_size=400000,  # Reduced buffer size
                batch_size=128,      # Increased batch size for efficiency
                initial_random_steps=2000,  # Reduced initial exploration
                update_every=2,      # Update less frequently
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
        self.total_steps = 0
        
        # Initialize actor (policy) network
        self.actor = GaussianPolicy(self.observation_space, self.action_space, hidden_dim=128).to(self.device)
        
        # Initialize critic (Q) networks with smaller networks
        self.critic = QNetwork(self.observation_space, self.action_space, hidden_dim=128).to(self.device)
        self.critic_target = QNetwork(self.observation_space, self.action_space, hidden_dim=128).to(self.device)
        
        # Copy weights from critic to critic_target
        self.hard_update(self.critic_target, self.critic)
        
        # Set up optimizers with gradient checkpointing for memory efficiency
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
        
        # Determine observation shape for buffer
        # For stacked frames: [channels, height, width]
        obs_shape = (12, 96, 96)  # 4 frames x 3 channels, 96x96 resolution
        action_dim = self.action_space.shape[0]
        
        # Create memory-efficient replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, obs_shape, action_dim, self.device)
        
        # For tracking progress
        self.rewards_history = []
        self.memory_usage_history = []
        self.best_mean_reward = float('-inf')
        
        # Memory management
        self._clean_memory()
    
    def _clean_memory(self):
        """Force garbage collection and clear CUDA cache"""
        # Clear any dangling tensors
        torch.cuda.empty_cache()
        
        # Run multiple GC passes
        for _ in range(3):
            gc.collect()
        
        # Report memory usage
        if torch.cuda.is_available():
            cuda_mem = torch.cuda.memory_allocated() / 1024**2
            print(f"CUDA Memory: {cuda_mem:.1f} MB")
        
        process = psutil.Process(os.getpid())
        ram_usage = process.memory_info().rss / 1024**2
        self.memory_usage_history.append(ram_usage)
        print(f"RAM Usage: {ram_usage:.1f} MB")
    
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
        
        # Clear references to tensors
        del states, actions, rewards, next_states, dones
        
        # Periodically clean memory to avoid fragmentation
        if self.total_steps % 1000 == 0:
            self._clean_memory()
    
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
    
    def train(self, num_steps=1500000, eval_freq=10000, save_path="models", checkpoint_freq=50000):
        """
        Memory-efficient training loop with checkpoints
        """
        os.makedirs(save_path, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Start from a known state
        if hasattr(self, 'best_mean_reward'):
            best_mean_reward = self.best_mean_reward
        else:
            best_mean_reward = float('-inf')
            self.best_mean_reward = best_mean_reward
        
        start_step = self.total_steps
        episode_count = 0
        episode_reward = 0
        episode_length = 0
        training_start_time = time.time()
        
        # For logging - use arrays instead of lists for memory efficiency
        recent_rewards = np.zeros(10)
        recent_rewards_idx = 0
        
        # Reset environment
        obs = self.env.reset()[0]
        
        print(f"Starting SAC training from step {start_step}...")
        
        try:
            for step in range(1, num_steps + 1):
                actual_step = start_step + step
                self.total_steps = actual_step
                
                # Fill the buffer with random actions initially
                if actual_step <= self.initial_random_steps:
                    action = np.random.uniform(-1.0, 1.0, size=self.action_space.shape)
                else:
                    action = self.select_action(obs)
                
                # Execute action in environment - handle both API versions
                step_result = self.env.step(action.reshape(1, -1))
                
                # Check if we have the new Gym API (5 return values) or old API (4 return values)
                if len(step_result) == 5:
                    next_obs, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    next_obs, reward, done, _ = step_result
                    terminated = done  # For compatibility
                    truncated = False  # For compatibility
                
                # Process reward
                reward_value = reward.item() if isinstance(reward, np.ndarray) and reward.size == 1 else reward
                
                # Store transition in buffer
                self.replay_buffer.push(obs, action, reward_value, next_obs, done)
                
                # Update observations and stats
                obs = next_obs
                episode_reward += reward_value
                episode_length += 1
                
                # Update parameters if it's time
                if actual_step > self.initial_random_steps and step % self.update_every == 0:
                    for _ in range(1):  # Number of updates per step
                        self.update_parameters()
                
                # Handle episode end
                if done:
                    # Log episode statistics
                    print(f"Episode {episode_count+1} finished: Reward={episode_reward:.2f}, Length={episode_length}")
                    
                    # Update running average
                    recent_rewards[recent_rewards_idx] = episode_reward
                    recent_rewards_idx = (recent_rewards_idx + 1) % 10
                    
                    # Reset for next episode
                    obs = self.env.reset()[0]
                    episode_count += 1
                    episode_reward = 0
                    episode_length = 0
                    
                    # Log every 10 episodes
                    if episode_count % 10 == 0:
                        current_avg = recent_rewards.mean()
                        elapsed_time = time.time() - training_start_time
                        print(f"Step: {actual_step}/{start_step+num_steps}, Episodes: {episode_count}, "
                              f"Mean reward (last 10): {current_avg:.2f}, "
                              f"Time elapsed: {elapsed_time:.2f}s, "
                              f"Steps/sec: {actual_step/elapsed_time:.2f}")
                        
                        # Clear memory periodically
                        self._clean_memory()
                
                # Evaluate and save checkpoints
                if step % eval_freq < self.update_every:
                    # Clean before evaluation
                    self._clean_memory()
                    
                    # Run evaluation
                    mean_reward = self.evaluate(n_episodes=2)  # Use fewer episodes to save memory
                    print(f"Evaluation at step {actual_step}: Mean reward = {mean_reward:.2f}")
                    
                    # Update best model
                    if mean_reward > best_mean_reward:
                        best_mean_reward = mean_reward
                        self.best_mean_reward = best_mean_reward
                        self.save(os.path.join(save_path, "sac_car_racing_best"))
                        print(f"New best model saved with mean reward: {best_mean_reward:.2f}")
                    
                    # Save checkpoint occasionally to resume training if needed
                    if step % checkpoint_freq < self.update_every:
                        checkpoint_path = os.path.join(save_path, f"sac_checkpoint_{actual_step}")
                        self.save(checkpoint_path)
                        print(f"Checkpoint saved at {checkpoint_path}")
                        
                        # Remove older checkpoints to save space
                        old_checkpoints = [f for f in os.listdir(save_path) 
                                         if f.startswith("sac_checkpoint_") 
                                         and f.endswith(".pt") 
                                         and f != f"sac_checkpoint_{actual_step}.pt"]
                        for old_cp in old_checkpoints:
                            try:
                                os.remove(os.path.join(save_path, old_cp))
                            except:
                                pass
                    
                    # Clean after evaluation
                    self._clean_memory()
        
        except Exception as e:
            print(f"Error during training: {e}")
            traceback.print_exc()
            # Save checkpoint on error
            error_checkpoint = os.path.join(save_path, "sac_error_checkpoint")
            self.save(error_checkpoint)
            print(f"Saved checkpoint at {error_checkpoint} after error")
        
        # Save final model
        self.save(os.path.join(save_path, "sac_car_racing_final"))
        print(f"Training completed: {self.total_steps} total steps")
        
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
            if deterministic:
                mean, _ = self.actor(obs_tensor)
                action = torch.tanh(mean)
            else:
                action, _, _ = self.actor.sample(obs_tensor)
                
            # Ensure action is properly shaped as an array
            action_np = action.cpu().numpy()
            
            # If the action has a batch dimension, remove it for single observations
            if len(action_np.shape) == 2 and action_np.shape[0] == 1:
                action_np = action_np.squeeze(0)
            
            # Make sure we have a properly shaped [3,] action vector
            if len(action_np) != 3:
                # Pad or truncate action to have 3 dimensions
                temp = np.zeros(3)
                temp[:min(len(action_np), 3)] = action_np[:min(len(action_np), 3)]
                action_np = temp
                
            # Ensure proper value ranges for car racing
            action_np[0] = np.clip(action_np[0], -1.0, 1.0)  # Steering [-1,1]
            action_np[1] = np.clip(action_np[1], 0.0, 1.0)   # Gas [0,1]
            action_np[2] = np.clip(action_np[2], 0.0, 1.0)   # Brake [0,1]
                
        return action_np, None
    
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

# Training script at the bottom for when the file is run directly
if __name__ == "__main__":
    import os
    from stable_baselines3.common.vec_env import VecTransposeImage
    
    # Create directories for saving models and logs
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    def make_env():
        """Create and wrap the CarRacing environment"""
        def _init():
            env = gym.make("CarRacing-v3", continuous=True)
            env = Monitor(env, "logs/car_racing_sac")
            return env
        return _init
    
    # Create and wrap the training environment
    env = DummyVecEnv([make_env()])
    env = VecFrameStack(env, n_stack=4)  # Stack 4 frames like in PPO
    env = VecTransposeImage(env)  # Convert from (H,W,C) to (C,H,W)
    
    # Create the SAC agent with settings more aligned with PPO
    agent = SAC(
        env=env,
        lr_actor=3e-4,           # Same as PPO (already matched)
        lr_critic=3e-4,          # Same as PPO (already matched)
        lr_alpha=3e-4,           # Same as PPO (already matched)
        gamma=0.99,              # Same as PPO (already matched)
        tau=0.005,               # Keep default SAC value
        alpha=0.2,               # Start with default SAC value
        auto_entropy_tuning=True,# Auto-tune to match PPO's exploration
        buffer_size=100000,      # Keep larger buffer for off-policy
        batch_size=64,           # Match PPO's batch size
        initial_random_steps=10000,  # Scale back to be more like PPO
        update_every=1,          # Update every step like PPO
        device="auto"
    )
    
    # Train the agent for the same number of steps as PPO
    agent.train(
        num_steps=1500000,       # Same as PPO
        eval_freq=10000,         # Same evaluation frequency as PPO
        save_path="models"
    )
    
    # Ensure best model is copied with standard naming for easier loading
    if os.path.exists("models/sac_car_racing_best.pt"):
        print("Copying best model to standard name format...")
        import shutil
        shutil.copy("models/sac_car_racing_best.pt", "models/sac_car_racing_best.pt")