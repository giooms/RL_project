import os
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

# Create directories for saving models and logs
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

class SaveTorchModelCallback(BaseCallback):
    """
    Callback for saving the policy network as a PyTorch model (.pt)
    """
    def __init__(self, save_path, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
    
    def _on_step(self):
        return True
    
    def _on_training_end(self):
        # Extract policy network from PPO
        policy_net = self.model.policy

        # Save the PyTorch model state dict
        torch.save({
            'policy_state_dict': policy_net.state_dict(),
            'optimizer_state_dict': self.model.policy.optimizer.state_dict()
        }, f"{self.save_path}/ppo_car_racing_final.pt")
        
        if self.verbose > 0:
            print(f"Saved PyTorch model to {self.save_path}/ppo_car_racing_final.pt")
        
        # Also save the best model in PyTorch format
        if os.path.exists(f"{self.save_path}/best_model.zip"):
            best_model = PPO.load(f"{self.save_path}/best_model")
            torch.save({
                'policy_state_dict': best_model.policy.state_dict(),
                'optimizer_state_dict': best_model.policy.optimizer.state_dict()
            }, f"{self.save_path}/ppo_car_racing_best.pt")
            
            if self.verbose > 0:
                print(f"Saved best PyTorch model to {self.save_path}/ppo_car_racing_best.pt")

def make_env():
    """Create and wrap the CarRacing environment"""
    def _init():
        env = gym.make("CarRacing-v3", continuous=True)
        env = Monitor(env, "logs/car_racing_ppo")
        return env
    return _init

# Create and wrap the training environment
env = DummyVecEnv([make_env()])
env = VecFrameStack(env, n_stack=4)  # Stack 4 frames to capture motion information

# Separate environment for evaluation with the same wrappers
eval_env = DummyVecEnv([make_env()])
eval_env = VecFrameStack(eval_env, n_stack=4)
eval_env = VecTransposeImage(eval_env) 

# Initialize the PPO agent
model = PPO(
    policy="CnnPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,  # Encourage exploration
    verbose=1,
    tensorboard_log="./logs/tensorboard/"
)

# Set up evaluation callback to save best model
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/",
    log_path="./logs/",
    eval_freq=10000,
    deterministic=True,
    render=False,
    n_eval_episodes=3,
)

# Create the PyTorch save callback
torch_save_callback = SaveTorchModelCallback(save_path="./models")

# Train the model with both callbacks
print("Starting training...")
model.learn(
    total_timesteps=1000000,
    callback=[eval_callback, torch_save_callback],
    progress_bar=True
)

print("Training complete!")

# Save the final model in SB3 format too
model.save("models/ppo_car_racing_final")