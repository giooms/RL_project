import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

# Create directories for saving models and logs
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def make_env():
    """Create and wrap the CarRacing environment"""
    def _init():
        env = gym.make("CarRacing-v3", continuous=True)
        env = Monitor(env, "logs/car_racing")
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

# Train the model
print("Starting training...")
model.learn(
    total_timesteps=500000,
    callback=eval_callback,
    progress_bar=True
)

print("Training complete!")