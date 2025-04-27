import os
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

# Import SAC implementation
from sac import SAC

def load_best_agent(algorithm="ppo"):
    """
    Load the best agent from the specified path based on algorithm.
    
    Args:
        algorithm (str): Either 'ppo' or 'sac'
        
    Returns:
        model: The loaded agent model
    """
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    
    if algorithm.lower() == "ppo":
        # Create environment with proper wrappers for PPO
        env = DummyVecEnv([lambda: gym.make("CarRacing-v3", continuous=True)])
        env = VecFrameStack(env, n_stack=4)
        env = VecTransposeImage(env)
        
        model_path = os.path.join(model_dir, "ppo_car_racing_best")
        if not os.path.exists(model_path + ".pt"):
            raise FileNotFoundError(f"PPO model not found at {model_path}.pt")
        
        return PPO.load(model_path, env=env)
        
    elif algorithm.lower() == "sac":
        # Create environment to initialize the agent
        env = DummyVecEnv([lambda: gym.make("CarRacing-v3", continuous=True)])
        env = VecFrameStack(env, n_stack=4)
        env = VecTransposeImage(env)
        
        model_path = os.path.join(model_dir, "sac_car_racing_best.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SAC model not found at {model_path}")
        
        # Initialize and load the SAC agent
        agent = SAC(env=env)
        agent.load(model_path)
        return agent
    
    else:
        raise ValueError("Algorithm must be either 'ppo' or 'sac'")
