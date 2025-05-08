import os
import torch
import numpy as np
import gymnasium as gym
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

# Import SAC implementation
from sac import SAC

def create_frame_stack_preprocessor(orig_predict):
    """Create a preprocessor that stacks frames for the model's predict method"""
    # Buffer to store last 4 frames
    frame_stack = deque(maxlen=4)
    
    def preprocess_and_predict(obs, deterministic=True):
        # Add new frame to the stack
        frame_stack.append(obs)
        # If not enough frames yet, pad with copies of the first frame
        while len(frame_stack) < 4:
            frame_stack.appendleft(obs)
        # Stack frames along the channel axis
        stacked_obs = np.concatenate(frame_stack, axis=2)  # (96, 96, 12)
        # Call the original predict
        return orig_predict(stacked_obs, deterministic=deterministic)
        
    return preprocess_and_predict

def load_best_agent(algorithm="ppo"):
    """
    Load the best agent from the specified path based on algorithm.
    
    Args:
        algorithm (str): Either 'ppo' or 'sac'
        
    Returns:
        model: The loaded agent model
    """
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    
    # Ensure the model directory exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Create properly wrapped environment for both algorithms
    def make_env():
        def _init():
            env = gym.make("CarRacing-v3", continuous=True)
            return env
        return _init
    
    env = DummyVecEnv([make_env()])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    
    if algorithm.lower() == "ppo":
        # Try loading from SB3 format first
        zip_path = os.path.join(model_dir, "ppo_car_racing_best.zip")
        
        if os.path.exists(zip_path):
            print(f"Loading PPO model from {zip_path}")
            model = PPO.load(zip_path, env=env)
            
            # Patch the predict method to handle single-frame observations
            model.predict = create_frame_stack_preprocessor(model.predict)
            return model
        
        # Fall back to PyTorch format
        pt_path = os.path.join(model_dir, "ppo_car_racing_best.pt")
        if os.path.exists(pt_path):
            print(f"Loading PPO model from PyTorch file: {pt_path}")
            # Initialize a new PPO model
            model = PPO("CnnPolicy", env)
            # Load the PyTorch state dict
            checkpoint = torch.load(pt_path, map_location=torch.device('cpu'))
            model.policy.load_state_dict(checkpoint['policy_state_dict'])
            
            # Patch the predict method
            model.predict = create_frame_stack_preprocessor(model.predict)
            return model
        
        # If we get here, no suitable model was found
        raise FileNotFoundError(f"PPO model file not found. Expected either:\n"
                              f"- {zip_path}\n"
                              f"- {pt_path}")
        
    elif algorithm.lower() == "sac":
        # Only look for SAC PyTorch model
        pt_path = os.path.join(model_dir, "sac_car_racing_best.pt")
        if os.path.exists(pt_path):
            print(f"Loading SAC model from {pt_path}")
            # Initialize and load the SAC agent
            agent = SAC(env=env)
            agent.load(pt_path)
            
            # Apply the frame stacking preprocessor
            agent.predict = create_frame_stack_preprocessor(agent.predict)
            
            return agent
                
        # If we get here, no suitable model was found
        raise FileNotFoundError(f"SAC model file not found. Expected: {pt_path}")
    
    else:
        raise ValueError("Algorithm must be either 'ppo' or 'sac'")