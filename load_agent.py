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
    
    # Ensure the model directory exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    print(f"Looking for models in: {model_dir}")
    print(f"Available files: {os.listdir(model_dir)}")
    
    if algorithm.lower() == "ppo":
        # Create environment with proper wrappers for PPO - standard 96x96 size
        def make_env():
            # Return a function that creates the environment when called
            def _init():
                env = gym.make("CarRacing-v3", continuous=True)
                return env
            return _init  # Return the function, not the result of calling it
        
        # Create properly wrapped environment
        env = DummyVecEnv([make_env()])
        env = VecFrameStack(env, n_stack=4)
        env = VecTransposeImage(env)
        
        # Try loading from SB3 format first (our preferred format)
        zip_path = os.path.join(model_dir, "ppo_car_racing_best.zip")
        zip_path_no_ext = os.path.join(model_dir, "ppo_car_racing_best")
        
        if os.path.exists(zip_path):
            print(f"Loading PPO model from {zip_path}")
            return PPO.load(zip_path, env=env)
        elif os.path.exists(zip_path_no_ext + ".zip"):
            print(f"Loading PPO model from {zip_path_no_ext}.zip")
            return PPO.load(zip_path_no_ext, env=env)
        
        # Fallback to PyTorch format if needed
        pt_path = os.path.join(model_dir, "ppo_car_racing_best.pt")
        if os.path.exists(pt_path):
            print(f"Loading PPO model from PyTorch file: {pt_path}")
            # Initialize a new PPO model
            model = PPO("CnnPolicy", env)
            # Load the PyTorch state dict
            checkpoint = torch.load(pt_path, map_location=torch.device('cpu'))
            model.policy.load_state_dict(checkpoint['policy_state_dict'])
            return model
        
        # Check for any other PPO model files - WITHOUT TRAINING NEW ONES
        print("Warning: Could not find ppo_car_racing_best model. Looking for alternatives...")
        ppo_files = [f for f in os.listdir(model_dir) if f.startswith("ppo_") and (f.endswith(".zip") or f.endswith(".pt"))]
        
        if ppo_files:
            print(f"Found alternative PPO model files: {ppo_files}")
            for file in ppo_files:
                if file.endswith(".zip"):
                    print(f"Loading alternative PPO model from {file}")
                    return PPO.load(os.path.join(model_dir, file[:-4]), env=env)
                elif file.endswith(".pt"):
                    print(f"Loading alternative PPO model from {file}")
                    model = PPO("CnnPolicy", env)
                    checkpoint = torch.load(os.path.join(model_dir, file), map_location=torch.device('cpu'))
                    model.policy.load_state_dict(checkpoint['policy_state_dict'])
                    return model
        
        # If we get here, no suitable model was found
        raise FileNotFoundError(f"No PPO model found in {model_dir}. Please train a model first.")
        
    elif algorithm.lower() == "sac":
        # Create environment to initialize the agent - standard 96x96 size
        def make_env():
            # Return a function that creates the environment when called
            def _init():
                env = gym.make("CarRacing-v3", continuous=True)
                return env
            return _init  # Return the function, not the result of calling it
            
        env = DummyVecEnv([make_env()])
        env = VecFrameStack(env, n_stack=4)
        env = VecTransposeImage(env)
        
        # Look for SAC PyTorch model
        pt_path = os.path.join(model_dir, "sac_car_racing_best.pt")
        if os.path.exists(pt_path):
            print(f"Loading SAC model from {pt_path}")
            # Initialize and load the SAC agent
            agent = SAC(env=env)
            agent.load(pt_path)
            return agent
        
        # Check for any alternative SAC model files - WITHOUT TRAINING NEW ONES
        print("Warning: Could not find sac_car_racing_best.pt. Looking for alternatives...")
        sac_files = [f for f in os.listdir(model_dir) if f.startswith("sac_") and f.endswith(".pt")]
        
        if sac_files:
            print(f"Found alternative SAC model files: {sac_files}")
            file = sac_files[0]  # Take the first one
            print(f"Loading alternative SAC model from {file}")
            agent = SAC(env=env)
            agent.load(os.path.join(model_dir, file))
            return agent
                
        # If we get here, no suitable model was found
        raise FileNotFoundError(f"No SAC model found in {model_dir}. Please train a model first.")
    
    else:
        raise ValueError("Algorithm must be either 'ppo' or 'sac'")
