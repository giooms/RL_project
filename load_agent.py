import os
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from a2c import A2C

def load_best_agent():
    """
    Load the best agent from the specified path.

    This function is designed to load your pre-trained agent model so that it can be used to
    interact with the environment. Follow these steps to implement the function:

    1) Choose the right library for model loading:
       - Depending on whether you used PyTorch, TensorFlow, or another framework to train your model,
         import the corresponding library (e.g., `import torch` or `import tensorflow as tf`).

    2) Specify the correct file path:
       - Define the path where your trained model is saved.
       - Ensure the path is correct and accessible from your script.

    3) Load the model:
       - Use the appropriate loading function from your library.
         For example, with PyTorch you might use:
           ```python
           model = torch.load('path/to/your_model.pth')
           ```

    4) Ensure the model is callable:
       - The loaded model should be usable like a function. When you call:
           ```python
           action = model(observation)
           ```
         it should output an action based on the input observation.

    Returns:
        model: The loaded model. It must be callable so that when you pass an observation to it,
               it returns the corresponding action.

    Example usage:
        >>> model = load_best_agent()
        >>> observation = get_current_observation()  # Your method to fetch the current observation.
        >>> action = model(observation)
    """
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    
    if algorithm.lower() == "ppo":
        model_path = os.path.join(model_dir, "ppo_car_racing_best")
        if not os.path.exists(f"{model_path}.zip"):
            raise FileNotFoundError(f"PPO model not found at {model_path}.zip")
        return PPO.load(model_path)
        
    elif algorithm.lower() == "a2c":
        model_path = os.path.join(model_dir, "a2c_car_racing_best.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"A2C model not found at {model_path}")
        
        # Create environment to initialize the agent
        env = DummyVecEnv([lambda: gym.make("CarRacing-v3", continuous=True)])
        env = VecFrameStack(env, n_stack=4)
        env = VecTransposeImage(env)
        
        # Initialize and load the A2C agent
        agent = A2C(env=env)
        agent.load(model_path)
        return agent
    
    else:
        raise ValueError("Algorithm must be either 'ppo' or 'a2c'")
