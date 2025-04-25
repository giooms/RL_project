import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import glob

def extract_rewards_from_a2c_log(log_file):
    """Extract episode rewards from A2C log file"""
    rewards = []
    episodes = []
    steps = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Pattern to match the evaluation prints
            eval_match = re.search(r'Evaluation at step (\d+): Mean reward = ([-\d\.]+)', line)
            if eval_match:
                step = int(eval_match.group(1))
                reward = float(eval_match.group(2))
                steps.append(step)
                rewards.append(reward)
                episodes.append(step // 1000)  # Approximate episode number
    
    return pd.DataFrame({'episode': episodes, 'step': steps, 'reward': rewards})

def read_car_racing_monitor(file_path):
    """Read the monitor CSV file from car racing environment"""
    # Skip the first line which contains metadata
    df = pd.read_csv(file_path, skiprows=1)
    df.columns = ['reward', 'length', 'time']  # Set column names
    df['episode'] = np.arange(len(df))  # Add episode number
    return df

def plot_learning_curves():
    """Plot learning curves from logs"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot PPO learning curve from monitor logs
    ppo_log = read_car_racing_monitor("logs/car_racing.monitor.csv")
    ax.plot(ppo_log['episode'], ppo_log['reward'].rolling(window=10).mean(), 
            label='PPO (10-episode rolling avg)', color='blue')
    
    # Plot A2C learning curve
    a2c_log = read_car_racing_monitor("logs/car_racing_a2c.monitor.csv")
    ax.plot(a2c_log['episode'], a2c_log['reward'].rolling(window=10).mean(), 
            label='A2C (10-episode rolling avg)', color='red')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Learning Curves: PPO vs A2C')
    ax.legend()
    ax.grid(True)
    
    plt.savefig('learning_curves.png')
    plt.show()

if __name__ == "__main__":
    plot_learning_curves()