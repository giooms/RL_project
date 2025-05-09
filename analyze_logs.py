import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import glob
import os
import argparse

def extract_rewards_from_log(log_file):
    """Extract episode rewards from log file"""
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

def plot_learning_curves(output_dir="analysis_results"):
    """Plot learning curves from logs"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot PPO learning curve from monitor logs
    try:
        ppo_log = read_car_racing_monitor("logs/car_racing_ppo.monitor.csv")
        ax.plot(ppo_log['episode'], ppo_log['reward'].rolling(window=10).mean(), 
                label='PPO (10-episode rolling avg)', color='blue')
    except FileNotFoundError:
        print("Warning: PPO monitor file not found.")
    
    # Plot SAC learning curve
    try:
        sac_log = read_car_racing_monitor("logs/car_racing_sac.monitor.csv")
        ax.plot(sac_log['episode'], sac_log['reward'].rolling(window=10).mean(), 
                label='SAC (10-episode rolling avg)', color='red')
    except FileNotFoundError:
        print("Warning: SAC monitor file not found.")
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Learning Curves: PPO vs SAC')
    ax.legend()
    ax.grid(True)
    
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
    plt.close()  # Close the plot instead of showing it (better for headless environments)
    print(f"Learning curves saved to {os.path.join(output_dir, 'learning_curves.png')}")

def plot_statistical_comparison(output_dir="analysis_results_new"):
    """Generate statistical comparison plots between algorithms"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Read data
    try:
        ppo_data = read_car_racing_monitor("logs/car_racing_ppo.monitor.csv")
        sac_data = read_car_racing_monitor("logs/car_racing_sac.monitor.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Reward Distribution (Box plot)
    axes[0,0].boxplot([ppo_data['reward'], sac_data['reward']], 
                      tick_labels=['PPO', 'SAC'], showfliers=True)
    axes[0,0].set_title('Reward Distribution')
    axes[0,0].set_ylabel('Episode Reward')
    
    # 2. Learning Curves (Moving Average)
    window_size = 10
    axes[0,1].plot(ppo_data['episode'], 
                  ppo_data['reward'].rolling(window=window_size).mean(),
                  label='PPO', color='blue')
    axes[0,1].plot(sac_data['episode'], 
                  sac_data['reward'].rolling(window=window_size).mean(),
                  label='SAC', color='red')
    axes[0,1].set_title(f'Learning Curves ({window_size}-episode rolling avg)')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Reward')
    axes[0,1].legend()
    
    # 3. Episode Length Comparison
    axes[1,0].boxplot([ppo_data['length'], sac_data['length']], 
                      tick_labels=['PPO', 'SAC'])
    axes[1,0].set_title('Episode Length Distribution')
    axes[1,0].set_ylabel('Steps per Episode')
    
    # 4. Performance Metrics (Bar chart)
    metrics = {
        'Mean Reward': [ppo_data['reward'].mean(), sac_data['reward'].mean()],
        'Max Reward': [ppo_data['reward'].max(), sac_data['reward'].max()],
        'Min Reward': [ppo_data['reward'].min(), sac_data['reward'].min()],
        'Std Dev': [ppo_data['reward'].std(), sac_data['reward'].std()]
    }
    
    # Convert the dictionary to a DataFrame for easier plotting
    metrics_df = pd.DataFrame(metrics, index=['PPO', 'SAC'])
    metrics_df.plot(kind='bar', ax=axes[1,1])
    axes[1,1].set_title('Performance Metrics')
    axes[1,1].set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_comparison_statistics.png'))
    plt.close()  # Close the plot instead of showing it
    
    # Also save the metrics data as CSV
    metrics_df.to_csv(os.path.join(output_dir, 'performance_metrics.csv'))
    print(f"Statistical comparison saved to {os.path.join(output_dir, 'algorithm_comparison_statistics.png')}")
    print(f"Metrics data saved to {os.path.join(output_dir, 'performance_metrics.csv')}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze and visualize RL algorithm performance")
    parser.add_argument("--output_dir", type=str, default="analysis_results",
                        help="Directory to save analysis results")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving analysis results to: {args.output_dir}")
    
    plot_learning_curves(args.output_dir)
    plot_statistical_comparison(args.output_dir)