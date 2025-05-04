import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import argparse
from load_agent import load_best_agent

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Compare RL algorithms for Car Racing")
    parser.add_argument("--n_episodes", type=int, default=10, 
                        help="Number of episodes to evaluate each algorithm")
    parser.add_argument("--render", type=lambda x: str(x).lower() == 'true', default=False,
                        help="Whether to render the environment (True/False)")
    parser.add_argument("--save_video", type=lambda x: str(x).lower() == 'true', default=False,
                        help="Whether to save videos of episodes (True/False)")
    parser.add_argument("--output_dir", type=str, default="comparison_results",
                        help="Directory to save results")
    
    return parser.parse_args()

def compare_algorithms(n_episodes=20, render=False, save_video=True, output_dir="comparison_results"):
    """Compare PPO and SAC algorithms with detailed metrics"""
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directories for videos if needed
    if save_video:
        video_dir = os.path.join(output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
    
    # Evaluation metrics to track
    metrics = {
        "ppo": {"rewards": [], "lengths": [], "timeouts": 0, "crashes": 0, "completions": 0, "trajectories": []},
        "sac": {"rewards": [], "lengths": [], "timeouts": 0, "crashes": 0, "completions": 0, "trajectories": []},
    }
    
    for algo in ["ppo", "sac"]:
        print(f"\nEvaluating {algo.upper()}...")
        
        for episode in range(n_episodes):
            # Set up video recording if enabled
            if save_video:
                video_path = os.path.join(video_dir, f"{algo}_episode_{episode}")
                env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
                env = gym.wrappers.RecordVideo(env, video_path, episode_trigger=lambda x: True)
            else:
                render_mode = "human" if render else None
                env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=True)
            
            # Prepare environment for the specific algorithm
            if algo == "ppo" or algo == "sac":
                # Need to wrap in DummyVecEnv and other wrappers for both algorithms
                from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
                env.close()  # Close the unwrapped env
                env = DummyVecEnv([lambda: gym.make("CarRacing-v3", 
                                                    render_mode="rgb_array" if save_video else render_mode, 
                                                    continuous=True)])
                env = VecFrameStack(env, n_stack=4)
                env = VecTransposeImage(env)
            
            # Load the appropriate agent
            model = load_best_agent(algo)
            
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            trajectory = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Handle vector environment outputs
                if isinstance(reward, np.ndarray):
                    reward = reward[0]
                if isinstance(terminated, np.ndarray):
                    terminated = terminated[0]
                if isinstance(truncated, np.ndarray):
                    truncated = truncated[0]
                
                episode_reward += reward
                episode_length += 1
                
                # Track position when possible (this will need adjustments for vector envs)
                if hasattr(env, 'envs'):
                    # For VecEnv
                    pos = None
                    try:
                        if hasattr(env.envs[0].unwrapped, 'car'):
                            pos = env.envs[0].unwrapped.car.hull.position
                            trajectory.append((float(pos[0]), float(pos[1])))
                    except (AttributeError, IndexError):
                        pass
                
                done = terminated or truncated
                
                # Track reason for episode end
                if done:
                    if episode_length >= 1000:
                        metrics[algo]["timeouts"] += 1
                    elif episode_reward < -50:  # Threshold for crashes
                        metrics[algo]["crashes"] += 1
                    elif episode_reward > 800:  # Likely completed
                        metrics[algo]["completions"] += 1
            
            # Record episode results
            metrics[algo]["rewards"].append(episode_reward)
            metrics[algo]["lengths"].append(episode_length)
            if trajectory:
                metrics[algo]["trajectories"].append(trajectory)
            
            print(f"{algo.upper()} - Episode {episode+1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
            
            env.close()
    
    # Save metrics to CSV
    for algo in metrics:
        df = pd.DataFrame({
            'episode': range(1, n_episodes+1),
            'reward': metrics[algo]['rewards'],
            'length': metrics[algo]['lengths']
        })
        df.to_csv(os.path.join(output_dir, f"{algo}_results.csv"), index=False)
    
    # Generate comprehensive statistics
    print("\n=== Performance Summary ===")
    with open(os.path.join(output_dir, "comparison_summary.txt"), 'w') as f:
        f.write("=== Performance Summary ===\n")
        for algo in ["ppo", "sac"]:
            rewards = metrics[algo]["rewards"]
            summary = f"\n{algo.upper()} Statistics:\n"
            summary += f"  Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}\n"
            summary += f"  Median Reward: {np.median(rewards):.2f}\n"
            summary += f"  Min/Max Reward: {np.min(rewards):.2f} / {np.max(rewards):.2f}\n"
            summary += f"  Mean Episode Length: {np.mean(metrics[algo]['lengths']):.2f}\n"
            summary += f"  Timeouts: {metrics[algo]['timeouts']}/{n_episodes}\n"
            summary += f"  Crashes: {metrics[algo]['crashes']}/{n_episodes}\n"
            summary += f"  Track Completions: {metrics[algo]['completions']}/{n_episodes}\n"
            
            print(summary)
            f.write(summary)
    
    # Create visualizations
    create_comparison_plots(metrics, n_episodes, output_dir)
    
    return metrics

def create_comparison_plots(metrics, n_episodes, output_dir):
    """Create detailed comparison visualizations"""
    import pandas as pd
    
    # Set up the plots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Reward Distribution (Box plot)
    ax1 = fig.add_subplot(221)
    ax1.boxplot([metrics['ppo']['rewards'], metrics['sac']['rewards']], 
               labels=['PPO', 'SAC'], showfliers=True)
    ax1.set_title('Reward Distribution')
    ax1.set_ylabel('Total Episode Reward')
    
    # 2. Episode Rewards (Line plot for each episode)
    ax2 = fig.add_subplot(222)
    ep_nums = range(1, n_episodes+1)
    ax2.plot(ep_nums, metrics['ppo']['rewards'], 'b-', label='PPO')
    ax2.plot(ep_nums, metrics['sac']['rewards'], 'r-', label='SAC')
    ax2.set_title('Episode Rewards')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax2.legend()
    
    # 3. Episode Length Comparison
    ax3 = fig.add_subplot(223)
    ax3.boxplot([metrics['ppo']['lengths'], metrics['sac']['lengths']], 
               labels=['PPO', 'SAC'])
    ax3.set_title('Episode Length Distribution')
    ax3.set_ylabel('Steps per Episode')
    
    # 4. Success Metrics (Bar chart)
    ax4 = fig.add_subplot(224)
    algorithms = ['PPO', 'SAC']
    completions = [metrics['ppo']['completions'], metrics['sac']['completions']]
    timeouts = [metrics['ppo']['timeouts'], metrics['sac']['timeouts']]
    crashes = [metrics['ppo']['crashes'], metrics['sac']['crashes']]
    
    x = np.arange(len(algorithms))
    width = 0.25
    
    ax4.bar(x - width, completions, width, label='Completions', color='g')
    ax4.bar(x, timeouts, width, label='Timeouts', color='y')
    ax4.bar(x + width, crashes, width, label='Crashes', color='r')
    ax4.set_xticks(x)
    ax4.set_xticklabels(algorithms)
    ax4.set_title('Episode Outcomes')
    ax4.set_ylabel('Count')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_detailed_comparison.png'))
    plt.close()
    
    # Create a helpful README in the output directory
    with open(os.path.join(output_dir, "README.txt"), 'w') as f:
        f.write("Car Racing Algorithm Comparison\n")
        f.write("===============================\n\n")
        f.write(f"Comparison run on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of episodes per algorithm: {n_episodes}\n\n")
        f.write("Files in this directory:\n")
        f.write("- ppo_results.csv: Raw results for PPO algorithm\n")
        f.write("- sac_results.csv: Raw results for SAC algorithm\n")
        f.write("- comparison_summary.txt: Statistical summary of algorithm performance\n")
        f.write("- algorithm_detailed_comparison.png: Visual comparison of algorithms\n")
        if os.path.exists(os.path.join(output_dir, "videos")):
            f.write("- videos/: Directory containing episode recordings\n")

if __name__ == "__main__":
    args = parse_arguments()
    compare_algorithms(
        n_episodes=args.n_episodes, 
        render=args.render, 
        save_video=args.save_video,
        output_dir=args.output_dir
    )