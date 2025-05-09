import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import argparse
import pandas as pd
import torch
import psutil
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
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
        
        # Create separate directories for each algorithm
        for algo in ["ppo", "sac"]:
            algo_video_dir = os.path.join(video_dir, algo)
            os.makedirs(algo_video_dir, exist_ok=True)
    
    # Evaluation metrics to track
    metrics = {
        "ppo": {"rewards": [], "lengths": [], "timeouts": 0, "crashes": 0, "completions": 0, "trajectories": []},
        "sac": {"rewards": [], "lengths": [], "timeouts": 0, "crashes": 0, "completions": 0, "trajectories": []},
    }
    
    # Generate a list of seeds to use for each episode
    # This ensures both algorithms face identical tracks
    episode_seeds = [np.random.randint(0, 10000) for _ in range(n_episodes)]
    print(f"Using fixed seeds for track generation: {episode_seeds}")
    
    # Pre-load models to avoid repeated loading
    print("Loading models...")
    models = {
        "ppo": load_best_agent("ppo"),
        "sac": load_best_agent("sac")
    }
    print("Models loaded successfully.")
    
    for episode in range(n_episodes):
        episode_seed = episode_seeds[episode]
        print(f"\n--- Episode {episode+1}/{n_episodes} (Seed: {episode_seed}) ---")
        
        # Track metrics for this specific episode/track
        episode_metrics = {
            "ppo": {"reward": 0, "length": 0, "timeout": False, "crash": False, "completion": False, "trajectory": []},
            "sac": {"reward": 0, "length": 0, "timeout": False, "crash": False, "completion": False, "trajectory": []}
        }
        
        for algo in ["ppo", "sac"]:
            print(f"\nRunning {algo.upper()}...")
            
            # Use the SAME approach as interface.py (which works)
            if save_video:
                current_video_dir = os.path.join(video_dir, algo, f"episode_{episode}")
                os.makedirs(current_video_dir, exist_ok=True)
                
                env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
                env = gym.wrappers.RecordVideo(env, current_video_dir, episode_trigger=lambda x: True)
                print(f"Recording video to {current_video_dir}")
            else:
                render_mode = "human" if render else None
                env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=True)
            
            # Get the pre-loaded model
            model = models[algo]
            
            try:
                # Reset with seed for same track
                observation, info = env.reset(seed=episode_seed)
                
                # Initialize episode tracking
                done = False
                episode_reward = 0
                episode_length = 0
                trajectory = []
                
                # Run episode
                while not done:
                    # Get action from model
                    action, _ = model.predict(observation, deterministic=True)
                    
                    # Take step in environment
                    observation, reward, terminated, truncated, info = env.step(action)
                    
                    # Track metrics
                    episode_reward += reward
                    episode_length += 1
                    
                    # Track position for trajectory
                    try:
                        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'car'):
                            pos = env.unwrapped.car.hull.position
                            trajectory.append((float(pos[0]), float(pos[1])))
                    except (AttributeError, TypeError):
                        pass
                    
                    # Check if episode is done
                    done = terminated or truncated
                    
                    # Track reason for episode end
                    if done:
                        if episode_reward < 600:  # First check for crashes - lowered threshold
                            metrics[algo]["crashes"] += 1
                            episode_metrics[algo]["crash"] = True
                        elif episode_length >= 1000 and episode_reward < 800:  # Then check for actual timeouts
                            metrics[algo]["timeouts"] += 1
                            episode_metrics[algo]["timeout"] = True
                        else:  # Otherwise it's a completion (either fast or timed out with high reward)
                            metrics[algo]["completions"] += 1
                            episode_metrics[algo]["completion"] = True
                
                # Record episode metrics
                metrics[algo]["rewards"].append(episode_reward)
                metrics[algo]["lengths"].append(episode_length)
                if trajectory:
                    metrics[algo]["trajectories"].append(trajectory)
                
                episode_metrics[algo]["reward"] = episode_reward
                episode_metrics[algo]["length"] = episode_length
                
                print(f"{algo.upper()} - Episode {episode+1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
                
            except Exception as e:
                print(f"Error in {algo} episode {episode}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Always close the environment
                env.close()
                time.sleep(1)  # Give time for video saving
                
        # Print episode comparison
        print(f"\n--- Episode {episode+1} Results (Seed: {episode_seed}) ---")
        print(f"PPO: Reward = {episode_metrics['ppo']['reward']:.2f}, Length = {episode_metrics['ppo']['length']}")
        print(f"SAC: Reward = {episode_metrics['sac']['reward']:.2f}, Length = {episode_metrics['sac']['length']}")
        print(f"Difference: {episode_metrics['ppo']['reward'] - episode_metrics['sac']['reward']:.2f}")
    
    # Save metrics and create visualizations
    for algo in metrics:
        df = pd.DataFrame({
            'episode': range(1, len(metrics[algo]['rewards'])+1),
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
            if rewards:  # Check there are results for this algorithm
                summary = f"\n{algo.upper()} Statistics:\n"
                summary += f"  Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}\n"
                summary += f"  Median Reward: {np.median(rewards):.2f}\n"
                summary += f"  Min/Max Reward: {np.min(rewards):.2f} / {np.max(rewards):.2f}\n"
                summary += f"  Mean Episode Length: {np.mean(metrics[algo]['lengths']):.2f}\n"
                summary += f"  Timeouts: {metrics[algo]['timeouts']}/{len(rewards)}\n"
                summary += f"  Crashes: {metrics[algo]['crashes']}/{len(rewards)}\n"
                summary += f"  Track Completions: {metrics[algo]['completions']}/{len(rewards)}\n"
                
                print(summary)
                f.write(summary)
    
    # Create visualizations
    create_comparison_plots(metrics, n_episodes, output_dir)
    
    return metrics

# Keep the existing visualization function unchanged
def create_comparison_plots(metrics, n_episodes, output_dir):
    """Create detailed comparison visualizations"""
    # Existing implementation...
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
    ep_nums = range(1, len(metrics['ppo']['rewards'])+1)
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
        f.write(f"Number of episodes per algorithm: {len(metrics['ppo']['rewards'])}\n\n")
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