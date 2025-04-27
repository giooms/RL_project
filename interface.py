import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from load_agent import load_best_agent


def parse_args():
    parser = argparse.ArgumentParser(description="Run trained agent on Car Racing environment")
    parser.add_argument("--algorithm", type=str, default="ppo", choices=["ppo", "sac"],
                        help="Algorithm to use (ppo or sac)")
    parser.add_argument("--render", type=lambda x: str(x).lower() == 'true', default=True,
                        help="Whether to render the environment (True/False)")
    parser.add_argument("--episodes", type=int, default=1, 
                        help="Number of episodes to run")
    parser.add_argument("--output_dir", type=str, default="interface_results",
                        help="Directory to save results")
    parser.add_argument("--save_video", type=lambda x: str(x).lower() == 'true', default=False,
                        help="Whether to save videos of episodes (True/False)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be saved to: {args.output_dir}")
    
    # Configure render mode and video recording
    if args.save_video:
        video_dir = os.path.join(args.output_dir, "videos", args.algorithm)
        os.makedirs(video_dir, exist_ok=True)
        env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
        env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda x: True)
        print(f"Recording videos to {video_dir}")
    else:
        render_mode = "human" if args.render else None
        env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=True)
    
    # Load agent
    model = load_best_agent(algorithm=args.algorithm)
    
    all_rewards = []
    
    for episode in range(args.episodes):
        observation, info = env.reset()
        episode_over = False
        rewards = []
        
        while not episode_over:
            action, _states = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            episode_over = terminated or truncated
        
        total_reward = sum(rewards)
        all_rewards.append(total_reward)
        print(f"Episode {episode+1}: Total reward = {total_reward:.2f}")

    # Print the average reward
    print(f"\nAverage reward over {args.episodes} episodes: {np.mean(all_rewards):.2f}")
    
    # Close the environment
    env.close()
    
    # Plot rewards if multiple episodes were run
    if args.episodes > 1:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, args.episodes + 1), all_rewards, marker='o')
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"{args.algorithm.upper()} Performance over {args.episodes} Episodes")
        plt.grid(True)
        
        # Save the plot to the output directory
        plot_path = os.path.join(args.output_dir, f"{args.algorithm}_episode_rewards.png")
        plt.savefig(plot_path)
        print(f"Performance plot saved to {plot_path}")
        
        # Also save the raw data
        data_path = os.path.join(args.output_dir, f"{args.algorithm}_episode_rewards.csv")
        with open(data_path, 'w') as f:
            f.write("episode,reward\n")
            for i, reward in enumerate(all_rewards, 1):
                f.write(f"{i},{reward}\n")
        print(f"Raw data saved to {data_path}")
        
        plt.close()  # Close the plot without showing it (better for non-GUI environments)

if __name__ == "__main__":
    main()
