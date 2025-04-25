import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from load_agent import load_best_agent

def evaluate_agent(algorithm, n_episodes=10, render=False):
    """Evaluate an agent over multiple episodes"""
    # Load the agent
    model = load_best_agent(algorithm)
    
    # Create environment
    render_mode = "human" if render else None
    env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=True)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        observation, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"{algorithm.upper()} - Episode {episode+1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    env.close()
    return episode_rewards, episode_lengths

def compare_algorithms(n_episodes=10, render=False):
    """Compare PPO and A2C algorithms"""
    print("Evaluating PPO...")
    ppo_rewards, ppo_lengths = evaluate_agent("ppo", n_episodes, render)
    
    print("\nEvaluating A2C...")
    a2c_rewards, a2c_lengths = evaluate_agent("a2c", n_episodes, render)
    
    # Print summary statistics
    print("\n=== Performance Summary ===")
    print(f"PPO - Mean Reward: {np.mean(ppo_rewards):.2f} ± {np.std(ppo_rewards):.2f}")
    print(f"A2C - Mean Reward: {np.mean(a2c_rewards):.2f} ± {np.std(a2c_rewards):.2f}")
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reward comparison
    ax1.boxplot([ppo_rewards, a2c_rewards], labels=["PPO", "A2C"])
    ax1.set_title("Reward Distribution")
    ax1.set_ylabel("Total Episode Reward")
    
    # Episode length comparison
    ax2.boxplot([ppo_lengths, a2c_lengths], labels=["PPO", "A2C"])
    ax2.set_title("Episode Length Distribution")
    ax2.set_ylabel("Steps per Episode")
    
    plt.tight_layout()
    plt.savefig("algorithm_comparison.png")
    plt.show()

if __name__ == "__main__":
    compare_algorithms(n_episodes=5, render=False)