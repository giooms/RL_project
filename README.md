# RL_project

> This repository contains implementations of two reinforcement learning algorithms for solving the CarRacing-v3 environment from Gymnasium: Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC).

## Installation

1. Create a Python environment (Python 3.8+ recommended):
   ```
   conda create -n car_racing_env python=3.10
   conda activate car_racing_env
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

- train_ppo.py: Training script for PPO
- sac.py: Custom SAC implementation and training script
- interface.py: Script for evaluating trained agents
- load_agent.py: Helper for loading trained models
- compare_agents.py: Script for comparing PPO and SAC performance
- models: Directory containing trained models

## Training

### Training PPO

To train a PPO agent on the CarRacing-v3 environment:

```python
python train_ppo.py
```

This will:
- Train for 1.5 million timesteps
- Save the best model as ppo_car_racing_best.zip and ppo_car_racing_best.pt
- Log training progress in logs directory
- Create tensorboard logs in tensorboard

### Training SAC

To train a SAC agent on the CarRacing-v3 environment:

```python
python sac.py
```

This will:
- Train for 1.5 million timesteps
- Save the best model as sac_car_racing_best.pt
- Log training progress in logs directory

## Evaluation

### Running a Single Algorithm

To evaluate a trained agent:

```python
python interface.py --algorithm ppo --render True --episodes 5 --output_dir results --save_video True
```

Arguments:
- `--algorithm`: Choose between `ppo` or `sac`
- `--render`: Set to `True` to visualize the agent (set to `False` for headless environments)
- `--episodes`: Number of episodes to run
- `--output_dir`: Directory to save results
- `--save_video`: Whether to record videos of episodes

### Comparing Both Algorithms

To compare PPO and SAC performance:

```python
python compare_agents.py --n_episodes 20 --render False --save_video True --output_dir comparison_results
```

Arguments:
- `--n_episodes`: Number of episodes to evaluate each algorithm
- `--render`: Whether to render the environment during evaluation
- `--save_video`: Whether to save videos of episodes
- `--output_dir`: Directory to save comparison results

The comparison script will:
- Run both algorithms on identical tracks
- Generate statistics on rewards, episode lengths, and success rates
- Create visualizations comparing performance
- Save videos of each episode (if enabled)

## Results

After running the comparison, you'll find:
- Performance summary in `output_dir/comparison_summary.txt`
- Reward statistics in CSV files for each algorithm
- Visualization plots in `output_dir/algorithm_detailed_comparison.png`
- Videos in `output_dir/videos/` (if save_video is enabled)

## Notes

- Both algorithms use frame stacking (4 frames) to provide temporal information
- The models are automatically loaded with the appropriate preprocessing 
- The SAC implementation includes memory optimizations for handling large replay buffers
- To observe performance differences, it's recommended to run at least 10 evaluation episodes

For detailed logs during training, check the tensorboard logs:
```python
tensorboard --logdir logs/tensorboard/
```