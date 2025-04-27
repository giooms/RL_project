# A2C for Car Racing: Protocol Analysis and Improvement Recommendations

## Current Implementation Analysis
Our implementation of Advantage Actor-Critic (A2C) for the CarRacing-v3 environment follows this training protocol:

### Architecture
- **CNN Feature Extractor**: Processes 4 stacked RGB frames (12 channels) through 3 convolutional layers
- **Actor-Critic Network**: Shares a feature extraction backbone with separate heads for:
    - **Actor**: Outputs action means for a Gaussian policy
    - **Critic**: Outputs state value estimates

### Training Protocol
- **Update Frequency**: Collects 2048 steps before each parameter update
- **Experience Collection**: Gathers trajectories with non-deterministic actions
- **Return Calculation**: Uses bootstrapped TD(0) returns with discount factor γ=0.99
- **Loss Components**:
    - Policy loss: -log_prob * advantage
    - Value loss: MSE between predicted values and returns
    - Entropy loss: Negative entropy to encourage exploration
- **Hyperparameters**:
    - Learning rate: 3e-4
    - Entropy coefficient: 0.01
    - Value coefficient: 0.5
    - Gradient norm clipping: 0.5

### Evaluation & Checkpointing
- **Evaluation Frequency**: Every 10,000 steps with deterministic actions
- **Model Saving**: Saves best model based on evaluation reward
- **Final Model**: Saves model at end of training

## Performance Analysis
The training logs reveal several issues:

- **Inconsistent Rewards**: The agent shows highly unstable performance
- **Poor Learning**: Most rewards are negative (-90 to -50), with only two positive spikes
- **Reward Oscillation**: Rewards fluctuate between better and worse periods without clear improvement
- **Instability**: Large differences in performance between consecutive evaluations