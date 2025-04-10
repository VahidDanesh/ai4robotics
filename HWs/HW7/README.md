# Homework 7 - Behavior Cloning

This homework implements behavior cloning for robotics tasks using neural networks. The goal is to train a neural network to imitate expert demonstrations in the MuJoCo pusher environment.

## Overview

The homework consists of two main parts:

1. **Neural Network Implementation**
   - Fully-connected neural network (FCNN) implementation
   - Data normalization for training stability
   - Training function with batch processing

2. **Behavior Cloning**
   - Loading and processing demonstration data from Minari
   - Implementing policy execution in the environment
   - Training and evaluating the policy

## Requirements

Install the required packages:
```bash
pip install torch numpy minari gymnasium mujoco
```

## Files

- `hw7.py`: Main implementation file containing all the required code
- `train_model.py`: Script for training the model
- `visualize_best_model.py`: Script for visualizing the trained model
- `trained_pusher_policy.pt`: Saved trained policy
- `best_trained_pusher_policy.pt`: Best performing trained policy

## Training the Model

To train the model, run:
```bash
python train_model.py
```

The training script will:
1. Load the MuJoCo pusher dataset
2. Train a neural network with the following default parameters:
   - Hidden size: 256
   - Number of layers: 2
   - Batch size: 64
   - Number of epochs: 100
3. Save the trained model to `trained_pusher_policy.pt`

## Visualizing the Model

To visualize the trained model in action, run:
```bash
python visualize_best_model.py
```

This will:
1. Load the best trained model
2. Run the policy in the environment with visualization enabled
3. Display the robot's behavior in real-time

## Notes

- The model is trained on expert demonstrations from the `mujoco/pusher/expert-v0` dataset
- Training uses MSE loss and Adam optimizer
- The environment is rendered in real-time during visualization
- The policy's performance is measured by the total reward accumulated during an episode

## Submission

Submit the following files:
1. `hw7.ipynb` with your code implementations
2. `trained_pusher_policy.pt` containing your trained model

The submission will be evaluated based on the performance of your trained policy in the pusher environment. 