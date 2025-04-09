import torch
import glob
import os
import argparse
import numpy as np
import minari
import gymnasium as gym

# Import required components from train_model.py
from train_model import FCNN, Normalizer, run_policy, load_minari_data

def load_model(model_path, env_observation_dim, env_action_dim):
    """Load a saved model and create the network and normalizers"""
    # Check if file exists
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the model state
    try:
        checkpoint = torch.load(model_path)
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None, None, None
    
    # Create network with the same architecture
    hid_size = checkpoint['hid_size']
    num_layers = checkpoint['num_layers']
    
    # Double check input and output dimensions
    try:
        net = FCNN(env_observation_dim, env_action_dim, hid_size, num_layers)
        net.load_state_dict(checkpoint['net'])
        
        # Create normalizers with pre-loaded values
        class PreloadedNormalizer(Normalizer):
            def __init__(self, mean, std):
                # Skip the regular initialization
                self.mean = mean
                self.std = std
        
        # Load normalizer parameters
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_normalizer = PreloadedNormalizer(
            mean=checkpoint['X_mean'].to(device),
            std=checkpoint['X_std'].to(device)
        )
        
        Y_normalizer = PreloadedNormalizer(
            mean=checkpoint['Y_mean'].to(device),
            std=checkpoint['Y_std'].to(device)
        )
        
        # Move network to appropriate device
        net = net.to(device)
        
        return net, X_normalizer, Y_normalizer
    except Exception as e:
        print(f"Error setting up model {model_path}: {e}")
        return None, None, None

def main(args):
    # Load environment to get dimensions
    dataset_name = "mujoco/pusher/expert-v0"
    _, _, env = load_minari_data(dataset_name)
    
    # Get observation and action dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Environment observation dimension: {obs_dim}")
    print(f"Environment action dimension: {action_dim}")
    
    # Find all model files
    model_files = glob.glob(os.path.join(args.model_dir, "*.pt"))
    if not model_files:
        print(f"No model files found in {args.model_dir}")
        return
    
    print(f"Found {len(model_files)} model files")
    
    # Evaluate each model
    results = []
    for model_path in model_files:
        model_name = os.path.basename(model_path)
        print(f"\nEvaluating model: {model_name}")
        
        # Load the model
        net, X_normalizer, Y_normalizer = load_model(model_path, obs_dim, action_dim)
        if net is None:
            print(f"Skipping model {model_name} due to loading errors")
            continue
        
        # Evaluate the model multiple times
        rewards = []
        for seed in range(args.num_trials):
            reward = run_policy(net, env, X_normalizer, Y_normalizer, seed=seed)
            rewards.append(reward)
            print(f"  Trial {seed+1}/{args.num_trials}: Reward = {reward:.2f}")
        
        # Calculate statistics
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)
        
        print(f"  Results: Mean = {mean_reward:.2f}, Std = {std_reward:.2f}, Min = {min_reward:.2f}, Max = {max_reward:.2f}")
        
        # Store results
        results.append({
            'model_path': model_path,
            'model_name': model_name,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'min_reward': min_reward,
            'max_reward': max_reward,
            'rewards': rewards
        })
    
    # Find best model
    if results:
        # Sort by mean reward (descending)
        results.sort(key=lambda x: x['mean_reward'], reverse=True)
        
        print("\n" + "="*80)
        print("Models ranked by mean reward:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['model_name']}: Mean = {result['mean_reward']:.2f}, "
                  f"Std = {result['std_reward']:.2f}, Min = {result['min_reward']:.2f}, "
                  f"Max = {result['max_reward']:.2f}")
        
        # Get the best model
        best_model = results[0]
        print("\n" + "="*80)
        print(f"Best model: {best_model['model_name']}")
        print(f"Mean reward: {best_model['mean_reward']:.2f}")
        print(f"Path: {best_model['model_path']}")
        
        # Copy the best model to the target location if specified
        if args.output_path:
            import shutil
            print(f"\nCopying best model to {args.output_path}")
            shutil.copy2(best_model['model_path'], args.output_path)
            print("Copy completed")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained behavior cloning models')
    parser.add_argument('--model_dir', type=str, default='HWs/HW7/models', help='Directory containing trained models')
    parser.add_argument('--num_trials', type=int, default=5, help='Number of evaluation trials per model')
    parser.add_argument('--output_path', type=str, default='HWs/HW7/trained_pusher_policy.pt', 
                        help='Path to save the best model (optional)')
    
    args = parser.parse_args()
    main(args) 