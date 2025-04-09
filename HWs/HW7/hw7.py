import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import minari
import gymnasium as gym
import time
import argparse
import os.path

# Fix for Windows path issues with Hugging Face downloads
# Monkey patch huggingface_hub to handle backslashes correctly
try:
    import huggingface_hub
    original_hf_hub_download = huggingface_hub.hf_hub_download
    
    def patched_hf_hub_download(repo_id, filename, **kwargs):
        # Replace backslashes with forward slashes
        fixed_filename = filename.replace("\\", "/")
        return original_hf_hub_download(repo_id, fixed_filename, **kwargs)
    
    # Apply the patch
    huggingface_hub.file_download.hf_hub_download = patched_hf_hub_download
    huggingface_hub.hf_hub_download = patched_hf_hub_download
    print("Applied HuggingFace URL path fix for Windows")
except Exception as e:
    print(f"Warning: Could not apply HuggingFace URL path fix: {e}")

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Part 1.1: Fully-connected network implementation
class FCNN(nn.Module):
    def __init__(self, input_dim, output_dim, hid_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hid_size))
        self.layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hid_size, hid_size))
            self.layers.append(nn.ReLU())
        
        # Output layer (no activation)
        self.layers.append(nn.Linear(hid_size, output_dim))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Part 1.2: Data normalization
class Normalizer:
    def __init__(self, X):
        # Compute stats on the input device, then move to target device
        self.mean = torch.mean(X, dim=0).to(device)
        self.std = torch.std(X, dim=0).to(device) + 1e-5  # Add small constant to avoid division by zero
    
    def normalize(self, X):
        # Ensure X is on the same device as mean and std
        if X.device != self.mean.device:
            X = X.to(self.mean.device)
        return (X - self.mean) / self.std
    
    def denormalize(self, X_normalized):
        # Ensure X_normalized is on the same device as mean and std
        if X_normalized.device != self.mean.device:
            X_normalized = X_normalized.to(self.mean.device)
        return X_normalized * self.std + self.mean

# Part 1.3: Training function
def train(X, Y, net, num_epochs, batchsize, model_path='trained_pusher_policy.pt'):
    # First move to device, then normalize
    X = X.to(device)
    Y = Y.to(device)
    
    # Normalize data
    X_normalizer = Normalizer(X)
    X_norm = X_normalizer.normalize(X)
    Y_normalizer = Normalizer(Y)
    Y_norm = Y_normalizer.normalize(Y)
    
    # Print shapes for debugging
    print(f"X_norm shape: {X_norm.shape}, device: {X_norm.device}")
    print(f"Y_norm shape: {Y_norm.shape}, device: {Y_norm.device}")
    
    # Create DataLoader
    dataset = TensorDataset(X_norm, Y_norm)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    
    # Initialize best loss tracking
    best_loss = float('inf')
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        batches = 0
        
        for batch_X, batch_Y in dataloader:
            # No need to move batch tensors to device as they're already on it
            optimizer.zero_grad()
            
            # Forward pass
            Yhat = net(batch_X)
            loss = criterion(Yhat, batch_Y)
            epoch_loss += loss.item()
            batches += 1
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        
        # Calculate average loss for this epoch
        avg_epoch_loss = epoch_loss / batches
        
        # Print progress
        epoch_time = time.time() - epoch_start
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}, Time: {epoch_time:.2f}s")
        
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Best model loss: {best_loss:.6f}")
    
    return X_normalizer, Y_normalizer

# Part 2.1: Loading Minari dataset and environment
def load_minari_data(dataset_name, render_mode=None):
    # Fix Windows path issue - replace backslashes with forward slashes
    dataset_name = dataset_name.replace("\\", "/")
    
    # Load dataset
    dataset = minari.load_dataset(dataset_name, download=True)
    env = dataset.recover_environment(render_mode=render_mode)
    
    # Extract all observations and actions
    observations = []
    actions = []
    
    for episode in dataset:
        # Only use corresponding observations for which we have actions
        # In most environments, the last observation doesn't have a corresponding action
        episode_obs = episode.observations[:-1]  # Exclude the last observation
        observations.extend(episode_obs)
        actions.extend(episode.actions)
    
    # Print for debugging
    print(f"Number of observations: {len(observations)}")
    print(f"Number of actions: {len(actions)}")
    
    # Ensure same length
    min_length = min(len(observations), len(actions))
    observations = observations[:min_length]
    actions = actions[:min_length]
    
    # Convert to tensors
    X = torch.tensor(np.array(observations), dtype=torch.float32)
    Y = torch.tensor(np.array(actions), dtype=torch.float32)
    
    return X, Y, env

# Part 2.2: Running a policy
def run_policy(net, env, X_normalizer, Y_normalizer, seed=None):
    obs, _ = env.reset(seed=seed)
    total_reward = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        # Normalize observation
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        obs_norm = X_normalizer.normalize(obs_tensor)
        
        # Get action from network
        with torch.no_grad():
            action_norm = net(obs_norm)
            action = Y_normalizer.denormalize(action_norm).squeeze(0).cpu().numpy()
        
        # Step environment
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
    
    return total_reward

# Main function to train and save the policy
def main(visualize=False):
    import os.path
    
    model_path = './HWs/HW7/trained_pusher_policy.pt'
    
    # Define dataset name with proper path format
    dataset_name = "mujoco/pusher/expert-v0"
    
    # Check if model exists
    if os.path.isfile(model_path):
        print(f"Loading existing model from {model_path}")
        # Load the model
        checkpoint = torch.load(model_path)
        
        # Create network with the same architecture
        input_dim = 23  # Known input dimension for pusher environment
        output_dim = 7  # Known output dimension for pusher environment
        hid_size = checkpoint['hid_size']
        num_layers = checkpoint['num_layers']
        
        net = FCNN(input_dim, output_dim, hid_size, num_layers)
        net.load_state_dict(checkpoint['net'])
        net = net.to(device)
        
        # Create normalizers with pre-loaded values
        class PreloadedNormalizer(Normalizer):
            def __init__(self, mean, std):
                # Skip the parent class init to avoid computing stats on empty tensor
                self.mean = mean
                self.std = std
        
        # Load normalizer parameters
        X_normalizer = PreloadedNormalizer(
            mean=checkpoint['X_mean'].to(device),
            std=checkpoint['X_std'].to(device)
        )
        
        Y_normalizer = PreloadedNormalizer(
            mean=checkpoint['Y_mean'].to(device),
            std=checkpoint['Y_std'].to(device)
        )
        
        print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')} with loss {checkpoint.get('loss', 'unknown')}")
        
        # Load environment for testing only
        _, _, env = load_minari_data(dataset_name)
    else:
        # Load data for training
        print("No existing model found. Training from scratch.")
        X, Y, env = load_minari_data(dataset_name)
        
        # Print shapes for debugging
        print(f"Original X shape: {X.shape}")
        print(f"Original Y shape: {Y.shape}")
        
        # Create and train network
        input_dim = X.shape[1]
        output_dim = Y.shape[1]
        hid_size = 256
        num_layers = 2
        
        net = FCNN(input_dim, output_dim, hid_size, num_layers)
        # Move model to GPU if available
        net = net.to(device)
        
        # Full training with GPU acceleration
        print(f"Training with {device} acceleration...")
        X_normalizer, Y_normalizer = train(X, Y, net, num_epochs=50, batchsize=1024, model_path=model_path)
    
    # Test with visualization if requested
    if visualize:
        try:
            # Create a new environment with visualization for testing
            print("Loading visualization environment...")
            _, _, vis_env = load_minari_data(dataset_name, render_mode="human")
            
            # Test the policy with visualization
            print("Running policy test with visualization...")
            reward = run_policy(net, vis_env, X_normalizer, Y_normalizer)
            print(f"Visualization test reward: {reward}")
            
            # Close the visualization environment
            vis_env.close()
        except Exception as e:
            print(f"Visualization failed with error: {e}")
            print("Continuing with non-visualization test...")
    
    # Test without visualization
    print("Running policy test without visualization...")
    reward = run_policy(net, env, X_normalizer, Y_normalizer)
    print(f"Test reward: {reward}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and run a behavior cloning policy')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization of the policy')
    args = parser.parse_args()
    
    main(visualize=True) 