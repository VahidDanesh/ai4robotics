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

# Fully-connected network implementation
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

# Data normalization
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

# Training function
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
    best_model_state = None
    
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
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_state = net.state_dict().copy()
        
        # Print progress
        epoch_time = time.time() - epoch_start
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}, Time: {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Best model loss: {best_loss:.6f}")
    
    # Restore best model
    if best_model_state is not None:
        net.load_state_dict(best_model_state)
    
    # Save the model
    save_dict = {
        'net': net.state_dict(),
        'hid_size': net.layers[0].out_features,  # Get hidden size from first linear layer
        'num_layers': sum(1 for layer in net.layers if isinstance(layer, nn.Linear)) - 1,  # Count linear layers minus output
        'X_mean': X_normalizer.mean,
        'X_std': X_normalizer.std,
        'Y_mean': Y_normalizer.mean,
        'Y_std': Y_normalizer.std,
        'loss': best_loss,
        'epoch': num_epochs
    }
    
    torch.save(save_dict, model_path)
    print(f"Model saved to {model_path}")
    
    return X_normalizer, Y_normalizer

# Load Minari dataset and environment
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

# Run policy function
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

# Main function to train and test the model
def main(args):
    # Define dataset name
    dataset_name = "mujoco/pusher/expert-v0"
    model_path = args.model_path
    
    # Load data for training
    print("Loading data...")
    X, Y, env = load_minari_data(dataset_name)
    
    # Print shapes for debugging
    print(f"Original X shape: {X.shape}")
    print(f"Original Y shape: {Y.shape}")
    
    # Create network
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    hid_size = args.hid_size
    num_layers = args.num_layers
    
    net = FCNN(input_dim, output_dim, hid_size, num_layers)
    net = net.to(device)
    
    print(f"Training network with {num_layers} hidden layers of size {hid_size}")
    X_normalizer, Y_normalizer = train(X, Y, net, 
                                      num_epochs=args.epochs, 
                                      batchsize=args.batch_size, 
                                      model_path=model_path)
    
    # Test the trained policy
    print("Testing trained policy...")
    test_reward = run_policy(net, env, X_normalizer, Y_normalizer)
    print(f"Test reward without visualization: {test_reward}")
    
    # Test with visualization if requested
    if args.visualize:
        try:
            print("Testing with visualization...")
            _, _, vis_env = load_minari_data(dataset_name, render_mode="human")
            vis_reward = run_policy(net, vis_env, X_normalizer, Y_normalizer)
            print(f"Visualization test reward: {vis_reward}")
            vis_env.close()
        except Exception as e:
            print(f"Visualization failed with error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a behavior cloning model')
    parser.add_argument('--hid_size', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--model_path', type=str, default='trained_pusher_policy.pt', help='Path to save the trained model')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization after training')
    
    args = parser.parse_args()
    main(args) 