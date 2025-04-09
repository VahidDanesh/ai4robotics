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

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Fix for Windows path issues with Hugging Face downloads (if needed)
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

# Define a fully-connected neural network for behavior cloning
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

# Data normalization class
class Normalizer:
    def __init__(self, X):
        self.mean = torch.mean(X, dim=0).to(device)
        self.std = torch.std(X, dim=0).to(device) + 1e-5  # Add small constant to avoid division by zero
    
    def normalize(self, X):
        if X.device != self.mean.device:
            X = X.to(self.mean.device)
        return (X - self.mean) / self.std
    
    def denormalize(self, X_normalized):
        if X_normalized.device != self.mean.device:
            X_normalized = X_normalized.to(self.mean.device)
        return X_normalized * self.std + self.mean

# Function to load dataset
def load_dataset(dataset_name="mujoco/pusher/expert-v0", render_mode=None):
    # Fix Windows path issue - replace backslashes with forward slashes
    dataset_name = dataset_name.replace("\\", "/")
    
    print(f"Loading dataset: {dataset_name}")
    # Load dataset
    dataset = minari.load_dataset(dataset_name, download=True)
    env = dataset.recover_environment(render_mode=render_mode)
    
    # Extract all observations and actions
    observations = []
    actions = []
    
    for episode in dataset:
        # Exclude the last observation which doesn't have a corresponding action
        episode_obs = episode.observations[:-1]
        observations.extend(episode_obs)
        actions.extend(episode.actions)
    
    # Print data sizes
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

# Training function
def train_model(X, Y, net, num_epochs, batch_size, learning_rate=1e-3):
    # Move data to device
    X = X.to(device)
    Y = Y.to(device)
    
    # Create normalizers
    X_normalizer = Normalizer(X)
    Y_normalizer = Normalizer(Y)
    
    # Normalize data
    X_norm = X_normalizer.normalize(X)
    Y_norm = Y_normalizer.normalize(Y)
    
    # Print shapes for debugging
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"X_norm shape: {X_norm.shape}, Y_norm shape: {Y_norm.shape}")
    
    # Create DataLoader
    dataset = TensorDataset(X_norm, Y_norm)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    # Initialize best model tracking
    best_loss = float('inf')
    best_model_state = None
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs with batch size {batch_size}...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        # Training progress
        for batch_X, batch_Y in dataloader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = net(batch_X)
            loss = criterion(outputs, batch_Y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            batch_count += 1
        
        # Calculate average loss for this epoch
        avg_loss = epoch_loss / batch_count
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = net.state_dict().copy()
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            time_elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Time: {time_elapsed:.2f}s")
    
    # Restore best model
    if best_model_state is not None:
        net.load_state_dict(best_model_state)
        print(f"Restored best model with loss: {best_loss:.6f}")
    
    # Return normalizers for inference
    return X_normalizer, Y_normalizer, best_loss

# Function to run policy in the environment
def run_policy(net, env, X_normalizer, Y_normalizer, episodes=5, render_mode=None):
    # If render_mode is specified, create a new environment with rendering
    if render_mode is not None:
        dataset_name = "mujoco/pusher/expert-v0"
        dataset = minari.load_dataset(dataset_name)
        env = dataset.recover_environment(render_mode=render_mode)
    
    rewards = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Prepare observation
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Normalize
            obs_norm = X_normalizer.normalize(obs_tensor)
            
            # Get action from policy
            with torch.no_grad():
                action_norm = net(obs_norm)
                action = Y_normalizer.denormalize(action_norm).squeeze(0).cpu().numpy()
            
            # Take step in environment
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        
        rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}")
    
    avg_reward = sum(rewards) / len(rewards)
    print(f"Average reward over {episodes} episodes: {avg_reward:.2f}")
    
    # Close environment if it was created for rendering
    if render_mode is not None:
        env.close()
    
    return rewards

# Main function
def main(args):
    model_path = args.model_path
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    
    # Load dataset
    X, Y, env = load_dataset()
    
    # Get input and output dimensions
    input_dim = X.shape[1]  # Observation dimension
    output_dim = Y.shape[1]  # Action dimension
    
    # Create network with specified hyperparameters
    net = FCNN(
        input_dim=input_dim,
        output_dim=output_dim,
        hid_size=args.hidden_size,
        num_layers=args.num_layers
    )
    
    # Move network to device
    net = net.to(device)
    print(f"Created network with {args.num_layers} hidden layers of size {args.hidden_size}")
    
    # Train the model
    X_normalizer, Y_normalizer, best_loss = train_model(
        X, Y, net,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Save the model
    save_dict = {
        'net': net.state_dict(),
        'hid_size': args.hidden_size,
        'num_layers': args.num_layers,
        'X_mean': X_normalizer.mean,
        'X_std': X_normalizer.std,
        'Y_mean': Y_normalizer.mean,
        'Y_std': Y_normalizer.std,
        'loss': best_loss
    }
    
    torch.save(save_dict, model_path)
    print(f"Model saved to {model_path}")
    
    # Test the policy
    print("\nTesting policy...")
    rewards = run_policy(net, env, X_normalizer, Y_normalizer, episodes=args.test_episodes)
    
    # Visualize if requested
    if args.visualize:
        print("\nVisualizing policy...")
        try:
            rewards = run_policy(net, env, X_normalizer, Y_normalizer, 
                                episodes=1, render_mode="human")
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    # Return the best reward achieved
    return max(rewards) if rewards else 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an optimal behavior cloning model")
    
    # Model architecture
    parser.add_argument("--hidden_size", type=int, default=512, 
                        help="Size of hidden layers (default: 512)")
    parser.add_argument("--num_layers", type=int, default=2, 
                        help="Number of hidden layers (default: 2)")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=1024, 
                        help="Batch size for training (default: 1024)")
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--learning_rate", type=float, default=1e-3, 
                        help="Learning rate (default: 0.001)")
    
    # Output and testing
    parser.add_argument("--model_path", type=str, default="HWs/HW7/trained_pusher_policy.pt", 
                        help="Path to save the trained model")
    parser.add_argument("--test_episodes", type=int, default=5, 
                        help="Number of test episodes (default: 5)")
    parser.add_argument("--visualize", action="store_true", 
                        help="Visualize the trained policy")
    
    args = parser.parse_args()
    
    # Print configuration
    print("Training with the following configuration:")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Number of layers: {args.num_layers}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Model path: {args.model_path}")
    print(f"  Test episodes: {args.test_episodes}")
    print(f"  Visualization: {'Enabled' if args.visualize else 'Disabled'}")
    
    # Run training
    main(args) 