import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import minari
import gymnasium as gym

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
        self.mean = torch.mean(X, dim=0)
        self.std = torch.std(X, dim=0) + 1e-5  # Add small constant to avoid division by zero
    
    def normalize(self, X):
        return (X - self.mean) / self.std
    
    def denormalize(self, X_normalized):
        return X_normalized * self.std + self.mean

# Part 1.3: Training function
def train(X, Y, net, num_epochs, batchsize):
    # Normalize data
    X_normalizer = Normalizer(X)
    X_norm = X_normalizer.normalize(X)
    Y_normalizer = Normalizer(Y)
    Y_norm = Y_normalizer.normalize(Y)
    
    # Create DataLoader
    dataset = TensorDataset(X_norm, Y_norm)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    
    # Training loop
    for epoch in range(num_epochs):
        for batch_X, batch_Y in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            Yhat = net(batch_X)
            loss = criterion(Yhat, batch_Y)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
    
    return X_normalizer, Y_normalizer

# Part 2.1: Loading Minari dataset and environment
def load_minari_data(dataset_name):
    # Load dataset
    dataset = minari.load_dataset(dataset_name, download=True)
    env = dataset.recover_environment()
    
    # Extract all observations and actions
    observations = []
    actions = []
    
    for episode in dataset:
        observations.extend(episode.observations)
        actions.extend(episode.actions)
    
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
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        obs_norm = X_normalizer.normalize(obs_tensor)
        
        # Get action from network
        with torch.no_grad():
            action_norm = net(obs_norm)
            action = Y_normalizer.denormalize(action_norm).squeeze(0).numpy()
        
        # Step environment
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
    
    return total_reward

# Main function to train and save the policy
def main():
    # Load data
    X, Y, env = load_minari_data("mujoco/pusher/expert-v0")
    
    # Create and train network
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    hid_size = 256
    num_layers = 2
    
    net = FCNN(input_dim, output_dim, hid_size, num_layers)
    X_normalizer, Y_normalizer = train(X, Y, net, num_epochs=100, batchsize=64)
    
    # Save trained network and normalizers
    save_dict = {
        'net': net.state_dict(),
        'hid_size': hid_size,
        'num_layers': num_layers,
        'X_mean': X_normalizer.mean,
        'X_std': X_normalizer.std,
        'Y_mean': Y_normalizer.mean,
        'Y_std': Y_normalizer.std,
    }
    torch.save(save_dict, 'trained_pusher_policy.pt')
    
    # Test the policy
    reward = run_policy(net, env, X_normalizer, Y_normalizer)
    print(f"Test reward: {reward}")

if __name__ == "__main__":
    main() 