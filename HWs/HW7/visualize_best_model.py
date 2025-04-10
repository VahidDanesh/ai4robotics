import torch
import minari
import gymnasium as gym
import numpy as np
import os

# Import the necessary components from train_model.py
from train_model import FCNN, Normalizer, run_policy, load_minari_data

# Function to load the best model

def load_best_model(model_path):
    # Load the model state
    checkpoint = torch.load(model_path)
    
    # Create network with the same architecture
    input_dim = checkpoint['X_mean'].shape[0]
    output_dim = checkpoint['Y_mean'].shape[0]
    hid_size = checkpoint['hid_size']
    num_layers = checkpoint['num_layers']
    
    net = FCNN(input_dim, output_dim, hid_size, num_layers)
    net.load_state_dict(checkpoint['net'])
    
    # Create normalizers with pre-loaded values
    class PreloadedNormalizer(Normalizer):
        def __init__(self, mean, std):
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

# Main function to visualize the best model
def main(model_path, num_episodes=10):
    # Load the best model
    net, X_normalizer, Y_normalizer = load_best_model(model_path)
    
    # Load the environment
    dataset_name = "mujoco/pusher/expert-v0"
    _, _, env = load_minari_data(dataset_name, render_mode="human")
    
    # Run the policy with visualization for multiple episodes
    print("Visualizing the best model...")
    try:
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}/{num_episodes}")
            run_policy(net, env, X_normalizer, Y_normalizer, seed=None)
    except Exception as e:
        print(f"Visualization failed: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    # Path to the best model
    model_path = "HWs/HW7/trained_pusher_policy.pt"
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
    else:
        main(model_path, num_episodes=20)  # Run 5 episodes for better observation 