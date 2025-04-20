import os
import torch
import numpy as np
import gymnasium as gym
import argparse
from hw8 import GaussianPolicy_11, train_PG_15, FCNN_11, visualize_policy

def main(args):
    # Set device to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create results directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Hyperparameters
    env_name = args.env_name
    hid_size = args.hid_size
    num_layers = args.num_layers
    gamma = args.gamma
    
    # If not doing hyperparameter search, use provided hyperparameters
    if not args.hyper_search:
        print(f"\n{'='*80}")
        print(f"Training with use_baseline={args.use_baseline}, batch_size={args.batch_size}, learning_rate={args.learning_rate}")
        print(f"{'='*80}\n")
        
        # Train policy with specified hyperparameters
        policy = train_PG_15(
            env_name=env_name,
            hid_size=hid_size,
            num_layers=num_layers,
            use_baseline=args.use_baseline,
            num_iterations=args.num_iterations,
            batch_size=args.batch_size,
            gamma=gamma,
            normalize_returns=args.normalize_returns,
            learning_rate=args.learning_rate,
            seed=args.seed
        )
        
        # Save the trained policy
        model_path = os.path.join(args.output_dir, f"trained_{env_name.split('-')[0].lower()}_policy.pt")
        torch.save(policy.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Evaluate the policy
        evaluate_policy(env_name, policy, args.seed, num_episodes=30)
        
        # Visualize if requested
        if args.visualize:
            visualize_policy(env_name, policy, num_episodes=3)
    
    else:
        # Hyperparameter search
        best_return = -np.inf
        best_hyperparams = None
        best_policy = None
        
        # Define hyperparameter options for grid search
        use_baseline_options = [True, False]
        batch_size_options = [10000, 30000, 50000]
        learning_rate_options = [1e-3, 3e-3, 1e-2, 3e-2]
        
        # Log file to track all results
        log_file_path = os.path.join(args.output_dir, "hyperparameter_search_log.txt")
        with open(log_file_path, "w") as log_file:
            log_file.write(f"Hyperparameter Search Results for {env_name}\n")
            log_file.write("=" * 50 + "\n")
            log_file.write("Format: use_baseline, batch_size, learning_rate -> max_return\n\n")
        
        # Perform grid search
        for use_baseline in use_baseline_options:
            for batch_size in batch_size_options:
                for lr in learning_rate_options:
                    print(f"\n{'='*80}")
                    print(f"Testing with use_baseline={use_baseline}, batch_size={batch_size}, learning_rate={lr}")
                    print(f"{'='*80}\n")
                    
                    # Train policy with current hyperparameters
                    policy = train_PG_15(
                        env_name=env_name,
                        hid_size=hid_size,
                        num_layers=num_layers,
                        use_baseline=use_baseline,
                        num_iterations=args.num_iterations,
                        batch_size=batch_size,
                        gamma=gamma,
                        normalize_returns=True,  # Always true for search
                        learning_rate=lr,
                        seed=args.seed
                    )
                    
                    # Evaluate the trained policy
                    avg_return = evaluate_policy(env_name, policy, args.seed)
                    
                    # Log result
                    with open(log_file_path, "a") as log_file:
                        log_file.write(f"use_baseline={use_baseline}, batch_size={batch_size}, lr={lr} -> {avg_return:.2f}\n")
                    
                    # Save the current model
                    model_path = os.path.join(args.output_dir, f"cheetah_policy_{use_baseline}_{batch_size}_{lr}.pt")
                    torch.save(policy.state_dict(), model_path)
                    print(f"Model saved to {model_path}")
                    
                    # Update best model if necessary
                    if avg_return > best_return:
                        best_return = avg_return
                        best_hyperparams = (use_baseline, batch_size, lr)
                        best_policy = policy
                        
                        # Save the current best model
                        best_model_path = os.path.join(args.output_dir, "trained_cheetah_policy.pt")
                        torch.save(policy.state_dict(), best_model_path)
                        print(f"New best model saved with return: {avg_return:.2f}")
        
        # Print final results
        print("\n" + "="*50)
        print("Hyperparameter search complete!")
        print(f"Best hyperparameters: use_baseline={best_hyperparams[0]}, batch_size={best_hyperparams[1]}, learning_rate={best_hyperparams[2]}")
        print(f"Best average return: {best_return:.2f}")
        print("="*50)
        
        # Save best hyperparameters to a file
        with open(os.path.join(args.output_dir, "best_hyperparameters.txt"), "w") as f:
            f.write(f"Best hyperparameters for {env_name}:\n")
            f.write(f"use_baseline: {best_hyperparams[0]}\n")
            f.write(f"batch_size: {best_hyperparams[1]}\n")
            f.write(f"learning_rate: {best_hyperparams[2]}\n")
            f.write(f"Best average return: {best_return:.2f}\n")
        
        # Visualize the best policy if requested
        if args.visualize:
            print("\nVisualizing the best policy...")
            visualize_policy(env_name, best_policy, num_episodes=3)

def evaluate_policy(env_name, policy, seed, num_episodes=30):
    """Evaluate the policy over multiple episodes and return the average reward."""
    eval_env = gym.make(env_name)
    returns = []
    
    for i in range(num_episodes):
        obs, _ = eval_env.reset(seed=seed+i*1000)
        episode_return = 0
        done = False
        
        while not done:
            # Move observation to device
            obs_tensor = torch.from_numpy(obs.astype(np.float32))
            
            # Sample action
            with torch.no_grad():
                action = policy.sample(obs_tensor)
                action = action.detach().numpy()
            
            # Take step in environment
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_return += reward
            done = terminated or truncated
            
        returns.append(episode_return)
    
    # Calculate average return
    avg_return = np.mean(returns)
    print(f"Average return over {num_episodes} evaluation episodes: {avg_return:.2f}")
    
    return avg_return

def load_and_test_policy(model_path, env_name, hid_size=32, num_layers=2):
    """Load a trained policy and visualize its performance."""
    # Create environment to get dimensions
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create policy
    policy = GaussianPolicy_11(obs_dim, action_dim, hid_size, num_layers)
    
    # Load weights
    policy.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    policy.eval()
    
    # Evaluate and visualize
    avg_return = evaluate_policy(env_name, policy, seed=0)
    visualize_policy(env_name, policy, num_episodes=3)
    
    return avg_return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train policy gradient agent")
    
    # Environment
    parser.add_argument('--env_name', type=str, default="HalfCheetah-v5", 
                        help='Name of the gym environment')
    
    # Model architecture
    parser.add_argument('--hid_size', type=int, default=32, 
                        help='Size of hidden layers')
    parser.add_argument('--num_layers', type=int, default=2, 
                        help='Number of hidden layers')
    
    # Training parameters
    parser.add_argument('--use_baseline', type=bool, default=True, 
                        help='Whether to use a baseline')
    parser.add_argument('--num_iterations', type=int, default=100, 
                        help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=30000, 
                        help='Batch size (number of steps)')
    parser.add_argument('--gamma', type=float, default=0.95, 
                        help='Discount factor')
    parser.add_argument('--normalize_returns', type=bool, default=True, 
                        help='Whether to normalize returns')
    parser.add_argument('--learning_rate', type=float, default=1e-2, 
                        help='Learning rate')
    
    # Other
    parser.add_argument('--seed', type=int, default=0, 
                        help='Random seed')
    parser.add_argument('--hyper_search', action='store_true',
                        help='Whether to perform hyperparameter search')
    parser.add_argument('--visualize', action='store_true',
                        help='Whether to visualize the trained policy')
    parser.add_argument('--output_dir', type=str, default='HWs/HW8', 
                        help='Directory to save outputs')
    parser.add_argument('--test_model', type=str, default=None,
                        help='Path to a model to test (skips training)')
    
    args = parser.parse_args()
    
    if args.test_model:
        # Test a pre-trained model
        load_and_test_policy(args.test_model, args.env_name, args.hid_size, args.num_layers)
    else:
        # Train a new model
        main(args)
