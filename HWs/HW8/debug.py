import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
import time

# Copy all the necessary classes and functions from hw8.ipynb
class GaussianPolicy_11(nn.Module):
    def __init__(self, obs_dim, action_dim, hid_size, num_layers):
        super().__init__()

        layers = nn.ModuleList()
        current_dim = obs_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hid_size))
            layers.append(nn.Tanh())
            current_dim = hid_size

        layers.append(nn.Linear(hid_size, action_dim))

        self.mu = nn.Sequential(*layers)
        # Initialize the log std to all zeros
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        ''' 
        obs: Tensor of shape (batch_size, obs_dim)
        return: mu: Tensor of shape (batch_size, action_dim)
                std: Tensor of shape (action_dim,)
        '''
        mu = self.mu(obs)
        std = torch.exp(self.log_std)
        return mu, std

    def sample(self, obs):
        '''
        obs: Tensor of shape (batch_size, obs_dim)
        return: action: Tensor of shape (batch_size, action_dim)

        Hint: Look into torch.distributions.Normal
        '''
        mu, std = self(obs)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        return action
    
    def log_prob(self, obs, action):
        '''
        obs: Tensor of shape (batch_size, obs_dim)
        action: Tensor of shape (batch_size, action_dim)
        return: log_prob: Tensor of shape (batch_size,)

        Hint 1: You may use torch.distributions.Normal.log_prob
        Hint 2: Think about how the joint probability of independent variables is 
        computed and then how you may do this in log space.
        '''
        mu, std = self.forward(obs)
        normal = torch.distributions.Normal(mu, std)
        log_prob = normal.log_prob(action)
        # Sum across action dimensions to get joint probability in log space
        return log_prob.sum(dim=1)

def sample_trajectory_12(env, policy, seed=None):
    obs, _ = env.reset(seed=seed)
    observations, actions, rewards = [], [], []
    steps = 0
    done = False

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action = policy.sample(obs_tensor).numpy().flatten()
        
        next_obs, reward, terminated, truncated, _ = env.step(action)

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)

        obs = next_obs
        done = terminated or truncated
        steps += 1

    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    path = {'observations': torch.from_numpy(observations),
            'actions': torch.from_numpy(actions),
            'rewards': torch.from_numpy(rewards),
            'length': steps}
    return path

def sample_trajectories_12(env, policy, min_batch_size, seed=None):
    timesteps = 0
    paths = []
    itr = 0
    while timesteps < min_batch_size:
        with torch.no_grad():   # accelerate computation by turning off unnecessary gradients
            path = sample_trajectory_12(env, policy, seed=seed+itr*1000)
        itr += 1
        paths.append(path)
        timesteps += path['length']
    return paths, timesteps

def sum_of_rewards_13(rewards, gamma):
    all_rtgs = []
    
    for trajectory_rewards in rewards:
        path_length = len(trajectory_rewards)
        rtgs = torch.zeros_like(trajectory_rewards)
        
        # Compute reward-to-go for each timestep
        for t in reversed(range(path_length)):
            rtgs[t] = trajectory_rewards[t]
            if t + 1 < path_length:
                rtgs[t] += gamma * rtgs[t+1]
                
        all_rtgs.append(rtgs)
    
    # Concatenate all rewards-to-go
    return torch.cat(all_rtgs)

def pg_loss_fn_14(policy, observations, actions, returns):
    log_probs = policy.log_prob(observations, actions)
    loss = -torch.mean(log_probs * returns)
    return loss

class FCNN_11(nn.Module):
    def __init__(self, input_dim, output_dim, hid_size, num_layers):
        super().__init__()
        layers = nn.ModuleList()
        current_dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hid_size))
            layers.append(nn.Tanh())
            current_dim = hid_size

        layers.append(nn.Linear(hid_size, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def compute_baseline_21(nn_baseline, observations, rewards_to_go):
    with torch.no_grad():   # We don't backprop through the baseline when computing PGs
        baseline_values = nn_baseline(observations).squeeze()

        rtg_mean = rewards_to_go.mean()
        rtg_std = rewards_to_go.std()

        baseline_mean = baseline_values.mean()
        baseline_std = baseline_values.std() + 1e-5

        baseline = (baseline_values - baseline_mean) / baseline_std * rtg_std + rtg_mean
    return baseline

def baseline_loss_22(nn_baseline, observations, rewards_to_go):
    # Normalize rewards_to_go to have mean 0 and std 1
    rtg_mean = rewards_to_go.mean()
    rtg_std = rewards_to_go.std() + 1e-5  # Add small constant to avoid division by zero
    normalized_rtg = (rewards_to_go - rtg_mean) / rtg_std
    
    # Get baseline predictions
    baseline_predictions = nn_baseline(observations).squeeze()
    
    # Compute mean squared error
    mse = torch.mean((baseline_predictions - normalized_rtg) ** 2)
    
    return mse

def update_parameters_15(policy, observations, actions, returns, optimizer):
    loss = pg_loss_fn_14(policy, observations, actions, returns)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach().item()

def update_baseline_parameters_23(nn_baseline, observations, rewards_to_go, optimizer):
    loss = baseline_loss_22(nn_baseline, observations, rewards_to_go)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach().item()

def train_PG_15(env_name, 
                hid_size, 
                num_layers, 
                use_baseline=False, 
                num_iterations=100, 
                batch_size=1000,
                gamma=1.0,
                normalize_returns=False,
                learning_rate=1e-3,
                seed=0):
    # Make the gym environment
    env = gym.make(env_name)

    # Set the random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = GaussianPolicy_11(obs_dim, action_dim, hid_size, num_layers)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    if use_baseline:
        nn_baseline = FCNN_11(obs_dim, 1, hid_size, num_layers)
        optimizer_baseline = torch.optim.Adam(nn_baseline.parameters(), lr=learning_rate)
    
    total_timesteps = 0
    returns_history = []
    highest_returns = -np.inf
    best_policy = None
    
    for iter in range(num_iterations):
        print(f"********** Iteration {iter} **********")

        # Sample trajectories
        paths, timesteps = sample_trajectories_12(env, policy, batch_size, seed=seed)
        total_timesteps += timesteps

        # Build tensors
        observations = torch.cat([path['observations'] for path in paths], dim=0)
        actions = torch.cat([path['actions'] for path in paths], dim=0)
        rewards = [path['rewards'] for path in paths]

        # Print the average undiscounted return
        undiscounted_return = torch.cat(rewards).sum() / len(paths)
        returns_history.append(undiscounted_return.item())
        print(f"\tAverage return: {undiscounted_return:.3f}")
        
        # Save best policy
        if undiscounted_return > highest_returns:
            highest_returns = undiscounted_return
            best_policy = policy.state_dict().copy()
            print(f"\tNew best policy with return: {highest_returns:.3f}")

        # Compute the reward-to-go
        rewards_to_go = sum_of_rewards_13(rewards, gamma)

        if use_baseline:
            baseline = compute_baseline_21(nn_baseline, observations, rewards_to_go)
            returns = rewards_to_go - baseline
        else:
            returns = rewards_to_go
            
        if normalize_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Update the policy
        loss = update_parameters_15(policy, observations, actions, returns, optimizer)
        print(f"\tPolicy loss: {loss:.5f}")

        if use_baseline:
            baseline_loss = update_baseline_parameters_23(nn_baseline, observations, rewards_to_go, optimizer_baseline)
            print(f"\tBaseline loss: {baseline_loss:.5f}")
    
    # Load the best policy
    if best_policy is not None:
        policy.load_state_dict(best_policy)
        print(f"Loaded best policy with return: {highest_returns:.3f}")
    
    return policy, returns_history

def visualize_policy(env_name, policy, num_episodes=1):
    env = gym.make(env_name, render_mode="human")
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = policy.sample(torch.from_numpy(obs.astype(np.float32)))
            action = action.detach().numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            env.render()
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {episode+1} reward: {total_reward:.2f}")
    env.close()

def test_inverted_pendulum():
    print("\n========== Testing InvertedPendulum-v5 ==========")
    # Parameters from task 1.6
    env_name = "InvertedPendulum-v5"
    hid_size = 64
    num_layers = 2
    num_iterations = 100
    batch_size = 1000  # From task 1.6
    gamma = 0.99
    normalize_returns = True  # From task 1.6
    learning_rate = 3e-3  # From task 1.6
    seed = 0
    
    start_time = time.time()
    
    policy, returns_history = train_PG_15(
        env_name=env_name,
        hid_size=hid_size,
        num_layers=num_layers,
        use_baseline=False,
        num_iterations=num_iterations,
        batch_size=batch_size,
        gamma=gamma,
        normalize_returns=normalize_returns,
        learning_rate=learning_rate,
        seed=seed
    )
    
    training_time = time.time() - start_time
    print(f"Training took {training_time:.2f} seconds")
    
    # Save the trained policy
    torch.save(policy.state_dict(), 'trained_pendulum_policy.pt')
    print("Saved policy to trained_pendulum_policy.pt")
    
    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(returns_history)
    plt.xlabel('Iteration')
    plt.ylabel('Average Return')
    plt.title('Learning Curve for InvertedPendulum-v5')
    plt.savefig('pendulum_learning_curve.png')
    plt.close()
    
    return policy

def test_half_cheetah():
    print("\n========== Testing HalfCheetah-v5 ==========")
    # Parameters from task 2.4
    env_name = "HalfCheetah-v5"
    hid_size = 32
    num_layers = 2
    use_baseline = True  
    num_iterations = 200
    batch_size = 20000 
    gamma = 0.95
    normalize_returns = True
    learning_rate = 1e-2  
    seed = 0
    
    start_time = time.time()
    
    policy, returns_history = train_PG_15(
        env_name=env_name,
        hid_size=hid_size,
        num_layers=num_layers,
        use_baseline=use_baseline,
        num_iterations=num_iterations,
        batch_size=batch_size,
        gamma=gamma,
        normalize_returns=normalize_returns,
        learning_rate=learning_rate,
        seed=seed
    )
    
    training_time = time.time() - start_time
    print(f"Training took {training_time:.2f} seconds")
    
    # Save the trained policy
    model_path = f'cheetah_{use_baseline}_{batch_size}_{learning_rate}.pt'
    torch.save(policy.state_dict(), model_path)
    print(f"Saved policy to {model_path}")
    
    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(returns_history)
    plt.xlabel('Iteration')
    plt.ylabel('Average Return')
    plt.title('Learning Curve for HalfCheetah-v5')
    plt.savefig(f'cheetah_{use_baseline}_{batch_size}_{learning_rate}.png')
    plt.close()
    
    return policy

def load_and_visualize(env_name, model_path, hid_size=64, num_layers=2):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    policy = GaussianPolicy_11(obs_dim, action_dim, hid_size, num_layers)
    policy.load_state_dict(torch.load(model_path))
    
    visualize_policy(env_name, policy, num_episodes=3)

if __name__ == "__main__":
    print("Policy Gradient Debugging Script")
    print("Choose an option:")
    print("1. Train InvertedPendulum-v5")
    print("2. Train HalfCheetah-v5")
    print("3. Visualize trained InvertedPendulum-v5")
    print("4. Visualize trained HalfCheetah-v5")
    
    choice = input("Enter your choice (1-4): ")
    
    if choice == '1':
        pendulum_policy = test_inverted_pendulum()
        visualize = input("Visualize trained policy? (y/n): ")
        if visualize.lower() == 'y':
            visualize_policy("InvertedPendulum-v5", pendulum_policy)
    
    elif choice == '2':
        cheetah_policy = test_half_cheetah()
        visualize = input("Visualize trained policy? (y/n): ")
        if visualize.lower() == 'y':
            visualize_policy("HalfCheetah-v5", cheetah_policy)
    
    elif choice == '3':
        try:
            load_and_visualize("InvertedPendulum-v5", "trained_pendulum_policy.pt", hid_size=64, num_layers=2)
        except FileNotFoundError:
            print("Policy file not found. Train the pendulum policy first.")
    
    elif choice == '4':
        try:
            load_and_visualize("HalfCheetah-v5", "trained_cheetah_policy.pt", hid_size=32, num_layers=2)
        except FileNotFoundError:
            print("Policy file not found. Train the cheetah policy first.")
    
    else:
        print("Invalid choice")
