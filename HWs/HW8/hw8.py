import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

# 1.1 Gaussian Policy
class GaussianPolicy_11(nn.Module):
    def __init__(self, obs_dim, action_dim, hid_size, num_layers):
        super().__init__()
        
        # Create a neural network for mu with Tanh activation
        layers = []
        input_size = obs_dim
        
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, hid_size))
            layers.append(nn.Tanh())
            input_size = hid_size
            
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
        '''
        mu, std = self.forward(obs)
        normal = torch.distributions.Normal(mu, std)
        action = normal.sample()
        return action

    def log_prob(self, obs, action):
        '''
        obs: Tensor of shape (batch_size, obs_dim)
        action: Tensor of shape (batch_size, action_dim)
        return: log_prob: Tensor of shape (batch_size,)
        '''
        mu, std = self.forward(obs)
        normal = torch.distributions.Normal(mu, std)
        log_prob = normal.log_prob(action)
        # Sum across action dimensions to get joint probability in log space
        return log_prob.sum(dim=1)

# 1.2 Sample a trajectory
def sample_trajectory_12(env, policy, seed=None):
    obs, _ = env.reset(seed=seed)
    observations, actions, rewards = [], [], []
    steps = 0
    
    done = False
    while not done:
        # Convert observation to tensor
        obs_tensor = torch.from_numpy(obs.astype(np.float32))
        
        # Sample action using policy
        with torch.no_grad():
            action = policy.sample(obs_tensor).numpy()
        
        # Take action in environment
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        # Store data
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        
        # Update observation and done flag
        obs = next_obs
        done = terminated or truncated
        steps += 1
    
    # Note: we discard the final observation as specified
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

# 1.3 Computing the cumulative rewards
def sum_of_rewards_13(rewards, gamma):
    ''' 
    rewards: list of torch tensors, each of which is the rewards for a single trajectory
    gamma: float, the discount factor
    return: torch tensor of shape (num_trajectories * num_steps, ), the reward-to-go for each time step, flattened
    '''
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

# 1.4 Loss function
def pg_loss_fn_14(policy, observations, actions, returns):
    ''' 
    policy: GaussianPolicy_11 object
    observations: Tensor of shape (num_trajectories * num_steps, obs_dim)
    actions: Tensor of shape (num_trajectories * num_steps, action_dim)
    returns: Tensor of shape (num_trajectories * num_steps, )
    return: loss: Tensor of shape (1, ), a loss whose gradient can be used for gradient *descent*
    '''
    # Get log probabilities of the actions
    log_probs = policy.log_prob(observations, actions)
    
    # Multiply by returns and average
    # Note: We negate because we want to maximize returns, but optimizers perform minimization
    loss = -torch.mean(log_probs * returns)
    
    return loss

# 1.5 Putting it all together: PG training
def update_parameters_15(policy, observations, actions, returns, optimizer):
    loss = pg_loss_fn_14(policy, observations, actions, returns)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach().item()

# FCNN for baseline
class FCNN_11(nn.Module):
    def __init__(self, input_dim, output_dim, hid_size, num_layers):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hid_size))
            layers.append(nn.Tanh())
            current_dim = hid_size
            
        layers.append(nn.Linear(hid_size, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

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

    # Initialize policy and optimizer
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    policy = GaussianPolicy_11(obs_dim, action_dim, hid_size, num_layers)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    if use_baseline:
        nn_baseline = FCNN_11(obs_dim, 1, hid_size, num_layers)
        optimizer_baseline = torch.optim.Adam(nn_baseline.parameters(), lr=1e-3)
    
    total_timesteps = 0
    highest_returns = -np.inf
    
    for iter in range(num_iterations):
        print(f"********** Iteration {iter} **********")

        # Sample trajectories
        paths, timesteps = sample_trajectories_12(env, policy, batch_size, seed=seed+iter*100)
        total_timesteps += timesteps

        # Build tensors
        observations = torch.cat([path['observations'] for path in paths], dim=0)
        actions = torch.cat([path['actions'] for path in paths], dim=0)
        rewards = [path['rewards'] for path in paths]

        # Print the average undiscounted return
        undiscounted_return = torch.cat(rewards).sum() / len(paths)
        print(f"\tAverage return: {undiscounted_return:.3f}")
        
        # Save best policy (optional)
        if undiscounted_return > highest_returns and env_name == "HalfCheetah-v5":
            print(f"\tSaving best policy with return: {undiscounted_return:.3f}")
            torch.save(policy.state_dict(), 'trained_cheetah_policy.pt')
            highest_returns = undiscounted_return

        # Compute the reward-to-go
        rewards_to_go = sum_of_rewards_13(rewards, gamma)

        if use_baseline:
            baseline = compute_baseline_21(nn_baseline, observations, rewards_to_go)
            returns = rewards_to_go - baseline
        else:
            returns = rewards_to_go
            
        if normalize_returns:
            # Normalize the returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Update the policy
        loss = update_parameters_15(policy, observations, actions, returns, optimizer)
        print(f"\tPolicy loss: {loss:.5f}")

        if use_baseline:
            # Update the baseline
            baseline_loss = update_baseline_parameters_23(nn_baseline, observations, rewards_to_go, optimizer_baseline)
            print(f"\tBaseline loss: {baseline_loss:.5f}")

    return policy

# 2.1 Computing the baseline
def compute_baseline_21(nn_baseline, observations, rewards_to_go):
    ''' 
    nn_baseline: FCNN_11 object
    observations: Tensor of shape (num_trajectories * num_steps, obs_dim)
    rewards_to_go: Tensor of shape (num_trajectories * num_steps,)
    return: Tensor of shape (num_trajectories * num_steps,) the baseline values computed from nn_baseline, 
            with the same mean and standard dev as returns
    '''
    with torch.no_grad():
        # Get raw baseline predictions
        baseline_raw = nn_baseline(observations).squeeze()
        
        # Adjust to match mean and std of rewards_to_go
        rtg_mean = rewards_to_go.mean()
        rtg_std = rewards_to_go.std() + 1e-5  # Add small constant to avoid division by zero
        
        baseline_mean = baseline_raw.mean()
        baseline_std = baseline_raw.std() + 1e-5
        
        # Scale baseline to match rewards_to_go distribution
        baseline = (baseline_raw - baseline_mean) / baseline_std * rtg_std + rtg_mean
        
        return baseline

# 2.2 Computing the baseline loss
def baseline_loss_22(nn_baseline, observations, rewards_to_go):
    ''' 
    nn_baseline: FCNN_11 object
    observations: Tensor of shape (num_trajectories * num_steps, obs_dim)
    rewards_to_go: Tensor of shape (num_trajectories * num_steps,)
    return: singleton Tensor of shape ([]), the mean square error
    '''
    # Normalize rewards_to_go to have mean 0 and std 1
    rtg_mean = rewards_to_go.mean()
    rtg_std = rewards_to_go.std() + 1e-5  # Add small constant to avoid division by zero
    normalized_rtg = (rewards_to_go - rtg_mean) / rtg_std
    
    # Get baseline predictions
    baseline_predictions = nn_baseline(observations).squeeze()
    
    # Compute mean squared error
    mse = torch.mean((baseline_predictions - normalized_rtg) ** 2)
    
    return mse

# 2.3 Putting it all together: PG training with baseline
def update_baseline_parameters_23(nn_baseline, observations, rewards_to_go, optimizer):
    loss = baseline_loss_22(nn_baseline, observations, rewards_to_go)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach().item()

# Helper function to visualize policy
def visualize_policy(env_name, policy, num_episodes=10):
    env = gym.make(env_name, render_mode="human")
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = policy.sample(torch.from_numpy(obs.astype(np.float32)))
            action = action.detach().numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            env.render()
            done = terminated or truncated
    env.close()

# Hyperparameters for 1.6 (InvertedPendulum)
batch_size_16 = 1000
normalize_returns_16 = True
learning_rate_16 = 3e-3

# Hyperparameters for 2.4 (HalfCheetah)
use_baseline_24 = True
batch_size_24 = 30000
learning_rate_24 = 1e-2

# Main execution example (commented out)
if __name__ == "__main__":
    # Example usage for InvertedPendulum
    # policy = train_PG_15(
    #     env_name="InvertedPendulum-v5",
    #     hid_size=64,
    #     num_layers=2,
    #     num_iterations=100,
    #     batch_size=batch_size_16,
    #     gamma=0.99,
    #     normalize_returns=normalize_returns_16,
    #     learning_rate=learning_rate_16,
    #     seed=0
    # )
    
    # Example usage for HalfCheetah
    policy = train_PG_15(
        env_name="HalfCheetah-v5",
        hid_size=32,
        num_layers=2,
        use_baseline=use_baseline_24,
        num_iterations=100,
        batch_size=batch_size_24,
        gamma=0.95,
        normalize_returns=True,
        learning_rate=learning_rate_24,
        seed=0
    )
    
    # To visualize a trained policy:
    # visualize_policy("InvertedPendulum-v5", policy)

    
    # To load and visualize a saved policy:
    # saved_policy = GaussianPolicy_11(obs_dim, action_dim, hid_size, num_layers)
    # saved_policy.load_state_dict(torch.load('trained_cheetah_policy.pt'))
    visualize_policy("HalfCheetah-v5", policy)
    pass
