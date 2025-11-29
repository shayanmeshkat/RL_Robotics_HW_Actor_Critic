import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import pandas as pd
import os

Gamma = 0.99
learning_rate = 0.002
hidden_size = 128
max_episodes = 1200
max_steps = 500
env_name = 'CartPole-v1'
test_interval = 200 
num_test_episodes = 3  
num_runs = 3 


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_actor = nn.Linear(hidden_size, action_dim)
        self.fc_critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc_actor(x), dim=-1)
        state_value = self.fc_critic(x)

        return action_probs, state_value
    

env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n


model = ActorCritic(state_dim, action_dim, hidden_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

episode_rewards = []

for episode in range(max_episodes):
    state, _ = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0)
    episode_reward = 0
    log_probs = []
    values = []
    rewards = []

    for step in range(max_steps):
        action_probs, state_value = model(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        next_state = torch.FloatTensor(next_state)

        done = terminated or truncated
        
        episode_reward += reward
        log_probs.append(log_prob)
        values.append(state_value)
        rewards.append(reward)

        state = next_state

        if done:
            break

    episode_rewards.append(episode_reward)

    returns = []
    R = 0

    for r in reversed(rewards):
        R = r + Gamma * R
        returns.insert(0, R)
    
    returns = torch.FloatTensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-5)

    actor_loss = 0
    critic_loss = 0
    for log_prob, value, R in zip(log_probs, values, returns):
        advantage = R - value.item()
        actor_loss += -log_prob * advantage
        critic_loss += F.mse_loss(value.squeeze(), torch.tensor([R]))

    actor_loss /= len(log_probs)
    critic_loss /= len(values)

    optimizer.zero_grad()
    total_loss = actor_loss + critic_loss
    total_loss.backward()
    optimizer.step()

    
        


    if episode % 50 == 0:
        avg_reward = np.mean(episode_rewards[-50:])
        print(f"Episode {episode+1}/{max_episodes}, Reward: {episode_reward:.2f}")
        print(f"Average Reward over last 50 episodes: {avg_reward:.2f}")


def test_policy(model, env_name, num_episodes=5, render=True):
    """Test the policy and optionally render the environment."""
    if render:
        test_env = gym.make(env_name, render_mode='human')
    else:
        test_env = gym.make(env_name)
    
    test_rewards = []
    
    for episode in range(num_episodes):
        state, _ = test_env.reset(seed=10)
        state = torch.FloatTensor(state).unsqueeze(0)
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                action_probs, _ = model(state)
                action = torch.argmax(action_probs, dim=-1)
            
            next_state, reward, terminated, truncated, _ = test_env.step(action.item())
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            episode_reward += reward
            state = next_state
            done = terminated or truncated
        
        test_rewards.append(episode_reward)
    
    test_env.close()
    return test_rewards 


# Store results from all runs
all_runs_rewards = {}
final_test_rewards = {}
all_test_rewards = {}  

for run in range(num_runs):
    print(f"\n{'='*50}")
    print(f"Starting Run {run + 1}/{num_runs}")
    print(f"{'='*50}\n")
    
    # Set seeds for reproducibility
    seed = run
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = gym.make(env_name)
    env.reset(seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(state_dim, action_dim, hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    episode_rewards = []
    test_rewards_history = []  

    for episode in range(max_episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        episode_reward = 0
        log_probs = []
        values = []
        rewards = []

        for step in range(max_steps):
            action_probs, state_value = model(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            next_state = torch.FloatTensor(next_state).unsqueeze(0)

            done = terminated or truncated
            
            episode_reward += reward
            log_probs.append(log_prob)
            values.append(state_value)
            rewards.append(reward)

            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)

        returns = []
        R = 0

        for r in reversed(rewards):
            R = r + Gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        actor_loss = 0
        critic_loss = 0
        for log_prob, value, R in zip(log_probs, values, returns):
            advantage = R - value.item()
            actor_loss += -log_prob * advantage
            critic_loss += F.mse_loss(value.squeeze(), torch.tensor([R]))

        actor_loss /= len(log_probs)
        critic_loss /= len(values)

        optimizer.zero_grad()
        total_loss = actor_loss + critic_loss
        total_loss.backward()
        optimizer.step()

        # Test policy at intervals
        if (episode + 1) % test_interval == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Run {run + 1}, Episode {episode+1}/{max_episodes}, Reward: {episode_reward:.2f}")
            print(f"Average Reward over last 50 episodes: {avg_reward:.2f}")
            
            # Test with visualization 
            if run == 0:
                print("Testing policy with visualization...")
                test_rewards_list = test_policy(model, env_name, num_test_episodes, render=False)
                print(f"Average Test Reward: {np.mean(test_rewards_list):.2f}\n")
            else:
                test_rewards_list = test_policy(model, env_name, num_test_episodes, render=False)
                print(f"Average Test Reward: {np.mean(test_rewards_list):.2f}\n")
            
           
            test_rewards_history.extend(test_rewards_list)

    # Final test after training completes
    print(f"Running final test for Run {run + 1}...")
    final_test = test_policy(model, env_name, num_test_episodes, render=False)
    final_test_rewards[f'Run_{run + 1}'] = final_test
    print(f"Final test rewards for Run {run + 1}: {final_test}")
    
    # final test to hisory
    test_rewards_history.extend(final_test)
    all_test_rewards[f'Run_{run + 1}'] = test_rewards_history
    
    env.close()
    
    # Store rewards for this run
    all_runs_rewards[f'Run_{run + 1}'] = episode_rewards
    
    print(f"Run {run + 1} completed. Final average reward: {np.mean(episode_rewards[-50:]):.2f}")

# Save all runs to CSV
output_dir = './Results'
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, 'episode_rewards.csv')

df = pd.DataFrame(all_runs_rewards)
df.to_csv(csv_path, index=False)
print(f"\nEpisode rewards saved to: {csv_path}")

# Save all test rewards to CSV
all_test_csv_path = os.path.join(output_dir, 'all_test_rewards.csv')
df_all_test = pd.DataFrame(all_test_rewards)
df_all_test.to_csv(all_test_csv_path, index=False)
print(f"All test rewards (from all intervals) saved to: {all_test_csv_path}")

# Save final test rewards to CSV 
test_csv_path = os.path.join(output_dir, 'final_test_rewards.csv')
df_test = pd.DataFrame(final_test_rewards)
df_test.to_csv(test_csv_path, index=False)
print(f"Final test rewards saved to: {test_csv_path}")

# Plot all runs
plt.figure(figsize=(12, 6))
for run_name, rewards in all_runs_rewards.items():
    plt.plot(rewards, alpha=0.6, label=run_name)

# Calculate and plot mean across runs
all_rewards_array = np.array([all_runs_rewards[f'Run_{i+1}'] for i in range(num_runs)])
mean_rewards = np.mean(all_rewards_array, axis=0)
plt.plot(mean_rewards, 'k-', linewidth=2, label='Mean')

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title(f'Episode Rewards over Time ({num_runs} runs)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(output_dir, 'episode_rewards_plot.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")
plt.show()