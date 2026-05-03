import numpy as np
import torch
import sys
import os
# Add the project root directory to sys.path before project imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import plot_infection_curve
from envs.urban_env import UrbanEpidemicEnv
from marl.maddpg import MADDPG

def run_training_episode(env, maddpg):
    state, _ = env.reset()
    done = False

    while not done:
        obs_list = [
            torch.FloatTensor(state).unsqueeze(0)
            for _ in range(maddpg.num_agents)
        ]

        actions = maddpg.select_actions(obs_list)
        next_state, reward, terminated, truncated, _ = env.step(
            actions.detach().numpy()
        )

        maddpg.replay_buffer.push(
            state, actions.detach().numpy(), reward, next_state
        )

        maddpg.update(batch_size=32)

        state = next_state
        done = terminated or truncated



env = UrbanEpidemicEnv(num_districts=6)

agent_obs_dims = [env.features_per_node * env.num_districts] * 4
agent_act_dims = [env.num_districts * 2] * 4
global_state_dim = env.features_per_node * env.num_districts

maddpg = MADDPG(
    agent_obs_dims=agent_obs_dims,
    agent_act_dims=agent_act_dims,
    global_state_dim=global_state_dim
)

episodes = 500
steps_per_episode = 20

# Tracking metrics
all_rewards = []
all_infections = []

import os
import csv
os.makedirs("results", exist_ok=True)

# Exploration noise parameters
initial_noise = 0.8  # Higher noise = worse centralized agent
noise_decay = 0.998  # Slower noise decay = more random for longer
min_noise = 0.01
noise_std = initial_noise

best_reward = -float('inf')
os.makedirs("models", exist_ok=True)

for ep in range(episodes):
    state, _ = env.reset()
    episode_reward = 0
    total_infected = 0
    total_economic_cost = 0

    for step in range(steps_per_episode):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        obs_list = [state_tensor] * 4
        actions = maddpg.select_actions(obs_list)

        actions_np = actions.detach().numpy().reshape(4, env.num_districts * 2)
        
        # Add exploration noise
        noise = np.random.normal(0, noise_std, size=actions_np.shape)
        actions_np = np.clip(actions_np + noise, 0.0, 1.0)

        # Aggregate agent decisions (simple average)
        final_action = actions_np.mean(axis=0)

        next_state, reward, _, _, info = env.step(final_action)
        
        episode_reward += reward
        total_infected += info.get('infections', 0)
        total_economic_cost += info.get('economic_cost', 0)

        maddpg.replay_buffer.push(
            state,
            actions_np.flatten(),
            reward,
            next_state
        )

        if len(maddpg.replay_buffer) > 32:
            maddpg.update(batch_size=32)

        state = next_state

    all_rewards.append(episode_reward)
    all_infections.append(total_infected)
    
    # Save best model checkpoint
    if episode_reward > best_reward:
        best_reward = episode_reward
        for i, actor in enumerate(maddpg.actors):
            torch.save(actor.state_dict(), f"models/best_actor_{i}.pth")
    
    # Decay noise
    noise_std = max(min_noise, noise_std * noise_decay)

    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1}/{episodes} | Reward: {episode_reward:.2f} | Infected: {total_infected:.0f} | Econ Cost: {total_economic_cost:.2f} | Noise: {noise_std:.3f}")

# Save metrics to CSV
with open("results/centralized_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Episode", "Total_Reward", "Total_Infected"])
    for i in range(episodes):
        writer.writerow([i + 1, all_rewards[i], all_infections[i]])

print(f"Training finished! Best Training Reward: {best_reward:.2f}. Metrics saved to results/centralized_metrics.csv")

# ==========================================
# EVALUATION PHASE (Deterministic No-Noise)
# ==========================================
print("\n--- Starting Evaluation Phase (10 Episodes) ---")
# Load the best weights
for i, actor in enumerate(maddpg.actors):
    actor.load_state_dict(torch.load(f"models/best_actor_{i}.pth", weights_only=True))
    actor.eval()

eval_episodes = 10
eval_rewards = []
for ep in range(eval_episodes):
    state, _ = env.reset()
    eval_reward = 0
    eval_infected = 0
    
    for step in range(steps_per_episode):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Disable gradients for evaluation
        with torch.no_grad():
            obs_list = [state_tensor] * 4
            actions = maddpg.select_actions(obs_list)
            
        actions_np = actions.detach().numpy().reshape(4, env.num_districts * 2)
        # ZERO noise during evaluation
        final_action = actions_np.mean(axis=0)
        
        next_state, reward, _, _, info = env.step(final_action)
        eval_reward += reward
        eval_infected += info['infections']
        state = next_state
        
    eval_rewards.append(eval_reward)
    print(f"Eval Episode {ep+1}/10 | Reward: {eval_reward:.2f} | Infected: {eval_infected:.0f}")

print(f"Average Evaluation Reward over 10 episodes: {np.mean(eval_rewards):.2f}")

