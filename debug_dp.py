import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from envs.urban_env import UrbanEpidemicEnv
from marl.maddpg import MADDPG

env = UrbanEpidemicEnv(num_districts=6)
agent_obs_dims = [env.features_per_node * env.num_districts] * 4
agent_act_dims = [env.num_districts * 2] * 4
global_state_dim = env.features_per_node * env.num_districts

maddpg = MADDPG(
    agent_obs_dims=agent_obs_dims,
    agent_act_dims=agent_act_dims,
    global_state_dim=global_state_dim,
    noise_multiplier=1.0,  
    max_grad_norm=1.0      
)

print("Starting DP test...")

for i in range(70):
    state = np.random.randn(global_state_dim)
    actions = np.random.rand(env.num_districts * 2 * 4)
    reward = np.random.rand()
    next_state = np.random.randn(global_state_dim)
    
    maddpg.replay_buffer.push(state, actions, reward, next_state)

print("Buffer populated. Calling updates...")
for step in range(200):
    maddpg.update(batch_size=64)
    if step % 20 == 0:
        print(f"Update {step} complete.")

for name, param in maddpg.critic.named_parameters():
    print(f"Critic {name} Has NaN: {torch.isnan(param).any().item()}")

for i, actor in enumerate(maddpg.actors):
    for name, param in actor.named_parameters():
        print(f"Actor {i} {name} Has NaN: {torch.isnan(param).any().item()}")

print("Done.")
