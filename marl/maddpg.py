import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset

# Assuming Actor and Critic are defined in actor.py and critic.py
try:
    from marl.actor import Actor
    from marl.critic import Critic
except ImportError:
    pass

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = map(np.stack, zip(*batch))
        return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)

class MADDPG:
    """
    MADDPG with:
    - Centralized Critic
    - Differential Privacy applied ONLY to the critic
    - RL-safe Opacus integration using dummy DataLoader
    - Supports both federated learning mode (injecting critic) and centralized regular mode
    """

    def __init__(
        self,
        critic: nn.Module = None,
        critic_optimizer: optim.Optimizer = None,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
        agent_obs_dims=None,
        agent_act_dims=None,
        global_state_dim=None
    ):
        self.device = device

        if critic is not None and critic_optimizer is not None:
            # Federated Mode Client Initialization
            self.critic = critic.to(self.device)
            self.critic_optimizer = critic_optimizer

            # Dummy DataLoader for Opacus
            dummy_x = torch.zeros((1, 1))
            dummy_y = torch.zeros((1, 1))
            dummy_dataset = TensorDataset(dummy_x, dummy_y)
            dummy_loader = DataLoader(dummy_dataset, batch_size=1)

            self.privacy_engine = PrivacyEngine()
            self.critic, self.critic_optimizer, _ = self.privacy_engine.make_private(
                module=self.critic,
                optimizer=self.critic_optimizer,
                data_loader=dummy_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
            )
        else:
            # Centralized Mode Initialization
            assert agent_obs_dims is not None
            assert agent_act_dims is not None
            assert global_state_dim is not None
            
            self.num_agents = len(agent_obs_dims)
            self.agent_obs_dims = agent_obs_dims
            self.agent_act_dims = agent_act_dims
            self.global_state_dim = global_state_dim
            
            self.actors = [Actor(obs_dim, act_dim).to(self.device) for obs_dim, act_dim in zip(agent_obs_dims, agent_act_dims)]
            # Target learning rate for "Best" performance
            # weight_decay=1e-4 prevents overfitting over the long federated training horizon
            self.actor_optimizers = [optim.Adam(actor.parameters(), lr=2e-4, weight_decay=1e-4) for actor in self.actors]
            
            # Target Actors
            self.target_actors = [Actor(obs_dim, act_dim).to(self.device) for obs_dim, act_dim in zip(agent_obs_dims, agent_act_dims)]
            for actor, target_actor in zip(self.actors, self.target_actors):
                target_actor.load_state_dict(actor.state_dict())
            
            total_act_dim = sum(agent_act_dims)
            self.critic = Critic(global_state_dim, total_act_dim).to(self.device)
            # More conservative critic learning rate for stability
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=5e-5)

            # Target Critic
            self.target_critic = Critic(global_state_dim, total_act_dim).to(self.device)
            self.target_critic.load_state_dict(self.critic.state_dict())
            
            # Hook-free Critic for safe Actor updates
            self.actor_update_critic = Critic(global_state_dim, total_act_dim).to(self.device)
            self.actor_update_critic.load_state_dict(self.critic.state_dict())
            for p in self.actor_update_critic.parameters():
                p.requires_grad = False
            
            self.replay_buffer = ReplayBuffer()

            # Apply Differential Privacy to Centralized Critic if requested
            if noise_multiplier > 0.0:
                # Use a realistic dataset size representing the ReplayBuffer so the DP sample rate q < 1.0
                dummy_x = torch.zeros((10000, 1))
                dummy_y = torch.zeros((10000, 1))
                dummy_dataset = TensorDataset(dummy_x, dummy_y)
                dummy_loader = DataLoader(dummy_dataset, batch_size=64)

                self.privacy_engine = PrivacyEngine()
                self.critic, self.critic_optimizer, _ = self.privacy_engine.make_private(
                    module=self.critic,
                    optimizer=self.critic_optimizer,
                    data_loader=dummy_loader,
                    noise_multiplier=noise_multiplier,
                    max_grad_norm=max_grad_norm,
                )
            else:
                self.privacy_engine = None

    def select_actions(self, obs_list):
        actions = []
        for i, actor in enumerate(self.actors):
            obs = obs_list[i].to(self.device)
            action = actor(obs)
            actions.append(action)
        return torch.cat(actions, dim=-1)

    def update(self, batch_size=32, gamma=0.99, tau=0.005):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        
        # -------------------
        # Centralized Critic Update
        # -------------------
        next_actions = []
        with torch.no_grad():
            for i, target_actor in enumerate(self.target_actors):
                next_actions.append(target_actor(next_states))
                
            next_actions = torch.cat(next_actions, dim=-1)
            target_q = self.target_critic(next_states, next_actions)
            expected_q = rewards + gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, expected_q.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping for Critic to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # -------------------
        # Actor Update
        # -------------------
        # Update the pristine clone with the latest Critic weights
        if hasattr(self, 'privacy_engine') and self.privacy_engine is not None:
            self.actor_update_critic.load_state_dict(self.critic._module.state_dict())
        else:
            self.actor_update_critic.load_state_dict(self.critic.state_dict())

        for i, actor in enumerate(self.actors):
            curr_actions = []
            for j, act in enumerate(self.actors):
                if i == j:
                    curr_actions.append(actor(states))
                else:
                    curr_actions.append(act(states).detach())
            
            curr_actions = torch.cat(curr_actions, dim=-1)
            
            # Maximize Q-value = Minimize negative Q-value
            # Route through the hook-free Critic clone to guarantee Opacus isolation
            actor_loss = -self.actor_update_critic(states, curr_actions).mean()
            
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            # Gradient clipping for Actor
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            self.actor_optimizers[i].step()

        # -------------------
        # Soft Update of Target Networks
        # -------------------
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
        for i in range(self.num_agents):
            for target_param, param in zip(self.target_actors[i].parameters(), self.actors[i].parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def update_critic(self, loss: torch.Tensor):
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def get_privacy_budget(self, delta: float = 1e-5) -> float:
        if hasattr(self, 'privacy_engine') and self.privacy_engine is not None:
            try:
                if len(self.privacy_engine.accountant.history) > 0:
                    return self.privacy_engine.get_epsilon(delta)
            except Exception:
                return -1.0
        return 0.0
