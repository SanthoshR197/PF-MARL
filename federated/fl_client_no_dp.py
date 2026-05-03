import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import os
import sys

# Add the project root directory to sys.path before project imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.urban_env import UrbanEpidemicEnv
from marl.maddpg import MADDPG

# =========================================================
# METRICS STORAGE
# =========================================================
infection_history = []
reward_history = []
economic_cost_history = []

def log_metrics(infected, reward, economic_cost):
    infection_history.append(infected)
    reward_history.append(reward)
    economic_cost_history.append(economic_cost)

def save_metrics_to_csv(client_id="no_dp"):
    os.makedirs("results", exist_ok=True)
    filename = f"results/fl_client_{client_id}_metrics.csv"
    
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["infected", "reward", "economic_cost"])
        for i in range(len(infection_history)):
            writer.writerow([
                infection_history[i],
                reward_history[i],
                economic_cost_history[i],
            ])

# =========================================================
# FEDERATED CLIENT WRAPPER for MADDPG (NO DP BASELINE)
# =========================================================
class FederatedMADDPGClientNoDP(fl.client.NumPyClient):

    def __init__(self, num_episodes_per_round=50, max_steps=20):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_episodes_per_round = num_episodes_per_round
        self.max_steps = max_steps
        
        # Instantiate real environment
        self.env = UrbanEpidemicEnv(num_districts=6)
        
        # Action space = 2x districts (Lockdowns + Health Measures)
        agent_obs_dims = [self.env.features_per_node * self.env.num_districts] * 4
        agent_act_dims = [self.env.num_districts * 2] * 4
        global_state_dim = self.env.features_per_node * self.env.num_districts

        # Create actual MADDPG Brain WITHOUT DP
        self.maddpg = MADDPG(
            agent_obs_dims=agent_obs_dims,
            agent_act_dims=agent_act_dims,
            global_state_dim=global_state_dim,
            noise_multiplier=0.0,  # CRITICAL CHANGE: 0.0 Noise = No DP
            max_grad_norm=1.0
        )
        
        self.noise_std = 0.5

    # -----------------------------------------------------
    def get_parameters(self, config=None):
        return [p.detach().cpu().numpy() for p in self.maddpg.critic.parameters()]

    def set_parameters(self, parameters):
        for p, new_p in zip(self.maddpg.critic.parameters(), parameters):
            p.data = torch.tensor(new_p, device=self.device)
            for target_p, source_p in zip(self.maddpg.target_critic.parameters(), self.maddpg.critic.parameters()):
                target_p.data.copy_(source_p.data)

    # -----------------------------------------------------
    def fit(self, parameters, config):
        print("\n[FL Client NO-DP Baseline] Received updated Critic from Server. Beginning Local Training...")
        self.set_parameters(parameters)

        for ep in range(self.num_episodes_per_round):
            state, _ = self.env.reset()
            episode_reward = 0
            ep_infections = 0
            ep_economic = 0
            
            for step in range(self.max_steps):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                obs_list = [state_tensor] * 4
                
                actions = self.maddpg.select_actions(obs_list)
                actions_np = actions.detach().numpy().reshape(4, self.env.num_districts * 2)
                
                noise = np.random.normal(0, self.noise_std, size=actions_np.shape)
                noisy_actions = np.clip(actions_np + noise, 0.0, 1.0)
                final_action = noisy_actions.mean(axis=0)

                next_state, reward, terminated, truncated, info = self.env.step(final_action)

                self.maddpg.replay_buffer.push(
                    state,
                    noisy_actions.flatten(),
                    reward,
                    next_state
                )

                state = next_state
                episode_reward += reward
                ep_infections += info.get('infections', 0)
                ep_economic += info.get('economic_cost', 0)
                
                if len(self.maddpg.replay_buffer) > 32:
                    self.maddpg.update(batch_size=32)

                if terminated or truncated:
                    break
                    
            self.noise_std = max(0.01, self.noise_std * 0.999)
            
            log_metrics(ep_infections, episode_reward, ep_economic)
            if (ep + 1) % 10 == 0:
                 print(f"   Local Ep {ep+1}/{self.num_episodes_per_round} | Reward: {episode_reward:.2f} | Infected: {int(ep_infections)}")

        save_metrics_to_csv()
        
        # Save Model Checkpoints
        os.makedirs("models", exist_ok=True)
        for idx, actor in enumerate(self.maddpg.actors):
            torch.save(actor.state_dict(), f"models/baselines_nodp_actor_{idx}.pth")
        torch.save(self.maddpg.critic.state_dict(), "models/baselines_nodp_critic.pth")
        
        print(f"[FL Client NO-DP Baseline] Local Training complete. Models saved. Sending learned Critic back to Server.\n")
        return self.get_parameters(), self.num_episodes_per_round, {}

    def evaluate(self, parameters, config):
        return 0.0, 0, {}

# =========================================================
# CLIENT START
# =========================================================
def main():
    print("Starting NON-PRIVATE PFMARL Client (Baseline)...")
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8081",
        client=FederatedMADDPGClientNoDP(num_episodes_per_round=50),
    )

if __name__ == "__main__":
    main()
