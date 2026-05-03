import numpy as np
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.urban_env import UrbanEpidemicEnv
from marl.maddpg import MADDPG

def run_baseline(env, strategy="total_lockdown", steps=20):
    state, _ = env.reset()
    total_infected = 0
    total_economic_cost = 0
    peak_infections = 0
    
    for _ in range(steps):
        if strategy == "total_lockdown":
            # Maximum lockdown (1.0) and health measures (1.0)
            actions = np.ones((4, env.num_districts * 2))
        elif strategy == "no_lockdown":
            # Zero lockdown (0.0) and zero health measures (0.0)
            actions = np.zeros((4, env.num_districts * 2))
        elif strategy == "random":
            actions = np.random.rand(4, env.num_districts * 2)
            
        final_action = actions.mean(axis=0)
        next_state, reward, _, _, info = env.step(final_action)
        
        infections_step = info.get('infections', 0)
        total_infected += infections_step
        peak_infections = max(peak_infections, infections_step)
        total_economic_cost += info.get('economic_cost', 0)
        
        state = next_state
        
    # Economic activity is inversely proportional to cost. 
    # If max cost per step is 1.0 per district per agent, max total cost = steps * districts
    max_possible_cost = steps * env.num_districts
    economic_activity = max_possible_cost - total_economic_cost
    
    return total_infected, peak_infections, total_economic_cost, economic_activity

def evaluate_maddpg(env, maddpg, steps=20):
    state, _ = env.reset()
    total_infected = 0
    total_economic_cost = 0
    peak_infections = 0
    
    for _ in range(steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            obs_list = [state_tensor] * maddpg.num_agents
            actions = maddpg.select_actions(obs_list)
            
        actions_np = actions.detach().numpy().reshape(4, env.num_districts * 2)
        final_action = actions_np.mean(axis=0)
        
        next_state, reward, _, _, info = env.step(final_action)
        
        infections_step = info.get('infections', 0)
        total_infected += infections_step
        peak_infections = max(peak_infections, infections_step)
        total_economic_cost += info.get('economic_cost', 0)
        
        state = next_state
        
    max_possible_cost = steps * env.num_districts
    economic_activity = max_possible_cost - total_economic_cost
        
    return total_infected, peak_infections, total_economic_cost, economic_activity

def main():
    print("--- BENCHMARK EVALUATION ---")
    env = UrbanEpidemicEnv(num_districts=6)
    
    agent_obs_dims = [env.features_per_node * env.num_districts] * 4
    agent_act_dims = [env.num_districts * 2] * 4
    global_state_dim = env.features_per_node * env.num_districts

    maddpg = MADDPG(
        agent_obs_dims=agent_obs_dims,
        agent_act_dims=agent_act_dims,
        global_state_dim=global_state_dim
    )
    
    # Load best weights
    models_exist = True
    for i in range(4):
        path = f"models/best_actor_{i}.pth"
        if os.path.exists(path):
            maddpg.actors[i].load_state_dict(torch.load(path, weights_only=True))
            maddpg.actors[i].eval()
        else:
            models_exist = False
            
    if not models_exist:
        print("Warning: Could not find 'best_actor_X.pth' models. The MADDPG evaluation will be untrained.")

    num_trials = 10
    
    results = {
        "total_lockdown": {"peak_inf": [], "econ_act": []},
        "no_lockdown": {"peak_inf": [], "econ_act": []},
        "random": {"peak_inf": [], "econ_act": []},
        "maddpg": {"peak_inf": [], "econ_act": []}
    }
    
    for _ in range(num_trials):
        _, peak, _, econ = run_baseline(env, "total_lockdown")
        results["total_lockdown"]["peak_inf"].append(peak)
        results["total_lockdown"]["econ_act"].append(econ)
        
        _, peak, _, econ = run_baseline(env, "no_lockdown")
        results["no_lockdown"]["peak_inf"].append(peak)
        results["no_lockdown"]["econ_act"].append(econ)
        
        _, peak, _, econ = run_baseline(env, "random")
        results["random"]["peak_inf"].append(peak)
        results["random"]["econ_act"].append(econ)
        
        _, peak, _, econ = evaluate_maddpg(env, maddpg)
        results["maddpg"]["peak_inf"].append(peak)
        results["maddpg"]["econ_act"].append(econ)

    for k in results:
        results[k]["peak_inf"] = np.mean(results[k]["peak_inf"])
        results[k]["econ_act"] = np.mean(results[k]["econ_act"])
        
    print(f"\n[1] Infection Control Benchmark (Target: >30% reduction vs non-cooperative/no-lockdown)")
    print(f"No Lockdown Peak Infections: {results['no_lockdown']['peak_inf']:.1f}")
    print(f"Random (Non-coop) Peak Infections: {results['random']['peak_inf']:.1f}")
    print(f"MADDPG Peak Infections: {results['maddpg']['peak_inf']:.1f}")
    
    red_from_no = (1 - results['maddpg']['peak_inf'] / results['no_lockdown']['peak_inf']) * 100
    red_from_rand = (1 - results['maddpg']['peak_inf'] / results['random']['peak_inf']) * 100
    print(f"-> Reduction vs No Lockdown: {red_from_no:.1f}%")
    print(f"-> Reduction vs Random: {red_from_rand:.1f}%")
    
    print(f"\n[2] Economic Stability Benchmark (Target: >50% higher activity vs Total Lockdown)")
    print(f"Total Lockdown Economic Activity: {results['total_lockdown']['econ_act']:.1f}")
    print(f"MADDPG Economic Activity: {results['maddpg']['econ_act']:.1f}")
    
    inc_from_lockdown = (results['maddpg']['econ_act'] / results['total_lockdown']['econ_act'] - 1) * 100
    print(f"-> Activity Increase vs Total Lockdown: {inc_from_lockdown:.1f}%")

if __name__ == "__main__":
    main()
