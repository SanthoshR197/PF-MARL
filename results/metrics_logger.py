# results/metrics_logger.py

infection_history = []
reward_history = []
economic_cost_history = []


def log_metrics(num_infected, reward, economic_cost):
    """
    Called during RL training (inside fit())
    """
    infection_history.append(num_infected)
    reward_history.append(reward)
    economic_cost_history.append(economic_cost)


def get_metrics():
    """
    Used by plot_results.py
    """
    return infection_history, reward_history, economic_cost_history