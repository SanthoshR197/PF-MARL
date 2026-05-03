import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Actor network for a single agent.
    Maps agent-specific observation -> action.
    """

    def __init__(self, obs_dim, act_dim, hidden_dim1=256, hidden_dim2=256, hidden_dim3=128):
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim1)
        self.ln1 = nn.LayerNorm(hidden_dim1)
        
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, act_dim)

    def forward(self, obs):
        x = F.relu(self.ln1(self.fc1(obs)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # Action bounded in [0, 1]
        action = torch.sigmoid(self.fc4(x))
        return action
