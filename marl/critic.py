import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """
    Centralized critic.
    Input: global state + all agent actions
    Output: Q-value
    """

    def __init__(self, state_dim, total_act_dim, hidden_dim1=256, hidden_dim2=256, hidden_dim3=128):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + total_act_dim, hidden_dim1)
        self.ln1 = nn.LayerNorm(hidden_dim1)
        
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, 1)

    def forward(self, state, actions):
        """
        state: (batch, state_dim)
        actions: (batch, total_act_dim)
        """
        x = torch.cat([state, actions], dim=-1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q = self.fc4(x)
        return q
