import torch
from torch import nn
import torch.nn.functional as F

from .utils import build_mlp


class GAILDiscrim(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(100, 100),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states, actions):
        return self.net(torch.cat([states, actions], dim=-1))

    def calculate_reward(self, states, actions):
        # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D)].
        with torch.no_grad():
            return -F.logsigmoid(-self.forward(states, actions))


class AIRLDiscrim(nn.Module):

    def __init__(self, state_shape, gamma,
                 hidden_units_r=(64, 64),
                 hidden_units_v=(64, 64),
                 hidden_activation_r=nn.Tanh(),
                 hidden_activation_v=nn.Tanh()):
        super().__init__()

        self.g = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units_r,
            hidden_activation=hidden_activation_r
        )
        self.h = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units_v,
            hidden_activation=hidden_activation_v
        )

        self.gamma = gamma

    def f(self, states, dones, next_states):
        rs = self.g(states)
        vs = self.h(states)
        next_vs = self.h(next_states)
        return rs + (1 - dones) * (self.gamma * next_vs - vs)

    def get_d(self, states,  dones, log_pis, next_states):
        # Discriminator's output is sigmoid(f - log_pi).
        exp_f = torch.exp(self.f(states, dones, next_states))
        return (exp_f/(exp_f + torch.exp(log_pis)))

    def forward(self, states, dones, log_pis, next_states):
        d = self.get_d(states,  dones, log_pis, next_states)
        return d

    def calculate_reward(self, states, dones, log_pis, next_states):

        d = self.get_d(states,  dones, log_pis, next_states).detach()
        return (torch.log(d + 1e-3) - torch.log((1-d)+1e-3))
#
#
#         # with torch.no_grad():
#         #     rs = self.g(states)
#         #     return rs
# class AIRLDiscrim(nn.Module):
#
#     def __init__(self, state_shape, gamma,
#                  hidden_units_r=(64, 64),
#                  hidden_units_v=(64, 64),
#                  hidden_activation_r=nn.ReLU(inplace=True),
#                  hidden_activation_v=nn.ReLU(inplace=True)):
#         super().__init__()
#
#         self.g = build_mlp(
#             input_dim=state_shape[0],
#             output_dim=1,
#             hidden_units=hidden_units_r,
#             hidden_activation=hidden_activation_r
#         )
#         self.h = build_mlp(
#             input_dim=state_shape[0],
#             output_dim=1,
#             hidden_units=hidden_units_v,
#             hidden_activation=hidden_activation_v
#         )
#
#         self.gamma = gamma
#
#     def f(self, states, dones, next_states):
#         rs = self.g(states)
#         vs = self.h(states)
#         next_vs = self.h(next_states)
#         return rs + (1 - dones) * self.gamma * next_vs - vs
#
#     def forward(self, states, dones, log_pis, next_states):
#         # Discriminator's output is sigmoid(f - log_pi).
#         return self.f(states, dones, next_states) - log_pis
#
#     def calculate_reward(self, states, dones, log_pis, next_states):
#         with torch.no_grad():
#             logits = self.forward(states, dones, log_pis, next_states)
#             return -F.logsigmoid(-logits)

# class AIRLDiscrim_sa(nn.Module):
#
#     def __init__(self, state_shape, action_shape, gamma,
#                  hidden_units_r=(64, 64),
#                  hidden_units_v=(64, 64),
#                  hidden_activation_r=nn.ReLU(inplace=True),
#                  hidden_activation_v=nn.ReLU(inplace=True)):
#         super().__init__()
#
#         self.g = build_mlp(
#             input_dim=state_shape[0] + action_shape[0],
#             output_dim=1,
#             hidden_units=hidden_units_r,
#             hidden_activation=hidden_activation_r
#         )
#         self.h = build_mlp(
#             input_dim=state_shape[0],
#             output_dim=1,
#             hidden_units=hidden_units_v,
#             hidden_activation=hidden_activation_v
#         )
#
#         self.gamma = gamma
#
#     def f(self, states, actions, dones, next_states):
#         x = torch.cat([states, actions], -1)
#         rs = self.g(x)
#         vs = self.h(states)
#         next_vs = self.h(next_states)
#         return rs + (1 - dones) * (self.gamma * next_vs - vs)
#
#     def get_d(self, states, actions, dones, log_pis, next_states):
#         # Discriminator's output is sigmoid(f - log_pi).
#         exp_f = torch.exp(self.f(states, actions, dones, next_states))
#         return (exp_f/(exp_f + torch.exp(log_pis)))
#
#     def forward(self, states, actions, dones, log_pis, next_states):
#         d = self.get_d(states,  actions, dones, log_pis, next_states)
#         return d
#
#     def calculate_reward(self, states, actions, dones, log_pis, next_states):
#
#         d = self.get_d(states,  actions, dones, log_pis, next_states).detach()
#         return (torch.log(d + 1e-3) - torch.log((1-d)+1e-3))


        # with torch.no_grad():
        #     rs = self.g(states)
        #     return rs