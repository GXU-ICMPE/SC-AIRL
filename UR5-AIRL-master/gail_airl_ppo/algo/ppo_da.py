import torch
from torch import nn
from torch.optim import Adam
import os
import numpy as np
import math
from .base import Algorithm
from gail_airl_ppo.buffer import RolloutBuffer, RolloutBuffer1, RolloutBuffer2
from gail_airl_ppo.network import StateIndependentPolicy, StateFunction, StateIndependentPolicy1, StateDependentPolicy
from gail_airl_ppo.utils import disable_gradient


def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)

    # Calculate gae recursively from behind.loss_critic = (self.critic(states) - targets).pow_(2).mean()
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


class PPO(Algorithm):

    def __init__(self, state_shape,  action_shape, device, seed, path0, pathc, gamma=0.99,
                 rollout_length=2048, mix_buffer=20, lr_actor=1e-4,
                 lr_critic=1e-4, units_actor=(64, 64), units_critic=(64, 64),
                 epoch_ppo=50, clip_eps=0.2, lambd=0.97, coef_ent=0.0,
                 max_grad_norm=10.0):
        super().__init__(state_shape, action_shape, device, seed, path0, pathc, gamma)

        # Rollout buffer.
        self.rollout_length = rollout_length


        self.buffer0 = RolloutBuffer1(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer
        )

        self.buffer1 = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer
        )

        #change here to train different task
        self.update_0 = 1
        self.update_1 = 1

        # Actor.
        if self.update_0:
            self.actor0 = StateIndependentPolicy1(
                state_shape=state_shape,
                action_shape=action_shape,
                hidden_units=units_actor,
                hidden_activation=nn.Tanh()
            ).to(device)
        else:
            self.actor0 = StateIndependentPolicy(
                state_shape=state_shape,
                action_shape=action_shape,
                hidden_units=units_actor,
                hidden_activation=nn.Tanh()
            ).to(device)
            self.actor0.load_state_dict(torch.load(path0))

            disable_gradient(self.actor0)


        if self.update_1:
            self.actor1 = StateIndependentPolicy(
                state_shape=state_shape,
                action_shape=action_shape,
                hidden_units=units_actor,
                hidden_activation=nn.Tanh()
            ).to(device)
        else:
            self.actor1 = StateIndependentPolicy(
                state_shape=state_shape,
                action_shape=action_shape,
                hidden_units=units_actor,
                hidden_activation=nn.Tanh()
            ).to(device)
            self.actor1.load_state_dict(torch.load(path0))

            disable_gradient(self.actor1)


        # 这里输出的就是action对应的mean


        # Critic.
        if self.update_0:
            self.critic = StateFunction(
                state_shape=state_shape,
                hidden_units=units_critic,
                hidden_activation=nn.Tanh()
            ).to(device)
        else:
            self.critic = StateFunction(
                state_shape=state_shape,
                hidden_units=units_critic,
                hidden_activation=nn.Tanh()
            ).to(device)

            self.critic.load_state_dict(torch.load(pathc))


        self.optim_actor0 = Adam(self.actor0.parameters(), lr=lr_actor, eps=1e-5)
        self.optim_actor1 = Adam(self.actor1.parameters(), lr=lr_actor, eps=1e-5)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)


        self.learning_steps_ppo = 0
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.step1 = 0
        self.step1_1 = 1


    def is_update(self, step):
        return step % self.rollout_length == 0

    def is_update_0(self):
        return self.update_0

    def is_update_1(self):
        return self.update_1

    def step(self, env, state, t, step):
        t += 1

        info = env.infomation()

        self.is_update_0()
        self.is_update_1()

        if info == 0:
            if not self.update_0:
                action = self.exploit0(state)
                next_state, reward, done, _ = env.step(action)
            else:
                action, log_pi = self.explore0(state)
                next_state, reward, done, _ = env.step(action)
                G = env.G_info()
                self.buffer0.append(state, action, reward, done, log_pi, next_state)
        else:
            if not self.update_1:
                action = self.exploit1(state)
                next_state, reward, done, _ = env.step(action)
            else:
                action, log_pi = self.explore1(state)
                next_state, reward, done, _ = env.step(action)
                self.buffer1.append(state, action, reward, done, log_pi, next_state)

        step_counter = env.step_n()

        if done and info == 0 and G == 1:
            done = False

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self, writer):
        self.learning_steps += 1
        states0, actions0, rewards0, dones0, log_pis0, next_states0 = \
            self.buffer0.get()

        self.update_ppo0(
            states0, actions0, rewards0, dones0, log_pis0, next_states0, writer)

        if self.buffer1._n >= self.rollout_length - 1:
            states1, actions1, rewards1, dones1, log_pis1, next_states1 = \
                self.buffer1.get()
            self.update_ppo1(
                states1, actions1, rewards1, dones1, log_pis1, next_states1, writer)

    def update_ppo0(self, states, actions, rewards, dones, log_pis, next_states,
                   writer):
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(states, targets, writer)
            self.update_actor(states, actions, log_pis, gaes, rewards, writer)

    def update_ppo1(self, states, actions, rewards, dones, log_pis, next_states,
                   writer):
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(states, targets, writer)
            self.update_actor1(states, actions, log_pis, gaes, writer)

    def update_critic(self, states, targets, writer):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/critic', loss_critic.item(), self.learning_steps)

    def update_actor(self, states, actions, log_pis_old, gaes, rewards, writer):
        log_pis = self.actor0.evaluate_log_pi(states, actions)
        entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor0.zero_grad()
        (loss_actor - self.coef_ent * entropy).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor0.parameters(), self.max_grad_norm)
        self.optim_actor0.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/actor', loss_actor.item(), self.learning_steps)
            writer.add_scalar(
                'stats/entropy', entropy.item(), self.learning_steps)

    def update_actor1(self, states, actions, log_pis_old, gaes, writer):
        log_pis = self.actor1.evaluate_log_pi(states, actions)
        entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor1 = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor1.zero_grad()
        (loss_actor1 - self.coef_ent * entropy).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor1.parameters(), self.max_grad_norm)
        self.optim_actor1.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/actor1', loss_actor1.item(), self.learning_steps)
            writer.add_scalar(
                'stats/entropy', entropy.item(), self.learning_steps)


    def save_models(self, save_dir):
        super().save_models(save_dir)
        # We only save actor to reduce workloads.
        torch.save(
            self.actor0.state_dict(),
            os.path.join(save_dir, 'actor.pth')
        )
        torch.save(
            self.actor1.state_dict(),
            os.path.join(save_dir, 'actor1.pth')
        )


class PPOExpert(PPO):

    def __init__(self, state_shape, action_shape, device, path0, path1,
                 units_actor=(64, 64)):
        self.actor0 = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh()
        ).to(device)
        self.actor0.load_state_dict(torch.load(path0))

        disable_gradient(self.actor0)
        self.device = device

        self.actor1 = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh()
        ).to(device)
        self.actor1.load_state_dict(torch.load(path1))

        disable_gradient(self.actor1)
        self.device = device

