import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import os

from .ppo_da import PPO
from gail_airl_ppo.network.disc1 import AIRLDiscrim


class DAAIRL(PPO):

    def __init__(self, buffer_exp0, buffer_exp1, state_shape,  action_shape, device, seed, path0, pathc,
                 gamma=0.99, rollout_length=10000, mix_buffer=1,
                 batch_size=128, lr_actor=3e-4, lr_critic=3e-3, lr_disc=3e-4,
                 units_actor=(64, 64), units_critic=(64, 64),
                 units_disc_r=(100, 100), units_disc_v=(100, 100),
                 epoch_ppo=50, epoch_disc=10, clip_eps=0.2, lambd=0.97,
                 coef_ent=0.0, max_grad_norm=10.0):
        super().__init__(
            state_shape, action_shape, device, seed, path0, pathc, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm
        )

        # Expert's buffer.
        self.buffer_exp0 = buffer_exp0
        self.buffer_exp1 = buffer_exp1
        self.gamma = gamma
        # Discriminator.


        self.disc = AIRLDiscrim(
            state_shape=state_shape,
            gamma=gamma,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.Tanh(),
            hidden_activation_v=nn.Tanh()
        ).to(device)

        self.de = device

        self.learning_steps_disc = 0
        self.criterion = nn.BCEWithLogitsLoss()
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc

    def tempretrue(self, mean_reward):
        return mean_reward > 0

    def update(self, writer):
        self.learning_steps += 1
        for _ in range(self.epoch_disc):
            if self.update_0:
                self.learning_steps_disc += 1
                # Samples from current policy's trajectories.
                states0, _, _, dones0, log_pis0, next_states0 = self.buffer0.sample(self.batch_size)
                # Samples from expert's demonstrations.
                states_exp0, actions_exp0, _, dones_exp0, next_states_exp0 = \
                    self.buffer_exp0.sample(self.batch_size)
                # Calculate log probabilities of expert actions.
                with torch.no_grad():
                    log_pis_exp0 = self.actor0.evaluate_log_pi(
                        states_exp0, actions_exp0)
                # Update discriminator.

                self.update_disc(
                    states0, dones0, log_pis0, next_states0, states_exp0,
                    dones_exp0, log_pis_exp0, next_states_exp0, writer
                )

            if self.update_1 and self.buffer1._n >= self.rollout_length:
                self.learning_steps_disc += 1
                states1, _, _, dones1, log_pis1, next_states1 = self.buffer1.sample(self.batch_size)
                # Samples from expert's demonstrations.
                states_exp1, actions_exp1, _, dones_exp1, next_states_exp1 = \
                        self.buffer_exp1.sample(self.batch_size)
                # Calculate log probabilities of expert actions.
                with torch.no_grad():
                        log_pis_exp1 = self.actor1.evaluate_log_pi(
                            states_exp1, actions_exp1)
                # Update discriminator.
                self.update_disc(
                        states1,  dones1, log_pis1, next_states1, states_exp1,
                        dones_exp1, log_pis_exp1, next_states_exp1, writer
                )

        if self.update_0:
        # We don't use reward signals here,
            states0, actions0, res0, dones0, log_pis0, next_states0 = self.buffer0.get()
            # Calculate rewards.
            rewards0 = self.disc.calculate_reward(
                states0, dones0, log_pis0, next_states0)

            # mean = rewards0.mean(dim=0)
            # std = rewards0.std(dim=0)
            # std[std == 0] = 1
            # rewards0 = (rewards0 - mean) / std

            # rewards0.clip(-1, 1)
            # rewards0.add_(res0)

            self.update_ppo0(
                states0, actions0, rewards0, dones0, log_pis0, next_states0, writer)

        if self.update_1 and self.buffer1._n >= self.rollout_length:
            # We don't use reward signals here,

            states1, actions1, res1, dones1, log_pis1, next_states1 = self.buffer1.get()
            rewards1 = self.disc.calculate_reward(
                states1, dones1, log_pis1, next_states1)


            # mean = rewards1.mean(dim=0)
            # std = rewards1.std(dim=0)
            # std[std == 0] = 1
            # rewards1 = (rewards1 - mean) / std
            # rewards1.clip(-1, 1)
            # rewards1.add_(res1)
            # Update PPO using estimated rewards.
            self.update_ppo1(
                states1, actions1, rewards1, dones1, log_pis1, next_states1, writer)


    def update_disc(self, states, dones, log_pis, next_states,
                    states_exp, dones_exp, log_pis_exp,
                    next_states_exp, writer):
        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc(states, dones, log_pis, next_states)
        logits_exp = self.disc(
            states_exp, dones_exp, log_pis_exp, next_states_exp)
        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].

        loss_exp = self.criterion(logits_exp, torch.ones(logits_exp.shape[0], 1).to(self.de))
        loss_pi = self.criterion(logits_pi, torch.zeros(logits_pi.shape[0], 1).to(self.de))

        # loss_pi = -F.logsigmoid(-logits_pi).mean()
        # loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar(
                'loss/disc', loss_disc.item(), self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0.5).float().mean().item()
                acc_exp = (logits_exp > 0.5).float().mean().item()
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)


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
        torch.save(
            self.critic.state_dict(),
            os.path.join(save_dir, 'critic.pth')
        )
        # torch.save(
        #     self.g.state_dict(),
        #     os.path.join(save_dir, 'critic.pth')
        # )



