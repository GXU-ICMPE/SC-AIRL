from abc import ABC, abstractmethod
import os
import numpy as np
import torch


class Algorithm(ABC):

    def __init__(self, state_shape,  action_shape1, device, seed, path0, pathc, gamma):   # path
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.learning_steps = 0
        self.state_shape = state_shape
        self.action_shape1 = action_shape1

        # self.action_shape1 = action_shape1
        self.device = device
        self.gamma = gamma


    def explore(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state.unsqueeze_(0))
        return action.cpu().numpy()[0], log_pi.item()
    def explore2_a(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor_2.sample(state.unsqueeze_(0))
        return action.cpu().numpy()[0], log_pi.item()
    def explore3_a(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor_3.sample(state.unsqueeze_(0))
        return action.cpu().numpy()[0], log_pi.item()
    def explore0(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor0.sample(state.unsqueeze_(0))
        return action.cpu().numpy()[0], log_pi.item()
    def explore1(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor1.sample(state.unsqueeze_(0))
        return action.cpu().numpy()[0], log_pi.item()
    def explore2(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor2.sample(state.unsqueeze_(0))
        return action.cpu().numpy()[0], log_pi.item()
    def explore3(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor3.sample(state.unsqueeze_(0))
        return action.cpu().numpy()[0], log_pi.item()
    def explore4(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor4.sample(state.unsqueeze_(0))
        return action.cpu().numpy()[0], log_pi.item()

    def exploit(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action = self.actor(state.unsqueeze_(0))
        return action.cpu().numpy()[0]
    def exploit0(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action = self.actor0(state.unsqueeze_(0))
        return action.cpu().numpy()[0]

    def exploit1(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action = self.actor1(state.unsqueeze_(0))
        return action.cpu().numpy()[0]

    def exploit_SC(self, state, info):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        if info == 0:
            with torch.no_grad():
                action = self.actor0(state.unsqueeze_(0))
        elif info == 1:
            with torch.no_grad():
                action = self.actor1(state.unsqueeze_(0))
        return action.cpu().numpy()[0]

    def exploit1(self, state, info):
        state1 = np.append(state[:6], np.array([0.0, 0.0, 0.0, 0.0]))

        state1 = torch.tensor(state, dtype=torch.float, device=self.device) # peg in
        #
        state1 = np.append(state[:6], 0.0) # two
        state1 = torch.tensor(state1, dtype=torch.float, device=self.device)
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        if info == 1:
            with torch.no_grad():
                action = self.actor1(state1.unsqueeze_(0))
                # change, _ = self.actor.sample(state.unsqueeze_(0))  # here for guide-actor
                change, = self.actor(state.unsqueeze_(0))
        elif info == 0:
            with torch.no_grad():
                action = self.actor2(state1.unsqueeze_(0))
                change = self.actor(state.unsqueeze_(0))
        elif info == 2:
            with torch.no_grad():
                action = self.actor3(state1.unsqueeze_(0))
                change = self.actor0(state.unsqueeze_(0))
        elif info == 3:
            with torch.no_grad():
                action = self.actor4(state1.unsqueeze_(0))
                change = self.actor0(state.unsqueeze_(0))

        return action.cpu().numpy()[0], change.cpu().numpy()[0]

    def exploit_panda_two(self, state, info):

        state = torch.tensor(state, dtype=torch.float, device=self.device)

        if info == 0:
            with torch.no_grad():
                action = self.actor0(state.unsqueeze_(0))

        elif info == 1:
            with torch.no_grad():
                action = self.actor1(state.unsqueeze_(0))

        elif info == 2:
            with torch.no_grad():
                action = self.actor2(state.unsqueeze_(0))

        elif info == 3:
            with torch.no_grad():
                action = self.actor3(state.unsqueeze_(0))

        return action.cpu().numpy()[0]

    @abstractmethod
    def is_update(self, step):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
