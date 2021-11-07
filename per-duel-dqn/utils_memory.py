from typing import (
    Tuple,
)

import random
import torch
import numpy

from utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    TensorStack5,
    TorchDevice,
)

class SumTree:
    write = 0

    def __init__(self, channels, capacity, device):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.device = device
        self.n_entries = 0
        self.__m_states = torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8)
        self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, state, action, reward, done):
        idx = self.write + self.capacity - 1

        self.__m_states[self.write] = state
        self.__m_actions[self.write] = action
        self.__m_rewards[self.write] = reward
        self.__m_dones[self.write] = done
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return idx, self.tree[idx], \
            self.__m_states[dataIdx, :4].to(self.device).float(), \
            self.__m_states[dataIdx, 1:].to(self.device).float(), \
            self.__m_actions[dataIdx].to(self.device), \
            self.__m_rewards[dataIdx].to(self.device).float(), \
            self.__m_dones[dataIdx].to(self.device)


class ReplayMemory(object):
    e = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.0

    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice,
    ) -> None:
        self.tree = SumTree(channels, capacity, device)
        self.__pos = 0


    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
    ) -> None:

        self.__pos = self.__pos + 1
        max_p = numpy.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, folded_state, action, reward, done)

    def sample(self, batch_size: int):
        idxs = numpy.empty((batch_size,), dtype=numpy.int32)
        ISWeights = numpy.empty((batch_size, 1))

        b_state_list = []
        b_next_list = []
        b_action_list = []
        b_reward_list = []
        b_done_list = []

        pri_seg = self.tree.total() / batch_size
        self.beta = numpy.min([1., self.beta + self.beta_increment_per_sampling])
        
        min_prob = numpy.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total()
        if min_prob == 0:
            min_prob = 0.00001

        for i in range(batch_size):
            a, b = pri_seg * i, pri_seg * (i + 1)

            while True:
                v = numpy.random.uniform(a, b)
                idx, p, b_state, b_next, b_action, b_reward, b_done = self.tree.get(v)
                if p != 0.0:
                    break

            idxs[i] = idx
            prob = p / self.tree.total()
            ISWeights[i, 0] = numpy.power(prob / min_prob, -self.beta)

            b_state_list.append(b_state)
            b_next_list.append(b_next)
            b_action_list.append(b_action)
            b_reward_list.append(b_reward)
            b_done_list.append(b_done)
        
        b_state_list = torch.stack(b_state_list)
        b_next_list = torch.stack(b_next_list)
        b_action_list = torch.stack(b_action_list)
        b_reward_list = torch.stack(b_reward_list)
        b_done_list = torch.stack(b_done_list)

        batch = (b_state_list, b_next_list, b_action_list, b_reward_list, b_done_list)

        return idxs, batch, ISWeights
      
    def batch_update(self, idx, error):
        error += self.e
        clipped_error = numpy.minimum(error.cpu().data.numpy(), self.abs_err_upper)
        ps = numpy.power(clipped_error, self.alpha)
        for ti, p in zip(idx, ps):
            self.tree.update(ti, p)

    def __len__(self) -> int:
        return self.__pos
