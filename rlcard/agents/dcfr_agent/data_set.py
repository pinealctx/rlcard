import torch

from torch.utils.data import Dataset


class MemoryDataset(Dataset):
    def __init__(self, buffers):
        self.buffers = buffers
        self.num_samples = len(buffers)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        (state, value, iteration, mask) = self.buffers[idx]
        return (
            torch.from_numpy(state).float(),
            torch.from_numpy(value).float(),
            torch.from_numpy(iteration).float(),
            torch.from_numpy(mask).float(),
        )


class AdvantageDataset(MemoryDataset):
    def __init__(self, advantage_buffers):
        super().__init__(advantage_buffers)

    def __getitem__(self, idx):
        (state, advantage, iteration, mask) = super().__getitem__(idx)
        return (
            state,
            advantage,
            iteration,
            mask,
        )


class PolicyDataset(MemoryDataset):
    def __init__(self, policy_buffers):
        super().__init__(policy_buffers)

    def __getitem__(self, idx):
        (state, action_prob, iteration, mask) = super().__getitem__(idx)
        return (
            state,
            action_prob,
            iteration,
            mask,
        )
