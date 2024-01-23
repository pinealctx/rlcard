import torch
import numpy as np
import random


class ReservoirBuffer(object):
    """Allows uniform sampling over a stream of data.

    This class supports the storage of arbitrary elements, such as observation
    tensors, integer actions, etc.

    See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
    """
    def __init__(self, reservoir_buffer_capacity: int):
        """Initializes the buffer.

        Args:
            reservoir_buffer_capacity: The number of elements that can be stored
                in the buffer.
        """
        self._reservoir_buffer_capacity = reservoir_buffer_capacity
        self._data = []
        self._add_calls = 0

    def add(self, item):
        """Potentially add an item to the buffer.

        Args:
            item: The item to add to the buffer.
        """
        if len(self._data) < self._reservoir_buffer_capacity:
            # Convert numpy arrays or lists to PyTorch tensors when adding to buffer
            item = torch.tensor(item) if isinstance(item, (np.ndarray, list)) else item
            self._data.append(item)
        else:
            idx = np.random.randint(0, self._add_calls + 1)
            if idx < self._reservoir_buffer_capacity:
                item = torch.tensor(item) if isinstance(item, (np.ndarray, list)) else item
                self._data[idx] = item
        self._add_calls += 1

    def sample(self, num_samples: int):
        """Returns a sample of items from the buffer.

        Args:
            num_samples: The number of items to sample from the buffer.

        Returns:
            A list of sampled items.
        """
        if len(self._data) < num_samples:
            raise ValueError('Not enough items in the buffer, want {} but size is {}.'.format(
                num_samples, len(self._data)
            ))
        else:
            return [torch.tensor(e) if isinstance(e, (np.ndarray, list))
                    else e for e in random.sample(self._data, num_samples)]

    def clear(self):
        """Clears the buffer."""
        self._data = []
        self._add_calls = 0

    @property
    def data(self):
        return self._data

    def shuffle_data(self):
        random.shuffle(self._data)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

