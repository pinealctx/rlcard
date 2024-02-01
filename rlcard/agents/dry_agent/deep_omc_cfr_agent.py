import os
import torch
import numpy as np

from datetime import datetime

"""from torch.optim.lr_scheduler import ReduceLROnPlateau
from rlcard.utils.utils import remove_illegal"""
from rlcard.agents.model import SoftMaxRstNet
from .omc_cfr_agent import OSMCCFRAgent
from .eval_agent import EvalAgent

"""#, EarlyStopping"""


class PolicyNet(SoftMaxRstNet):
    def __init__(
            self,
            state_shape,
            action_shape,
            mlp_layers=[512, 512, 512, 512, 512, 512, 512, 512],
            activation='leakyrelu',
            lr=0.00003,
            l2_lambda=0.00001,
    ):
        super().__init__(state_shape, action_shape, mlp_layers, activation, lr)
        if l2_lambda > 0:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2_lambda)
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)


class DeepOSMCCFRAgent(OSMCCFRAgent):
    def __init__(self,
                 env,
                 max_lru_size,
                 device_num=0,
                 batch_size=64,
                 process_id=0,
                 activation='leakyrelu',
                 lr=0.00003,
                 early_stop_patience=20,
                 l2_lambda=0.00001,
                 min_training_times=2000,
                 model_path='./deep_omc_cfr_model'):
        super().__init__(env, max_lru_size, model_path)

        self.device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.process_id = process_id
        self.average_policy_net = PolicyNet(self.env.state_shape[0][0], self.env.num_actions,
                                            activation=activation,
                                            lr=lr,
                                            l2_lambda=l2_lambda).to(self.device)
        self.average_policy_net.share_memory()
        self.early_stop_patience = early_stop_patience
        self.min_training_times = min_training_times
        self.eval_agent = EvalAgent(self.average_policy_net, self.device)

    def train(self):
        average_policy_counter = self.average_policy_update_count
        print("Start training model, current average policy update count: {}".format(average_policy_counter))
        """loop train if average policy has been countered more than max_lru_size"""
        """then train model"""
        prev_policy_counter = average_policy_counter
        while self.average_policy_update_count - average_policy_counter < self.max_lru_size / 10:
            super().train()
            if self.iteration % 1000 == 0:
                now = datetime.now()
                print("Time: {}, Iteration: {}, average policy update count: {}, delta :{}".
                      format(now.strftime("%Y-%m-%d %H:%M:%S"),
                             self.iteration,
                             self.average_policy_update_count,
                             self.average_policy_update_count - prev_policy_counter))
                prev_policy_counter = self.average_policy_update_count
        print("Start training model, previous average policy update count: {}, current average policy update count: {}".
              format(average_policy_counter, self.average_policy_update_count))
        """train model"""
        self.train_model()
        self.average_policy_pool.clear()
        self.save()

    def train_model(self):
        average_policy_pool_count = len(self.average_policy_pool)
        if average_policy_pool_count < self.batch_size:
            return

        # Unpack the data from average_policy_pool
        states, targets = zip(*list(self.average_policy_pool.items()))
        states = [np.frombuffer(state, dtype=np.float64) for state in states]
        states = np.array(states)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        targets = np.array(targets)
        targets = torch.tensor(targets, dtype=torch.float32, device=self.device)

        # Initialize the learning rate scheduler
        # scheduler = ReduceLROnPlateau(self.average_policy_net.optimizer, 'min')

        # Initialize the EarlyStopping object
        # early_stopping = EarlyStopping(self.early_stop_patience)
        count = len(states)
        print("Start training model, total data size: {}, average policy pool count: {}".
              format(count, average_policy_pool_count))

        # min_iterations = (count // self.batch_size) * self.min_training_times
        for i in range(self.min_training_times):
            # shuffle the data
            # perm = torch.randperm(count)
            # states = states[perm]
            # targets = targets[perm]
            curr_loss = 0
            batch_count = 0
            for j in range(0, count, self.batch_size):
                inputs = states[j:j + self.batch_size]
                labels = targets[j:j + self.batch_size]
                # Train the model and get the current loss
                curr_loss += self.train_batch(inputs, labels)
                batch_count += 1
            print("Training model, episode: {}, loss: {}".format(i, curr_loss / batch_count))
            # Update the learning rate
            # scheduler.step(curr_loss)
            """if i % 2000 == 0:
                print("Training model, episode: {}, loss: {}".format(i, curr_loss))"""
        # print("Finish training model, total episode: {}".format(episode))

    def train_batch(self, inputs, labels):
        self.average_policy_net.lock.acquire()
        try:
            curr_loss = self.average_policy_net.train_model(inputs, labels)
        finally:
            self.average_policy_net.lock.release()
        return curr_loss

    def save(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        file_name = "deep_omc_cfr_model_{}_{}.pth".format(self.process_id, self.iteration)
        self.save_model(os.path.join(self.model_path, file_name))

    def load(self):
        if not os.path.exists(self.model_path):
            return
        file_name = "deep_omc_cfr_model_{}.pth".format(self.process_id)
        file_full_path = os.path.join(self.model_path, file_name)
        self.load_model(file_full_path)
        print("Load model from {}".format(file_full_path))

    def eval_step(self, state):
        return self.eval_agent.eval_step(state)

    def save_model(self, path):
        """Save the model to the specified path."""
        torch.save({
            'average_net_state_dict': self.average_policy_net.state_dict(),
        }, path)

    def load_model(self, path):
        """Load the model from the specified path."""
        checkpoint = torch.load(path)
        self.average_policy_net.load_state_dict(checkpoint['average_net_state_dict'])
        self.eval_agent = EvalAgent(self.average_policy_net, self.device)
