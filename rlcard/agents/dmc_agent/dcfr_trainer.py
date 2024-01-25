import os
import logging
import threading
import time
import timeit
import pprint
from collections import deque

import torch
from torch import multiprocessing as mp
from torch import nn

from .file_writer import FileWriter
from .dcfr_model import DCFRModel
from .dcfr_utils import (
    create_buffers_for_cfr,
)

from .model import DMCModel
from .pettingzoo_model import DMCModelPettingZoo
from .utils import (
    get_batch,
    create_buffers,
    create_optimizers,
    act,
)

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('deep_cfr')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

class DCFRTrainer(object):
    def __init(
            self,
            env,
            cuda='',
            load_model=False,
            xpid='dcfr',
            save_interval=30,
            num_actor_devices=1,
            num_actors=8,
            training_device='0',
            save_dir='experiments/dcfr_result',
            total_frames=100000000000,
            batch_size=128,
            num_buffers=256,
            num_threads=8,
            max_grad_norm=40,
            learning_rate=0.0001,
            momentum=0,
    ):
        self.env = env

        self.plogger = FileWriter(
            xpid=xpid,
            rootdir=save_dir,
        )

        self.checkpointpath = os.path.expandvars(
            os.path.expanduser('%s/%s/%s' % (save_dir, xpid, 'model.tar')))

        self.B = batch_size

        self.xpid = xpid
        self.load_model = load_model
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.num_actor_devices = num_actor_devices
        self.num_actors = num_actors
        self.training_device = training_device
        self.total_frames = total_frames
        self.num_buffers = num_buffers
        self.num_threads = num_threads
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.num_players = self.env.num_players
        self.action_shape = self.env.action_shape
        if self.action_shape[0] == None:  # One-hot encoding
            self.action_shape = [[self.env.num_actions] for _ in range(self.num_players)]

        def model_func(device):
            return DCFRModel(
                self.env.state_shape,
                self.action_shape,
                self.num_actors,
                lr=self.learning_rate,
                device=str(device),
            )

        self.model_func = model_func

        self.mean_episode_return_buf = [deque(maxlen=100) for _ in range(self.num_players)]

        if cuda == "":  # Use CPU
            self.device_iterator = ['cpu']
            self.training_device = "cpu"
        else:
            self.device_iterator = range(num_actor_devices)

    def start(self):
        # Initialize actor models
        models = {}
        for device in self.device_iterator:
            model = self.model_func(device)
            model.share_memory()
            model.eval()
            models[device] = model

        # Initialize buffers
        buffers = create_buffers_for_cfr(
            self.num_actors,
            self.num_buffers,
            self.env.state_shape[0],
            self.action_shape[0],
            self.device_iterator,
        )
        print()

        # Initialize queues
        actor_processes = []
        ctx = mp.get_context('spawn')
        free_queue = {}
        full_queue = {}

        for device in self.device_iterator:
            _free_queue = [ctx.SimpleQueue() for _ in range(self.num_actors)]
            _full_queue = [ctx.SimpleQueue() for _ in range(self.num_actors)]
            free_queue[device] = _free_queue
            full_queue[device] = _full_queue

        # Learner model for training
        learner_model = self.model_func(self.training_device)

        # Load models if any
        if self.load_model and os.path.exists(self.checkpointpath):
            checkpoint_states = torch.load(
                self.checkpointpath,
                map_location="cuda:"+str(self.training_device) if self.training_device != "cpu" else "cpu",
            )
            for actor in range(self.num_actors):
                agent = learner_model.get_agent(actor)
                agent.load_state_dict(checkpoint_states['model_state_dict'][actor])
                agent.optimizer.load_state_dict(checkpoint_states['optimizer_state_dict'][actor])
            stats = checkpoint_states['stats']
            frames = checkpoint_states['frames']
            log.info(f"Resuming preempted job, current stats:\n{stats}")

        # Starting actor processes
        for device in self.device_iterator:
            for i in range (self.num_actors):
                args = (i, device, free_queue[device], full_queue[device], models[device], buffers[device], self.env)
                actor = ctx.Process(target=act, args=args)
                actor.start()
                actor_processes.append(actor)

