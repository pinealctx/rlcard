from rlcard.agents import CFRAgent
from cachetools import LRUCache
from rlcard.agents.dmc_agent import FileWriter


class DeepCFRAgent(CFRAgent):
    def __init__(self, env, average_policy_lru:LRUCache, max_lru_size, model_path='./deep_cfr_model'):
        super().__init__(env, model_path)

        # 固定大小的字典
        self.policy = LRUCache(maxsize=max_lru_size)
        self.average_policy = average_policy_lru
        self.regrets = LRUCache(maxsize=max_lru_size)


class DeepCFRTrainer(object):
    def __init__(
            self,
            env,
            cuda='',
            num_actor_devices=1,
            xpid='deep_cfr',
            savedir='./deep_cfr_model',
            max_lru_size=1000000,

            train_device="0",
            load_model=False,

            batch_size=128,

            num_traversals=150,
            learning_rate=1e-3,
            save_interval=10,
    ):
        self.env = env
        if cuda == "":
            self.device_iterator = ["cpu"]
            self.training_device = "cpu"
        else:
            self.device_iterator = ["cuda:" + str(i) for i in range(num_actor_devices)]
            self.training_device = self.device_iterator[0]
        self.plogger = FileWriter(
            xpid=xpid,
            rootdir=savedir,
        )
        self.average_policy = LRUCache(maxsize=max_lru_size)
        self.agent = DeepCFRAgent(
            env,
            self.average_policy,
            max_lru_size,
            model_path=savedir,
        )
