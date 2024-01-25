from rlcard.agents import CFRAgent
from cachetools import LRUCache
from rlcard.agents.dmc_agent import FileWriter


class DeepCFRAgent(CFRAgent):
    def __init__(self, env, max_lru_size, model_path='./deep_cfr_model'):
        super().__init__(env, model_path)

        # 固定大小的字典
        self.policy = LRUCache(maxsize=max_lru_size)
        self.average_policy = LRUCache(maxsize=max_lru_size)
        self.regrets = LRUCache(maxsize=max_lru_size)

