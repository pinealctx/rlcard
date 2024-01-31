from .omc_cfr_agent import OSMCCFRAgent


class DebugOSMCCFRAgent(OSMCCFRAgent):
    def __init__(self,
                 env,
                 max_lru_size,
                 model_path='./debug_omc_cfr_model'):
        super().__init__(env, max_lru_size, model_path)

    def train(self):
        super().train()
        print("Average policy update count: {}".format(self.average_policy_update_count))
