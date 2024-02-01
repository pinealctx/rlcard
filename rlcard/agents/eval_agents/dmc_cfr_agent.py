import torch

from rlcard.agents.dry_agent.deep_omc_cfr_agent import PolicyNet
from rlcard.agents.dry_agent.eval_agent import EvalAgent


class DMCCFRAgent(object):
    def __init__(self,
                 env,
                 model_path,
                 device_num=1,
                 activation='leakyrelu',
                 ):
        device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')
        average_policy_net = PolicyNet(env.state_shape[0][0],
                                       env.num_actions,
                                       activation=activation).to(device)
        checkpoint = torch.load(model_path)
        average_policy_net.load_state_dict(checkpoint['average_net_state_dict'])
        self.eval_agent = EvalAgent(average_policy_net, device)
        self.use_raw = False
        self.num_actions = env.num_actions

    def eval_step(self, state):
        return self.eval_agent.eval_step(state)

