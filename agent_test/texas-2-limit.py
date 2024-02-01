import os
import argparse
import rlcard

from rlcard.utils import (
    set_seed,
    tournament,
    Logger,
)

from rlcard.agents import RandomAgent
from rlcard.agents.eval_agents.dmc_cfr_agent import DMCCFRAgent

"""outside agents"""
from new_agents.our_agent.Calling_stations_Player import CallingStationAgent
from new_agents.our_agent.Donkeys_Player import DonkeyAgent
from new_agents.our_agent.Loose_Aggressive_Player import LooseAggressiveAgent
from new_agents.our_agent.Maniacs_player import ManiacAgent
from new_agents.our_agent.Nits_player import NitsAgent
from new_agents.our_agent.Probability_player import ProbabilityAgent
from new_agents.our_agent.Raise_player import RaiseAgent
from new_agents.our_agent.Random_player import RandomAgent as RandomAgentV2
from new_agents.our_agent.Rock_player import RockAgent
from new_agents.our_agent.Tight_Aggressive_Player import TightAggressiveAgent
from new_agents.our_agent.Weak_Tight_Players import WeakTightAgent

agent_key_funcs = {
    'cs': (CallingStationAgent, "Calling Station"),
    'dk': (DonkeyAgent, "Donkey"),
    'la': (LooseAggressiveAgent, "Loose Aggressive"),
    'mc': (ManiacAgent, "Maniac"),
    'nt': (NitsAgent, "Nits"),
    'pt': (ProbabilityAgent, "Probability"),
    'rs': (RaiseAgent, "Raise"),
    "rd": (RandomAgent, "Random"),
    'rd2': (RandomAgentV2, "Random V2"),
    'rk': (RockAgent, "Rock"),
    'ta': (TightAggressiveAgent, "Tight Aggressive"),
    'wt': (WeakTightAgent, "Weak Tight"),
}


def combat(args):
    seed = args.seed

    eval_env = rlcard.make(
        args.game,
        config={
            'seed': seed,
        }
    )

    set_seed(seed)
    agent1, name1 = gen_agent(args.play1, eval_env, device_num=args.device_num)
    agent2, name2 = gen_agent(args.play2, eval_env, device_num=args.device_num)
    print("{} VS {}".format(name1, name2))
    eval_env.set_agents([agent1, agent2])
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            print('\rIteration {}'.format(episode), end='')
            # Evaluate the performance. Play with Random agents.
            logger.log_performance(
                episode,
                tournament(eval_env, args.num_eval_games)[0]
            )


def gen_agent(key, env, **kwargs):
    # if key not in agent_key_funcs use DMCCFRAgent
    if "device_num" not in kwargs:
        kwargs['device_num'] = 1
    if key not in agent_key_funcs:
        return DMCCFRAgent(env, key, device_num=kwargs['device_num']), "DMC CFR Agent{}".format(key)
    agent, name = agent_key_funcs[key]
    return agent(num_actions=env.num_actions), name


def gen_help():
    str_buf = "Available agents:\n"
    for key, (agent, name) in agent_key_funcs.items():
        str_buf.join("{}: {}\n".format(key, name))
    str_buf.join("Net Agent: your_model_path\n")
    return str_buf


if __name__ == '__main__':
    agent_help_str = gen_help()
    parser = argparse.ArgumentParser("Agent Combat")
    parser.add_argument(
        '--game',
        type=str,
        default='limit-holdem',
    )
    parser.add_argument(
        '--play1',
        type=str,
        default='dk',
        help=agent_help_str,
    )
    parser.add_argument(
        '--play2',
        type=str,
        default='dk',
        help=agent_help_str,
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--num-eval-games',
        type=int,
        default=1000,
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--device-num',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='agent_combat',
    )
    args = parser.parse_args()
    combat(args)
