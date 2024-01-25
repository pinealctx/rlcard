''' An example of solve Leduc Hold'em with CFR (chance sampling)
'''
import os
import argparse

import rlcard
from rlcard.agents import (
    CFRAgent,
    RandomAgent,
)
from rlcard.utils import (
    set_seed,
    tournament,
    Logger,
    plot_curve,
)
from rlcard.agents.dmc_agent.dcfr_model import DCFRAgent


def train(args):
    # Make environments, CFR only supports Leduc Holdem
    env = rlcard.make(
        args.game,
        config={
            'seed': 0,
            'allow_step_back': True,
        }
    )
    eval_env = rlcard.make(
        args.game,
        config={
            'seed': 0,
        }
    )

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Initilize CFR Agent
    if args.deep:
        agent = DCFRAgent(
            env,
            1000000,
            os.path.join(
                args.log_dir,
                'deep_cfr_model',
            ),
        )
    else:
        agent = CFRAgent(
            env,
            os.path.join(
                args.log_dir,
                'cfr_model',
            ),
        )
        agent.load()  # If we have saved model, we first load the model


    # Evaluate CFR against random
    eval_env.set_agents([
        agent,
        RandomAgent(num_actions=env.num_actions),
    ])

    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            agent.train()
            print('\rIteration {}'.format(episode), end='')
            # Evaluate the performance. Play with Random agents.
            if episode % args.evaluate_every == 0:
                agent.save() # Save model
                logger.log_performance(
                    episode,
                    tournament(
                        eval_env,
                        args.num_eval_games
                    )[0]
                )

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path
    # Plot the learning curve
    plot_curve(csv_path, fig_path, 'cfr')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CFR example in RLCard")
    parser.add_argument(
        '--game',
        type=str,
        default='limit-holdem',
    )
    parser.add_argument(
        '--deep',
        type=bool,
        default=False,
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=50000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=2000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/limit-holdem_cfr_result/',
    )

    args = parser.parse_args()

    train(args)

