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
from rlcard.agents.dry_agent.deep_omc_cfr_agent import DeepOSMCCFRAgent


def train(args):
    # Make environments, CFR only supports Leduc Holdem
    seed = args.seed

    env = rlcard.make(
        args.game,
        config={
            'seed': seed,
            'allow_step_back': True,
        }
    )
    eval_env = rlcard.make(
        args.game,
        config={
            'seed': seed,
        }
    )

    # Seed numpy, torch, random
    set_seed(seed)

    # Initilize CFR Agent
    if args.agent_type == 1:
        agent = DeepOSMCCFRAgent(
            env,
            100000,
            epochs=args.num_trains,
            process_id=seed,
            lr=args.lr,
        )
    else:
        agent = CFRAgent(
            env,
            os.path.join(
                "{}_{}".format(args.log_dir, seed),
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
            if episode % 100 == 0:
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
        '--agent_type',
        type=int,
        help='0 for cfr, 1 for dmcrcf',
        default=1,
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=5000,
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
    parser.add_argument(
        '--num_trains',
        type=int,
        default=5,
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0003,
    )

    args = parser.parse_args()

    train(args)

