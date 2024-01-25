import logging
import traceback
import numpy as np
import torch

from copy import deepcopy


shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('deep_cfr')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)


def create_buffers_for_cfr(
    num_actors,
    num_buffers,
    state_shape,
    action_shape,
    device_iterator,
):
    buffers = {}
    for device in device_iterator:
        buffers[device] = []
        for _ in range(num_actors):
            specs = dict(
                state=dict(size=state_shape, dtype=torch.float32),
                action_probs=dict(size=action_shape, dtype=torch.float32),
            )
            _buffers = {key: [] for key in specs}
            for _ in range(num_buffers):
                for key in _buffers:
                    if device == "cpu":
                        _buffer = torch.empty(**specs[key]).to('cpu').share_memory_()
                    else:
                        _buffer = torch.empty(**specs[key]).to('cuda:'+str(device)).share_memory_()
                    _buffers[key].append(_buffer)
            buffers[device].append(_buffers)
    return buffers


def act_for_cfr(
        i,
        device,
        free_queue,
        full_queue,
        model,
        buffers,
        env
):
    try:
        log.info('Device %s Actor %i started.', str(device), i)

        # Configure environment
        env = deepcopy(env)
        env.seed(i)
        # env.set_agents(model.get_actors())

        while True:
            trajectories, payoffs = env.run(is_training=True)
            for p in range(env.num_players):
                size[p] += len(trajectories[p][:-1]) // 2
                diff = size[p] - len(target_buf[p])
                if diff > 0:
                    done_buf[p].extend([False for _ in range(diff - 1)])
                    done_buf[p].append(True)
                    episode_return_buf[p].extend([0.0 for _ in range(diff - 1)])
                    episode_return_buf[p].append(float(payoffs[p]))
                    target_buf[p].extend([float(payoffs[p]) for _ in range(diff)])
                    # State and action
                    for i in range(0, len(trajectories[p]) - 2, 2):
                        state = trajectories[p][i]['obs']
                        action = env.get_action_feature(trajectories[p][i + 1])
                        state_buf[p].append(torch.from_numpy(state))
                        action_buf[p].append(torch.from_numpy(action))

                while size[p] > T:
                    index = free_queue[p].get()
                    if index is None:
                        break
                    for t in range(T):
                        buffers[p]['done'][index][t, ...] = done_buf[p][t]
                        buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
                        buffers[p]['target'][index][t, ...] = target_buf[p][t]
                        buffers[p]['state'][index][t, ...] = state_buf[p][t]
                        buffers[p]['action'][index][t, ...] = action_buf[p][t]
                    full_queue[p].put(index)
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    state_buf[p] = state_buf[p][T:]
                    action_buf[p] = action_buf[p][T:]
                    size[p] -= T

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e
