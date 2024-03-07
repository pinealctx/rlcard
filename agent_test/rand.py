import random

# integer list for 0 to 51
idx_52 = list(range(52))


def rand_sample(n: int):
    """random sample n cards from 52"""
    return random.sample(idx_52, n)


if __name__ == '__main__':
    print(rand_sample(5))
