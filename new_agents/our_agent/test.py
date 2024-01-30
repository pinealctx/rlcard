def encode_the_action_list(actions):

    encoding = [[]] * 4
    encoding[0] = []
    encoding[1] = []
    encoding[2] = []
    encoding[3] = []

    current_stage = 0

    for action in actions:

        if action[1] == 'raise':
            encoding[current_stage].append(1)

        elif action[1] == 'call':
            if encoding[0] == []:
                encoding[current_stage].append(2)
            else:
                encoding[current_stage].append(2)
                current_stage += 1

        elif action[1] == 'check':
            if encoding[current_stage] != []:
                if encoding[current_stage][-1] == 3 or encoding[current_stage][-1] == 2:
                    encoding[current_stage].append(3)
                    current_stage += 1
            else:
                encoding[current_stage].append(3)

        elif action[1] == 'fold':
            encoding[current_stage].append(4)
            break

    encoding[0] += [-1] * (6 - len(encoding[0]))
    encoding[1] += [-1] * (6 - len(encoding[1]))
    encoding[2] += [-1] * (6 - len(encoding[2]))
    encoding[3] += [-1] * (6 - len(encoding[3]))
    encodings = encoding[0] + encoding[1] + encoding[2] + encoding[3]

    return encodings


if __name__ == "__main__":

    actions = [(1, 'call'), (0, 'check'), (1, 'check'), (0, 'raise'), (1, 'fold')]
    print(encode_the_action_list(actions))
