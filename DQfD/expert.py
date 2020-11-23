import gym
import nel
import numpy as np
from numpy.linalg import norm


def recognize_item(px, py, vision):
    """Recognize the item located at (px, py) in the vision."""
    color = vision[px, py, :]
    if norm(color - np.array([0, 1, 0])) < 1e-3:
        return 'diamond'
    elif norm(color - np.array([1, 0, 0])) < 1e-3:
        return 'tongs'
    elif norm(color - np.array([0, 0, 1])) < 1e-3:
        # Avoid confusion that agent and jellybean share the same color.
        if (px, py) == (5, 5):
            return 'agent'
        else:
            return 'jellybean'
    elif norm(color - np.array([0.5, 0.5, 0.5])) < 1e-3:
        return 'wall'
    else:
        return 'blank'


def simulate_step(px, py, d, a, vision, n_tongs):
    """Simulate a step.
    Returns: px_next, py_next, d_next, r
    """
    r = -1
    if a == 0:  # action: FORWARD
        if d == 0:  # direction: UP
            px_next, py_next = px, py + 1
        elif d == 1:  # direction: DOWN
            px_next, py_next = px, py - 1
        elif d == 2:  # direction: LEFT
            px_next, py_next = px - 1, py
        else:  # direction: RIGHT
            px_next, py_next = px + 1, py
        d_next = d
        if 0 <= px_next < 11 and 0 <= py_next < 11:  # next position in vision
            item_next = recognize_item(px_next, py_next, vision)
            if item_next == 'diamond' and n_tongs > 0:
                r += 100
            elif item_next == 'tongs':
                r += 40  # reward to encourage collecting tongs
            elif item_next == 'jellybean':
                r += 20
            elif item_next == 'wall':
                px_next, py_next = px, py
                r += -1  # penalty to discourage forwarding to wall
    elif a == 1:  # action: LEFT
        px_next, py_next = px, py
        if d == 0:  # direction: UP
            d_next = 2
        elif d == 1:  # direction: DOWN
            d_next = 3
        elif d == 2:  # direction: LEFT
            d_next = 1
        else:  # direction: RIGHT
            d_next = 0
    else:  # action: RIGHT
        px_next, py_next = px, py
        if d == 0:  # direction: UP
            d_next = 3
        elif d == 1:  # direction: DOWN
            d_next = 2
        elif d == 2:  # direction: LEFT
            d_next = 0
        else:  # direction: RIGHT
            d_next = 1
    return px_next, py_next, d_next, r


def value_iteration(vision, n_tongs):
    """Run value iteration algorithm to get the optimal Q-value.
    Returns: Q-value for locations in the vision.
    """
    q_value = np.zeros([11, 11, 4, 3], dtype=np.float32)
    for _ in range(1000):
        delta = 0
        for px in range(11):
            for py in range(11):
                item = recognize_item(px, py, vision)
                # Not update for locations collecting items, to avoid endless loops.
                if (item == 'diamond' and n_tongs > 0) or item == 'tongs' or item == 'jellybean':
                    continue
                q_new = np.zeros([4, 3], dtype=np.float32)
                for d in range(4):
                    for a in range(3):
                        px_next, py_next, d_next, r = simulate_step(px, py, d, a, vision, n_tongs)
                        q_new[d, a] = r
                        if 0 <= px_next < 11 and 0 <= py_next < 11:  # next position in vision
                            q_new[d, a] += np.max(q_value[px_next, py_next, d_next])
                delta = max(delta, norm(q_new - q_value[px, py], ord=np.inf))
                q_value[px, py] = q_new
        if delta < 1e-3:
            break
    return q_value


def run_expert_policy(env):
    env.reset()
    observation, reward, _, _ = env.step(0)
    reward_list = [reward]
    for step in range(1000):
        env.render()
        vision = observation['vision']
        n_tongs = env._agent._items[1]  # number of tongs carried
        q_value = value_iteration(vision, n_tongs)
        action = np.argmax(q_value[5, 5, 0])
        observation, reward, _, _ = env.step(action)
        reward_list.append(reward)
    return reward_list


if __name__ == '__main__':
    env = gym.make('NEL-v0')
    env_r = gym.make('NEL-render-v0')
    reward_list = run_expert_policy(env_r)
