"""
qlearning_lab.py
==================
YOUR TASK: implement Q-learning control on the PackBot environment.

Run test_correctness.py first -- it checks qlearning_update() against a
hand-computed example before you spend any time on the full training loop.

Do not import anything from qlearning_solution.py.
"""

import random
import numpy as np
from packbot_env import PackBotEnv, state_index
from rl_utils import epsilon_greedy, linear_epsilon_schedule, init_Q, wear_threshold


def qlearning_update(Q, s, a, r, s_next, alpha, gamma):
    """
    TODO: implement the Q-learning update rule.

    Inputs:
        Q       : the (N_STATES, N_ACTIONS) value table
        s, a    : the state index and action just taken
        r       : the reward received
        s_next  : the resulting state index, or None if the episode ended
        alpha, gamma : learning rate and discount factor

    Returns:
        The NEW value that Q[s, a] should be updated to. Do not mutate Q
        inside this function -- just return the new scalar.

    Reminder from lecture: Q-learning bootstraps using max_a' Q[s_next, a'] --
    the BEST available action at s_next -- regardless of which action your
    behavior policy would actually take there. Notice there is no a_next
    parameter at all: Q-learning never needs to know what your ε-greedy
    policy actually sampled next. That is the entire difference between
    this file and sarsa_lab.py.

    Special case: if s_next is None (terminal transition), there is
    nothing to bootstrap from -- the target is just r.
    """
    if s_next is None:
        target = r
    else:
        target = r + gamma * np.max(Q[s_next])
    return Q[s, a] + alpha * (target - Q[s, a])


def train_qlearning(num_episodes=4000, alpha=0.1, gamma=0.95, seed=0,
                     eps_start=1.0, eps_end=0.2, log_every=50):
    """
    TODO: implement the Q-learning training loop using qlearning_update() above.

    Structure to follow (the standard Q-learning control loop from lecture):

        Initialize S
        Repeat until S is terminal:
            Choose A from S using epsilon-greedy(Q)
            Take action A, observe R, S'
            Q(S, A) <- qlearning_update(Q, S, A, R, S' if S' not terminal else None, alpha, gamma)
            S <- S'

    Return (Q, episode_returns, threshold_log) to match qlearning_solution.py's
    signature -- compare_policies.py and test scripts expect this shape.
    threshold_log should be a list of (episode, {item_type: wear_threshold(...)})
    tuples, logged every `log_every` episodes.
    """
    env = PackBotEnv(seed=seed)
    rng = random.Random(seed)
    Q = init_Q()


    # TODO: implement the training loop described above.
    episode_returns = []
    threshold_log = []

    for ep in range(num_episodes):
        eps = linear_epsilon_schedule(ep, num_episodes, eps_start, eps_end)

       
        S = env.reset()
        total_reward = 0

        
        while True:
           
            A = epsilon_greedy(Q, S, eps, rng)
            S_next, R, done, _ = env.step(A)
            total_reward += R
            S_next_idx = None if done else state_index(*S_next)
            Q[state_index(*S), A] = qlearning_update(Q, state_index(*S), A, R, S_next_idx, alpha, gamma)

            S = S_next

            if done:
                break

        episode_returns.append(total_reward)
        if (ep + 1) % log_every == 0:
            threshold_log.append((ep + 1, {t: wear_threshold(Q, t) for t in range(3)}))

    return Q, episode_returns, threshold_log

    


if __name__ == "__main__":
    Q, returns, thresholds = train_qlearning()
    print("Final wear thresholds by item type:", thresholds[-1][1])
    print("Mean return, last 200 episodes:", np.mean(returns[-200:]))
