import random
import numpy as np
from dataclasses import dataclass
from campus_graph import EDGES, astar
from qlearning_lab import qlearning_update
from rl_utils import linear_epsilon_schedule

# Adapted from Q learning/qlearning_lab.py
# Domain changed: PackBotEnv -> DroneEnv on the campus graph
# qlearning_update and linear_epsilon_schedule imported unchanged
# epsilon_greedy, greedy_action, init_Q copied from rl_utils.py,
# only change is the state is already an int so no state_index
# DroneEnv adapted from PackBotEnv in packbot_env.py

places = ["Dock", "NAB", "JSW", "Kalai", "Narsi", "Bhagat Ji",
          "Library", "RH-1", "RH-5", "SAZ"]
n = len(places)
N_STATES = n
N_ACTIONS = n   # action = which place to fly to

# drone flies about 300 metres a minute
speed = 300

# open stretches where wind keeps pushing the drone back
windy = [("Dock", "Library"), ("Library", "RH-1")]
windy_fail = 0.5
calm_fail = 0.1

neighbours = {}
dist_between = {}
fail_prob = {}
for p in places:
    neighbours[p] = []
for a, b, d in EDGES:
    neighbours[a].append(b)
    neighbours[b].append(a)
    dist_between[(a, b)] = d
    dist_between[(b, a)] = d
    if (a, b) in windy or (b, a) in windy:
        p = windy_fail
    else:
        p = calm_fail
    fail_prob[(a, b)] = p
    fail_prob[(b, a)] = p


@dataclass
class DroneStepInfo:
    blown_back: bool
    pos_before: int
    pos_after: int


class DroneEnv:
    """
    Adapted from PackBotEnv in packbot_env.py.
    State: index of the place the drone is at.
    Action: index of the place it tries to fly to.
    Wind on some edges can push it back so it stays where it was.
    """

    def __init__(self, goal, seed=None):
        self.rng = random.Random(seed)
        self.goal = places.index(goal)
        self.max_steps = 30
        self.pos = None
        self.steps = 0
        self.reset()

    def seed(self, seed):
        self.rng = random.Random(seed)

    def reset(self):
        # start somewhere random that isnt the goal
        start = self.rng.randrange(n)
        while start == self.goal:
            start = self.rng.randrange(n)
        self.pos = start
        self.steps = 0
        return self.pos

    def _fail_probability(self, here, there):
        # like _drop_probability in PackBotEnv
        return fail_prob[(here, there)]

    def step(self, action):
        assert 0 <= action < N_ACTIONS
        pos_before = self.pos
        here = places[self.pos]
        there = places[action]
        blown_back = False

        if there not in neighbours[here]:
            # cant fly there directly, just hovers and wastes a minute
            reward = -1
        else:
            minutes = dist_between[(here, there)] / speed
            if self.rng.random() < self._fail_probability(here, there):
                # wind pushed it back, time is wasted and it stays where it was
                blown_back = True
            else:
                self.pos = action
            reward = -minutes

        self.steps += 1
        done = self.pos == self.goal
        if self.steps >= self.max_steps and not done:
            # took way too long, give up
            reward -= 20
            done = True

        info = DroneStepInfo(blown_back=blown_back, pos_before=pos_before,
                             pos_after=self.pos)
        return self.pos, reward, done, info


# same as epsilon_greedy in rl_utils, state is already an int here
def epsilon_greedy(Q, state, epsilon, rng):
    s = state
    if rng.random() < epsilon:
        return rng.randrange(N_ACTIONS)
    return int(np.argmax(Q[s]))


# same as greedy_action in rl_utils
def greedy_action(Q, state):
    s = state
    return int(np.argmax(Q[s]))


# same as init_Q in rl_utils
def init_Q():
    return np.zeros((N_STATES, N_ACTIONS))


def train_drone(goal, num_episodes=3000, alpha=0.1, gamma=1.0, seed=0,
                eps_start=1.0, eps_end=0.1):
    env = DroneEnv(goal, seed=seed)
    rng = random.Random(seed)
    Q = init_Q()
    episode_returns = []

    for ep in range(num_episodes):
        eps = linear_epsilon_schedule(ep, num_episodes, eps_start, eps_end)
        S = env.reset()
        total_reward = 0

        while True:
            A = epsilon_greedy(Q, S, eps, rng)
            S_next, R, done, _ = env.step(A)
            total_reward += R
            S_next_idx = None if done else S_next
            Q[S, A] = qlearning_update(Q, S, A, R, S_next_idx, alpha, gamma)
            S = S_next
            if done:
                break

        episode_returns.append(total_reward)

    return Q, episode_returns


def best_route(Q, start, goal):
    route = [start]
    here = places.index(start)
    g = places.index(goal)
    while here != g and len(route) < 15:
        here = greedy_action(Q, here)
        route.append(places[here])
    return route


if __name__ == "__main__":
    goal = "RH-5"
    Q, returns = train_drone(goal, num_episodes=10000)

    print("expected minutes to reach", goal, "from each place:")
    for i in range(n):
        if places[i] == goal:
            continue
        nxt = places[greedy_action(Q, i)]
        print(" ", places[i], round(-np.max(Q[i]), 2), "min, next stop", nxt)

    path, d = astar("Dock", goal)
    print("A* route: ", path, d, "m")
    print("MDP route:", best_route(Q, "Dock", goal))