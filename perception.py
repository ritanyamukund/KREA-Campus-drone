import numpy as np
from campus_graph import NODE_POSITIONS, EDGES
from hmm_filter import HMMStateEstimator

# list of places, the index of each one is its state number
places = list(NODE_POSITIONS.keys())
n = len(places)

# who is connected to who
neighbours = {}
for p in places:
    neighbours[p] = []
for edge in EDGES:
    a = edge[0]
    b = edge[1]
    neighbours[a].append(b)
    neighbours[b].append(a)

# chance the drone actually reaches the next stop (wind can hold it back)
reach_prob = 0.8

# how bad the gps is, same units as NODE_POSITIONS
gps_noise = 60


def make_transition():
    # action = index of the place the drone is trying to fly to
    T = np.zeros((n, n, n))
    for target in range(n):
        for i in range(n):
            here = places[i]
            there = places[target]
            if there in neighbours[here]:
                T[target, i, target] = reach_prob
                T[target, i, i] = 1 - reach_prob
            else:
                # cant fly there directly so it just stays where it is
                T[target, i, i] = 1.0
    return T


def make_emission():
    # E[gps reading, true place]
    E = np.zeros((n, n))
    for s in range(n):
        x1, y1 = NODE_POSITIONS[places[s]]
        for k in range(n):
            x2, y2 = NODE_POSITIONS[places[k]]
            # times 100 to get metres, same as heuristic() in campus_graph
            d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2) * 100
            E[k, s] = np.exp(-(d**2) / (2 * gps_noise**2))
        # each column should add up to 1
        E[:, s] = E[:, s] / np.sum(E[:, s])
    return E


T = make_transition()
E = make_emission()


def fake_gps(true_place):
    # pick a noisy reading based on where the drone really is
    s = places.index(true_place)
    k = np.random.choice(n, p=E[:, s])
    return k


def make_estimator(start_place=None):
    est = HMMStateEstimator(n, T, E)
    # if we know where it took off from, start the belief there
    if start_place is not None:
        est.belief_state = np.zeros(n)
        est.belief_state[places.index(start_place)] = 1.0
    return est


if __name__ == "__main__":
    np.random.seed(1)
    route = ["Dock", "Library", "RH-1", "RH-5"]
    est = make_estimator("Dock")
    true_place = "Dock"

    for nxt in route[1:]:
        action = places.index(nxt)
        # does the drone actually make it this time
        if nxt in neighbours[true_place] and np.random.rand() < reach_prob:
            true_place = nxt
        reading = fake_gps(true_place)
        belief = est.bayesian_filter_step(action, reading)
        guess = places[est.get_most_likely_state()]
        print("trying to reach", nxt, "| actually at", true_place,
              "| gps says", places[reading], "| best guess", guess,
              "|", round(belief.max(), 2))