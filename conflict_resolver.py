import copy
from campus_graph import MIN_RESERVE

# Landing pad conflict at RH-5, two drones want to land at the same time
# minimax written from the Class 3 adversarial search slides:
#   minimax(t, player):
#     if t is a leaf -> value of t
#     if player is agent -> max of minimax applied to child trees
#     else -> min of minimax applied to child trees
#
# MAX = dispatcher, picks which drone lands first
# MIN = wind, picks calm or gusty for each landing (worst case for us)
# leaf value = battery left in the weakest drone after both have landed

HOVER_DRAIN = 2    # battery % used per minute while in the air
CALM_TIME = 1      # minutes to land with no wind
GUST_TIME = 3      # minutes to land in a gust


def is_leaf(state):
    # both drones are down
    return state["landed"] == 2


def leaf_value(state):
    # how much battery the worst off drone has left
    return min(state["batt"].values())


def children(state, player):
    kids = []
    if player == "dispatcher":
        # dispatcher chooses the landing order
        names = list(state["batt"].keys())
        for first in names:
            for second in names:
                if first != second:
                    kid = copy.deepcopy(state)
                    kid["order"] = [first, second]
                    kids.append(kid)
    else:
        # wind decides how long the next landing takes
        for t in [CALM_TIME, GUST_TIME]:
            kid = copy.deepcopy(state)
            # everyone still in the air burns battery while this one lands
            for name in kid["order"][kid["landed"]:]:
                kid["batt"][name] = kid["batt"][name] - t * HOVER_DRAIN
            kid["landed"] += 1
            kids.append(kid)
    return kids


def minimax(state, player):
    if is_leaf(state):
        return leaf_value(state)

    if player == "dispatcher":
        best = -1000
        for kid in children(state, player):
            val = minimax(kid, "wind")
            if val > best:
                best = val
        return best
    else:
        worst = 1000
        for kid in children(state, player):
            val = minimax(kid, "wind")
            if val < worst:
                worst = val
        return worst


def resolve_conflict(drone1, batt1, drone2, batt2):
    start = {"batt": {drone1: batt1, drone2: batt2}, "order": None, "landed": 0}

    best_order = None
    best_val = -1000
    scores = {}
    # try each landing order and keep the one with the best worst case
    for kid in children(start, "dispatcher"):
        val = minimax(kid, "wind")
        label = kid["order"][0] + " first"
        scores[label] = val
        if val > best_val:
            best_val = val
            best_order = kid["order"]

    # even the best order might leave someone under the reserve
    safe = best_val >= MIN_RESERVE
    return best_order, best_val, scores, safe


if __name__ == "__main__":
    order, val, scores, safe = resolve_conflict("Drone-1", 25, "Drone-2", 70)
    print("worst case battery for each choice:", scores)
    print("land order:", order)
    print("weakest drone ends with", val, "% | safe:", safe)