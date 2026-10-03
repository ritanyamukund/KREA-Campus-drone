"""
packbot_env.py
================
The PackBot environment (PROVIDED — do not modify).

A robotic gripper at a warehouse pick station processes a queue of items,
one per timestep. For each item it must choose how to grasp it:

    FAST_GRASP    (action 0) - quick, good throughput reward, but risk of
                                dropping/damaging the item. Risk rises sharply
                                with the gripper's accumulated WEAR.
    CAREFUL_GRASP (action 1) - slow, small reward, very low drop risk.
                                Also RELIEVES wear.

State:  (item_type, wear)
    item_type in {0, 1, 2}   -- 0: Standard, 1: Fragile, 2: Bulky
    wear      in {0, ..., WEAR_MAX}

This is a genuinely *sequential* control problem, not a bandit: the wear
level carries over from one item to the next, so today's grasp choice
changes tomorrow's risk. There is no notion of position or destination —
nothing here is a pathfinding / navigation problem.

Episode = one shift = ITEMS_PER_EPISODE items processed back-to-back.
"""

import random
from dataclasses import dataclass


N_ITEM_TYPES = 3           # 0=Standard, 1=Fragile, 2=Bulky
ITEM_NAMES = ["Standard", "Fragile", "Bulky"]

WEAR_MAX = 10               # wear ranges 0..WEAR_MAX inclusive
N_WEAR_LEVELS = WEAR_MAX + 1

FAST_GRASP = 0
CAREFUL_GRASP = 1
N_ACTIONS = 2
ACTION_NAMES = ["FastGrasp", "CarefulGrasp"]

ITEMS_PER_EPISODE = 30

# Drop-probability model. Ordinary (non-breakdown) drop risk is small and
# does NOT scale with wear, wear's only real danger is the cliff-edge
# breakdown risk defined below. This keeps the "when do I start being
# careful" decision cleanly attributable to the cliff-edge mechanism
# rather than a second, confounding source of risk.
BASE_FAST_DROP = 0.03        # drop prob for FastGrasp, wear-independent
FRAGILITY_BONUS = [0.00, 0.05, 0.02]   # added to FastGrasp drop prob, by item_type
BASE_CAREFUL_DROP = 0.01    # drop prob for CarefulGrasp (wear-independent)

# Wear dynamics
FAST_WEAR_INC = 2           # wear increase after a FastGrasp
CAREFUL_WEAR_DEC = 3        # wear decrease after a CarefulGrasp
DROP_EXTRA_WEAR = 1         # a mishandled item stresses the gripper further

# Rewards
FAST_SUCCESS_REWARD = 3.0
CAREFUL_SUCCESS_REWARD = 1.0
DROP_PENALTY = -5.0         # applied on top of the (absent) success reward

# --- The "cliff edge": a breakdown risk that exists ONLY at maximum wear.
# FastGrasp-ing a gripper that is already maxed out on wear can jam the
# mechanism entirely: the shift ends immediately (all remaining items in
# the queue are lost) with a large penalty. CarefulGrasp is always safe.
# This mirrors Cliff Walking's cliff: a catastrophe reachable from one
# specific boundary state, which an on-policy learner will build a safety
# margin against and an off-policy learner will happily approach.
BREAKDOWN_PROB_AT_MAX_WEAR = 0.85
BREAKDOWN_PENALTY = -30.0


def state_index(item_type: int, wear: int) -> int:
    """Map (item_type, wear) -> a single integer index, for tabular Q tables."""
    return item_type * N_WEAR_LEVELS + wear


def index_to_state(idx: int):
    """Inverse of state_index."""
    item_type, wear = divmod(idx, N_WEAR_LEVELS)
    return item_type, wear


N_STATES = N_ITEM_TYPES * N_WEAR_LEVELS


@dataclass
class StepInfo:
    dropped: bool
    item_type: int
    wear_before: int
    wear_after: int


class PackBotEnv:
    """
    A minimal Gym-style environment. No pathfinding, no coordinates:
    state is (item_type, wear); action is FAST_GRASP or CAREFUL_GRASP.
    """

    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)
        self.wear = 0
        self.item_type = None
        self.items_done = 0
        self.reset()

    def seed(self, seed: int):
        self.rng = random.Random(seed)

    def _draw_item(self):
        self.item_type = self.rng.randrange(N_ITEM_TYPES)

    def reset(self):
        self.wear = 0
        self.items_done = 0
        self._draw_item()
        return (self.item_type, self.wear)

    def _drop_probability(self, action: int) -> float:
        if action == FAST_GRASP:
            p = BASE_FAST_DROP + FRAGILITY_BONUS[self.item_type]
        else:
            p = BASE_CAREFUL_DROP
        return min(max(p, 0.0), 0.95)

    def step(self, action: int):
        assert action in (FAST_GRASP, CAREFUL_GRASP)
        wear_before = self.wear

        # The cliff edge: FastGrasp-ing at max wear risks an immediate,
        # shift-ending breakdown.
        if action == FAST_GRASP and wear_before == WEAR_MAX:
            if self.rng.random() < BREAKDOWN_PROB_AT_MAX_WEAR:
                info = StepInfo(dropped=True, item_type=self.item_type,
                                 wear_before=wear_before, wear_after=wear_before)
                return (self.item_type, self.wear), BREAKDOWN_PENALTY, True, info

        p_drop = self._drop_probability(action)
        dropped = self.rng.random() < p_drop

        if action == FAST_GRASP:
            reward = DROP_PENALTY if dropped else FAST_SUCCESS_REWARD
            self.wear = min(WEAR_MAX, self.wear + FAST_WEAR_INC)
        else:
            reward = DROP_PENALTY if dropped else CAREFUL_SUCCESS_REWARD
            self.wear = max(0, self.wear - CAREFUL_WEAR_DEC)

        if dropped:
            self.wear = min(WEAR_MAX, self.wear + DROP_EXTRA_WEAR)

        info = StepInfo(dropped=dropped, item_type=self.item_type,
                         wear_before=wear_before, wear_after=self.wear)

        self.items_done += 1
        done = self.items_done >= ITEMS_PER_EPISODE

        prev_item_type = self.item_type
        if not done:
            self._draw_item()

        next_state = (self.item_type if not done else prev_item_type, self.wear)
        return next_state, reward, done, info
