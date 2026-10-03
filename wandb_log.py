"""
wandb_log.py
============
Adapted from Q learning/compare_policies.py
Trains the drone Q-learning for both halls (RH-1 and RH-5) and makes:
  1. learning curves (episode return vs episode) for both goals
  2. state values V(s) = max_a Q(s, a) for every place on campus

Logs both plots and the raw returns to W&B project "krea-campus-drones".
If wandb isn't installed it still runs and saves the plots locally.

Usage:
    python wandb_log.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drone_mdp import train_drone, places

try:
    import wandb
    HAVE_WANDB = True
except ImportError:
    HAVE_WANDB = False

GOALS = ["RH-1", "RH-5"]
NUM_EPISODES = 10000


def smooth(x, window=100):
    x = np.array(x, dtype=float)
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def main():
    if HAVE_WANDB:
        run = wandb.init(project="krea-campus-drones",
                         config={"goals": GOALS, "num_episodes": NUM_EPISODES,
                                 "windy_fail": 0.5, "calm_fail": 0.1})

    all_returns = {}
    all_values = {}
    for goal in GOALS:
        print("Training Q-learning to", goal, "...")
        Q, returns = train_drone(goal, num_episodes=NUM_EPISODES)
        all_returns[goal] = returns
        # value of each place = best action value there
        values = []
        for s in range(len(places)):
            values.append(np.max(Q[s]))
        all_values[goal] = values

    # also log the raw returns so the curve shows up as a chart in wandb
    if HAVE_WANDB:
        for i in range(NUM_EPISODES):
            row = {"episode": i}
            for goal in GOALS:
                row["return_" + goal] = all_returns[goal][i]
            run.log(row)

    # plot 1: learning curves
    fig, ax = plt.subplots(figsize=(7, 5))
    for goal in GOALS:
        ax.plot(smooth(all_returns[goal]), label="to " + goal)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return (smoothed)")
    ax.set_title("Drone Q-learning: learning curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig("learning_curves.png", dpi=150)
    print("Saved learning_curves.png")

    # plot 2: state values per place (what the prof asked for)
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(places))
    width = 0.4
    ax2.bar(x - width / 2, all_values["RH-1"], width, label="goal RH-1")
    ax2.bar(x + width / 2, all_values["RH-5"], width, label="goal RH-5")
    ax2.set_xticks(x)
    ax2.set_xticklabels(places, rotation=45)
    ax2.set_ylabel("V(s) = max_a Q(s, a)")
    ax2.set_title("Learned state values for each place")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig("state_values.png", dpi=150)
    print("Saved state_values.png")

    if HAVE_WANDB:
        wandb.log({
            "learning_curves": wandb.Image("learning_curves.png"),
            "state_values": wandb.Image("state_values.png"),
        })
        run.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()