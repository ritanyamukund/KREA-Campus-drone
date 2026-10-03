import heapq
import math
from typing import Optional

# Adapted from Lab1/planner.py (ForkliftPlanner.a_star)
# Domain changed: grid (x,y) coords -> named campus nodes, edge weights in metres

DRAIN_RATE = 0.005
MIN_RESERVE = 15

NODE_POSITIONS = {
    "Dock":      (0.0, 2.0),
    "NAB":       (1.5, 3.0),
    "JSW":       (3.0, 3.5),
    "Kalai":     (4.5, 2.5),
    "Narsi":     (5.0, 2.0),
    "Bhagat Ji": (5.5, 1.5),
    "Library":   (3.5, 1.0),
    "RH-1":      (6.0, 3.0),
    "RH-5":      (6.5, 1.5),
    "SAZ":       (5.0, 0.5),
}

EDGES = [
    ("Dock",      "NAB",       180),
    ("Dock",      "Library",   364),
    ("NAB",       "JSW",       158),
    ("NAB",       "Kalai",     304),
    ("JSW",       "Library",   255),
    ("JSW",       "Kalai",     180),
    ("Kalai",     "Narsi",      71),
    ("Narsi",     "Bhagat Ji",  71),
    ("Bhagat Ji", "SAZ",       112),
    ("SAZ",       "RH-5",      180),
    ("SAZ",       "RH-1",      269),
    ("RH-1",      "RH-5",      158),
    ("Library",   "RH-1",      320),
    ("Kalai",     "SAZ",       206),
]

def build_graph(edges):
    graph = {}
    for a, b, dist in edges:
        graph.setdefault(a, []).append((b, dist))
        graph.setdefault(b, []).append((a, dist))
    return graph

GRAPH = build_graph(EDGES)

def heuristic(node_a, node_b):
    x1, y1 = NODE_POSITIONS[node_a]
    x2, y2 = NODE_POSITIONS[node_b]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * 100

def astar(start, goal):
    if start not in GRAPH or goal not in GRAPH:
        return None, float('inf')

    open_set = [(0, start)]                          # CHANGED: removed path-in-tuple
    g_scores = {start: 0}
    parents = {start: None}                          # CHANGED: parent dict like Lab1
    open_dict = {start: 0}                           # CHANGED: staleness check like Lab1

    while open_set:
        f, current = heapq.heappop(open_set)

        # stale node check (same as Lab1)
        if open_dict.get(current, float('inf')) < g_scores.get(current, float('inf')):
            continue

        if current == goal:
            # reconstruct by walking parent pointers then reversing (same as Lab1)
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parents[node]
            path.reverse()
            return path, g_scores[goal]

        for neighbour, dist in GRAPH.get(current, []):
            tentative_g = g_scores[current] + dist
            if tentative_g < g_scores.get(neighbour, float('inf')):
                parents[neighbour] = current
                g_scores[neighbour] = tentative_g
                f_score = tentative_g + heuristic(neighbour, goal)
                heapq.heappush(open_set, (f_score, neighbour))
                open_dict[neighbour] = tentative_g

    return None, float('inf')

def trip_battery_cost(path):
    total = 0
    for i in range(len(path) - 1):
        for neighbour, dist in GRAPH[path[i]]:
            if neighbour == path[i + 1]:
                total += dist * DRAIN_RATE
                break
    return round(total, 2)