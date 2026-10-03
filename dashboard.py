import json
import random
import numpy as np
import streamlit as st
import plotly.graph_objects as go

import llm_interface
from campus_graph import NODE_POSITIONS, EDGES, astar, trip_battery_cost
from memory_db import init_db, get_bots, get_tasks, add_task, update_bot
from drone_mdp import train_drone, best_route, fail_prob, windy
from perception import make_estimator, fake_gps, places
from conflict_resolver import resolve_conflict

st.set_page_config(page_title="Krea Campus Delivery Drones", layout="wide")
init_db()

# the two halls have one landing pad each
PADS = ["RH-1", "RH-5"]


# q learning is quick but no point redoing it every click
@st.cache_resource
def get_Q(goal):
    Q, returns = train_drone(goal, num_episodes=10000)
    return Q


def read_trace(history):
    # turn the raw chat history from the react loop into steps for the page
    steps = []
    for msg in history[2:]:
        if msg["role"] == "assistant":
            try:
                step = json.loads(llm_interface.strip_fences(msg["content"]))
            except Exception:
                step = {"thought": msg["content"], "action": "?", "action_input": {}}
            step["observation"] = None
            steps.append(step)
        else:
            if steps:
                try:
                    steps[-1]["observation"] = json.loads(msg["content"])
                except Exception:
                    steps[-1]["observation"] = msg["content"]
    return steps


def last_verify(steps):
    # the most recent verifier call and what it said
    for step in reversed(steps):
        if step.get("action") == "verify_delivery":
            return step
    return None


def mdp_leg(start, goal):
    if start == goal:
        return [start]
    return best_route(get_Q(goal), start, goal)


def fly(route, seed):
    # actually fly the route with wind, gps noise and the hmm tracking it
    rng = random.Random(seed)
    np.random.seed(seed)
    est = make_estimator(route[0])
    true_place = route[0]
    used = 0.0
    log = [{"trying": "-", "actual": true_place, "gps": "-", "guess": true_place,
            "conf": 1.0, "blown_back": False, "belief": est.belief_state.copy()}]

    for target in route[1:]:
        tries = 0
        while true_place != target and tries < 6:
            tries += 1
            used += trip_battery_cost([true_place, target])
            blown = rng.random() < fail_prob[(true_place, target)]
            if not blown:
                true_place = target
            reading = fake_gps(true_place)
            belief = est.bayesian_filter_step(places.index(target), reading)
            log.append({
                "trying": target,
                "actual": true_place,
                "gps": places[reading],
                "guess": places[est.get_most_likely_state()],
                "conf": round(float(belief.max()), 2),
                "blown_back": blown,
                "belief": belief.copy(),
            })
    return log, round(used, 2)


def draw_map(astar_path=None, mdp_path=None, belief=None, true_place=None):
    fig = go.Figure()

    # normal edges grey, windy ones red dotted
    for a, b, d in EDGES:
        x1, y1 = NODE_POSITIONS[a]
        x2, y2 = NODE_POSITIONS[b]
        is_windy = (a, b) in windy or (b, a) in windy
        fig.add_trace(go.Scatter(
            x=[x1, x2], y=[y1, y2], mode="lines", hoverinfo="none", showlegend=False,
            line=dict(color="#ef4444" if is_windy else "#475569",
                      width=2, dash="dot" if is_windy else "solid")))

    def add_path(path, color, name, width):
        xs, ys = [], []
        for p in path:
            xs.append(NODE_POSITIONS[p][0])
            ys.append(NODE_POSITIONS[p][1])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name,
                                 line=dict(color=color, width=width)))

    if astar_path:
        add_path(astar_path, "#f97316", "A* (shortest)", 8)
    if mdp_path:
        add_path(mdp_path, "#22c55e", "MDP (fastest on average)", 4)

    names = list(NODE_POSITIONS.keys())
    xs = [NODE_POSITIONS[p][0] for p in names]
    ys = [NODE_POSITIONS[p][1] for p in names]

    if belief is not None:
        # node colour = how sure the hmm is that the drone is there
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text", text=names, textposition="top center",
            marker=dict(size=[16 + 40 * belief[places.index(p)] for p in names],
                        color=[belief[places.index(p)] for p in names],
                        colorscale="YlOrRd", cmin=0, cmax=1, showscale=True,
                        colorbar=dict(title="belief")),
            hovertext=[f"{p}: {belief[places.index(p)]:.2f}" for p in names],
            hoverinfo="text", name="HMM belief"))
    else:
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=names,
                                 textposition="top center", hoverinfo="text",
                                 marker=dict(size=18, color="#3b82f6"), name="Locations"))

    if true_place:
        x, y = NODE_POSITIONS[true_place]
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", name="true position",
                                 marker=dict(symbol="x", size=18, color="#a855f7",
                                             line=dict(width=2))))

    fig.update_layout(
        height=460, margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", y=-0.05),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font=dict(color="white"))
    return fig


# ---------------- sidebar: dispatch form ----------------

drones = get_bots()
drone_names = [d["bot_id"] for d in drones]

with st.sidebar:
    st.header("Dispatch a delivery")
    with st.form("dispatch"):
        request = st.text_area("What should the drone do?",
                               "bring lunch from Kalai to RH-5, floor 2")
        drone_id = st.selectbox("Drone", drone_names)
        payload = st.number_input("Payload (kg)", 0.1, 20.0, 1.0, 0.5)
        answer = st.text_input("If the AI needs to ask you something, answer:",
                               "charge it to 50% first")
        go_btn = st.form_submit_button("Send to AI dispatcher")

    if st.button("Reset drones"):
        update_bot("Drone-1", 85.0, "Dock", "idle")
        update_bot("Drone-2", 62.0, "Dock", "idle")
        st.session_state.pop("res", None)
        st.rerun()


# ---------------- run everything when the form is sent ----------------

if go_btn:
    me = [d for d in drones if d["bot_id"] == drone_id][0]
    full_request = (f"{request}. {drone_id} is at {me['location']}, has "
                    f"{me['battery_pct']}% battery and payload is {payload}kg.")

    # the dashboard cant use input() so the sidebar answer is used instead
    llm_interface.ask_user = lambda q: answer

    with st.spinner("AI dispatcher is thinking..."):
        try:
            final, history = llm_interface.run(llm_interface.LLMClient(), full_request)
            error = None
        except Exception as e:
            final, history, error = None, [], str(e)

    steps = read_trace(history)
    check = last_verify(steps)
    res = {"request": full_request, "drone": drone_id, "steps": steps,
           "final": final, "error": error, "approved": False}

    obs = check["observation"] if check else None
    if final and isinstance(obs, dict) and obs.get("status") == "SATISFIABLE":
        res["approved"] = True
        plan = check["action_input"]
        pickup, dropoff = plan["pickup"], plan["dropoff"]
        start = me["location"]
        # if the ai planned a charge first, use that battery, else the real one
        battery = max(float(plan.get("battery_pct", 0)), me["battery_pct"])
        res["charged"] = battery > me["battery_pct"]

        # A* route vs learned MDP route for the same trip
        a1, _ = astar(start, pickup)
        a2, _ = astar(pickup, dropoff)
        res["astar"] = a1 + a2[1:]
        res["astar_m"] = sum(d for a, b, d in EDGES for i in range(len(res["astar"]) - 1)
                             if {a, b} == {res["astar"][i], res["astar"][i + 1]})
        m1 = mdp_leg(start, pickup)
        m2 = mdp_leg(pickup, dropoff)
        res["mdp"] = m1 + m2[1:]

        # fly the mdp route for real, hmm tracks it from noisy gps
        log, used = fly(res["mdp"], seed=random.randint(0, 10000))
        res["log"] = log
        res["used"] = used
        res["battery_before"] = battery
        res["battery_after"] = round(battery - used, 2)
        res["pickup"], res["dropoff"] = pickup, dropoff

        # landing pad conflict if another drone wants the same pad
        if dropoff in PADS:
            other = [d for d in drones if d["bot_id"] != drone_id][0]
            order, val, scores, safe = resolve_conflict(
                drone_id, res["battery_after"], other["bot_id"], other["battery_pct"])
            res["conflict"] = {"other": other["bot_id"], "other_batt": other["battery_pct"],
                               "order": order, "val": val, "scores": scores, "safe": safe}

        # memory: save the new battery, location and the task
        update_bot(drone_id, res["battery_after"], dropoff, "idle")
        add_task(drone_id, pickup, dropoff, final.get("floor", 1),
                 final.get("item", "item"), "delivered")
    else:
        add_task(drone_id, "-", "-", 0, request[:40], "rejected")

    st.session_state["res"] = res
    st.session_state["step_idx"] = len(res.get("log", [1])) - 1


# ---------------- page ----------------

st.title("Krea Campus Delivery Drones")
res = st.session_state.get("res")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Campus map")
    if res and res["approved"]:
        idx = min(st.session_state.get("step_idx", 0), len(res["log"]) - 1)
        row = res["log"][idx]
        st.plotly_chart(draw_map(res["astar"], res["mdp"], row["belief"], row["actual"]),
                        use_container_width=True)
        st.caption("Red dotted edges are windy. Node colour and size show where the "
                   "HMM thinks the drone is. The purple X is where it really is.")
    else:
        st.plotly_chart(draw_map(), use_container_width=True)

with col2:
    st.subheader("Drone status")
    for d in get_bots():
        b = d["battery_pct"]
        color = "green" if b > 50 else "orange" if b > 25 else "red"
        st.markdown(f"**{d['bot_id']}** at {d['location']} | :{color}[{b:.1f}% battery] | {d['status']}")
    st.subheader("Task log")
    st.dataframe(get_tasks(), hide_index=True, use_container_width=True)

if not res:
    st.info("Type a request in the sidebar to dispatch a drone.")
    st.stop()

st.divider()
tab1, tab2, tab3, tab4 = st.tabs(["AI reasoning", "Routes: A* vs MDP",
                                  "Flight + HMM", "Landing pad conflict"])

with tab1:
    st.markdown(f"**Request sent to the AI:** {res['request']}")
    if res["error"]:
        st.error(res["error"])
    for i, step in enumerate(res["steps"], 1):
        with st.container(border=True):
            st.markdown(f"**Step {i}: `{step.get('action')}`**")
            st.markdown(f"*Thought:* {step.get('thought', '')}")
            if step.get("action_input"):
                st.json(step["action_input"], expanded=False)
            obs = step.get("observation")
            if isinstance(obs, dict) and obs.get("status") == "SATISFIABLE":
                st.success("Z3 verifier: SATISFIABLE. " + obs.get("feedback", ""))
            elif isinstance(obs, dict) and obs.get("status") in ("UNSATISFIABLE", "REJECTED"):
                st.error("Z3 verifier: " + obs.get("feedback", obs.get("reason", "")))
            elif isinstance(obs, dict) and "user_reply" in obs:
                st.warning("You answered: " + obs["user_reply"])
            elif obs is not None:
                st.code(json.dumps(obs) if isinstance(obs, dict) else str(obs))
    if not res["approved"]:
        st.error("Delivery not approved by the verifier, so the drone did not fly.")

if not res["approved"]:
    st.stop()

with tab2:
    a, b = st.columns(2)
    a.metric("A* route (shortest)", f"{res['astar_m']} m")
    a.write(" → ".join(res["astar"]))
    b.metric("MDP route (Q-learning)", f"{len(res['mdp']) - 1} hops")
    b.write(" → ".join(res["mdp"]))
    if res["astar"] != res["mdp"]:
        st.info("The two disagree. A* only looks at distance, the Q-learning drone "
                "learned that the windy edges keep blowing it back, so it avoids them.")
    else:
        st.info("Both routes agree for this trip.")

with tab3:
    if res.get("charged"):
        st.warning(f"The AI's plan assumed the drone was charged to {res['battery_before']}% first.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Battery before", f"{res['battery_before']}%")
    c2.metric("Battery used (incl. wind)", f"{res['used']}%")
    c3.metric("Battery after", f"{res['battery_after']}%")

    st.slider("Flight step (moves the map above)", 0, len(res["log"]) - 1,
              key="step_idx")
    table = []
    for i, row in enumerate(res["log"]):
        table.append({"step": i, "trying to reach": row["trying"],
                      "actually at": row["actual"], "gps says": row["gps"],
                      "HMM best guess": row["guess"], "confidence": row["conf"],
                      "blown back": "yes" if row["blown_back"] else ""})
    st.dataframe(table, hide_index=True, use_container_width=True)

    row = res["log"][st.session_state["step_idx"]]
    bar = go.Figure(go.Bar(x=places, y=row["belief"], marker_color="#f97316"))
    bar.update_layout(height=260, margin=dict(l=0, r=0, t=30, b=0),
                      title="HMM belief at this step", yaxis=dict(range=[0, 1]))
    st.plotly_chart(bar, use_container_width=True)

with tab4:
    c = res.get("conflict")
    if not c:
        st.write(f"{res['dropoff']} has no shared landing pad, so no conflict this time.")
    else:
        st.write(f"**{res['drone']}** ({res['battery_after']}%) and **{c['other']}** "
                 f"({c['other_batt']}%) both want the {res['dropoff']} landing pad.")
        st.write("Minimax (dispatcher = MAX picks the landing order, wind = MIN picks "
                 "calm or gusty landings). Value = battery left in the weakest drone, worst case:")
        st.table([{"choice": k, "worst case battery %": v} for k, v in c["scores"].items()])
        st.success(f"Landing order: {' then '.join(c['order'])}")
        if not c["safe"]:
            st.error("Even the best order leaves a drone under the 15% reserve.")