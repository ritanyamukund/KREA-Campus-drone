"""
llm_interface.py -- ReAct execution loop for the Campus Delivery Drone.
Adapted from Lab/agent.py. Loop structure, schema, and helpers unchanged.
""" 

from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from campus_graph import astar, trip_battery_cost, NODE_POSITIONS  
from safety_verifier import verify_delivery_smt                    

load_dotenv()



MAX_TURNS = 12
DEFAULT_MODEL = "gpt-5-mini"  





class AgentStep(BaseModel):
    thought: str
    action: str
    action_input: Dict[str, Any]



# CHANGED: campus delivery test cases


TEST_CASES = {
    "default": (
        "Deliver food from Kalai to RH-5, floor 2. Drone has 80% battery and payload is 1kg."
    ),
    "low_battery": (
        "Deliver books from Dock to RH-1, floor 1. Drone has 18% battery and payload is 2kg."
    ),
    "heavy": (
        "Deliver parcel from NAB to SAZ, floor 3. Drone has 90% battery and payload is 7kg."
    ),
}



# CHANGED: system prompt for campus delivery context


SYSTEM_PROMPT = """
you are a campus delivery drone dispatcher at Krea University.
every turn reply with ONE json object and nothing else. no markdown fences,
no extra text before/after. shape:

{
  "thought": "...",
  "action": "find_path" | "verify_delivery" | "ask_user" | "finish",
  "action_input": { ... }
}

the 4 actions:

1. find_path
   {"pickup": "<location>", "dropoff": "<location>"}
   valid locations: Dock, NAB, JSW, Kalai, Narsi, Bhagat Ji, Library, RH-1, RH-5, SAZ

2. verify_delivery
   {"pickup": "...", "dropoff": "...", "battery_pct": <number>, "payload_kg": <number>, "start": "<where the drone is now>"}
   always call this before finish. it checks the battery reserve and payload limit.

3. ask_user
   {"question": "..."} - only if the fix needs a real human choice (charge first
   vs use another drone, split the order etc), not stuff you can figure out

4. finish
   {"summary": "...", "pickup": "...", "dropoff": "...", "floor": <int>, "item": "..."}

rule 1 - if verify_delivery says UNSATISFIABLE you cannot resend the same
action_input again. read the diagnostics/feedback first, then either send a
genuinely different payload or ask_user if it's a judgment call.

rule 2 - dont call finish saying the delivery works unless the LAST
verify_delivery call for that exact plan said SATISFIABLE. if its impossible
just say that directly using the diagnostic reason, dont be vague.
"""


# JSON helpers (from Lab/agent.py)

def strip_fences(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j != -1:
        t = t[i:j + 1]
    return t


def freeze(action_input: Dict[str, Any]) -> str:
    return json.dumps(action_input, sort_keys=True)


class SeenPayloads:
    def __init__(self) -> None:
        self._seen: set[str] = set()

    def seen(self, action_input: Dict[str, Any]) -> bool:
        return freeze(action_input) in self._seen

    def add(self, action_input: Dict[str, Any]) -> None:
        self._seen.add(freeze(action_input))


def parse_step(raw: str) -> AgentStep:
    cleaned = strip_fences(raw)
    return AgentStep.model_validate_json(cleaned)



# LLM client (from Lab/agent.py)

class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.client = OpenAI()
        self.model = model

    def complete(self, messages: List[Dict[str, str]]) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return r.choices[0].message.content


# CHANGED: campus tool functions 

def find_path(pickup: str, dropoff: str) -> Dict[str, Any]:
    if pickup not in NODE_POSITIONS or dropoff not in NODE_POSITIONS:
        return {"error": f"Unknown location. Valid: {list(NODE_POSITIONS.keys())}"}
    path, dist = astar(pickup, dropoff)
    if path is None:
        return {"error": "No path found"}
    cost = trip_battery_cost(path)
    return {"path": path, "distance_m": dist, "battery_cost_pct": cost}



# ask_user (Lab/agent.py)

def ask_user(question: str) -> str:
    print(f"\nagent asks: {question}")
    if sys.stdin.isatty():
        return input("your answer: ")
    return "proceed with the safest available option"

# ReAct loop (unchanged structure, CHANGED actions)

def run(llm, user_request: str) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_request},
    ]
    seen = SeenPayloads()

    for t in range(MAX_TURNS):
        raw = llm.complete(history)
        history.append({"role": "assistant", "content": raw})

        try:
            step = parse_step(raw)
        except (ValidationError, json.JSONDecodeError) as e:
            history.append({"role": "user", "content": f"bad output, didnt match schema: {e}. send ONLY the json object next time."})
            continue

        act = step.action

        if act == "finish":
            return step.action_input, history

        elif act == "find_path":                          # CHANGED
            obs = find_path(**step.action_input)          # CHANGED
            history.append({"role": "user", "content": json.dumps(obs)})

        elif act == "verify_delivery":                              # CHANGED
            payload = step.action_input
            if seen.seen(payload):
                obs = {"status": "REJECTED", "reason": "already tried this exact payload. think of something actually different."}
            else:
                seen.add(payload)
                obs = verify_delivery_smt(payload)                  # CHANGED
            history.append({"role": "user", "content": json.dumps(obs)})

        elif act == "ask_user":
            reply = ask_user(step.action_input.get("question", ""))
            history.append({"role": "user", "content": json.dumps({"user_reply": reply})})

        else:
            history.append({"role": "user", "content": f"'{act}' isnt one of the 4 actions, use find_path / verify_delivery / ask_user / finish"})

    raise TimeoutError(f"ran out of turns (MAX_TURNS={MAX_TURNS}) without a finish action")


def save_trace(path: str, messages: List[Dict[str, str]]) -> None:
    with open(path, "w") as f:
        json.dump(messages, f, indent=2)
    print(f"Saved trace to {path}")




def main() -> None:
    args = sys.argv[1:]
    case = args[0] if args else "default"          # CHANGED
    if case not in TEST_CASES:
        raise SystemExit(f"Unknown test case '{case}', expected one of {list(TEST_CASES)}")

    user_request = TEST_CASES[case]
    llm = LLMClient()
    trace_path = f"trace_{case}.json"

    result, messages = run(llm, user_request)
    save_trace(trace_path, messages)

    print("\nFinal result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()