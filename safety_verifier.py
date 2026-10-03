"""
safety_verifier.py -- Z3 SMT verifier for the campus delivery drone.
Adapted from Lab/verifier.py. Same shape as verify_dispatch_smt:
pydantic validation -> z3 solver -> SATISFIABLE or UNSATISFIABLE with diagnostics.
Domain changed: AGV package assignment -> one drone, one delivery.
"""

from __future__ import annotations

from typing import Any, Dict

import z3
from pydantic import BaseModel, Field

from campus_graph import astar, trip_battery_cost, MIN_RESERVE

MAX_PAYLOAD = 5   # kg


class DeliverySpec(BaseModel):
    pickup: str
    dropoff: str
    battery_pct: float = Field(ge=0, le=100)
    payload_kg: float = Field(gt=0)
    start: str = "Dock"   # where the drone is right now


def verify_delivery_smt(raw_payload: dict) -> Dict[str, Any]:
    try:
        spec = DeliverySpec.model_validate(raw_payload)
    except Exception as e:
        return {"status": "SYNTACTIC_ERROR", "message": str(e)}

    # drone has to fly to the pickup first, then to the dropoff
    leg1, d1 = astar(spec.start, spec.pickup)
    leg2, d2 = astar(spec.pickup, spec.dropoff)
    if leg1 is None or leg2 is None:
        return {"status": "SYNTACTIC_ERROR", "message": "no route between those locations"}

    full_path = leg1 + leg2[1:]
    trip_cost = trip_battery_cost(full_path)

    solver = z3.Solver()
    battery = z3.Real("battery")
    cost = z3.Real("trip_cost")
    payload = z3.Real("payload")

    # tie the variables to the real numbers from this request
    solver.add(battery == spec.battery_pct)
    solver.add(cost == trip_cost)
    solver.add(payload == spec.payload_kg)

    # 1. battery left after the trip has to stay above the reserve
    solver.add(battery - cost >= MIN_RESERVE)

    # 2. payload has to be under the limit
    solver.add(payload <= MAX_PAYLOAD)

    if solver.check() == z3.sat:
        return {
            "status": "SATISFIABLE",
            "path": full_path,
            "trip_cost_pct": trip_cost,
            "battery_after": round(spec.battery_pct - trip_cost, 2),
            "feedback": "Delivery verified mathematically.",
        }

    # ---- UNSAT: work out which rule broke ----
    battery_after = round(spec.battery_pct - trip_cost, 2)
    battery_bad = battery_after < MIN_RESERVE
    payload_bad = spec.payload_kg > MAX_PAYLOAD

    diagnostics = {
        "trip_cost_pct": trip_cost,
        "battery_after": battery_after,
        "min_reserve": MIN_RESERVE,
        "payload_kg": spec.payload_kg,
        "max_payload_kg": MAX_PAYLOAD,
    }

    if battery_bad and payload_bad:
        feedback = (
            f"UNSAT: both rules fail. battery would drop to {battery_after}% "
            f"(needs >= {MIN_RESERVE}%) and payload {spec.payload_kg}kg is over "
            f"the {MAX_PAYLOAD}kg limit. charge first and split the load."
        )
    elif battery_bad:
        feedback = (
            f"UNSAT: this trip costs {trip_cost}% so battery would end at "
            f"{battery_after}%, under the {MIN_RESERVE}% reserve. charge the drone "
            f"or use a drone with more battery."
        )
    else:
        feedback = (
            f"UNSAT: payload {spec.payload_kg}kg is over the {MAX_PAYLOAD}kg limit. "
            f"split it into smaller trips or drop part of it."
        )

    return {
        "status": "UNSATISFIABLE",
        "diagnostics": diagnostics,
        "feedback": feedback,
    }


if __name__ == "__main__":
    import json

    ok_case = {"pickup": "Kalai", "dropoff": "RH-5", "battery_pct": 80, "payload_kg": 1}
    low_battery = {"pickup": "Dock", "dropoff": "RH-1", "battery_pct": 18, "payload_kg": 2}
    too_heavy = {"pickup": "NAB", "dropoff": "SAZ", "battery_pct": 90, "payload_kg": 7}

    print("OK case:")
    print(json.dumps(verify_delivery_smt(ok_case), indent=2))
    print("\nLow battery case:")
    print(json.dumps(verify_delivery_smt(low_battery), indent=2))
    print("\nToo heavy case:")
    print(json.dumps(verify_delivery_smt(too_heavy), indent=2))