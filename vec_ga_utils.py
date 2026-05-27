
from collections import defaultdict

def build_resources_by_act(log):
    res_by_act = defaultdict(set)

    for trace in log:
        for ev in trace:
            act = ev["concept:name"]
            res = ev.get("org:resource", "other")

            if res is None:
                res = "other"

            res_by_act[act].add(str(res))

    return dict(res_by_act)

def event_plausibility(event, resources_by_act):
    act = event["concept:name"]
    res = event.get("org:resource", "other")

    if res is None:
        res = "other"

    score = 0.0

    # Activity exists
    if act in resources_by_act:
        score += 1.0
    else:
        return -2.0   # impossible activity

    # Resource matches observed
    if res in resources_by_act[act]:
        score += 1.0
    else:
        score -= 1.0

    return score

def trace_plausibility(trace, resources_by_act):
    if len(trace["events"]) == 0:
        return -5

    scores = [
        event_plausibility(ev, resources_by_act)
        for ev in trace["events"]
    ]

    return sum(scores) / len(scores)
