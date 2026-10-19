"""SENTINEL-EDGE analyst: shift summaries from the event store.

The analyst reads structured events — never raw video. `llm`, when given,
is any callable taking the prompt string and returning text (see
ollama_adapter.py for the local-Ollama wiring). Without one, a
deterministic rules fallback renders the report — including a real
trend-vs-previous-shift comparison and repeat-track listing.
"""

import json

SHIFT_REPORT_PROMPT = """You are a construction-site safety analyst. Given the
JSON compliance summary below, write a 60-second shift report for the foreman:

- Overall transition-compliance rate and trend vs. the previous shift (if given)
- Zones ranked by violations per hour (already sorted, worst first)
- Repeat tracks worth a toolbox talk
- One concrete action for next shift

Be terse. No hedging. Numbers first.

SUMMARY:
{summary_json}

PREVIOUS SHIFT (optional):
{previous_json}
"""

STREAK_RULE = (
    "Escalate immediately: 3+ violation events for the same track_id "
    "within 10 minutes, or any zone exceeding 5 violations/hour. "
    "(Implemented in pipeline._check_streak; escalations are logged "
    "as 'escalation' events.)"
)


def render_shift_report(summary: dict, llm=None,
                        previous: dict | None = None) -> str:
    """Render the report with an LLM callable if provided, else rules."""
    if llm is not None:
        return llm(SHIFT_REPORT_PROMPT.format(
            summary_json=json.dumps(summary, indent=2),
            previous_json=json.dumps(previous, indent=2)
            if previous else "n/a"))
    prev_zones = {z["zone"]: z for z in (previous or {}).get("zones", [])}
    lines = ["SHIFT SAFETY REPORT",
             f"(observed {summary.get('observed_hours', '?')} h)",
             ""]
    for z in summary.get("zones", []):
        pz = prev_zones.get(z["zone"])
        trend = ""
        if pz is not None:
            d = z["violations_per_hour"] - pz["violations_per_hour"]
            trend = (f", {'up' if d > 0 else 'down'} {abs(d):.1f}/hr "
                     f"vs last shift")
        lines.append(
            f"- {z['zone']}: {z['transition_compliance_rate']:.0%} "
            f"transition-compliant, {z['violations_per_hour']:.1f} "
            f"violations/hr ({z['violations']} violations / "
            f"{z['events']} events{trend})")
    reps = summary.get("repeat_tracks", [])
    if reps:
        lines += ["", "Repeat tracks (2+ violations this run):"]
        lines += [f"  - run {r['run_id']} track {r['track_id']}: "
                  f"{r['violations']} violations" for r in reps]
    lines += ["", f"Escalation rule: {STREAK_RULE}"]
    return "\n".join(lines)
