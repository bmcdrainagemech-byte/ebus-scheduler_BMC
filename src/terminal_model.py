"""
terminal_model.py — Concurrent-occupancy model for shared bus terminals.

Built to surface the GANGAJALIYA accumulation problem: dispatchers report
buses piling up at certain terminals during the day, exceeding the available
parking. This module:

  1. Auto-detects the top-N most-shared terminals across all routes in a
     CitySchedule (terminals that appear as start_point or end_point on the
     largest number of routes).
  2. Simulates concurrent bus occupancy at each shared terminal, in 10-min
     bins, by walking each bus's trip history and computing arrival/departure
     pairs.
  3. Reports peak count, average count, dwell-time stats, and a list of
     time bins where occupancy exceeded a configurable capacity threshold.

API mirrors `depot_model.py` so the dashboard can consume both consistently.

Usage:
    from src.terminal_model import auto_detect_shared_terminals, simulate_terminals
    terms = auto_detect_shared_terminals(city_schedule, top_n=2)
    logs  = simulate_terminals(city_schedule, terminals=terms,
                               capacity_threshold=8, bin_minutes=10)
"""

from __future__ import annotations
__version__ = "2026-04-27-p1"

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta


# ── Output dataclass ─────────────────────────────────────────────────────────

@dataclass
class TerminalLog:
    """Concurrent-occupancy simulation output for one terminal."""
    terminal_name: str
    routes_using: list[str]                              # route codes that touch this terminal
    capacity_threshold: int                              # configurable, default 8

    # Concurrency metrics
    peak_concurrent: int = 0
    peak_time_bin: str = ""                              # "HH:MM" of the peak
    avg_concurrent: float = 0.0
    capacity_exceeded_bins: int = 0                      # count of bins where count > threshold
    excess_windows: list[str] = field(default_factory=list)  # time bin labels exceeding threshold

    # Dwell-time metrics — per-visit (one visit = one arrival→departure pair)
    visits: int = 0
    avg_dwell_min: float = 0.0
    peak_dwell_min: float = 0.0

    # Full profile for plotting
    occupancy_profile: list[dict] = field(default_factory=list)
    # Each dict: {"time_bin": "HH:MM", "count": int, "buses": list[str]}

    @property
    def has_capacity_breach(self) -> bool:
        return self.capacity_exceeded_bins > 0


# ── Auto-detect ──────────────────────────────────────────────────────────────

def auto_detect_shared_terminals(city_schedule, top_n: int = 2) -> list[str]:
    """
    Return the top-N terminal names by route-frequency.

    Counts how many routes use each location as start_point or end_point.
    Ties broken by alphabetical order for stability.

    Excludes the depot — the depot has its own model in `depot_model.py`.
    """
    counter: Counter = Counter()
    depots = set()

    for code, rr in city_schedule.results.items():
        cfg = rr.config
        if getattr(cfg, "depot", None):
            depots.add(cfg.depot)
        if cfg.start_point:
            counter[cfg.start_point] += 1
        if cfg.end_point:
            counter[cfg.end_point] += 1

    # Drop depots from candidates
    for d in depots:
        counter.pop(d, None)

    # Sort by (-count, name) so ties are alphabetical for stable output
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return [name for name, _ in ranked[:top_n]]


def routes_using_terminal(city_schedule, terminal_name: str) -> list[str]:
    """Return sorted list of route codes that use this terminal as start or end point."""
    out = []
    for code, rr in city_schedule.results.items():
        cfg = rr.config
        if cfg.start_point == terminal_name or cfg.end_point == terminal_name:
            out.append(code)
    return sorted(out)


# ── Visit extraction ─────────────────────────────────────────────────────────

def _extract_visits_for_terminal(city_schedule, terminal_name: str) -> list[dict]:
    """
    Walk every bus on every route. Emit one record per terminal visit:
      {bus_id, route, arrive, depart, dwell_min}

    A visit is defined as: bus arrives at the terminal (a trip ends there) and
    later departs (the next trip starts there). If the next trip is the same
    location (turnaround), it counts as a visit. End-of-day stays without a
    subsequent departure are clipped to operating_end.
    """
    visits: list[dict] = []

    for code, rr in city_schedule.results.items():
        for bus in rr.buses:
            trips = [t for t in bus.trips if t.actual_arrival is not None]
            if not trips:
                continue
            # Sort by arrival just to be safe
            trips_sorted = sorted(trips, key=lambda t: t.actual_arrival)

            for i, t in enumerate(trips_sorted):
                if t.end_location != terminal_name:
                    continue
                arrive = t.actual_arrival
                # Find next trip that *departs* from this terminal
                depart = None
                for nxt in trips_sorted[i + 1:]:
                    if nxt.actual_departure is None:
                        continue
                    if nxt.start_location == terminal_name:
                        depart = nxt.actual_departure
                        break
                    # If the next trip starts somewhere else, the bus left this
                    # terminal at some unobserved point — treat it as having
                    # left at the next departure time.
                    depart = nxt.actual_departure
                    break

                if depart is None:
                    # Bus stayed here through end-of-day — clip to operating_end
                    op_end = rr.config.operating_end
                    depart = arrive.replace(hour=op_end.hour, minute=op_end.minute,
                                             second=0, microsecond=0)
                    if depart < arrive:
                        depart = arrive  # safety

                dwell_min = max(0.0, (depart - arrive).total_seconds() / 60.0)
                visits.append({
                    "bus_id": f"{code}-{bus.bus_id}",
                    "route": code,
                    "arrive": arrive,
                    "depart": depart,
                    "dwell_min": dwell_min,
                })

    return visits


# ── Simulation ───────────────────────────────────────────────────────────────

def _bin_label(t: datetime) -> str:
    """Round down to a 10-min bin label."""
    return t.strftime("%H:%M")


def _floor_to_bin(t: datetime, bin_minutes: int) -> datetime:
    """Floor a datetime to the nearest bin boundary."""
    minute = (t.minute // bin_minutes) * bin_minutes
    return t.replace(minute=minute, second=0, microsecond=0)


def simulate_terminal(city_schedule, terminal_name: str,
                       capacity_threshold: int = 8,
                       bin_minutes: int = 10) -> TerminalLog:
    """
    Build a TerminalLog for one terminal.

    Algorithm:
      1. Collect every visit (arrive/depart pair) at this terminal.
      2. Build the operational time range from min(arrive) to max(depart).
      3. For each 10-min bin, count how many visits overlap the bin.
      4. Aggregate peak / avg / threshold-exceeded stats.
    """
    visits = _extract_visits_for_terminal(city_schedule, terminal_name)
    routes = routes_using_terminal(city_schedule, terminal_name)

    log = TerminalLog(
        terminal_name=terminal_name,
        routes_using=routes,
        capacity_threshold=capacity_threshold,
    )

    if not visits:
        return log

    # Time range
    range_start = _floor_to_bin(min(v["arrive"] for v in visits), bin_minutes)
    range_end = max(v["depart"] for v in visits)
    if range_end <= range_start:
        return log

    # Bin walk
    profile: list[dict] = []
    counts_running: list[int] = []
    cur = range_start
    while cur < range_end:
        nxt = cur + timedelta(minutes=bin_minutes)
        # Count visits whose [arrive, depart) overlaps [cur, nxt)
        present = [v for v in visits if v["arrive"] < nxt and v["depart"] > cur]
        cnt = len(present)
        profile.append({
            "time_bin": _bin_label(cur),
            "count": cnt,
            "buses": sorted(v["bus_id"] for v in present),
        })
        counts_running.append(cnt)
        cur = nxt

    log.occupancy_profile = profile

    # Aggregate metrics
    peak_idx = max(range(len(counts_running)), key=lambda i: counts_running[i])
    log.peak_concurrent = counts_running[peak_idx]
    log.peak_time_bin = profile[peak_idx]["time_bin"]
    log.avg_concurrent = round(sum(counts_running) / len(counts_running), 2)

    excess = [p for p in profile if p["count"] > capacity_threshold]
    log.capacity_exceeded_bins = len(excess)
    log.excess_windows = [p["time_bin"] for p in excess]

    # Dwell stats
    log.visits = len(visits)
    log.avg_dwell_min = round(sum(v["dwell_min"] for v in visits) / len(visits), 1)
    log.peak_dwell_min = round(max(v["dwell_min"] for v in visits), 1)

    return log


def simulate_terminals(city_schedule,
                        terminals: list[str] | None = None,
                        capacity_threshold: int = 8,
                        bin_minutes: int = 10,
                        top_n: int = 2) -> list[TerminalLog]:
    """
    Build TerminalLog for each terminal in `terminals`, or auto-detect top-N
    if not provided.

    Returns a list ordered the same as the input (or by detection order).
    """
    if terminals is None:
        terminals = auto_detect_shared_terminals(city_schedule, top_n=top_n)
    return [
        simulate_terminal(city_schedule, t,
                          capacity_threshold=capacity_threshold,
                          bin_minutes=bin_minutes)
        for t in terminals
    ]
