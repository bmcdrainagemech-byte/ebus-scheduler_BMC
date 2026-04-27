"""
recommender.py — Actionable recommendation engine for eBus Scheduler.

Analyses a completed CitySchedule and produces prioritised, actionable
recommendations. Each recommendation has a category, action, reason,
expected impact, and confidence level.

This is the primary product differentiator — metrics without actions are
not useful to planners.

Usage:
    from src.recommender import generate_recommendations, Recommendation
    recs = generate_recommendations(city_schedule)
    for r in recs[:5]:
        print(f"[P{r.priority}] {r.action} — {r.reason}")
"""

from __future__ import annotations
__version__ = "2026-04-17-p2"

from dataclasses import dataclass, field


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Recommendation:
    """Single actionable recommendation."""
    category: str           # fleet_adjustment | headway_change | charging_window |
                            # depot_infrastructure | corridor_coordination | soc_risk
    priority: int           # 1 = safety/compliance, 5 = minor optimisation
    route_codes: list[str]  # affected routes (empty = citywide)
    action: str             # what to do (imperative, specific)
    reason: str             # why (data-driven explanation)
    expected_impact: str    # what changes if action is taken
    confidence: str         # "high" | "medium" | "low"

    def __repr__(self):
        routes = ", ".join(self.route_codes) if self.route_codes else "citywide"
        return f"[P{self.priority}|{self.confidence}] {self.category}: {self.action} ({routes})"


# ── LOS thresholds per route category ─────────────────────────────────────────

_LOS_TARGETS = {
    "trunk":    "B",    # trunk routes should achieve LOS A or B
    "standard": "C",    # standard routes should achieve LOS B or C
    "feeder":   "D",    # feeder routes can tolerate LOS C or D
}

_GRADE_NUM = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}


def _grade_below(actual: str, target: str) -> bool:
    """True if actual LOS grade is worse than target."""
    return _GRADE_NUM.get(actual, 6) > _GRADE_NUM.get(target, 3)


# ── Main entry point ─────────────────────────────────────────────────────────

def generate_recommendations(city_schedule) -> list[Recommendation]:
    """
    Analyse CitySchedule and return prioritised recommendations.

    Checks (in priority order):
      P1 — SOC risk (bus near floor during revenue service)
      P2 — Headway infeasibility (physics > configured)
      P2 — Fleet deficit (allocated < PVR)
      P3 — LOS below target for route category
      P3 — Fleet surplus/deficit transfer opportunities
      P4 — Depot charger utilisation
      P5 — Minor optimisations (km balance, dead-km)
      P3 — min_km_per_bus policy violation (Phase D)
      P2 — charging trips after p5_charging_end (Phase D)
    """
    recs: list[Recommendation] = []

    _check_soc_risk(city_schedule, recs)
    _check_headway_infeasibility(city_schedule, recs)
    _check_fleet_deficit(city_schedule, recs)
    _check_los_grade(city_schedule, recs)
    _check_fleet_transfers(city_schedule, recs)
    _check_depot_utilisation(city_schedule, recs)
    _check_charging_window(city_schedule, recs)
    _check_km_balance(city_schedule, recs)
    _check_dead_km(city_schedule, recs)
    _check_min_km_per_bus(city_schedule, recs)        # Phase D #4, #8
    _check_p5_window_charging(city_schedule, recs)    # Phase D Bug #1 signal

    # Sort by priority (lowest = most critical)
    recs.sort(key=lambda r: (r.priority, r.category))
    return recs


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_soc_risk(cs, recs: list):
    """P1: Any bus within 5% of SOC floor = safety risk."""
    for code, r in cs.results.items():
        margin = r.metrics.min_soc_seen - r.config.min_soc_percent
        if margin < 5:
            trigger = r.config.trigger_soc_percent
            recs.append(Recommendation(
                category="soc_risk",
                priority=1,
                route_codes=[code],
                action=f"Increase trigger_soc from {trigger:.0f}% to {trigger + 5:.0f}% "
                       f"or add terminal charger on {code}",
                reason=f"Bus reached {r.metrics.min_soc_seen:.1f}% SOC "
                       f"(floor={r.config.min_soc_percent}%, margin only {margin:.1f}%)",
                expected_impact=f"Prevents SOC floor violation. "
                               f"Earlier charging trigger adds ~1 extra charge stop per day.",
                confidence="high",
            ))


def _check_headway_infeasibility(cs, recs: list):
    """P2: Headway configured below physics minimum."""
    for code, r in cs.results.items():
        status = getattr(r, "headway_feasibility_status", "UNKNOWN")
        if status != "INFEASIBLE":
            continue
        details = getattr(r, "headway_feasibility_details", []) or []
        for d in details:
            if not isinstance(d, dict) or d.get("status") != "INFEASIBLE":
                continue
            cfg_hw = d.get("cfg_hw", "?")
            phys_min = d.get("physics_min", "?")
            band = d.get("band", "?")
            rec_hw = d.get("rec", phys_min)
            recs.append(Recommendation(
                category="headway_change",
                priority=2,
                route_codes=[code],
                action=f"Increase {code} headway in band {band} from {cfg_hw} to {rec_hw} min",
                reason=f"Configured headway ({cfg_hw} min) is below physics minimum "
                       f"({phys_min} min). Scheduler silently uses {phys_min} min.",
                expected_impact="Eliminates silent headway override. "
                               "Dashboard shows actual scheduled headway.",
                confidence="high",
            ))


def _check_fleet_deficit(cs, recs: list):
    """P2: Route allocated fewer buses than PVR."""
    for code, r in cs.results.items():
        if r.fleet_allocated < r.pvr:
            deficit = r.pvr - r.fleet_allocated
            recs.append(Recommendation(
                category="fleet_adjustment",
                priority=2,
                route_codes=[code],
                action=f"Add {deficit} bus(es) to {code} "
                       f"(current: {r.fleet_allocated}, PVR: {r.pvr})",
                reason=f"Fleet ({r.fleet_allocated}) is below Peak Vehicle Requirement ({r.pvr}). "
                       f"Service coverage will be degraded.",
                expected_impact=f"Meets PVR. Headway gaps should reduce by ~{deficit * 15:.0f}%.",
                confidence="high",
            ))


def _check_los_grade(cs, recs: list):
    """P3: LOS grade below target for route category."""
    for code, r in cs.results.items():
        grade = getattr(r.metrics, "los_grade", "")
        if not grade:
            continue
        category = getattr(r.config, "route_category", "standard")
        target = _LOS_TARGETS.get(category, "C")
        if _grade_below(grade, target):
            cv = r.metrics.headway_cv
            max_gap = r.metrics.max_headway_gap_min
            # Suggest specific action based on gap analysis
            if r.fleet_allocated <= r.pvr:
                action = (f"Add 1–2 buses to {code} to improve LOS from {grade} to {target} "
                         f"(current fleet: {r.fleet_allocated})")
                impact = "Additional bus reduces headway CV and max gap."
            else:
                action = (f"Review {code} headway profile — CV={cv:.2f}, max gap={max_gap:.0f} min. "
                         f"Consider tightening peak headway or adjusting charging window.")
                impact = "Better headway regularity improves passenger wait times."
            recs.append(Recommendation(
                category="fleet_adjustment",
                priority=3,
                route_codes=[code],
                action=action,
                reason=f"LOS {grade} is below target {target} for {category} route. "
                       f"Headway CV={cv:.3f}, max gap={max_gap:.0f} min.",
                expected_impact=impact,
                confidence="medium",
            ))


def _check_fleet_transfers(cs, recs: list):
    """P3: Surplus on one route + deficit on another = transfer opportunity."""
    surplus_routes = []
    deficit_routes = []
    for code, r in cs.results.items():
        if r.fleet_allocated > r.pvr + 1:  # more than 1 extra
            surplus_routes.append((code, r.fleet_allocated - r.pvr))
        elif r.fleet_allocated < r.pvr:
            deficit_routes.append((code, r.pvr - r.fleet_allocated))

    if surplus_routes and deficit_routes:
        for s_code, s_count in surplus_routes:
            for d_code, d_count in deficit_routes:
                transfer = min(s_count, d_count)
                recs.append(Recommendation(
                    category="fleet_adjustment",
                    priority=3,
                    route_codes=[s_code, d_code],
                    action=f"Transfer {transfer} bus(es) from {s_code} to {d_code}",
                    reason=f"{s_code} has {s_count} surplus (fleet={cs.results[s_code].fleet_allocated}, "
                           f"PVR={cs.results[s_code].pvr}). "
                           f"{d_code} has {d_count} deficit.",
                    expected_impact=f"{d_code} meets PVR. {s_code} retains "
                                   f"{cs.results[s_code].fleet_allocated - transfer} buses (still ≥ PVR).",
                    confidence="high",
                ))


def _check_depot_utilisation(cs, recs: list):
    """P4: Depot charger utilisation warnings."""
    # Check from depot_log if available (Phase 2 depot_model integration)
    for code, r in cs.results.items():
        depot_log = getattr(r, "depot_log", None)
        if depot_log is None:
            continue
        util = getattr(depot_log, "utilisation_pct_slow", 0) or 0
        peak_q = getattr(depot_log, "peak_queue_depth_slow", 0) or 0
        if util > 85:
            recs.append(Recommendation(
                category="depot_infrastructure",
                priority=4,
                route_codes=[code],
                action=f"Add 1 charger slot at depot (utilisation={util:.0f}%, "
                       f"peak queue={peak_q} buses)",
                reason=f"Depot charger utilisation exceeds 85%. "
                       f"{peak_q} buses queued simultaneously at peak.",
                expected_impact="Reduces charging wait time. "
                               "Buses return to revenue service faster.",
                confidence="medium",
            ))


def _check_charging_window(cs, recs: list):
    """P4: P5 window may need adjustment."""
    for code, r in cs.results.items():
        # Check if last bus charges too close to window end
        p5_end = r.config.p5_charging_end
        if p5_end is None:
            continue
        last_charge_time = None
        for bus in r.buses:
            for trip in bus.trips:
                if trip.trip_type == "Charging" and trip.actual_arrival:
                    if last_charge_time is None or trip.actual_arrival > last_charge_time:
                        last_charge_time = trip.actual_arrival
        if last_charge_time:
            try:
                p5_end_min = p5_end.hour * 60 + p5_end.minute
                last_min = last_charge_time.hour * 60 + last_charge_time.minute
                if last_min > p5_end_min - 15:  # within 15 min of window end
                    recs.append(Recommendation(
                        category="charging_window",
                        priority=4,
                        route_codes=[code],
                        action=f"Extend {code} p5_charging_end from "
                               f"{p5_end.strftime('%H:%M')} to "
                               f"{(last_charge_time.hour * 60 + last_charge_time.minute + 30) // 60:02d}:"
                               f"{(last_charge_time.minute + 30) % 60:02d}",
                        reason=f"Last bus finished charging at "
                               f"{last_charge_time.strftime('%H:%M')}, "
                               f"only {p5_end_min - last_min:.0f} min before window closes.",
                        expected_impact="Prevents rushed/missed charging for late buses.",
                        confidence="medium",
                    ))
            except Exception:
                pass


def _check_km_balance(cs, recs: list):
    """P5: KM imbalance across buses on a route."""
    for code, r in cs.results.items():
        if r.metrics.km_range > 30:  # more than 30 km difference
            avg_km = (sum(r.metrics.km_per_bus) / len(r.metrics.km_per_bus)
                     if r.metrics.km_per_bus else 0)
            recs.append(Recommendation(
                category="fleet_adjustment",
                priority=5,
                route_codes=[code],
                action=f"Review {code} km balance — range is {r.metrics.km_range:.0f} km "
                       f"(avg {avg_km:.0f} km/bus)",
                reason=f"Some buses work significantly more than others. "
                       f"Max-min difference: {r.metrics.km_range:.0f} km.",
                expected_impact="Better fleet wear distribution. "
                               "May extend battery life.",
                confidence="low",
            ))


def _check_dead_km(cs, recs: list):
    """P5: High dead-km ratio."""
    for code, r in cs.results.items():
        if r.metrics.dead_km_ratio > 0.15:  # more than 15% dead km
            recs.append(Recommendation(
                category="fleet_adjustment",
                priority=5,
                route_codes=[code],
                action=f"Investigate {code} dead-km ratio ({r.metrics.dead_km_ratio:.1%}) — "
                       f"consider terminal charger to reduce depot round-trips",
                reason=f"Dead km ({r.metrics.dead_km:.0f} km) exceeds 15% of total "
                       f"({r.metrics.total_km:.0f} km). Most dead-km is depot charging trips.",
                expected_impact="Terminal charger eliminates depot round-trip dead-km. "
                               "Typical saving: 40–60% dead-km reduction.",
                confidence="medium",
            ))


def _check_min_km_per_bus(cs, recs: list):
    """
    P3: Buses on this route are running less than the configured min_km_per_bus
    policy. Emits up to two recommendation variants per under-policy route:

      Variant A — fleet reduction. If reducing the fleet by 1 keeps PVR
      satisfied, the same revenue trips would be redistributed across N-1
      buses, lifting avg km/bus by approximately:
          new_avg = current_avg * fleet / (fleet - 1)
      Confidence: high if the fleet reduction still meets PVR with margin.

      Variant B — headway tightening. If the route is at or near PVR (so
      fleet reduction is not viable), tightening peak headway by 2 min is
      the next lever — more revenue trips per bus. Confidence: medium since
      the achievable headway depends on route physics.

    Why this matters (planner-facing): min_km_per_bus is a policy obligation
    (CRDF: 180 km/bus/day default). Buses below this number are
    underutilised assets — the operator pays fixed costs (driver, energy
    standby, depreciation) without proportional revenue.
    """
    for code, r in cs.results.items():
        cfg = r.config
        min_km_policy = float(getattr(cfg, "min_km_per_bus", 0) or 0)
        if min_km_policy <= 0:
            continue  # not enforced for this route

        km_list = list(getattr(r.metrics, "km_per_bus", []) or [])
        if not km_list:
            continue

        avg_km = sum(km_list) / len(km_list)
        min_km_actual = min(km_list)
        below_count = sum(1 for k in km_list if k < min_km_policy)

        # Trigger: any bus below policy, OR route avg below policy.
        if below_count == 0 and avg_km >= min_km_policy:
            continue

        fleet = r.fleet_allocated or len(km_list)
        pvr = max(1, int(r.pvr or 1))

        # ── Variant A: fleet reduction (preferred when feasible) ───────────
        if fleet > pvr:
            new_fleet = fleet - 1
            # First-order estimate: same revenue km redistributed.
            new_avg = avg_km * fleet / new_fleet if new_fleet > 0 else avg_km
            meets_policy = new_avg >= min_km_policy

            recs.append(Recommendation(
                category="fleet_adjustment",
                priority=3,
                route_codes=[code],
                action=(f"Reduce {code} fleet from {fleet} to {new_fleet} "
                        f"(PVR={pvr}, so safe)"),
                reason=(f"Avg km/bus is {avg_km:.0f} km — below policy "
                        f"of {min_km_policy:.0f} km. {below_count} of {len(km_list)} "
                        f"bus(es) ran less than {min_km_policy:.0f} km. "
                        f"Fleet ({fleet}) exceeds PVR ({pvr}) by "
                        f"{fleet - pvr} bus(es), so reducing by 1 preserves "
                        f"peak service."),
                expected_impact=(
                    f"Avg km/bus rises from {avg_km:.0f} to ~{new_avg:.0f} km "
                    + ("(meets policy). " if meets_policy
                       else f"(still below {min_km_policy:.0f}). ")
                    + f"Releases 1 bus for redeployment elsewhere or maintenance. "
                    f"No headway impact."
                ),
                confidence="high" if meets_policy else "medium",
            ))
            continue  # Don't emit Variant B if A is viable.

        # ── Variant B: headway tightening (when fleet already at PVR) ──────
        # We don't have access to physics_min_headway in metrics here, but
        # rec_peak_headway on the RouteResult is an upper bound for what's
        # safe. Suggest 2-min tightening from current peak.
        try:
            cur_peak = int(r.headway_df["headway_min"].min())
        except Exception:
            cur_peak = 10
        target_peak = max(cur_peak - 2, int(getattr(r, "physics_min_headway", 0) or 0) or 5)
        if target_peak >= cur_peak:
            target_peak = max(5, cur_peak - 1)

        recs.append(Recommendation(
            category="headway_change",
            priority=3,
            route_codes=[code],
            action=(f"Tighten {code} peak headway from {cur_peak} to {target_peak} min "
                    f"(fleet already at PVR={pvr}, so cannot reduce)"),
            reason=(f"Avg km/bus is {avg_km:.0f} km — below policy "
                    f"of {min_km_policy:.0f} km. Fleet of {fleet} matches PVR "
                    f"({pvr}), so reducing fleet would break peak service. "
                    f"Tighter headway = more revenue trips per bus per day."),
            expected_impact=(
                f"More trips per bus during peak hours. "
                f"Estimated lift: +{(cur_peak/target_peak - 1) * 100:.0f}% peak trips "
                f"if physics allows. Verify on the Headways sub-tab — if the "
                f"physics minimum is above {target_peak} min, this won't be feasible."
            ),
            confidence="medium",
        ))


def _check_p5_window_charging(cs, recs: list):
    """
    P2: Charging trips that started after the configured p5_charging_end.

    This is the recommendation-side counterpart to the Phase A diagnostic
    XLSX export. The diagnostic tells you which trigger fired; this check
    tells the planner the count is non-zero and points them to the export.

    Only emits one recommendation per route, regardless of count.
    """
    for code, r in cs.results.items():
        cfg = r.config
        p5_end = getattr(cfg, "p5_charging_end", None)
        if p5_end is None:
            continue

        late_count = 0
        for bus in r.buses:
            for trip in bus.trips:
                if trip.trip_type != "Charging":
                    continue
                if trip.actual_departure is None:
                    continue
                dep = trip.actual_departure
                if (dep.hour, dep.minute) > (p5_end.hour, p5_end.minute):
                    late_count += 1

        if late_count == 0:
            continue

        recs.append(Recommendation(
            category="charging_window",
            priority=2,
            route_codes=[code],
            action=(f"Investigate {late_count} charging trip(s) on {code} "
                    f"that started after p5_charging_end "
                    f"({p5_end.strftime('%H:%M')})"),
            reason=(f"Late charging is usually driven by SOC_TRIGGER_OFFPEAK "
                    f"(bus crossed trigger after the P5 window closed) or "
                    f"SOC_P3_OVERRIDE (next trip would have breached SOC floor). "
                    f"Both are legitimate — but if frequent, the P5 stagger cap "
                    f"(fleet // 5) may be too restrictive, leaving too many "
                    f"buses to charge late."),
            expected_impact=(
                f"Download the decision-log XLSX from the Depot & Terminals "
                f"tab to see exactly which trigger fired for each late charge. "
                f"If most are SOC_TRIGGER_OFFPEAK, raise trigger_soc_percent "
                f"or relax the P5 stagger cap."
            ),
            confidence="medium",
        ))




# ── Phase E: operational-difficulty grouping ────────────────────────────────
# Per external review (#8): planners need to see recommendations sorted by
# how disruptive each one is, not just by priority. A "P5 km balance" tweak
# is fundamentally easier than a "P3 fleet reduction" — even though P5 < P3
# in priority terms.
#
# Difficulty buckets:
#   easy       — redistribute trips on the same route (no extra cost,
#                no fleet change, no schedule rebuild)
#   moderate   — adjust layover/dead-running patterns or single-route
#                headway tweak
#   significant — add/remove trips on the schedule, or change charging window
#   major       — fleet size change, infrastructure change, route redesign
_DIFFICULTY_RULES: list[tuple[str, str]] = [
    # (substring matched in category OR action, difficulty)
    ("km balance", "easy"),
    ("review", "easy"),
    ("trigger_soc", "moderate"),
    ("p5_charging", "moderate"),
    ("layover", "moderate"),
    ("recovery_buffer", "moderate"),
    ("headway_change", "moderate"),
    ("dead-km", "moderate"),
    ("dead_km", "moderate"),
    ("charging_window", "moderate"),
    ("transfer", "significant"),
    ("redistribute", "significant"),
    ("rebalance", "significant"),
    ("Reduce", "major"),
    ("Increase fleet", "major"),
    ("fleet_adjustment", "major"),
    ("depot", "major"),
    ("infrastructure", "major"),
    ("terminal charger", "major"),
]


def _classify_difficulty(rec: "Recommendation") -> str:
    """Return one of: easy, moderate, significant, major."""
    haystack = f"{rec.category} {rec.action}".lower()
    for needle, level in _DIFFICULTY_RULES:
        if needle.lower() in haystack:
            return level
    # Default conservative: priority 1-2 = major (safety/compliance), else moderate.
    if rec.priority <= 2:
        return "major"
    return "moderate"


def group_by_difficulty(recs: list["Recommendation"]) -> dict[str, list["Recommendation"]]:
    """
    Group recommendations into operational-difficulty buckets.

    Returns dict with keys "easy", "moderate", "significant", "major" — empty
    keys are still present so callers can render in fixed order.
    """
    groups: dict[str, list[Recommendation]] = {
        "easy": [], "moderate": [], "significant": [], "major": [],
    }
    for r in recs:
        groups[_classify_difficulty(r)].append(r)
    return groups

# ── Summary helpers ───────────────────────────────────────────────────────────

def top_recommendations(recs: list[Recommendation], n: int = 5) -> list[Recommendation]:
    """Return the top-N recommendations by priority."""
    return recs[:n]


def recommendations_by_route(recs: list[Recommendation], route_code: str) -> list[Recommendation]:
    """Filter recommendations for a specific route."""
    return [r for r in recs if route_code in r.route_codes or not r.route_codes]


def recommendation_summary_rows(recs: list[Recommendation]) -> list[dict]:
    """Format recommendations as rows for a dashboard table."""
    return [
        {
            "Priority": f"P{r.priority}",
            "Category": r.category.replace("_", " ").title(),
            "Route(s)": ", ".join(r.route_codes) if r.route_codes else "Citywide",
            "Action": r.action,
            "Reason": r.reason,
            "Impact": r.expected_impact,
            "Confidence": r.confidence.title(),
        }
        for r in recs
    ]
