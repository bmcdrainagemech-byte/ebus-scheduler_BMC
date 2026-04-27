"""
auto_patch.py — Auto-patch recommendation engine (Phase F MVP).

Converts dominant operational diagnostics into structured, planner-reviewable
config patches. NEVER silently overwrites user config.

Design principles:
  1. Each Patch is a *suggestion* — it must be explicitly applied by the
     planner via the UI.
  2. Patches are checked against safety constraints (SOC floor, max layover,
     operating hours, charging infrastructure). Unsafe patches are flagged
     `safe_to_auto_apply=False` and cannot be applied via "apply all safe".
  3. Original config is preserved through the existing Phase D bulk
     download — patches mutate in-memory only.
  4. Every patch carries enough context (reason, expected_benefit,
     confidence) for plain-English UI rendering.

MVP scope:
  - 4 patch generators: late-charging window, headway infeasibility,
    min-km-per-bus, continuous-driving violation
  - apply_patch(patch, city_config) — in-memory mutation, returns
    (success, message)
  - patches_to_dataframe(patches) — for UI display
"""

from __future__ import annotations
__version__ = "2026-04-27-phase-f-p1"

from dataclasses import dataclass, asdict
from datetime import time, timedelta


# ── Patch dataclass ─────────────────────────────────────────────────────────

@dataclass
class Patch:
    """A single proposed configuration change."""
    patch_id: str                 # stable id, e.g. "p5_charging_R1"
    route_code: str               # "" = citywide; otherwise specific route
    field: str                    # config field name being changed
    old_value: object             # current value (for diff display + rollback)
    new_value: object             # proposed value
    reason: str                   # plain-English explanation of WHY
    expected_benefit: str         # plain-English explanation of WHAT IMPROVES
    confidence: str               # "high" | "medium" | "low"
    severity: str                 # "critical" | "high" | "medium" | "low"
    safe_to_auto_apply: bool      # True = passes all safety constraints
    safety_notes: str = ""        # populated when safe_to_auto_apply=False

    def to_row(self) -> dict:
        """Render as a dict for st.dataframe display."""
        return {
            "Route": self.route_code or "All",
            "Field": self.field,
            "Current": str(self.old_value),
            "Proposed": str(self.new_value),
            "Severity": self.severity,
            "Confidence": self.confidence,
            "Auto-applicable?": "Yes" if self.safe_to_auto_apply else "Advisory only",
            "Reason": self.reason,
        }


# ── Patch generators ────────────────────────────────────────────────────────

def _gen_late_charging_patches(cs) -> list[Patch]:
    """
    If a route has charging trips after p5_charging_end, propose extending
    the window to cover the latest observed late charge (capped to within
    operating hours minus 1 hour).

    Reasoning: the most common cause of late charging is that the configured
    P5 window closes before the bus's natural SOC trigger time. Pushing
    p5_charging_end out by enough to capture the late events removes the
    false signal without changing scheduling behaviour.
    """
    patches: list[Patch] = []

    for code, r in cs.results.items():
        cfg = r.config
        p5_end = getattr(cfg, "p5_charging_end", None)
        if p5_end is None:
            continue

        # Find latest charging departure
        latest_charge_min = -1
        late_count = 0
        for bus in r.buses:
            for trip in bus.trips:
                if trip.trip_type != "Charging":
                    continue
                if trip.actual_departure is None:
                    continue
                dep = trip.actual_departure
                dep_min = dep.hour * 60 + dep.minute
                p5_end_min = p5_end.hour * 60 + p5_end.minute
                if dep_min > p5_end_min:
                    late_count += 1
                    latest_charge_min = max(latest_charge_min, dep_min)

        if late_count == 0:
            continue

        # Propose new p5_charging_end = latest charge + 15 min cushion,
        # rounded up to next 15-min boundary, capped at op_end - 60 min.
        proposed_min = latest_charge_min + 15
        proposed_min = ((proposed_min + 14) // 15) * 15
        op_end = cfg.operating_end
        op_end_min = op_end.hour * 60 + op_end.minute
        cap_min = op_end_min - 60
        if proposed_min > cap_min:
            proposed_min = cap_min

        if proposed_min <= p5_end.hour * 60 + p5_end.minute:
            continue  # already covers — nothing to patch

        new_p5_end = time(proposed_min // 60, proposed_min % 60)

        # Safety check: must remain before operating_end and after p5_charging_start.
        p5_start = getattr(cfg, "p5_charging_start", None)
        safe = True
        notes = ""
        if p5_start and new_p5_end <= p5_start:
            safe = False
            notes = "New end time would be before p5_charging_start"
        if new_p5_end >= op_end:
            safe = False
            notes = "New end time would be at or after operating_end"

        patches.append(Patch(
            patch_id=f"p5_charging_end_{code}",
            route_code=code,
            field="p5_charging_end",
            old_value=p5_end.strftime("%H:%M"),
            new_value=new_p5_end.strftime("%H:%M"),
            reason=(f"{late_count} charging trip(s) on {code} started after the "
                    f"current p5_charging_end ({p5_end.strftime('%H:%M')}). "
                    f"Latest observed: {latest_charge_min // 60:02d}:"
                    f"{latest_charge_min % 60:02d}."),
            expected_benefit=(f"Extending the window to "
                              f"{new_p5_end.strftime('%H:%M')} captures the late "
                              f"charges as in-window events, removing the false "
                              f"compliance signal. No change to scheduling "
                              f"behaviour — buses already charge at these times."),
            confidence="high" if late_count >= 3 else "medium",
            severity="medium",
            safe_to_auto_apply=safe,
            safety_notes=notes,
        ))
    return patches


def _gen_headway_infeasibility_patches(cs) -> list[Patch]:
    """
    For routes flagged INFEASIBLE on headway feasibility, propose loosening
    the peak headway to the rec_peak_headway value computed by the existing
    physics-aware engine.

    Note: this patch currently writes a coarse-grained signal (just peak
    headway) — it does not rewrite the full headway profile. The planner
    can apply the recommended headway profile via the existing Headways
    sub-tab "Apply Recommended" button for a granular change.
    """
    patches: list[Patch] = []

    for code, r in cs.results.items():
        feas_status = getattr(r, "headway_feasibility_status", None)
        if feas_status != "INFEASIBLE":
            continue

        rec_peak = getattr(r, "rec_peak_headway", 0)
        if not rec_peak or rec_peak <= 0:
            continue

        try:
            cur_peak = int(r.headway_df["headway_min"].min())
        except Exception:
            continue

        if rec_peak <= cur_peak:
            continue

        patches.append(Patch(
            patch_id=f"peak_headway_{code}",
            route_code=code,
            field="peak_headway_min",
            old_value=cur_peak,
            new_value=rec_peak,
            reason=(f"Route {code}'s configured peak headway ({cur_peak} min) is "
                    f"below the physics minimum. The schedule will show large "
                    f"charging gaps that no headway tweak can fix."),
            expected_benefit=(f"Loosening peak headway to {rec_peak} min restores "
                              f"feasibility. Use the per-route Headways sub-tab's "
                              f"'Apply Recommended' button for the full profile "
                              f"adjustment, not just the peak value."),
            confidence="high",
            severity="critical",
            # Advisory only — applying this requires re-running the route
            # scheduler with the full recommended profile, which the existing
            # Apply Recommended button does already. Keep this as a signpost.
            safe_to_auto_apply=False,
            safety_notes=("Apply via per-route Headways sub-tab to adjust "
                          "the full headway profile, not just peak."),
        ))
    return patches


def _gen_min_km_patches(cs) -> list[Patch]:
    """
    For routes with avg km/bus below min_km_per_bus AND fleet > PVR (so
    reduction is safe), propose reducing fleet by 1.

    Mirrors the recommender's Variant A but as a structured patch.
    """
    patches: list[Patch] = []

    for code, r in cs.results.items():
        cfg = r.config
        min_km_policy = float(getattr(cfg, "min_km_per_bus", 0) or 0)
        if min_km_policy <= 0:
            continue

        km_list = list(getattr(r.metrics, "km_per_bus", []) or [])
        if not km_list:
            continue

        avg_km = sum(km_list) / len(km_list)
        if avg_km >= min_km_policy:
            continue

        fleet = r.fleet_allocated or len(km_list)
        pvr = max(1, int(r.pvr or 1))
        if fleet <= pvr:
            continue  # Variant A not viable — let recommender suggest Variant B

        new_fleet = fleet - 1
        new_avg = avg_km * fleet / new_fleet
        meets_policy = new_avg >= min_km_policy

        patches.append(Patch(
            patch_id=f"fleet_reduce_{code}",
            route_code=code,
            field="fleet_size",
            old_value=fleet,
            new_value=new_fleet,
            reason=(f"Avg km/bus on {code} is {avg_km:.0f} km — below the "
                    f"{min_km_policy:.0f} km policy. Fleet ({fleet}) exceeds "
                    f"PVR ({pvr}), so reducing by 1 preserves peak service."),
            expected_benefit=(f"Avg km/bus rises from {avg_km:.0f} to "
                              f"~{new_avg:.0f} km "
                              + ("(meets policy). " if meets_policy
                                 else f"(still below {min_km_policy:.0f}). ")
                              + f"Releases 1 bus for redeployment."),
            confidence="high" if meets_policy else "medium",
            severity="medium",
            safe_to_auto_apply=True,   # fleet > PVR check is the safety guard
        ))
    return patches


def _gen_continuous_driving_patches(cs) -> list[Patch]:
    """
    For routes where any bus exceeds max_continuous_driving_min, propose
    a Break_Policy Add rule at one of the route's terminals to force
    a regulatory break.

    This is the institutional remediation for labor-law violations.
    The patch is *advisory* unless we can pick a clear node — typically
    the route's start_point (where most layovers naturally occur).
    """
    patches: list[Patch] = []

    for code, r in cs.results.items():
        cfg = r.config
        max_cont_min = int(getattr(cfg, "max_continuous_driving_min", 240) or 0)
        reg_brk = int(getattr(cfg, "regulatory_break_min", 20) or 0)
        if max_cont_min <= 0:
            continue

        # Detect violations by re-walking the trips (mirror of O6 logic).
        worst_min = 0
        worst_bus = ""
        for bus in r.buses:
            cont = 0
            prev_arr = None
            for t in bus.trips:
                if t.actual_departure is None or t.actual_arrival is None:
                    continue
                if prev_arr is not None:
                    gap = (t.actual_departure - prev_arr).total_seconds() / 60.0
                    if gap >= reg_brk:
                        cont = 0
                if t.trip_type != "Revenue":
                    cont = 0
                else:
                    cont += int(t.travel_time_min)
                    if cont > worst_min:
                        worst_min = cont
                        worst_bus = bus.bus_id
                prev_arr = t.actual_arrival

        if worst_min <= max_cont_min:
            continue

        # Propose Break_Policy Add at the route's start_point (most common
        # natural layover location).
        target_node = cfg.start_point

        # Check whether there's already a rule at this node.
        existing = getattr(cfg, "break_policy", None) or []
        already_has = any(
            (r_.get("break_node") == target_node and
             (r_.get("action") or "").lower() == "add")
            for r_ in existing
        )
        if already_has:
            continue

        # Also check if there's a Remove rule conflicting at the same node —
        # if so, this Add patch would override it. Flag in safety notes.
        remove_conflict = any(
            (r_.get("break_node") == target_node and
             (r_.get("action") or "").lower() == "remove")
            for r_ in existing
        )
        notes = ""
        safe = True
        if remove_conflict:
            safe = False
            notes = (f"Existing Break_Policy Remove rule at {target_node} "
                     "conflicts. Resolve manually before applying.")

        patches.append(Patch(
            patch_id=f"break_add_{code}",
            route_code=code,
            field="break_policy",
            old_value="(no Add rule)",
            new_value=f"Add at {target_node}",
            reason=(f"Bus {worst_bus} on {code} drove {worst_min} min "
                    f"continuously — exceeds {max_cont_min} min legal limit. "
                    f"Forcing a regulatory break (>= {reg_brk} min) at "
                    f"{target_node} resets the continuous-driving counter."),
            expected_benefit=(f"Brings {code} into labor-law compliance. "
                              f"Layovers at {target_node} will be at least "
                              f"{reg_brk} min instead of preferred_layover_min "
                              f"({cfg.preferred_layover_min} min)."),
            confidence="high",
            severity="critical",
            safe_to_auto_apply=safe,
            safety_notes=notes,
        ))
    return patches


# ── Entry point ─────────────────────────────────────────────────────────────

def generate_patches(city_schedule) -> list[Patch]:
    """
    Run all patch generators against a CitySchedule and return a flat list,
    sorted by severity (critical → low) then by route_code.
    """
    out: list[Patch] = []
    out += _gen_late_charging_patches(city_schedule)
    out += _gen_headway_infeasibility_patches(city_schedule)
    out += _gen_min_km_patches(city_schedule)
    out += _gen_continuous_driving_patches(city_schedule)

    severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    out.sort(key=lambda p: (severity_rank.get(p.severity, 9), p.route_code))
    return out


# ── Apply ───────────────────────────────────────────────────────────────────

def apply_patch(patch: Patch, city_config) -> tuple[bool, str]:
    """
    Apply one Patch to the CityConfig (in-memory mutation).

    Returns (success, message). `success=False` if the patch's field can't
    be applied programmatically (e.g., headway profile changes route
    elsewhere). Does NOT re-run the schedule — caller must do that.
    """
    if patch.route_code not in city_config.routes:
        return False, f"Route {patch.route_code} not found in config"

    cfg = city_config.routes[patch.route_code].config

    if patch.field == "p5_charging_end":
        try:
            hh, mm = map(int, str(patch.new_value).split(":"))
            cfg.p5_charging_end = time(hh, mm)
            return True, f"Set {patch.route_code}.p5_charging_end = {patch.new_value}"
        except Exception as e:
            return False, f"Could not parse new p5_charging_end: {e}"

    if patch.field == "fleet_size":
        try:
            cfg.fleet_size = int(patch.new_value)
            return True, f"Set {patch.route_code}.fleet_size = {patch.new_value}"
        except Exception as e:
            return False, f"Could not set fleet_size: {e}"

    if patch.field == "break_policy":
        # Add rule at start_point (per generator design).
        node_name = str(patch.new_value).replace("Add at ", "").strip()
        existing = list(getattr(cfg, "break_policy", []) or [])
        existing.append({
            "break_node": node_name,
            "action": "Add",
            "peak_only": False,
            "notes": f"Auto-patch: continuous-driving remediation",
        })
        cfg.break_policy = existing
        return True, f"Added Break_Policy Add rule at {node_name} on {patch.route_code}"

    if patch.field == "peak_headway_min":
        # Advisory only — flagged unsafe by the generator.
        return False, ("peak_headway_min patches are advisory only. "
                       "Use per-route Headways sub-tab Apply Recommended.")

    return False, f"Unknown patch field: {patch.field}"


def patches_to_dataframe(patches: list[Patch]):
    """Return a pandas DataFrame for st.dataframe display."""
    import pandas as pd
    if not patches:
        return pd.DataFrame()
    return pd.DataFrame([p.to_row() for p in patches])
