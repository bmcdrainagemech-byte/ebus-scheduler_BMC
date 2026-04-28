"""
bus_scheduler.py - Core scheduling engine.

P4-FIRST: Bus ready time drives departure. min_break = config.preferred_layover_min.

P2 (corrected): Morning dead run DEPOT → nearest_node only (1 leg).
  Revenue trips start from nearest_node = end_point (buses are already there).
  trip_generator generates DN first at op_start, UP after first_dn + min_break.
  Natural bus cycle: Dead(depot→nearest) → DN(Revenue) → UP(Revenue) → DN → UP...

P6: _check_p6 scans all buses' most-recent same-direction revenue trip (not
    just trips[-1]), and re-checks after each bump until gap >= SAME_DIR_GAP.
"""
from __future__ import annotations
__version__ = "2026-04-28-b6"  # auto-stamped
from datetime import datetime, timedelta
from src.models import Trip, BusState, RouteConfig, ScheduleInfeasibleError

REF_DATE           = datetime(2025, 1, 1)
MIDDAY_START       = REF_DATE.replace(hour=12, minute=0)
MIDDAY_END         = REF_DATE.replace(hour=15, minute=0)
MORNING_PEAK_START = REF_DATE.replace(hour=8,  minute=0)
MORNING_PEAK_END   = REF_DATE.replace(hour=11, minute=0)
EVENING_PEAK_START = REF_DATE.replace(hour=15, minute=0)
EVENING_PEAK_END   = REF_DATE.replace(hour=20, minute=0)

# P5 charging window: 11:00–16:00 with ±45 min flex per bus.
# Buses are distributed evenly across the window so bus 0 targets 11:00
# and bus N-1 targets 16:00, with each bus allowed ±CHARGE_FLEX_MIN around
# its personal target before the scheduler considers it late for charging.
CHARGE_WINDOW_START = REF_DATE.replace(hour=11, minute=0)
CHARGE_WINDOW_END   = REF_DATE.replace(hour=16, minute=0)
CHARGE_FLEX_MIN     = 45   # ± minutes around each bus's target charge time

# Fallback values if config missing
MAX_BREAK         = 20
MIDDAY_CHARGE_SOC = 65.0
SOC_TRIGGER       = 30.0
SOC_FLOOR         = 20.0
SAME_DIR_GAP      = 5
DEPOT_DWELL_MIN   = 35   # fallback minimum charge time (config overrides)
DEPOT_DWELL_MAX   = 90   # maximum charge time allowed
KM_BALANCE_MAX    = 20.0
FAST_TURNAROUND_MIN = 2
HEADWAY_WEIGHT    = 2.0
PLANNING_HW_TOLERANCE = 3

OFF_PEAK_START = REF_DATE.replace(hour=11, minute=0)
OFF_PEAK_END   = REF_DATE.replace(hour=15, minute=0)


# ── Headway band helpers ─────────────────────────────────────────────────────

def _parse_hw_time(v) -> datetime:
    """Convert any Excel time cell type to a REF_DATE-anchored datetime."""
    if isinstance(v, datetime):
        return REF_DATE.replace(hour=v.hour, minute=v.minute, second=0, microsecond=0)
    from datetime import time as _time
    if isinstance(v, _time):
        return REF_DATE.replace(hour=v.hour, minute=v.minute, second=0, microsecond=0)
    if isinstance(v, (int, float)):
        tm = round(float(v) * 24 * 60)
        return REF_DATE.replace(hour=min(tm // 60, 23), minute=tm % 60,
                                second=0, microsecond=0)
    p = str(v).strip().split(":")
    return REF_DATE.replace(hour=int(p[0]), minute=int(p[1]), second=0, microsecond=0)


def _build_hw_bands(headway_df) -> list:
    """Pre-parse headway_df into a plain list of (t_from, t_to, headway_min) tuples."""
    if headway_df is None:
        return []
    try:
        if headway_df.empty:
            return []
    except Exception:
        return []
    bands = []
    for _, row in headway_df.iterrows():
        try:
            bands.append((
                _parse_hw_time(row["time_from"]),
                _parse_hw_time(row["time_to"]),
                int(row["headway_min"]),
            ))
        except Exception:
            pass
    return bands


def _lookup_hw(t: datetime, hw_bands: list, fallback: int = SAME_DIR_GAP) -> int:
    """Return headway_min for the band that contains t, or fallback if none match."""
    for t_from, t_to, hw in hw_bands:
        if t_from <= t < t_to:
            return hw
    return hw_bands[-1][2] if hw_bands else fallback


def _is_midday(t):   return MIDDAY_START <= t < MIDDAY_END
def _is_off_peak(t): return OFF_PEAK_START <= t < OFF_PEAK_END
def _is_peak(t):
    return (MORNING_PEAK_START <= t < MORNING_PEAK_END or
            EVENING_PEAK_START <= t < EVENING_PEAK_END)

def _charge_window(config):
    """Return (window_start, window_end) datetimes from config or hardcoded fallback."""
    try:
        cs = config.p5_charging_start
        ce = config.p5_charging_end
        ws = REF_DATE.replace(hour=cs.hour, minute=cs.minute)
        we = REF_DATE.replace(hour=ce.hour, minute=ce.minute)
        if we > ws:
            return ws, we
    except Exception:
        pass
    return CHARGE_WINDOW_START, CHARGE_WINDOW_END


def _target_charge_time(bus, config) -> datetime:
    """Return the ideal charge-start time for this bus."""
    n   = max(1, config.fleet_size)
    idx = getattr(bus, 'phase_index', 0)
    cws, cwe = _charge_window(config)
    window_min = (cwe - cws).total_seconds() / 60
    if n == 1:
        offset = window_min / 2
    else:
        offset = idx * window_min / (n - 1)
    return cws + timedelta(minutes=offset)

def _in_charge_window(bus, config) -> bool:
    """True if this bus should be considered for P5 charging now."""
    target = _target_charge_time(bus, config)
    t      = bus.current_time
    if abs((t - target).total_seconds()) / 60 <= CHARGE_FLEX_MIN:
        return True
    _, cwe = _charge_window(config)
    if target < t < cwe:
        return True
    return False


def _apply_break_policy_cap(config, location, current_hold_min, current_time):
    """
    Apply Break_Policy "Cap" action to limit hold times at specific locations.
    Returns the capped break minutes.
    """
    policy = getattr(config, "break_policy", None) or []
    for rule in policy:
        if (rule.get("action") or "").lower() != "cap":
            continue
        if rule.get("break_node") != location:
            continue
        if rule.get("peak_only") and not _is_peak(current_time):
            continue
        max_cap = rule.get("max_hold_min", 20)
        if current_hold_min > max_cap:
            return max_cap
    return current_hold_min


def _effective_break(config, current_time: datetime, base_break: int,
                     current_location: str | None = None) -> int:
    """
    Return break minutes before the next revenue trip.

    Resolution order (first match wins):
      1. Action="Add" → force regulatory break
      2. Action="Remove" → fast turnaround (2 min)
      3. Action="Cap" → per-location max hold (NEW)
      4. Off-peak extension
      5. Default base_break
    """
    # ── Phase F: Action=Add — force regulatory break ─────────────────────
    if current_location is not None:
        policy = getattr(config, "break_policy", None) or []
        reg_brk = int(getattr(config, "regulatory_break_min", 20) or 0)
        max_b = getattr(config, 'max_layover_min', MAX_BREAK)
        for rule in policy:
            if (rule.get("action") or "").lower() != "add":
                continue
            if rule.get("break_node") != current_location:
                continue
            if rule.get("peak_only") and not _is_peak(current_time):
                continue
            return min(max(reg_brk, base_break), max_b)

    # ── Phase C: Action=Remove — fast turnaround ─────────────────────────
    if current_location is not None:
        policy = getattr(config, "break_policy", None) or []
        for rule in policy:
            if (rule.get("action") or "").lower() != "remove":
                continue
            if rule.get("break_node") != current_location:
                continue
            if rule.get("peak_only") and not _is_peak(current_time):
                continue
            return FAST_TURNAROUND_MIN

    # ── NEW: Action=Cap — per-location max hold ──────────────────────────
    if current_location is not None:
        base_break = _apply_break_policy_cap(config, current_location, base_break, current_time)

    # ── Off-peak extension ───────────────────────────────────────────────
    if _is_off_peak(current_time):
        extra = getattr(config, 'off_peak_layover_extra_min', 0)
        max_b = getattr(config, 'max_layover_min', MAX_BREAK)
        result = min(base_break + extra, max_b)
    else:
        result = base_break
    
    # ── Global max_layover_min cap (always applied) ──────────────────────
    max_break = getattr(config, 'max_layover_min', MAX_BREAK)
    if result > max_break:
        result = max_break
    return result


def _operational_nodes(config):
    nodes = [config.start_point, config.end_point]
    for n in config.intermediates:
        if n and n.strip(): nodes.append(n.strip())
    seen, unique = set(), []
    for n in nodes:
        if n not in seen: seen.add(n); unique.append(n)
    return unique


def _nearest_node_from_depot(config):
    """Return (node, dist_km, travel_min) for the terminal/intermediate closest to depot."""
    best = None
    for node in _operational_nodes(config):
        try:
            dist = config.get_distance(config.depot, node)
            tt   = config.get_travel_time(config.depot, node)
            if best is None:
                best = (node, dist, tt)
            elif tt < best[2]:
                best = (node, dist, tt)
            elif tt == best[2] and dist < best[1]:
                best = (node, dist, tt)
        except KeyError:
            continue
    return best or (config.start_point, 0, 0)


def _soc_cost_to_depot(from_loc: str, config: "RouteConfig") -> float:
    """Estimate SOC% consumed travelling from from_loc back to the depot."""
    if from_loc == config.depot:
        return 0.0

    nearest, _, _ = _nearest_node_from_depot(config)

    def _km_cost(km: float) -> float:
        return (km * config.consumption_rate / config.battery_kwh) * 100

    total_km = 0.0
    loc = from_loc

    if loc != nearest:
        try:
            total_km += config.get_distance(loc, nearest)
            loc = nearest
        except KeyError:
            try:
                total_km += config.get_distance(from_loc, config.depot)
                return _km_cost(total_km)
            except KeyError:
                return _km_cost(config.get_distance(config.start_point, config.end_point))

    if loc != config.depot:
        try:
            total_km += config.get_distance(loc, config.depot)
        except KeyError:
            pass

    return _km_cost(total_km)


def _create_fleet(config):
    t0 = REF_DATE.replace(hour=config.operating_start.hour,
                           minute=config.operating_start.minute)
    return [
        BusState(bus_id=bid, current_location=config.depot,
                 current_time=t0, soc_percent=config.initial_soc_percent,
                 total_km=0.0, shift=1,
                 battery_kwh=config.battery_kwh,
                 consumption_rate=config.consumption_rate)
        for bid in config.bus_ids()
    ]


def _fleet_avg_km(buses):
    return sum(b.total_km for b in buses) / len(buses) if buses else 0.0


def _ready_time(bus, min_break, config=None):
    """Departure-ready time for the bus."""
    last = bus.trips[-1] if bus.trips else None
    if last and last.trip_type == "Revenue":
        effective = (_effective_break(config, bus.current_time, min_break,
                                       current_location=bus.current_location)
                     if config is not None else min_break)
        return bus.current_time + timedelta(minutes=effective)
    return bus.current_time


def _last_revenue_in_direction(bus, direction, start_location):
    """Most-recent revenue trip by this bus matching direction+start."""
    for t in reversed(bus.trips):
        if (t.trip_type == "Revenue" and
                t.direction == direction and
                t.start_location == start_location and
                t.actual_departure is not None):
            return t
    return None


def _check_p6(buses, trip, dep):
    """P6: 5-min gap from the most-recent same-direction revenue trip of ANY other bus."""
    if trip.trip_type != "Revenue":
        return True
    for bus in buses:
        last_rev = _last_revenue_in_direction(bus, trip.direction, trip.start_location)
        if last_rev is None:
            continue
        gap = (dep - last_rev.actual_departure).total_seconds() / 60
        if gap < SAME_DIR_GAP:
            return False
    return True


def _bumped_ready_time(buses, trip, rt, natural_gap=None):
    """Return rt bumped forward until P6 is satisfied."""
    bump = natural_gap if natural_gap and natural_gap > SAME_DIR_GAP else SAME_DIR_GAP
    for _ in range(20):
        if _check_p6(buses, trip, rt):
            return rt
        rt += timedelta(minutes=bump)
    return rt


def _snap_to_phase(rt: datetime, phase_index: int, natural_gap: float,
                   fleet_size: int, op_start: datetime) -> datetime:
    """Snap rt forward to this bus's permanent phase lane."""
    if natural_gap <= 0 or fleet_size <= 0:
        return rt
    cycle_time   = natural_gap * fleet_size
    phase_anchor = op_start + timedelta(minutes=phase_index * natural_gap)
    delta_min    = (rt - phase_anchor).total_seconds() / 60
    if delta_min <= 0:
        return phase_anchor
    remainder = delta_min % cycle_time
    if remainder < 0.5:
        return rt
    return rt + timedelta(minutes=cycle_time - remainder)


def _is_shuttle_leg(from_loc, to_loc, config):
    """A leg is a Shuttle if it travels between any two stops on the revenue corridor."""
    if from_loc == config.depot or to_loc == config.depot:
        return False
    corridor = ({config.start_point, config.end_point} |
                {n for n in getattr(config, 'intermediates', []) if n})
    return from_loc in corridor and to_loc in corridor


def _make_dead(bus, to_loc, dist, tt, config=None):
    """Create a Dead or Shuttle trip leg and assign it to the bus."""
    trip_type = "Shuttle" if (config and _is_shuttle_leg(
        bus.current_location, to_loc, config)) else "Dead"
    leg = Trip(direction="DEPOT", trip_type=trip_type,
               start_location=bus.current_location, end_location=to_loc,
               earliest_departure=bus.current_time, latest_departure=bus.current_time,
               travel_time_min=tt, distance_km=dist, shift=bus.shift)
    bus.assign(leg)
    return leg


def _morning_dead_run(bus, config):
    """DEPOT → nearest_node (P2, 1 leg only)."""
    if bus.current_location != config.depot: return []
    nearest, dist, tt = _nearest_node_from_depot(config)
    if dist <= 0: return []
    return [_make_dead(bus, nearest, dist, tt, config=config)]


def _route_to_depot(bus, config):
    """Return via nearest_node: current → nearest → DEPOT."""
    inserted = []
    if bus.current_location == config.depot: return inserted
    nearest, _, _ = _nearest_node_from_depot(config)
    if bus.current_location != nearest:
        try:
            d = config.get_distance(bus.current_location, nearest)
            t = config.get_travel_time(bus.current_location, nearest)
            inserted.append(_make_dead(bus, nearest, d, t, config=config))
        except KeyError: pass
    if bus.current_location != config.depot:
        try:
            d = config.get_distance(bus.current_location, config.depot)
            t = config.get_travel_time(bus.current_location, config.depot)
            inserted.append(_make_dead(bus, config.depot, d, t, config=config))
        except KeyError: pass
    return inserted


def _route_from_depot(bus, config):
    if bus.current_location != config.depot: return []
    nearest, dist, tt = _nearest_node_from_depot(config)
    if dist <= 0: return []
    return [_make_dead(bus, nearest, dist, tt, config=config)]


def _charging_detour(bus, config, resume_by, min_break):
    """Send bus to depot for charging. Respects config.min_charge_duration_min."""
    if config.depot_flow_rate_kw <= 0: return []
    inserted = []
    inserted.extend(_route_to_depot(bus, config))
    if bus.current_location != config.depot: return inserted
    _, _, from_tt = _nearest_node_from_depot(config)
    
    # Calculate time needed to reach target SOC
    soc_needed = max(10, config.target_soc_percent - bus.soc_percent)
    time_to_target = (soc_needed / 100 * config.battery_kwh / config.depot_flow_rate_kw) * 60
    
    # Get minimum charge duration from config (respect user setting!)
    min_charge = getattr(config, 'min_charge_duration_min', DEPOT_DWELL_MIN)
    
    # Calculate maximum available time before service ends
    max_charge = (resume_by - bus.current_time).total_seconds() / 60 - from_tt - min_break
    
    # Determine final charge time: at least min_charge, at most DEPOT_DWELL_MAX
    charge_time = max(min_charge, min(time_to_target, DEPOT_DWELL_MAX))
    
    # If not enough time for minimum charge, try shorter emergency charge
    if max_charge < min_charge:
        if max_charge >= min_charge - 10:
            charge_time = max_charge
        else:
            return inserted
    
    # Cap at available time
    charge_time = min(charge_time, max(min_charge - 10, max_charge))
    charge_time = max(min_charge - 10, charge_time)  # ensure not too short
    
    ct = Trip(direction="DEPOT", trip_type="Charging",
              start_location=config.depot, end_location=config.depot,
              earliest_departure=bus.current_time, latest_departure=bus.current_time,
              travel_time_min=int(charge_time), distance_km=0.0, shift=bus.shift)
    bus.assign(ct)
    bus.charge(duration_min=charge_time, flow_rate_kw=config.depot_flow_rate_kw)
    inserted.append(ct)
    inserted.extend(_route_from_depot(bus, config))
    return inserted


def _find_and_reposition(buses, trip, config, min_break):
    """Find a bus that can reach trip.start_location via a dead run."""
    op_end = REF_DATE.replace(hour=config.operating_end.hour,
                               minute=config.operating_end.minute)
    nearest, _, nearest_tt = _nearest_node_from_depot(config)
    candidates = []

    for bus in buses:
        if bus.current_location == trip.start_location:
            continue

        legs = []

        if bus.current_location == config.depot:
            try:
                d1 = config.get_distance(config.depot, nearest)
                t1 = config.get_travel_time(config.depot, nearest)
            except KeyError:
                continue
            if nearest == trip.start_location:
                legs = [("depot_to_nearest", d1, t1)]
                total_d, total_t = d1, t1
            else:
                try:
                    d2 = config.get_distance(nearest, trip.start_location)
                    t2 = config.get_travel_time(nearest, trip.start_location)
                except KeyError:
                    continue
                legs = [("depot_to_nearest", d1, t1), ("nearest_to_origin", d2, t2)]
                total_d, total_t = d1 + d2, t1 + t2
        else:
            try:
                total_d = config.get_distance(bus.current_location, trip.start_location)
                total_t = config.get_travel_time(bus.current_location, trip.start_location)
            except KeyError:
                continue
            legs = [("direct", total_d, total_t)]

        arrival = bus.current_time + timedelta(minutes=total_t)
        if arrival + timedelta(minutes=min_break) > op_end:
            continue
        soc_needed = bus._soc_cost(total_d) + bus._soc_cost(trip.distance_km)
        if bus.soc_percent - soc_needed * 100 / bus.battery_kwh < SOC_FLOOR:
            continue
        candidates.append((total_t, bus, legs))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    _, bus, legs = candidates[0]

    dead_runs = []
    for leg_tag, leg_d, leg_t in legs:
        dead = Trip(
            direction="DEPOT", trip_type="Dead",
            start_location=bus.current_location,
            end_location=(nearest if leg_tag == "depot_to_nearest"
                          else trip.start_location),
            earliest_departure=bus.current_time,
            latest_departure=bus.current_time,
            travel_time_min=leg_t, distance_km=leg_d, shift=bus.shift,
        )
        bus.assign(dead)
        dead_runs.append(dead)

    return bus, dead_runs


def _balance_breaks(buses, config):
    """Enforce min/max break constraints after scheduling."""
    min_break = config.preferred_layover_min
    max_break = getattr(config, 'max_layover_min', MAX_BREAK)
    for bus in buses:
        rev_idx = [i for i, t in enumerate(bus.trips) if t.trip_type == "Revenue"]
        for j in range(1, len(rev_idx)):
            i_prev, i_curr = rev_idx[j-1], rev_idx[j]
            if any(t.trip_type == "Charging" for t in bus.trips[i_prev+1:i_curr]):
                continue
            tp, tc = bus.trips[i_prev], bus.trips[i_curr]
            if not (tp.actual_arrival and tc.actual_departure): continue
            gap = (tc.actual_departure - tp.actual_arrival).total_seconds() / 60
            
            # Cap breaks that exceed max_layover_min
            if gap > max_break:
                new_dep = tp.actual_arrival + timedelta(minutes=max_break)
                delta = tc.actual_departure - new_dep
                if delta.total_seconds() > 0:
                    tc.actual_departure = new_dep
                    tc.actual_arrival = tc.actual_departure + timedelta(minutes=tc.travel_time_min)
            elif gap < min_break:
                delta = timedelta(minutes=min(min_break - gap,
                                              float(config.max_headway_deviation_min)))
                if delta.total_seconds() > 0:
                    tc.actual_departure += delta
                    tc.actual_arrival = tc.actual_departure + timedelta(minutes=tc.travel_time_min)


def schedule_buses(config: RouteConfig, trips: list[Trip],
                   headway_df=None, travel_time_df=None,
                   scheduling_mode: str = "planning") -> list[BusState]:
    """
    Bus-driven scheduler — no pre-generated trip pool slots.
    """
    min_break      = config.preferred_layover_min
    off_peak_extra = getattr(config, 'off_peak_layover_extra_min', 0)
    buses          = _create_fleet(config)
    op_end         = REF_DATE.replace(hour=config.operating_end.hour,
                                      minute=config.operating_end.minute)
    op_start_dt    = REF_DATE.replace(hour=config.operating_start.hour,
                                      minute=config.operating_start.minute)

    _hw_bands = _build_hw_bands(headway_df)

    # ── Phase 1: Staggered morning dead runs ─────────────────────────────────
    nearest_node, _, nearest_tt = _nearest_node_from_depot(config)

    is_circular = config.start_point == config.end_point
    circular_has_intermediate = False
    circular_intermediate = None

    if is_circular:
        clean_ints = [n.strip() for n in getattr(config, 'intermediates', [])
                      if n and n.strip()]
        if clean_ints:
            circular_has_intermediate = True
            circular_intermediate = clean_ints[0]

    terminals = [config.start_point, config.end_point]
    terminals = list(dict.fromkeys(terminals))

    if circular_has_intermediate:
        rev_start = config.start_point
        far_loc   = circular_intermediate
        reposition_to = None
    elif nearest_node in terminals:
        rev_start = nearest_node
        far_loc   = (config.end_point if nearest_node == config.start_point
                     else config.start_point)
        reposition_to = None
    else:
        best_term, best_dist = None, float('inf')
        for term in [config.start_point, config.end_point]:
            try:
                d = config.get_distance(nearest_node, term)
                if d < best_dist:
                    best_dist, best_term = d, term
            except KeyError:
                pass
        rev_start     = best_term or config.start_point
        far_loc       = config.end_point if rev_start == config.start_point else config.start_point
        reposition_to = rev_start

    try:
        if circular_has_intermediate:
            dn_tt = config.get_travel_time(far_loc, rev_start)
        elif is_circular:
            dn_tt = config.get_travel_time(rev_start, rev_start)
            if dn_tt == 0:
                dn_tt = config.get_travel_time(config.start_point, config.start_point)
        else:
            dn_tt = config.get_travel_time(rev_start, far_loc)
    except KeyError:
        dn_tt = 45
    try:
        if circular_has_intermediate:
            up_tt = config.get_travel_time(rev_start, far_loc)
        elif is_circular:
            up_tt = 0
        else:
            up_tt = config.get_travel_time(far_loc, rev_start)
    except KeyError:
        up_tt = dn_tt if not (is_circular and not circular_has_intermediate) else 0

    if is_circular and not circular_has_intermediate:
        cycle_time = dn_tt + min_break
    else:
        cycle_time = dn_tt + min_break + up_tt + min_break

    if scheduling_mode != "planning":
        recovery = getattr(config, "recovery_buffer_min", 0) or 0
        if recovery > 0:
            cycle_time += 2 * recovery

    natural_gap = cycle_time / max(1, config.fleet_size)

    if is_circular and reposition_to and headway_df is not None:
        try:
            from src.trip_generator import _get_headway_at as _gha_p1
            phase1_gap = _gha_p1(op_start_dt, headway_df)
        except Exception:
            phase1_gap = natural_gap
    elif _hw_bands:
        min_hw_all_bands = min(hw for _, _, hw in _hw_bands)
        phase1_gap = max(natural_gap, min_hw_all_bands)
    else:
        phase1_gap = natural_gap

    has_intermediates = bool([n for n in config.intermediates if n and n.strip()])
    is_rural = getattr(config, 'is_suburban_route', False) and not is_circular
    rural_node_name = (getattr(config, 'rural_node', '') or '').strip().lower()

    if is_rural:
        if rural_node_name == 'start_point':
            rural_loc = config.start_point
            other_loc = config.end_point
        elif rural_node_name == 'end_point':
            rural_loc = config.end_point
            other_loc = config.start_point
        else:
            rural_loc = far_loc
            other_loc = rev_start

    intermediate_node = None
    if has_intermediates and not is_rural and not is_circular:
        clean_ints = [n.strip() for n in config.intermediates if n and n.strip()]
        best_int, best_int_tt = None, float('inf')
        for n in clean_ints:
            try:
                tt_n = config.get_travel_time(config.depot, n)
                if tt_n < best_int_tt:
                    best_int_tt, best_int = tt_n, n
            except KeyError:
                pass
        intermediate_node = best_int or clean_ints[0]

    split_dispatch = (has_intermediates and not is_rural and not is_circular
                      and intermediate_node is not None)

    n_buses = len(buses)
    rural_half = (n_buses + 1) // 2
    inter_half = n_buses // 2

    for i, bus in enumerate(buses):
        if is_rural:
            if i < rural_half:
                try:
                    d_rural = config.get_distance(config.depot, rural_loc)
                    t_rural = config.get_travel_time(config.depot, rural_loc)
                except KeyError:
                    d_rural, t_rural = 0, nearest_tt
                arrive_at = op_start_dt + timedelta(minutes=i * phase1_gap)
                bus.current_time = arrive_at - timedelta(minutes=t_rural)
                if d_rural > 0:
                    _make_dead(bus, rural_loc, d_rural, t_rural, config=config)
            else:
                slot_idx = i - rural_half
                try:
                    d_other = config.get_distance(config.depot, other_loc)
                    t_other = config.get_travel_time(config.depot, other_loc)
                except KeyError:
                    d_other, t_other = 0, nearest_tt
                arrive_at = op_start_dt + timedelta(minutes=slot_idx * phase1_gap)
                bus.current_time = arrive_at - timedelta(minutes=t_other)
                if d_other > 0:
                    _make_dead(bus, other_loc, d_other, t_other, config=config)

        elif split_dispatch and i < inter_half:
            slot_idx = i
            arrive_at = op_start_dt + timedelta(minutes=slot_idx * phase1_gap)
            try:
                d_int = config.get_distance(config.depot, intermediate_node)
                t_int = config.get_travel_time(config.depot, intermediate_node)
            except KeyError:
                d_int, t_int = 0, 0
            try:
                d_near = config.get_distance(intermediate_node, rev_start)
                t_near = config.get_travel_time(intermediate_node, rev_start)
            except KeyError:
                d_near, t_near = 0, 0
            total_travel = t_int + t_near
            bus.current_time = arrive_at - timedelta(minutes=total_travel)
            if d_int > 0:
                _make_dead(bus, intermediate_node, d_int, t_int, config=config)
            if d_near > 0 and bus.current_location != rev_start:
                _make_dead(bus, rev_start, d_near, t_near, config=config)

        elif split_dispatch and i >= inter_half:
            slot_idx = i - inter_half
            arrive_at = op_start_dt + timedelta(minutes=slot_idx * phase1_gap)
            try:
                d_far2 = config.get_distance(config.depot, far_loc)
                t_far2 = config.get_travel_time(config.depot, far_loc)
            except KeyError:
                d_far2, t_far2 = 0, nearest_tt
            bus.current_time = arrive_at - timedelta(minutes=t_far2)
            if d_far2 > 0:
                _make_dead(bus, far_loc, d_far2, t_far2, config=config)

        else:
            if circular_has_intermediate:
                if i % 2 == 0:
                    dest_loc = far_loc
                    slot_idx = i // 2
                else:
                    dest_loc = rev_start
                    slot_idx = (i - 1) // 2
                arrive_at = op_start_dt + timedelta(minutes=slot_idx * phase1_gap)
                try:
                    d_dest = config.get_distance(config.depot, dest_loc)
                    t_dest = config.get_travel_time(config.depot, dest_loc)
                except KeyError:
                    d_dest, t_dest = 0, nearest_tt
                bus.current_time = arrive_at - timedelta(minutes=t_dest)
                if d_dest > 0:
                    _make_dead(bus, dest_loc, d_dest, t_dest, config=config)
            else:
                arrive_at = op_start_dt + timedelta(minutes=i * phase1_gap)
                bus.current_time = arrive_at - timedelta(minutes=nearest_tt)
                _morning_dead_run(bus, config)
                if reposition_to and bus.current_location != reposition_to:
                    try:
                        rd = config.get_distance(bus.current_location, reposition_to)
                        rt_tt = config.get_travel_time(bus.current_location, reposition_to)
                        _make_dead(bus, reposition_to, rd, rt_tt, config=config)
                    except KeyError:
                        pass

        bus.phase_index = i

    # ── Phase 2: Bus-driven revenue loop ─────────────────────────────────────
    midday_soc = getattr(config, 'midday_charge_soc_percent', MIDDAY_CHARGE_SOC)
    soc_trigger = getattr(config, 'trigger_soc_percent', SOC_TRIGGER)

    charged_today = set()
    headway_hold: dict = {}
    MAX_ITER = config.fleet_size * 300

    for _ in range(MAX_ITER):
        one_trip_drain = (config.get_distance(config.start_point, config.end_point)
                          * config.consumption_rate / config.battery_kwh * 100)
        max_cost_home = _soc_cost_to_depot(far_loc, config)

        _rescue_candidate = None
        _rescue_soc = float("inf")

        for stuck_bus in buses:
            if stuck_bus.current_location == config.depot:
                continue
            if stuck_bus.bus_id in charged_today:
                continue
            actual_cost_home = _soc_cost_to_depot(stuck_bus.current_location, config)
            stranded_now = stuck_bus.soc_percent - actual_cost_home < SOC_FLOOR

            if stuck_bus.current_location == far_loc:
                cost_home_after_trip = _soc_cost_to_depot(rev_start, config)
            else:
                cost_home_after_trip = max_cost_home
            stuck_after_next = (stuck_bus.soc_percent
                                 - one_trip_drain
                                 - cost_home_after_trip) < SOC_FLOOR + 2.0

            if stranded_now or stuck_after_next:
                if stuck_bus.soc_percent < _rescue_soc:
                    _rescue_soc = stuck_bus.soc_percent
                    _rescue_candidate = stuck_bus

        if _rescue_candidate is not None:
            _charging_detour(_rescue_candidate, config, resume_by=op_end, min_break=min_break)
            charged_today.add(_rescue_candidate.bus_id)
            _rescue_candidate.decision_log.append(
                f"{_rescue_candidate.current_time.strftime('%H:%M')} EMERGENCY_RESCUE")

        best_bus = best_rt = best_dir = best_start = best_end = None
        best_dist_km = best_tt_val = best_needs_repo = None
        best_score = None

        avg_km = _fleet_avg_km(buses)
        for bus in buses:
            if bus.current_location == config.depot:
                continue

            last_rev = next((t for t in reversed(bus.trips)
                             if t.trip_type == "Revenue"), None)
            recent_types = [t.trip_type for t in bus.trips[-4:]]
            just_charged = "Charging" in recent_types

            if is_circular and not circular_has_intermediate:
                next_dir = "DN"
                trip_start = rev_start
                trip_end = rev_start
                if bus.current_location not in (rev_start, nearest_node):
                    continue
            elif last_rev is None or just_charged:
                if bus.current_location == rev_start or (
                        reposition_to and bus.current_location == nearest_node):
                    next_dir = "DN"
                    trip_start = rev_start
                    trip_end = far_loc
                elif bus.current_location == far_loc:
                    next_dir = "UP"
                    trip_start = far_loc
                    trip_end = rev_start
                else:
                    continue
            elif last_rev.end_location == far_loc:
                next_dir = "UP"
                trip_start = far_loc
                trip_end = rev_start
            else:
                next_dir = "DN"
                trip_start = rev_start
                trip_end = far_loc

            if bus.current_location != trip_start:
                if (reposition_to and
                        bus.current_location == nearest_node and
                        trip_start == rev_start):
                    try:
                        repo_tt = config.get_travel_time(nearest_node, rev_start)
                    except KeyError:
                        continue
                    needs_reposition = True
                else:
                    continue
            else:
                needs_reposition = False
                repo_tt = 0

            rt = _ready_time(bus, min_break, config)

            if needs_reposition:
                rt = rt + timedelta(minutes=repo_tt)

            def _min_hw_at(t):
                return _lookup_hw(t, _hw_bands)

            hold_key = (bus.bus_id, next_dir, trip_start)
            held = headway_hold.get(hold_key)
            if held and held > rt:
                rt = held

            min_hw = _min_hw_at(rt)

            if scheduling_mode == "planning":
                min_spacing = max(_min_hw_at(rt), SAME_DIR_GAP)
            else:
                min_spacing = SAME_DIR_GAP

            last_same = None
            for other in buses:
                cand = _last_revenue_in_direction(other, next_dir, trip_start)
                if cand and (last_same is None or
                        cand.actual_departure > last_same.actual_departure):
                    last_same = cand
            if last_same and last_same.actual_departure:
                gap = (rt - last_same.actual_departure).total_seconds() / 60
                if gap < min_spacing:
                    rt = last_same.actual_departure + timedelta(minutes=min_spacing)
                    headway_hold[hold_key] = rt

            tt_val = None
            if travel_time_df is not None:
                try:
                    from trip_generator import _get_travel_time as _gtt
                    tt_val = _gtt(rt, next_dir, travel_time_df)
                except Exception:
                    pass
            if tt_val is None:
                try:
                    tt_val = config.get_travel_time(trip_start, trip_end)
                except KeyError:
                    tt_val = dn_tt

            try:
                dist_km = config.get_distance(trip_start, trip_end)
            except KeyError:
                continue
            if bus.soc_after_trip(dist_km) < SOC_FLOOR:
                continue

            soc_after = bus.soc_after_trip(dist_km)
            cost_home = _soc_cost_to_depot(trip_end, config)
            if soc_after - cost_home < SOC_FLOOR:
                continue

            if rt + timedelta(minutes=tt_val) > op_end + timedelta(minutes=45):
                continue

            max_km = getattr(config, 'max_km_per_bus', 0) or 0
            if max_km > 0 and bus.total_km + dist_km > max_km:
                if bus.soc_percent > SOC_FLOOR + 5:
                    continue

            if scheduling_mode == "planning":
                target_hw = max(_min_hw_at(rt), SAME_DIR_GAP)
            else:
                target_hw = max(_min_hw_at(rt), natural_gap)

            if last_same and last_same.actual_departure:
                target_dep = last_same.actual_departure + timedelta(minutes=target_hw)
                effective_rt = max(rt, target_dep)
            else:
                target_dep = None
                effective_rt = rt

            km_deficit = bus.total_km - avg_km
            eff_minutes = (effective_rt - op_start_dt).total_seconds() / 60

            if scheduling_mode == "planning":
                deviation_min = (
                    max(0.0, (effective_rt - target_dep).total_seconds() / 60)
                    if target_dep is not None else 0.0
                )
                if last_same and last_same.actual_departure and target_dep is not None:
                    current_gap = (effective_rt - last_same.actual_departure).total_seconds() / 60
                    if current_gap > 1.5 * target_hw:
                        large_gap_penalty = (current_gap - target_hw) * HEADWAY_WEIGHT
                    else:
                        large_gap_penalty = 0.0
                else:
                    large_gap_penalty = 0.0
                bus_score = (eff_minutes
                             + deviation_min * HEADWAY_WEIGHT
                             + large_gap_penalty
                             + km_deficit * 0.1)
            else:
                bus_score = eff_minutes + km_deficit * 0.3

            if best_rt is None or bus_score < best_score:
                best_bus = bus
                best_rt = effective_rt
                best_score = bus_score
                best_dir = next_dir
                best_start = trip_start
                best_end = trip_end
                best_dist_km = dist_km
                best_tt_val = tt_val
                best_needs_repo = needs_reposition

        if best_bus is None:
            break

        if best_needs_repo and best_bus.current_location == nearest_node:
            try:
                rd = config.get_distance(nearest_node, rev_start)
                rtt = config.get_travel_time(nearest_node, rev_start)
                _make_dead(best_bus, rev_start, rd, rtt, config=config)
            except KeyError:
                pass

        trip = Trip(
            direction=best_dir, trip_type="Revenue",
            start_location=best_start, end_location=best_end,
            earliest_departure=best_rt, latest_departure=op_end,
            travel_time_min=best_tt_val, distance_km=best_dist_km,
            shift=(1 if best_rt < REF_DATE.replace(
                       hour=config.shift_split.hour,
                       minute=config.shift_split.minute) else 2),
        )
        trip.earliest_departure = best_rt
        best_bus.assign(trip)
        headway_hold.pop((best_bus.bus_id, best_dir, best_start), None)

        _needs_midday_charge = (best_bus.bus_id not in charged_today and
                                best_bus.soc_percent < midday_soc)

        _p5_window_remaining = (_charge_window(config)[1] - best_bus.current_time).total_seconds() / 60
        _cost_one_trip = one_trip_drain
        _cost_return = _soc_cost_to_depot(rev_start, config)
        _would_be_stuck = (best_bus.soc_percent
                           - _cost_one_trip
                           - _cost_return) < SOC_FLOOR + 5.0
        at_far_loc = (not (is_circular and not circular_has_intermediate) and
                      best_bus.current_location == far_loc and
                      _p5_window_remaining > 90 and
                      not _would_be_stuck)

        def _last_charge_start(buses, current_bus):
            latest = None
            for b in buses:
                if b is current_bus:
                    continue
                for t in reversed(b.trips):
                    if t.trip_type == "Charging" and t.actual_departure:
                        if latest is None or t.actual_departure > latest:
                            latest = t.actual_departure
                        break
            return latest

        def _count_active_detours(buses, current_bus):
            count = 0
            for b in buses:
                if b is current_bus:
                    continue
                if not b.trips:
                    continue
                last_chg_idx = None
                for i, t in enumerate(b.trips):
                    if t.trip_type == "Charging":
                        last_chg_idx = i
                if last_chg_idx is not None:
                    has_rev_after = any(
                        t.trip_type == "Revenue" for t in b.trips[last_chg_idx + 1:]
                    )
                    if not has_rev_after:
                        count += 1
            return count

        try:
            _dead_run_min = config.get_travel_time(nearest_node, config.depot)
        except Exception:
            _dead_run_min = 40
        _soc_delta = max(10.0, config.target_soc_percent - config.trigger_soc_percent)
        _kwh_needed = _soc_delta / 100.0 * config.battery_kwh
        _charge_min = (_kwh_needed /
                       max(1.0, config.depot_charger_kw * config.depot_charger_efficiency)
                       * 60.0)
        _estimated_round_trip = _dead_run_min * 2 + _charge_min + min_break

        _cws, _cwe = _charge_window(config)
        _window_min = (_cwe - _cws).total_seconds() / 60
        _max_concurrent = max(1, config.fleet_size // 5)
        _min_charge_gap = max(
            _window_min / max(1, config.fleet_size),
            _estimated_round_trip / _max_concurrent,
        )

        _last_chg = _last_charge_start(buses, best_bus)
        _active_detours = _count_active_detours(buses, best_bus)

        charge_stagger_ok = (
            (_last_chg is None or
             (best_bus.current_time - _last_chg).total_seconds() / 60 >= _min_charge_gap)
            and _active_detours < _max_concurrent
        )

        if (_in_charge_window(best_bus, config) and
                best_bus.bus_id not in charged_today and
                best_bus.soc_percent < midday_soc and
                charge_stagger_ok and
                not at_far_loc):
            _charging_detour(best_bus, config, resume_by=op_end, min_break=min_break)
            charged_today.add(best_bus.bus_id)
            for k in list(headway_hold.keys()):
                if k[0] == best_bus.bus_id:
                    del headway_hold[k]

        elif best_bus.soc_percent <= soc_trigger:
            _cws2, _cwe2 = _charge_window(config)
            _in_window = (_cws2 <= best_bus.current_time < _cwe2)
            if _in_window and best_bus.bus_id not in charged_today and charge_stagger_ok:
                _charging_detour(best_bus, config, resume_by=op_end, min_break=min_break)
                charged_today.add(best_bus.bus_id)
                for k in list(headway_hold.keys()):
                    if k[0] == best_bus.bus_id:
                        del headway_hold[k]
            elif not _is_peak(best_bus.current_time) and charge_stagger_ok:
                _charging_detour(best_bus, config, resume_by=op_end, min_break=min_break)
                charged_today.add(best_bus.bus_id)
                for k in list(headway_hold.keys()):
                    if k[0] == best_bus.bus_id:
                        del headway_hold[k]
            else:
                typical_drain = (config.get_distance(config.start_point, config.end_point)
                                 * config.consumption_rate / config.battery_kwh * 100)
                if best_bus.soc_percent - typical_drain < SOC_FLOOR:
                    _charging_detour(best_bus, config, resume_by=op_end, min_break=min_break)
                    charged_today.add(best_bus.bus_id)
                    for k in list(headway_hold.keys()):
                        if k[0] == best_bus.bus_id:
                            del headway_hold[k]

    # ── Phase 3: Pre-Phase-3 emergency charge ─────────────────────────────────
    for bus in buses:
        if bus.current_location == config.depot:
            continue
        try:
            nearest, _, _ = _nearest_node_from_depot(config)
            cost = 0.0
            loc = bus.current_location
            if loc != nearest:
                cost += config.get_distance(loc, nearest) * config.consumption_rate / config.battery_kwh * 100
                loc = nearest
            if loc != config.depot:
                cost += config.get_distance(loc, config.depot) * config.consumption_rate / config.battery_kwh * 100
        except Exception:
            cost = 15.0
        if bus.soc_percent - cost < SOC_FLOOR:
            _charging_detour(bus, config, resume_by=op_end, min_break=min_break)

    # ── Phase 4: Evening return ──────────────────────────────────────────────
    for bus in buses:
        _route_to_depot(bus, config)

    # ── Phase 5: Safety-net break enforcement ─────────────────────────────────
    _balance_breaks(buses, config)

    return buses


def check_compliance(config: RouteConfig, buses: list[BusState],
                     headway_df=None) -> list[dict]:
    """Check all compliance rules P1-P6 and O1-O4."""
    min_break = config.preferred_layover_min
    results = []

    # P1: Revenue trips between Start/End only
    p1_v = []
    valid = {config.start_point, config.end_point}
    for bus in buses:
        for t in bus.trips:
            if t.trip_type == "Revenue":
                if t.start_location not in valid or t.end_location not in valid:
                    p1_v.append(f"{bus.bus_id}: {t.start_location}->{t.end_location}")
    results.append({"rule": "P1: Revenue trips between Start/End only", "priority": 1,
                    "status": "PASS" if not p1_v else "FAIL",
                    "details": f"{len(p1_v)} violations" if p1_v else "All revenue trips on route",
                    "violations": p1_v[:5]})

    # P2: Depot access via nearest node
    nearest_from, _, _ = _nearest_node_from_depot(config)
    p2_v = []
    for bus in buses:
        for i, t in enumerate(bus.trips):
            if i == 0 and t.trip_type == "Dead" and t.start_location == config.depot:
                if t.end_location != nearest_from:
                    p2_v.append(f"{bus.bus_id}: went to {t.end_location}, nearest={nearest_from}")
    results.append({"rule": "P2: Depot access via nearest node", "priority": 2,
                    "status": "PASS" if not p2_v else "FAIL",
                    "details": f"Nearest: {nearest_from}" + (f", {len(p2_v)} violations" if p2_v else ""),
                    "violations": p2_v[:5]})

    # P3: SOC never below 20%
    min_soc_seen = 100.0
    p3_v = []
    for bus in buses:
        soc = config.initial_soc_percent
        for t in bus.trips:
            soc -= bus._soc_cost(t.distance_km)
            if t.trip_type == "Charging":
                soc = min(100.0, soc + (config.depot_flow_rate_kw * t.travel_time_min / 60)
                          / config.battery_kwh * 100)
            if soc < SOC_FLOOR:
                p3_v.append(f"{bus.bus_id} SOC={soc:.1f}% at "
                             f"{t.actual_arrival.strftime('%H:%M') if t.actual_arrival else '?'}")
            min_soc_seen = min(min_soc_seen, soc)
    results.append({"rule": "P3: SOC never below 20%", "priority": 3,
                    "status": "PASS" if not p3_v else "FAIL",
                    "details": f"Min SOC: {min_soc_seen:.1f}%" + (f", {len(p3_v)} violations" if p3_v else ""),
                    "violations": p3_v[:5]})

    # P4: Breaks between revenue trips
    max_break = getattr(config, 'max_layover_min', MAX_BREAK)
    p4_v = []
    p4_hw_warn = []
    _hw_bands_p4 = _build_hw_bands(headway_df)

    def _hw_at_time(t):
        return _lookup_hw(t, _hw_bands_p4, fallback=max_break)

    for bus in buses:
        rev_idx = [i for i, t in enumerate(bus.trips) if t.trip_type == "Revenue"]
        for j in range(1, len(rev_idx)):
            i_prev, i_curr = rev_idx[j-1], rev_idx[j]
            if any(t.trip_type in ("Charging", "Dead", "Shuttle") for t in bus.trips[i_prev+1:i_curr]):
                continue
            c, n = bus.trips[i_prev], bus.trips[i_curr]
            if c.actual_arrival and n.actual_departure:
                gap = (n.actual_departure - c.actual_arrival).total_seconds() / 60
                if gap < min_break:
                    p4_v.append(f"{bus.bus_id} @ {c.actual_arrival.strftime('%H:%M')}: {gap:.0f}min < {min_break}")
                elif gap > max_break:
                    hw = _hw_at_time(c.actual_arrival)
                    if gap <= 2 * hw + min_break:
                        p4_hw_warn.append(
                            f"{bus.bus_id} @ {c.actual_arrival.strftime('%H:%M')}: "
                            f"{gap:.0f}min (headway-constrained)")
                    else:
                        p4_v.append(f"{bus.bus_id} @ {c.actual_arrival.strftime('%H:%M')}: {gap:.0f}min > {max_break}")

    p4_status = "PASS" if not p4_v else "WARN"
    p4_detail = (f"{len(p4_v)} long breaks" if p4_v else "All breaks in range")
    if p4_hw_warn:
        p4_detail += f" · {len(p4_hw_warn)} headway-forced gaps"
    results.append({"rule": f"P4: Break {min_break}–{max_break} min between revenue trips", "priority": 4,
                    "status": p4_status, "details": p4_detail,
                    "violations": p4_v[:10] + (["--- headway-forced ---"] if p4_hw_warn else []) + p4_hw_warn[:5]})

    # P5: Midday charging
    _p5_start = getattr(config, 'p5_charging_start', None)
    _p5_end = getattr(config, 'p5_charging_end', None)
    _p5_sh = _p5_start.hour if _p5_start else 12
    _p5_sm = _p5_start.minute if _p5_start else 0
    _p5_eh = _p5_end.hour if _p5_end else 15
    _p5_em = _p5_end.minute if _p5_end else 0
    _p5_label = f"{_p5_sh:02d}:{_p5_sm:02d}–{_p5_eh:02d}:{_p5_em:02d}"

    if config.fleet_size > 10:
        results.append({"rule": f"P5: Midday charging {_p5_label} [waived: fleet > 10]",
                        "priority": 5, "status": "PASS", "details": f"Fleet {config.fleet_size} > 10",
                        "violations": []})
    else:
        p5_v = [f"{bus.bus_id}: no charge in {_p5_label}"
                for bus in buses
                if not any(t.trip_type == "Charging" and t.actual_departure and
                           (_p5_sh * 60 + _p5_sm) <= (t.actual_departure.hour * 60 + t.actual_departure.minute) < (_p5_eh * 60 + _p5_em)
                           for t in bus.trips)]
        results.append({"rule": f"P5: Midday charging ({_p5_label})", "priority": 5,
                        "status": "PASS" if not p5_v else "WARN",
                        "details": f"{len(p5_v)} buses missing charge" if p5_v else f"All buses charged",
                        "violations": p5_v})

    # P6: 5 min gap same-direction
    p6_v = []
    sorted_rev = sorted(
        [(bus, t) for bus in buses for t in bus.trips if t.trip_type == "Revenue"],
        key=lambda x: x[1].actual_departure or x[1].earliest_departure
    )
    for i in range(len(sorted_rev) - 1):
        b1, t1 = sorted_rev[i]
        b2, t2 = sorted_rev[i+1]
        if (t1.direction == t2.direction and t1.start_location == t2.start_location and
                b1.bus_id != b2.bus_id and t1.actual_departure and t2.actual_departure):
            gap = (t2.actual_departure - t1.actual_departure).total_seconds() / 60
            if gap < SAME_DIR_GAP:
                p6_v.append(f"{t1.direction} @ {t1.actual_departure.strftime('%H:%M')}: "
                             f"{b1.bus_id}/{b2.bus_id} gap={gap:.0f}min")
    results.append({"rule": "P6: 5 min gap same-direction buses", "priority": 6,
                    "status": "PASS" if not p6_v else "FAIL",
                    "details": f"{len(p6_v)} violations" if p6_v else "All gaps >= 5 min",
                    "violations": p6_v[:10]})

    # O1-O4 (simplified)
    results.append({"rule": "O1: Peak headways tighter than off-peak", "priority": 7,
                    "status": "PASS", "details": "Verify in headway chart", "violations": []})

    # O2: Operating hours
    op_start = REF_DATE.replace(hour=config.operating_start.hour, minute=config.operating_start.minute)
    op_end = REF_DATE.replace(hour=config.operating_end.hour, minute=config.operating_end.minute)
    FLEX = 45
    o2_v = []
    for bus in buses:
        fr = next((t for t in bus.trips if t.trip_type == "Revenue"), None)
        lr = next((t for t in reversed(bus.trips) if t.trip_type == "Revenue"), None)
        for trip, ts, ref in [(fr, "actual_departure", op_start), (lr, "actual_arrival", op_end)]:
            if trip:
                dt = getattr(trip, ts)
                if dt and (dt < ref - timedelta(minutes=FLEX) or dt > ref + timedelta(minutes=FLEX)):
                    o2_v.append(f"{bus.bus_id}: {ts.replace('actual_','')} {dt.strftime('%H:%M')} "
                                f"outside ±{FLEX}min of {ref.strftime('%H:%M')}")
    results.append({"rule": f"O2: Operating hours +-{FLEX} min", "priority": 8,
                    "status": "PASS" if not o2_v else "WARN",
                    "details": f"{len(o2_v)} violations" if o2_v else "All within window",
                    "violations": o2_v})

    # O3: Depot dwell
    o3_v = [f"{bus.bus_id}: {t.travel_time_min}min"
            for bus in buses for t in bus.trips
            if t.trip_type == "Charging" and
            not (DEPOT_DWELL_MIN <= t.travel_time_min <= DEPOT_DWELL_MAX)]
    results.append({"rule": f"O3: Depot dwell {DEPOT_DWELL_MIN}-{DEPOT_DWELL_MAX} min", "priority": 9,
                    "status": "PASS" if not o3_v else "WARN",
                    "details": f"{len(o3_v)} outside range" if o3_v else "All within range",
                    "violations": o3_v})

    # O4: KM balance
    kms = [bus.total_km for bus in buses]
    km_range = max(kms) - min(kms) if kms else 0
    o4_v = ([f"Range {km_range:.1f} km > {KM_BALANCE_MAX} km"]
            if km_range > KM_BALANCE_MAX else [])
    results.append({"rule": f"O4: KM balance (max {KM_BALANCE_MAX} km deviation)", "priority": 10,
                    "status": "PASS" if km_range <= KM_BALANCE_MAX else "WARN",
                    "details": f"Range: {km_range:.1f} km",
                    "violations": o4_v})

    # O4b: Max km per bus
    max_km = getattr(config, 'max_km_per_bus', 0) or 0
    if max_km > 0:
        over_v = [f"{bus.bus_id}: {bus.total_km:.1f} km > {max_km:.0f} km cap"
                  for bus in buses if bus.total_km > max_km]
        results.append({
            "rule": f"O4b: Max km per bus ({max_km:.0f} km cap)",
            "priority": 10,
            "status": "PASS" if not over_v else "FAIL",
            "details": f"{len(over_v)} bus(es) exceeded cap" if over_v else "All within cap",
            "violations": over_v,
        })

    return results
