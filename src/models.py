"""
Core data models for the electric bus scheduling system.

Three dataclasses:
- Trip: immutable container for a single scheduled trip
- BusState: mutable state tracker for each bus across the day
- RouteConfig: parsed route configuration (from YAML + CSV)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta


@dataclass
class Trip:
    """A single trip (revenue, dead-run, or charging) in the schedule."""

    direction: str          # "UP" | "DN" | "DEPOT"
    trip_type: str          # "Revenue" | "Dead" | "Charging"
    start_location: str
    end_location: str
    earliest_departure: datetime
    latest_departure: datetime
    travel_time_min: int
    distance_km: float

    assigned_bus: str | None = None
    actual_departure: datetime | None = None
    actual_arrival: datetime | None = None
    handover: bool = False  # True if trip straddles shift_split
    shift: int | None = None
    full_or_shuttle: str = "Full"
    route_code: str = ""            # citywide: which route this trip belongs to

    def compute_arrival(self) -> datetime:
        """Return arrival based on actual_departure + travel_time."""
        if self.actual_departure is None:
            raise ValueError("Cannot compute arrival: actual_departure not set")
        return self.actual_departure + timedelta(minutes=self.travel_time_min)

    def __repr__(self) -> str:
        bus = self.assigned_bus or "---"
        dep = self.actual_departure.strftime("%H:%M") if self.actual_departure else "??:??"
        return (
            f"Trip({bus} {self.direction} {self.start_location}->{self.end_location} "
            f"@ {dep} [{self.trip_type}])"
        )


class ScheduleInfeasibleError(Exception):
    """Raised when no valid bus assignment exists for a trip."""
    pass


@dataclass
class BusState:
    """
    Mutable state for one bus across the operating day.
    This is the critical class — tracks location, time, SOC, and trip history.
    """

    bus_id: str
    current_location: str
    current_time: datetime
    soc_percent: float
    total_km: float
    shift: int
    battery_kwh: float
    consumption_rate: float     # kWh per km
    trips: list[Trip] = field(default_factory=list)
    route_history: list[str] = field(default_factory=list)  # citywide: routes served [Phase 2 interlining]
    current_route: str = ""                                  # citywide: currently assigned route
    # Per-bus decision log — records WHY each key scheduling decision was made.
    # Populated by bus_scheduler.py at: bus selection, charge triggers, emergency rescue.
    # Use for debugging silent failures, audit trail, and future explainability UI.
    # Format: "<HH:MM> <REASON>" — append only, never mutate existing entries.
    decision_log: list[str] = field(default_factory=list)

    def _soc_cost(self, km: float) -> float:
        """SOC percentage consumed for a given distance."""
        return (km * self.consumption_rate / self.battery_kwh) * 100

    def soc_after_trip(self, distance_km: float) -> float:
        """Projected SOC after travelling a given distance."""
        return self.soc_percent - self._soc_cost(distance_km)

    def idle_gap_min(self, trip: Trip) -> float:
        """Minutes between bus becoming available and trip's earliest departure."""
        delta = trip.earliest_departure - self.current_time
        return delta.total_seconds() / 60

    def can_serve(self, trip: Trip, min_break_min: int = 5, min_soc: float = 20.0) -> bool:
        """
        Check whether this bus can feasibly serve a trip.
        Requirements:
          1. Bus is at the trip's start location
          2. Enough idle gap for the minimum driver break
          3. SOC stays >= min_soc after the trip
        """
        at_right_place = self.current_location == trip.start_location
        gap = self.idle_gap_min(trip)
        soc_after = self.soc_after_trip(trip.distance_km)
        return at_right_place and gap >= min_break_min and soc_after >= min_soc

    def assign(self, trip: Trip) -> None:
        """
        Commit a trip to this bus. Updates location, time, SOC, km, and trip list.
        Sets actual_departure / actual_arrival on the trip object.
        """
        trip.assigned_bus = self.bus_id
        trip.actual_departure = trip.earliest_departure
        trip.actual_arrival = trip.compute_arrival()
        trip.shift = self.shift

        self.current_location = trip.end_location
        self.current_time = trip.actual_arrival
        self.soc_percent -= self._soc_cost(trip.distance_km)
        self.total_km += trip.distance_km
        self.trips.append(trip)

    def charge(self, duration_min: float, flow_rate_kw: float) -> float:
        """
        Apply charging for a given duration. Returns new SOC (capped at 100%).
        flow_rate_kw = charger_capacity_kw * efficiency (already effective rate).
        """
        kwh_added = flow_rate_kw * (duration_min / 60)
        soc_added = (kwh_added / self.battery_kwh) * 100
        self.soc_percent = min(100.0, self.soc_percent + soc_added)
        return self.soc_percent

    def __repr__(self) -> str:
        return (
            f"Bus({self.bus_id} @ {self.current_location} "
            f"t={self.current_time.strftime('%H:%M')} "
            f"SOC={self.soc_percent:.1f}% km={self.total_km:.1f})"
        )


@dataclass
class RouteConfig:
    """All configuration for a single route, parsed from YAML + CSV."""

    # Identity
    route_code: str
    route_name: str

    # Locations
    depot: str
    start_point: str
    end_point: str
    intermediates: list[str]

    # Fleet
    fleet_size: int
    battery_kwh: float
    consumption_rate: float     # kWh per km
    initial_soc_percent: float

    # Charging
    depot_charger_kw: float
    depot_charger_efficiency: float
    terminal_charger_kw: float
    terminal_charger_efficiency: float
    trigger_soc_percent: float      # proactive charging threshold
    target_soc_percent: float       # charge up to this level
    min_soc_percent: float          # hard floor — never go below
    min_charge_duration_min: int

    # Operating hours
    operating_start: time
    operating_end: time
    shift_split: time

    # Layover
    min_layover_min: int
    preferred_layover_min: int

    # Dead run
    dead_run_buffer_min: int

    # Optimizer
    max_headway_deviation_min: int
    km_balance_tolerance_pct: float

    # Distances & times — keyed by "FROM->TO" full location names
    segment_distances: dict[str, float]     # {"DEPOT->GANGAJALIA BUS STAND": 7.2}
    segment_times: dict[str, int]           # {"DEPOT->GANGAJALIA BUS STAND": 25}

    # Lat/lon per location (optional, for OSRM auto-fetch)
    location_coords: dict[str, tuple[float, float]] = field(default_factory=dict)

    # S6: Minimum km each bus must run (0 = no minimum)
    min_km_per_bus: float = 0.0
    max_km_per_bus: float = 0.0          # 0 = disabled; hard cap per bus per day

    # v5: configurable break ceiling (replaces hardcoded MAX_BREAK = 20)
    max_layover_min: int = 20

    # v5: SOC threshold for P5 midday charging window (replaces hardcoded 65.0)
    midday_charge_soc_percent: float = 65.0

    # v5: extra break added during off-peak 11:00–16:00 (0 = disabled)
    off_peak_layover_extra_min: int = 0

    # v5: fallback speed for estimating travel time when segment time is missing
    avg_speed_kmph: float = 30.0

    # v6: configurable P5 midday charging window (replaces hardcoded 12:00-15:00)
    p5_charging_start: time = field(default_factory=lambda: time(12, 0))
    p5_charging_end:   time = field(default_factory=lambda: time(15, 0))

    # v6: rural route — ceil(fleet/2) buses dispatched to rural_node at service start
    # rural_node must be "start_point" or "end_point"
    is_suburban_route: bool = False
    rural_node: str = ""

    # v9: route classification — determines service expectations (min frequency, max gap)
    #   "trunk"    — high-frequency corridor (target LOS A–B)
    #   "standard" — regular urban route (target LOS B–C)
    #   "feeder"   — low-frequency connector (target LOS C–D acceptable)
    route_category: str = "standard"

    # v9: recovery buffer added to cycle_time in Resource Optimization mode
    # Accounts for minor delays (dwell time, traffic). 0 = disabled (default).
    # cycle_time_effective = up + dn + 2×preferred_layover + 2×recovery_buffer
    recovery_buffer_min: int = 0

    # v11 (Phase E): labor-law / driver-rest config.
    # Defaults match common driver-rest regulations: max 12 hours total
    # duty spread, max 4 hours continuous driving, 20-min regulatory
    # break required to reset the continuous-driving counter. Adjust per
    # local labor law via config; 0 = check disabled.
    max_duty_hours: float = 12.0
    max_continuous_driving_min: int = 240    # 4 hours
    regulatory_break_min: int = 20

    # v10 (Phase C): per-route break policy overrides.
    # Optional `Break_Policy` sheet in the route Excel:
    #   Break_Node | Action | Peak_Only | Notes
    # Action ∈ {"Remove", "Add"}.
    #   "Remove" — at this node, do not enforce preferred_layover_min;
    #              treat as fast turnaround instead.
    #   "Add"    — RESERVED for Phase D. Loader accepts the value but the
    #              scheduler currently ignores it (logged but no behaviour).
    # Peak_Only — when True, the rule applies only during peak hours.
    # Each entry stored as a dict for forward-compatibility.
    break_policy: list = field(default_factory=list)

    @property
    def depot_flow_rate_kw(self) -> float:
        return self.depot_charger_kw * self.depot_charger_efficiency

    @property
    def terminal_flow_rate_kw(self) -> float:
        return self.terminal_charger_kw * self.terminal_charger_efficiency

    def bus_ids(self) -> list[str]:
        return [f"B{str(i).zfill(2)}" for i in range(1, self.fleet_size + 1)]

    @staticmethod
    def segment_key(from_loc: str, to_loc: str) -> str:
        """Full-name key: 'DEPOT->GANGAJALIA BUS STAND'."""
        return f"{from_loc.strip()}->{to_loc.strip()}"

    def get_distance(self, from_loc: str, to_loc: str) -> float:
        key = self.segment_key(from_loc, to_loc)
        if key not in self.segment_distances:
            raise KeyError(f"No distance for segment '{key}'")
        return self.segment_distances[key]

    def get_travel_time(self, from_loc: str, to_loc: str) -> int:
        key = self.segment_key(from_loc, to_loc)
        if key not in self.segment_times:
            raise KeyError(f"No travel time for segment '{key}'")
        return self.segment_times[key]
