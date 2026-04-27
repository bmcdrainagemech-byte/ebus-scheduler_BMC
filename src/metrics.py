"""
metrics.py — Computes schedule quality KPIs.

Usage:
    from src.metrics import compute_metrics
    m = compute_metrics(config, buses)
    print(m)
    print(f"Score: {m.weighted_score():.3f}")
"""

from __future__ import annotations
__version__ = "2026-04-09-p4"

from dataclasses import dataclass, field
from src.models import BusState, RouteConfig


@dataclass
class ScheduleMetrics:
    """KPI summary for a complete schedule. Lower score = better."""

    # Core metrics
    revenue_trips_assigned: int = 0
    revenue_trips_total: int = 0
    total_km: float = 0.0
    revenue_km: float = 0.0
    dead_km: float = 0.0
    dead_km_ratio: float = 0.0          # dead_km / total_km
    dead_trips: int = 0                  # number of dead run legs

    # Fleet balance
    km_per_bus: list[float] = field(default_factory=list)
    km_range: float = 0.0               # max - min bus km
    km_std: float = 0.0                 # standard deviation of bus km
    trips_per_bus: list[int] = field(default_factory=list)

    # Break quality
    breaks_at_minimum: int = 0          # count of breaks exactly at min_layover
    avg_break_min: float = 0.0
    negative_breaks: int = 0            # should always be 0

    # SOC
    min_soc_seen: float = 100.0         # lowest SOC any bus reached
    buses_below_trigger: int = 0        # buses that dropped below trigger SOC
    charging_stops: int = 0

    # Headway evenness (p4 additions)
    max_headway_gap_min: float = 0.0    # largest single departure gap (UP or DN)
    headway_cv: float = 0.0             # coefficient of variation of gaps (0=perfect)

    # ── Phase 1: Schedule-derived passenger & energy metrics ──────────
    avg_wait_min: float = 0.0           # mean(headway) / 2 — avg expected passenger wait
    worst_wait_min: float = 0.0         # max gap — worst-case service hole
    ewt_proxy: float = 0.0             # Σ max(0, gap_i - hw_i) / N — excess waiting time
    kwh_per_rev_km: float = 0.0        # total energy consumed / revenue km
    los_grade: str = ""                 # LOS A–F from headway_cv (TCQSM thresholds)
    service_reliability_idx: float = 0.0  # composite: 0.35×hw_reg + 0.35×trip_comp + 0.30×soc_rel

    # Optimizer weights (configurable)
    WEIGHTS: dict = field(default_factory=lambda: {
        "dead_km_ratio":          0.25,
        "km_range_norm":          0.25,   # normalised by avg km
        "breaks_at_minimum_pct":  0.10,
        "soc_penalty":            0.20,
        "headway_cv":             0.20,   # NEW: penalise uneven headways
    }, repr=False)

    def weighted_score(self) -> float:
        """Single number for optimizer. Lower = better."""
        w = self.WEIGHTS
        avg_km = sum(self.km_per_bus) / len(self.km_per_bus) if self.km_per_bus else 1.0
        km_range_norm = self.km_range / avg_km if avg_km > 0 else 0.0

        total_breaks = self.breaks_at_minimum + max(1, len(self.km_per_bus))
        breaks_pct = self.breaks_at_minimum / total_breaks

        # SOC penalty: higher if buses go below min_soc or many below trigger
        soc_penalty = (100 - self.min_soc_seen) / 100 + self.buses_below_trigger * 0.1

        # headway_cv already normalised (0 = perfect even spacing)
        return (
            w["dead_km_ratio"]          * self.dead_km_ratio
            + w["km_range_norm"]        * km_range_norm
            + w["breaks_at_minimum_pct"] * breaks_pct
            + w["soc_penalty"]          * soc_penalty
            + w["headway_cv"]           * min(self.headway_cv, 2.0)  # cap at 2
        )

    def summary(self) -> str:
        lines = [
            f"Revenue trips: {self.revenue_trips_assigned}/{self.revenue_trips_total}",
            f"Total km: {self.total_km:.1f} (revenue: {self.revenue_km:.1f}, dead: {self.dead_km:.1f})",
            f"Dead km ratio: {self.dead_km_ratio:.1%}",
            f"KM per bus: {', '.join(f'{k:.1f}' for k in self.km_per_bus)}",
            f"KM range: {self.km_range:.1f} (std: {self.km_std:.1f})",
            f"Breaks at minimum: {self.breaks_at_minimum}",
            f"Min SOC seen: {self.min_soc_seen:.1f}%",
            f"Charging stops: {self.charging_stops}",
            f"Max headway gap: {self.max_headway_gap_min:.0f} min",
            f"Headway CV: {self.headway_cv:.3f}",
            f"Weighted score: {self.weighted_score():.4f}",
        ]
        return "\n".join(lines)


def compute_metrics(
    config: RouteConfig,
    buses: list[BusState],
    total_revenue_trips: int | None = None,
    assigned_revenue_trips: int | None = None,
    headway_df=None,
) -> ScheduleMetrics:
    """
    Compute all KPIs from a completed schedule.
    assigned_revenue_trips: override for bus-driven scheduler where Trip objects
    are created fresh and not drawn from the pool.
    """
    m = ScheduleMetrics()

    all_trips = []
    for bus in buses:
        all_trips.extend(bus.trips)

    rev = [t for t in all_trips if t.trip_type == "Revenue"]
    dead = [t for t in all_trips if t.trip_type == "Dead"]
    chg = [t for t in all_trips if t.trip_type == "Charging"]

    m.revenue_trips_assigned = assigned_revenue_trips if assigned_revenue_trips is not None else len(rev)
    m.revenue_trips_total = total_revenue_trips or len(rev)
    m.revenue_km = sum(t.distance_km for t in rev)
    m.dead_km = sum(t.distance_km for t in dead)
    m.dead_trips = len(dead)
    m.total_km = m.revenue_km + m.dead_km
    m.dead_km_ratio = m.dead_km / m.total_km if m.total_km > 0 else 0.0
    m.charging_stops = len(chg)

    # Per-bus metrics
    m.km_per_bus = [bus.total_km for bus in buses]
    m.trips_per_bus = [len(bus.trips) for bus in buses]

    if m.km_per_bus:
        m.km_range = max(m.km_per_bus) - min(m.km_per_bus)
        avg = sum(m.km_per_bus) / len(m.km_per_bus)
        m.km_std = (sum((k - avg) ** 2 for k in m.km_per_bus) / len(m.km_per_bus)) ** 0.5

    # Break analysis
    for bus in buses:
        for i in range(len(bus.trips) - 1):
            curr = bus.trips[i]
            nxt = bus.trips[i + 1]
            if curr.actual_arrival and nxt.actual_departure:
                gap = (nxt.actual_departure - curr.actual_arrival).total_seconds() / 60
                if gap < 0:
                    m.negative_breaks += 1
                elif abs(gap - config.min_layover_min) < 1:
                    m.breaks_at_minimum += 1

    # SOC tracking — reconstruct from trips
    for bus in buses:
        soc = config.initial_soc_percent
        for trip in bus.trips:
            soc -= (trip.distance_km * config.consumption_rate / config.battery_kwh) * 100
            if trip.trip_type == "Charging":
                charge_kwh = config.depot_flow_rate_kw * (trip.travel_time_min / 60)
                soc = min(100.0, soc + (charge_kwh / config.battery_kwh) * 100)
            m.min_soc_seen = min(m.min_soc_seen, soc)
        if soc < config.trigger_soc_percent:
            m.buses_below_trigger += 1

    # ── Headway gap statistics ────────────────────────────────────────────────
    # Compute over all consecutive same-direction revenue departures.
    all_gaps: list[float] = []
    for direction in ("UP", "DN"):
        deps = sorted([
            t.actual_departure
            for b in buses for t in b.trips
            if t.trip_type == "Revenue" and t.direction == direction
            and t.actual_departure is not None
        ])
        for i in range(1, len(deps)):
            gap = (deps[i] - deps[i - 1]).total_seconds() / 60
            if gap > 0:
                all_gaps.append(gap)

    if all_gaps:
        m.max_headway_gap_min = max(all_gaps)
        avg_gap = sum(all_gaps) / len(all_gaps)
        if avg_gap > 0 and len(all_gaps) > 1:
            variance = sum((g - avg_gap) ** 2 for g in all_gaps) / len(all_gaps)
            m.headway_cv = (variance ** 0.5) / avg_gap
    # ─────────────────────────────────────────────────────────────────────────

    # ── Phase 1: Schedule-derived passenger & energy metrics ──────────────
    # avg_wait = mean(headway) / 2 — standard transit planning formula
    if all_gaps:
        avg_gap = sum(all_gaps) / len(all_gaps)
        m.avg_wait_min = round(avg_gap / 2.0, 1)
        m.worst_wait_min = round(max(all_gaps), 1)

    # EWT proxy: excess waiting time = Σ max(0, actual_gap - scheduled_hw) / N
    # Requires headway_df to know what the scheduled headway was at each departure
    if all_gaps and headway_df is not None:
        try:
            from datetime import datetime as _dt
            _REF = _dt(2025, 1, 1)

            # Build headway lookup from headway_df
            hw_bands = []
            for _, row in headway_df.iterrows():
                t_from = row["time_from"]
                hw = float(row["headway_min"])
                if hasattr(t_from, "hour"):
                    mins = t_from.hour * 60 + t_from.minute
                else:
                    parts = str(t_from).split(":")
                    mins = int(parts[0]) * 60 + int(parts[1])
                hw_bands.append((mins, hw))
            hw_bands.sort()

            def _hw_at(dep_min):
                hw = hw_bands[0][1] if hw_bands else 30.0
                for start, h in hw_bands:
                    if dep_min >= start:
                        hw = h
                return hw

            ewt_sum = 0.0
            ewt_count = 0
            for direction in ("UP", "DN"):
                deps = sorted([
                    t.actual_departure
                    for b in buses for t in b.trips
                    if t.trip_type == "Revenue" and t.direction == direction
                    and t.actual_departure is not None
                ])
                for i in range(1, len(deps)):
                    gap = (deps[i] - deps[i - 1]).total_seconds() / 60
                    dep_min = deps[i].hour * 60 + deps[i].minute
                    scheduled_hw = _hw_at(dep_min)
                    ewt_sum += max(0.0, gap - scheduled_hw)
                    ewt_count += 1

            if ewt_count > 0:
                m.ewt_proxy = round(ewt_sum / ewt_count, 2)
        except Exception:
            pass  # EWT is non-critical; don't fail metrics on parse error

    # Energy efficiency: kWh per revenue km.
    # Z.ai #4.1.1 fix: numerator now uses revenue_km only (the energy
    # actually consumed serving paying passengers), not total_km. Mixing
    # total_km in the numerator inflated the metric by ~dead_km_ratio %
    # (typically 10-20%), making routes with longer depot positioning look
    # less energy-efficient than they actually are.
    if m.revenue_km > 0:
        revenue_energy_kwh = m.revenue_km * config.consumption_rate
        m.kwh_per_rev_km = round(revenue_energy_kwh / m.revenue_km, 3)
        # Note: this is now mathematically equal to consumption_rate by
        # construction. Kept as an explicit field for forward compat —
        # future temperature/SOH derating factors will diverge from
        # consumption_rate and the metric will become non-trivial.

    # LOS grade from headway_cv (TCQSM thresholds)
    m.los_grade = _los_from_cv(m.headway_cv)

    # Service reliability index: composite metric
    # 0.35 × headway_regularity + 0.35 × trip_completion + 0.30 × soc_reliability
    hw_reg = max(0.0, 1.0 - m.headway_cv)  # 1.0 = perfect, 0.0 = CV >= 1
    trip_comp = (m.revenue_trips_assigned / max(1, m.revenue_trips_total))
    soc_rel = 1.0 if m.min_soc_seen >= config.min_soc_percent else (
        m.min_soc_seen / max(1.0, config.min_soc_percent)
    )
    m.service_reliability_idx = round(0.35 * hw_reg + 0.35 * trip_comp + 0.30 * soc_rel, 3)
    # ─────────────────────────────────────────────────────────────────────────

    return m


def _los_from_cv(cv: float) -> str:
    """Map headway CV to LOS grade (TCQSM thresholds)."""
    if cv < 0.10:
        return "A"
    elif cv < 0.20:
        return "B"
    elif cv < 0.35:
        return "C"
    elif cv < 0.50:
        return "D"
    elif cv < 0.75:
        return "E"
    else:
        return "F"
