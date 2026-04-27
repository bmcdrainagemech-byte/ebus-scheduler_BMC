"""
diagnostics.py — Decision-log diagnostic exporter for eBus Scheduler.

Built to debug Bug #1: "Why are buses going for charging after 16:00?"

Exports per-bus decision_log entries to a multi-sheet Excel file, with a
focused sheet on charging events that started after the configured
p5_charging_end. This lets planners see exactly which trigger fired
(SOC_TRIGGER_OFFPEAK, SOC_P3_OVERRIDE, pre-Phase-3 emergency, P5 catch-up)
for every late charge.

Usage:
    from src.diagnostics import build_decision_log_xlsx
    blob = build_decision_log_xlsx(city_schedule)
    st.download_button("Decision log (XLSX)", blob, "decision_log.xlsx")
"""

from __future__ import annotations
__version__ = "2026-04-27-p1"

from datetime import datetime, time
from io import BytesIO

import pandas as pd


# ── Trigger taxonomy — keep in sync with bus_scheduler.py decision_log calls ─
_TRIGGER_KEYS = (
    "SOC_TRIGGER_OFFPEAK",
    "SOC_TRIGGER_PEAK",
    "SOC_P3_OVERRIDE",
    "P3_RESCUE_LOOKAHEAD",
    "P3_RESCUE_STRANDED",
    "EMERGENCY_RESCUE",
    "P5_MIDDAY",
    "PRE_PHASE3_EMERGENCY",
    "SELECTED",
)


def _classify(entry: str) -> str:
    """Return the trigger category for a decision_log entry, or 'OTHER'."""
    upper = entry.upper()
    for key in _TRIGGER_KEYS:
        if key in upper:
            return key
    if "CHARG" in upper:
        return "CHARGE_OTHER"
    return "OTHER"


def _entry_minutes(entry: str) -> int | None:
    """Extract HH:MM from an entry like '14:30 SOC_TRIGGER...'. None if not found."""
    if not entry or len(entry) < 5:
        return None
    try:
        hh = int(entry[0:2])
        mm = int(entry[3:5])
        return hh * 60 + mm
    except (ValueError, IndexError):
        return None


def collect_decision_log_rows(city_schedule) -> list[dict]:
    """
    Walk every bus on every route, emit one row per decision_log entry.

    Columns: Route, Bus, Time, Trigger, Entry, Time_min (sortable).
    """
    rows: list[dict] = []
    for code, rr in city_schedule.results.items():
        for bus in rr.buses:
            for entry in getattr(bus, "decision_log", []) or []:
                t_min = _entry_minutes(entry)
                rows.append({
                    "Route": code,
                    "Bus": bus.bus_id,
                    "Time": entry[:5] if t_min is not None else "",
                    "Trigger": _classify(entry),
                    "Entry": entry,
                    "Time_min": t_min if t_min is not None else 9999,
                })
    rows.sort(key=lambda r: (r["Route"], r["Bus"], r["Time_min"]))
    return rows


def collect_charging_trip_rows(city_schedule) -> list[dict]:
    """
    One row per Charging trip across all routes.

    Includes departure_time, p5_window_end (from config), is_late flag,
    and the *immediately preceding* decision_log entry (which usually
    identifies the trigger that sent the bus to charge).
    """
    rows: list[dict] = []
    for code, rr in city_schedule.results.items():
        cfg = rr.config
        p5_end = getattr(cfg, "p5_charging_end", None) or time(15, 0)
        p5_end_min = p5_end.hour * 60 + p5_end.minute

        for bus in rr.buses:
            log = getattr(bus, "decision_log", []) or []
            for trip in bus.trips:
                if trip.trip_type != "Charging" or trip.actual_departure is None:
                    continue
                dep = trip.actual_departure
                dep_min = dep.hour * 60 + dep.minute

                # Find the most recent decision_log entry at-or-before this charge
                prev_entry = ""
                prev_trigger = "OTHER"
                for e in reversed(log):
                    em = _entry_minutes(e)
                    if em is not None and em <= dep_min:
                        prev_entry = e
                        prev_trigger = _classify(e)
                        break

                rows.append({
                    "Route": code,
                    "Bus": bus.bus_id,
                    "Charge Start": dep.strftime("%H:%M"),
                    "Duration (min)": int(trip.travel_time_min),
                    "Charge End": (
                        trip.actual_arrival.strftime("%H:%M")
                        if trip.actual_arrival else ""
                    ),
                    "P5 Window End": f"{p5_end.hour:02d}:{p5_end.minute:02d}",
                    "Late?": "YES" if dep_min > p5_end_min else "NO",
                    "Min Past P5 End": max(0, dep_min - p5_end_min),
                    "Trigger (from log)": prev_trigger,
                    "Triggering Entry": prev_entry,
                    "_sort": dep_min,
                })

    rows.sort(key=lambda r: (-r["_sort"], r["Route"], r["Bus"]))
    for r in rows:
        r.pop("_sort", None)
    return rows


def build_decision_log_xlsx(city_schedule) -> bytes:
    """
    Build an Excel workbook with three sheets:

      1. All Decisions          — every decision_log entry (all routes, all buses)
      2. Charging Trips         — one row per Charging trip + the trigger entry
      3. Late Charging (Bug #1) — Charging trips that started after p5_charging_end
      4. Trigger Summary        — pivot of count(Charging Trips) by route × trigger

    Returns bytes ready for st.download_button.
    """
    all_rows = collect_decision_log_rows(city_schedule)
    chg_rows = collect_charging_trip_rows(city_schedule)
    late_rows = [r for r in chg_rows if r["Late?"] == "YES"]

    df_all = pd.DataFrame(all_rows).drop(columns=["Time_min"], errors="ignore")
    df_chg = pd.DataFrame(chg_rows)
    df_late = pd.DataFrame(late_rows)

    # Trigger summary pivot
    if not df_chg.empty:
        df_summary = (
            df_chg.assign(_count=1)
            .pivot_table(index="Route", columns="Trigger (from log)",
                         values="_count", aggfunc="sum", fill_value=0)
            .reset_index()
        )
    else:
        df_summary = pd.DataFrame(columns=["Route"])

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        (df_all if not df_all.empty
         else pd.DataFrame([{"Note": "No decision_log entries found."}])
         ).to_excel(xw, sheet_name="All Decisions", index=False)

        (df_chg if not df_chg.empty
         else pd.DataFrame([{"Note": "No charging trips found."}])
         ).to_excel(xw, sheet_name="Charging Trips", index=False)

        (df_late if not df_late.empty
         else pd.DataFrame([{"Note": f"No charging trips started after p5_charging_end."}])
         ).to_excel(xw, sheet_name="Late Charging (Bug 1)", index=False)

        df_summary.to_excel(xw, sheet_name="Trigger Summary", index=False)

    buf.seek(0)
    return buf.getvalue()


def late_charging_count(city_schedule) -> int:
    """Quick scalar — number of charging trips after configured p5_charging_end."""
    return sum(1 for r in collect_charging_trip_rows(city_schedule) if r["Late?"] == "YES")
