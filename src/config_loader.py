"""
config_loader.py — Reads the eBus Config Input Excel and returns:
    1. RouteConfig object (all route parameters + segment distances)
    2. headway_df (pandas DataFrame: time_from, time_to, headway_min)
    3. travel_time_df (pandas DataFrame: time_from, time_to, up_min, dn_min)

Supports both the legacy Excel layout and the v5 template layout.

Layout differences handled transparently:

  Field map columns
    Legacy  : column B = parameter name, column C = value
    v5      : column A = parameter name, column B = value

  Travel-time sheet name
    Legacy  : "TravelTime_Profile"
    v5      : "Travel_Time"   (either is accepted)

  Segment table
    Legacy  : embedded in Route_Config, "Sr." header, resolved names in cols G/H
    v5      : standalone "Distances" sheet (A=from, B=to, C=dist_km, D=time_min)

  Coordinates
    Legacy  : embedded in Route_Config with "Location Label" header (cols B-E)
    v5      : standalone "Coordinates" sheet (A=name, B=lat, C=lon)

New fields added in v5 (read with safe defaults so legacy files continue working):
    avg_speed_kmph              default 30.0  — fallback when segment time is missing
    max_layover_min             default 20    — P4 upper break limit
    midday_charge_soc_percent   default 65.0  — P5 trigger SOC
    off_peak_layover_extra_min  default 0     — extra break 11:00-16:00
"""

from __future__ import annotations
__version__ = "2026-03-24-b1"  # auto-stamped

from datetime import time
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from src.models import RouteConfig


class ConfigError(Exception):
    """Raised when the input Excel is missing required data."""
    pass


# ── Helpers ──────────────────────────────────────────────────────────

def _parse_time(val) -> time:
    """Convert 'HH:MM' string or datetime.time to time object."""
    if isinstance(val, time):
        return val
    if isinstance(val, str):
        parts = val.strip().split(":")
        return time(int(parts[0]), int(parts[1]))
    raise ConfigError(f"Cannot parse time from: {val!r}")


def _parse_float(val, field_name: str) -> float:
    if val is None:
        raise ConfigError(f"Missing value for '{field_name}'")
    try:
        return float(val)
    except (ValueError, TypeError):
        raise ConfigError(f"Cannot parse float for '{field_name}': {val!r}")


def _parse_int(val, field_name: str) -> int:
    return int(_parse_float(val, field_name))


def _find_sheet(wb, *names: str):
    """Return the first worksheet whose name matches any of *names (case-insensitive)."""
    name_map = {s.lower(): s for s in wb.sheetnames}
    for candidate in names:
        found = name_map.get(candidate.lower())
        if found:
            return wb[found]
    return None


# ── Field-map builder ────────────────────────────────────────────────

def _build_field_map(ws) -> dict[str, tuple]:
    """
    Scan the Route_Config sheet and build {field_name: (row, value)}.

    Supports two column layouts:
      v5 template : col A = parameter name, col B = value
      Legacy v4   : col B = parameter name, col C = value

    Detection: search the first 60 rows for 'route_code' — whichever column
    contains it defines the label column.  This is reliable because 'route_code'
    is a required field that only appears in the parameter column, unlike section
    headers or legend notes that may incidentally contain underscores.
    """
    label_col = 2   # default: legacy B/C layout
    for row in range(1, min(ws.max_row + 1, 60)):
        if str(ws.cell(row, 1).value or "").strip() == "route_code":
            label_col = 1   # v5 A/B layout
            break
        if str(ws.cell(row, 2).value or "").strip() == "route_code":
            label_col = 2   # legacy B/C layout
            break
    value_col = label_col + 1

    field_map: dict[str, tuple] = {}
    for row in range(1, ws.max_row + 1):
        label = ws.cell(row, label_col).value
        if label is None:
            continue
        # Strip ★ prefix (used in Excel for visual highlighting of new fields)
        # and any surrounding whitespace so lookups work without the prefix.
        label_clean = str(label).strip().lstrip("★").strip()
        field_map[label_clean] = (row, ws.cell(row, value_col).value)
    return field_map


def _get(field_map: dict, key: str):
    """Look up a field value by exact key. Raises ConfigError if missing."""
    if key not in field_map:
        raise ConfigError(f"Field '{key}' not found in Route_Config sheet")
    return field_map[key][1]


def _get_opt(field_map: dict, key: str, default=None):
    """Look up a field value, return default instead of raising if missing."""
    entry = field_map.get(key)
    if entry is None:
        return default
    val = entry[1]
    return val if val is not None else default


# ── Location parser ──────────────────────────────────────────────────

def _parse_locations(ws, wb, field_map: dict):
    """
    Read locations (depot, start, end, intermediates) and coordinates.

    Strategy (tries in order):
      1. Standalone "Coordinates" sheet (v5 layout)
      2. Embedded "Location Label" table in Route_Config (legacy layout)

    Returns:
        locations : dict  {label: name}
        coords    : dict  {name: (lat, lon)}
    """
    locations: dict[str, str] = {}
    coords:    dict[str, tuple] = {}

    # ── Strategy 1: standalone Coordinates sheet ─────────────────────────────
    coord_ws = _find_sheet(wb, "Coordinates", "Coords", "Locations")
    if coord_ws is not None:
        # Expect header row with "location_name" / "latitude" / "longitude"
        # or just read from row 2 onwards with A=name, B=lat, C=lon
        header_found = False
        for r in range(1, coord_ws.max_row + 1):
            v = coord_ws.cell(r, 1).value
            if v and str(v).lower() in ("location_name", "location", "name"):
                header_found = True
                continue
            if v is None:
                break
            name = str(v).strip()
            lat  = coord_ws.cell(r, 2).value
            lon  = coord_ws.cell(r, 3).value
            if name and lat is not None and lon is not None:
                try:
                    coords[name] = (float(lat), float(lon))
                except (ValueError, TypeError):
                    pass

    # Also read location labels from the field map (v5 stores them as plain fields)
    for key in ("depot", "start_point", "end_point",
                "intermediate_1", "intermediate_2"):
        val = _get_opt(field_map, key)
        if val:
            # Map to legacy location label style
            label_map = {
                "depot":          "Depot",
                "start_point":    "Start point",
                "end_point":      "End point",
                "intermediate_1": "Intermediate point 1",
                "intermediate_2": "Intermediate point 2",
            }
            locations[label_map[key]] = str(val).strip()

    # ── Strategy 2: legacy embedded "Location Label" table ───────────────────
    if not locations:
        header_row = None
        for row in range(1, ws.max_row + 1):
            if ws.cell(row, 2).value == "Location Label":
                header_row = row
                break
        if header_row is None:
            # Also try col 1 (v5 layout)
            for row in range(1, ws.max_row + 1):
                if ws.cell(row, 1).value == "Location Label":
                    header_row = row
                    break

        if header_row is not None:
            r = header_row + 1
            while r <= ws.max_row:
                # Determine offset from detection above
                label = ws.cell(r, 2).value
                name  = ws.cell(r, 3).value
                if label is None or name is None:
                    break
                label = str(label).strip()
                name  = str(name).strip()
                locations[label] = name
                lat = ws.cell(r, 4).value
                lon = ws.cell(r, 5).value
                if lat is not None and lon is not None:
                    try:
                        coords[name] = (float(lat), float(lon))
                    except (ValueError, TypeError):
                        pass
                r += 1

    return locations, coords


# ── Segment parser ───────────────────────────────────────────────────

def _parse_segments(ws, wb, avg_speed_kmph: float = 30.0, field_map: dict = None):
    """
    Read segment distances and travel times.
    Generic labels (Start point, End point, Intermediate 1, Intermediate 2, Depot)
    are resolved to actual location names via field_map in both strategies.
    """
    distances: dict[str, float] = {}
    times:     dict[str, int]   = {}

    # ── Generic label resolver (used by both strategies) ─────────────────────
    _label_map = {
        "depot":                "depot",
        "start point":          "start_point",
        "end point":            "end_point",
        "intermediate 1":       "intermediate_1",
        "intermediate 2":       "intermediate_2",
        "intermediate point 1": "intermediate_1",
        "intermediate point 2": "intermediate_2",
    }

    def _resolve(label) -> str | None:
        if not label: return None
        s = str(label).strip()
        key = _label_map.get(s.lower())
        if key and field_map:
            val = _get_opt(field_map, key)
            if val and str(val).strip():
                return str(val).strip()
        return s  # actual name — pass through unchanged

    # ── Strategy 1: standalone Distances sheet ────────────────────────────────
    dist_ws = _find_sheet(wb, "Distances", "Segment_Distances", "Segments")
    if dist_ws is not None:
        data_start = None
        for r in range(1, dist_ws.max_row + 1):
            v = dist_ws.cell(r, 1).value
            if v and str(v).lower() not in (
                "from_location", "from", "origin", "segment", "sr", "sr.",
                "segment distances & times", "segment distances and travel times",
            ):
                if dist_ws.cell(r, 2).value is None:
                    continue  # skip section-header rows (merged, no col B)
                data_start = r
                break

        if data_start is not None:
            for r in range(data_start, dist_ws.max_row + 1):
                fr_raw = dist_ws.cell(r, 1).value
                to_raw = dist_ws.cell(r, 2).value
                dist   = dist_ws.cell(r, 3).value
                tt     = dist_ws.cell(r, 4).value

                if fr_raw is None or to_raw is None or dist is None:
                    continue

                fr = _resolve(fr_raw)
                to = _resolve(to_raw)
                if not fr or not to:
                    continue

                try:
                    dist = float(dist)
                except (ValueError, TypeError):
                    continue

                if dist <= 0:
                    continue

                if tt is None or str(tt).strip() == "":
                    tt = round(dist / max(avg_speed_kmph, 1) * 60)
                else:
                    try:
                        tt = int(float(tt))
                    except (ValueError, TypeError):
                        tt = round(dist / max(avg_speed_kmph, 1) * 60)

                key = RouteConfig.segment_key(fr, to)
                distances[key] = dist
                times[key]     = tt

        if distances:
            return distances, times

    # ── Strategy 2: legacy embedded "Sr." table in Route_Config ──────────────
    # Generic label → actual name resolver using field_map
    # e.g. "Depot" → field_map["depot"][1], "Start point" → field_map["start_point"][1]
    generic_to_field = {
        "Depot":                "depot",
        "Start point":          "start_point",
        "End point":            "end_point",
        "Intermediate point 1": "intermediate_1",
        "Intermediate point 2": "intermediate_2",
    }

    def _resolve(label: str) -> str | None:
        """Resolve a generic label to an actual location name."""
        label = str(label).strip()
        fkey  = generic_to_field.get(label)
        if fkey and field_map:
            val = _get_opt(field_map, fkey)
            if val:
                return str(val).strip()
        return label   # fall back to label as-is (handles absolute names)

    header_row = None
    for row in range(1, ws.max_row + 1):
        if ws.cell(row, 2).value == "Sr.":
            header_row = row
            break
    if header_row is None:
        raise ConfigError(
            "No segment data found. Expected either a 'Distances' sheet "
            "or an embedded 'Sr.' table in Route_Config."
        )

    r = header_row + 1
    while r <= ws.max_row:
        sr = ws.cell(r, 2).value
        if sr is None:
            break
        try:
            int(sr)
        except (ValueError, TypeError):
            break

        # Try resolved names from cols G/H first (older legacy format),
        # then fall back to resolving generic labels from cols C/D via field_map.
        res_from_raw = ws.cell(r, 7).value
        res_to_raw   = ws.cell(r, 8).value
        generic_from = ws.cell(r, 3).value
        generic_to   = ws.cell(r, 4).value

        if res_from_raw and res_to_raw:
            res_from = str(res_from_raw).strip()
            res_to   = str(res_to_raw).strip()
        elif generic_from and generic_to:
            res_from = _resolve(generic_from)
            res_to   = _resolve(generic_to)
        else:
            r += 1
            continue

        dist = ws.cell(r, 5).value
        tt   = ws.cell(r, 6).value

        if dist is None:
            r += 1
            continue

        try:
            dist = float(dist)
        except (ValueError, TypeError):
            r += 1
            continue

        if dist <= 0:
            r += 1
            continue

        if tt is None:
            tt = round(dist / max(avg_speed_kmph, 1) * 60)
        else:
            try:
                tt = int(float(tt))
            except (ValueError, TypeError):
                tt = round(dist / max(avg_speed_kmph, 1) * 60)

        if res_from and res_to:
            key = RouteConfig.segment_key(res_from, res_to)
            distances[key] = dist
            times[key]     = tt

        r += 1

    return distances, times


# ── Headway parser ───────────────────────────────────────────────────

def _parse_headway(wb) -> pd.DataFrame:
    """Read Headway_Profile sheet into a DataFrame."""
    ws = _find_sheet(wb, "Headway_Profile", "Headway")
    if ws is None:
        raise ConfigError("Missing required sheet: 'Headway_Profile'")

    # Find header row containing "Time From" in col B or col A
    header_row = None
    for row in range(1, ws.max_row + 1):
        for col in (1, 2):
            if str(ws.cell(row, col).value or "").strip().lower() in ("time from", "time_from"):
                header_row = row
                # Detect which column the data starts in
                data_col = col
                break
        if header_row:
            break

    if header_row is None:
        raise ConfigError("Cannot find 'Time From' header in Headway_Profile")

    rows = []
    r = header_row + 1
    while r <= ws.max_row:
        tf = ws.cell(r, data_col).value
        tt = ws.cell(r, data_col + 1).value
        hw = ws.cell(r, data_col + 2).value
        if tf is None:
            break
        try:
            rows.append({
                "time_from":   str(tf).strip(),
                "time_to":     str(tt).strip(),
                "headway_min": int(float(hw)),
            })
        except (ValueError, TypeError):
            pass
        r += 1

    if not rows:
        raise ConfigError("Headway_Profile sheet has no data rows")

    return pd.DataFrame(rows)


# ── Travel time parser ───────────────────────────────────────────────

def _parse_travel_time(wb) -> pd.DataFrame:
    """Read TravelTime_Profile or Travel_Time sheet into a DataFrame."""
    ws = _find_sheet(wb, "TravelTime_Profile", "Travel_Time",
                     "TravelTime", "Travel Time")
    if ws is None:
        raise ConfigError(
            "Missing required sheet: expected 'TravelTime_Profile' or 'Travel_Time'"
        )

    header_row = None
    data_col = 2
    for row in range(1, ws.max_row + 1):
        for col in (1, 2):
            if str(ws.cell(row, col).value or "").strip().lower() in ("time from", "time_from"):
                header_row = row
                data_col = col
                break
        if header_row:
            break

    if header_row is None:
        raise ConfigError("Cannot find 'Time From' header in travel-time sheet")

    rows = []
    r = header_row + 1
    while r <= ws.max_row:
        tf = ws.cell(r, data_col).value
        tt = ws.cell(r, data_col + 1).value
        up = ws.cell(r, data_col + 2).value
        dn = ws.cell(r, data_col + 3).value
        if tf is None:
            break
        try:
            rows.append({
                "time_from": str(tf).strip(),
                "time_to":   str(tt).strip(),
                "up_min":    int(float(up)),
                "dn_min":    int(float(dn)),
            })
        except (ValueError, TypeError):
            pass
        r += 1

    if not rows:
        raise ConfigError("Travel-time sheet has no data rows")

    return pd.DataFrame(rows)


# ── Main entry point ─────────────────────────────────────────────────

def load_config(excel_path: str | Path) -> tuple[RouteConfig, pd.DataFrame, pd.DataFrame]:
    """
    Read the eBus Config Input Excel and return:
        config          RouteConfig object
        headway_df      DataFrame with columns: time_from, time_to, headway_min
        travel_time_df  DataFrame with columns: time_from, time_to, up_min, dn_min

    Accepts both legacy (v4) and v5 template layouts transparently.
    """
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    wb = load_workbook(path, data_only=True)

    # ── Find the main config sheet (accept both name variants) ───────────────
    ws = _find_sheet(wb, "Route_Config", "Config", "route_config")
    if ws is None:
        raise ConfigError(
            f"Missing required sheet: expected 'Route_Config' or 'Config'. "
            f"Found sheets: {wb.sheetnames}"
        )

    # ── Field map ─────────────────────────────────────────────────────────────
    fm = _build_field_map(ws)

    # ── Read avg_speed early — needed by segment parser ───────────────────────
    avg_speed = float(_get_opt(fm, "avg_speed_kmph", 30.0) or 30.0)

    # ── Parse locations and coordinates ───────────────────────────────────────
    locations, coords = _parse_locations(ws, wb, fm)

    depot       = locations.get("Depot")
    start_point = locations.get("Start point")
    end_point   = locations.get("End point")

    if not all([depot, start_point, end_point]):
        raise ConfigError(
            f"Missing core locations. Found: Depot={depot}, "
            f"Start={start_point}, End={end_point}"
        )

    intermediates = []
    for key in ["Intermediate point 1", "Intermediate point 2"]:
        name = locations.get(key, "")
        if name:
            intermediates.append(name)

    # ── Parse segment distances and times ─────────────────────────────────────
    seg_distances, seg_times = _parse_segments(ws, wb,
                                               avg_speed_kmph=avg_speed,
                                               field_map=fm)

    if not seg_distances:
        raise ConfigError("No valid segment distances found")

    # ── Build RouteConfig ─────────────────────────────────────────────────────
    # Legacy field name for consumption rate has a unit suffix; v5 uses plain name
    consumption_key = ("consumption_rate (kWh/km)"
                       if "consumption_rate (kWh/km)" in fm
                       else "consumption_rate")

    fleet_raw = _get(fm, "fleet_size")
    fleet_size = int(float(fleet_raw)) if fleet_raw is not None else 0

    config = RouteConfig(
        route_code=str(_get(fm, "route_code")).strip(),
        route_name=str(_get(fm, "route_name")).strip(),
        depot=depot,
        start_point=start_point,
        end_point=end_point,
        intermediates=intermediates,
        fleet_size=fleet_size,
        battery_kwh=_parse_float(_get(fm, "battery_kwh"), "battery_kwh"),
        consumption_rate=_parse_float(_get(fm, consumption_key), "consumption_rate"),
        initial_soc_percent=_parse_float(
            _get(fm, "initial_soc_percent"), "initial_soc_percent"
        ),
        depot_charger_kw=_parse_float(
            _get(fm, "depot_charger_kw"), "depot_charger_kw"
        ),
        depot_charger_efficiency=_parse_float(
            _get(fm, "depot_charger_efficiency"), "depot_charger_efficiency"
        ),
        terminal_charger_kw=_parse_float(
            _get(fm, "terminal_charger_kw"), "terminal_charger_kw"
        ),
        terminal_charger_efficiency=_parse_float(
            _get(fm, "terminal_charger_efficiency"), "terminal_charger_efficiency"
        ),
        trigger_soc_percent=_parse_float(
            _get(fm, "trigger_soc_percent"), "trigger_soc_percent"
        ),
        target_soc_percent=_parse_float(
            _get(fm, "target_soc_percent"), "target_soc_percent"
        ),
        min_soc_percent=_parse_float(
            _get(fm, "min_soc_percent"), "min_soc_percent"
        ),
        min_charge_duration_min=_parse_int(
            _get(fm, "min_charge_duration_min"), "min_charge_duration_min"
        ),
        operating_start=_parse_time(_get(fm, "operating_start")),
        operating_end=_parse_time(_get(fm, "operating_end")),
        shift_split=_parse_time(_get(fm, "shift_split")),
        min_layover_min=_parse_int(
            _get(fm, "min_layover_min"), "min_layover_min"
        ),
        preferred_layover_min=_parse_int(
            _get(fm, "preferred_layover_min"), "preferred_layover_min"
        ),
        dead_run_buffer_min=_parse_int(
            _get(fm, "dead_run_buffer_min"), "dead_run_buffer_min"
        ),
        max_headway_deviation_min=_parse_int(
            _get(fm, "max_headway_deviation_min"), "max_headway_deviation_min"
        ),
        km_balance_tolerance_pct=_parse_float(
            _get(fm, "km_balance_tolerance_pct"), "km_balance_tolerance_pct"
        ),
        segment_distances=seg_distances,
        segment_times=seg_times,
        location_coords=coords,
        # ── New v5 fields — safe defaults when not present in Excel ──────────
        min_km_per_bus=float(
            _get_opt(fm, "min_km_per_bus", 0) or 0
        ),
        max_km_per_bus=float(
            _get_opt(fm, "max_km_per_bus", 0) or 0
        ),
        max_layover_min=int(float(
            _get_opt(fm, "max_layover_min", 20) or 20
        )),
        midday_charge_soc_percent=float(
            _get_opt(fm, "midday_charge_soc_percent", 65.0) or 65.0
        ),
        off_peak_layover_extra_min=int(float(
            _get_opt(fm, "off_peak_layover_extra_min", 0) or 0
        )),
        avg_speed_kmph=avg_speed,
        # v6: configurable P5 charging window
        p5_charging_start=_parse_time(
            _get_opt(fm, "p5_charging_start", "12:00") or "12:00"
        ),
        p5_charging_end=_parse_time(
            _get_opt(fm, "p5_charging_end", "15:00") or "15:00"
        ),
        # v6: rural route dispatch
        is_suburban_route=(
            str(_get_opt(fm, "is_suburban_route", "No") or "No").strip().lower()
            in ("yes", "true", "1", "y")
        ),
        rural_node=str(_get_opt(fm, "rural_node", "") or "").strip().lower(),
    )

    # ── Parse headway and travel time profiles ─────────────────────────────────
    headway_df     = _parse_headway(wb)
    travel_time_df = _parse_travel_time(wb)

    # ── Phase C: parse optional Break_Policy sheet ─────────────────────────────
    # Schema: Break_Node | Action | Peak_Only | Notes
    # If the sheet is absent the field stays empty (no behaviour change for
    # routes that don't opt in).
    config.break_policy = _parse_break_policy(wb, config)

    wb.close()
    return config, headway_df, travel_time_df


def _parse_break_policy(wb, config) -> list[dict]:
    """
    Read optional `Break_Policy` sheet.

    Schema (header row, case-insensitive, any order):
      Break_Node | Action | Peak_Only | Notes

    Action values (case-insensitive): "Remove" | "Add"
      - Remove: at this node, fast turnaround instead of preferred_layover_min
      - Add:    take a break at this intermediate (RESERVED for Phase D — the
                loader accepts the value but the scheduler does not yet act
                on it. This keeps the schema forward-compatible.)
    Peak_Only: "Yes"/"No"/"True"/"False" (case-insensitive). Default No.
    Notes: free text — stored verbatim, never parsed.

    Returns a list of dicts:
      [{"break_node": str, "action": "Remove"|"Add", "peak_only": bool,
        "notes": str}, ...]
    Returns [] if the sheet is missing, empty, or has no valid rows.

    Validation rules:
      - break_node must match an operational node on this route
        (start_point, end_point, or an intermediate). Rules referencing
        unknown nodes are silently dropped — the route still loads.
      - action must be "Remove" or "Add" (case-insensitive). Other values
        cause the row to be dropped.
    """
    ws = _find_sheet(wb, "Break_Policy", "BreakPolicy", "break_policy")
    if ws is None:
        return []

    # Locate header row and column indices
    header_row = None
    col_idx: dict[str, int] = {}
    for r in range(1, min(ws.max_row, 10) + 1):
        row_vals = {
            c: str(ws.cell(r, c).value or "").strip().lower()
            for c in range(1, ws.max_column + 1)
        }
        if "break_node" in row_vals.values() and "action" in row_vals.values():
            header_row = r
            for c, v in row_vals.items():
                if v in ("break_node", "action", "peak_only", "notes"):
                    col_idx[v] = c
            break

    if header_row is None:
        return []

    if "break_node" not in col_idx or "action" not in col_idx:
        return []

    # Build the set of operational nodes for validation
    op_nodes = {config.start_point, config.end_point}
    for n in (config.intermediates or []):
        if n and str(n).strip():
            op_nodes.add(str(n).strip())

    out: list[dict] = []
    for r in range(header_row + 1, ws.max_row + 1):
        node_raw = ws.cell(r, col_idx["break_node"]).value
        action_raw = ws.cell(r, col_idx["action"]).value
        if node_raw is None or str(node_raw).strip() == "":
            # Empty row — treat as end-of-data sentinel
            break

        node = str(node_raw).strip()
        action = str(action_raw or "").strip().lower()
        if action not in ("remove", "add"):
            continue
        if node not in op_nodes:
            # Silently drop rules pointing at nodes we don't know about.
            continue

        peak_only = False
        if "peak_only" in col_idx:
            v = ws.cell(r, col_idx["peak_only"]).value
            peak_only = str(v or "").strip().lower() in ("yes", "true", "y", "1")

        notes = ""
        if "notes" in col_idx:
            v = ws.cell(r, col_idx["notes"]).value
            notes = str(v or "").strip()

        out.append({
            "break_node": node,
            "action": action.capitalize(),   # "Remove" | "Add"
            "peak_only": peak_only,
            "notes": notes,
        })

    return out
