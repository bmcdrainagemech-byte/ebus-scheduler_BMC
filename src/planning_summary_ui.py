"""
planning_summary_ui.py — Planning Summary renderer for eBus Scheduler.

Professional transit authority style. Data-dense, no decoration.
Renders as the first element on the citywide dashboard.
"""

from __future__ import annotations
__version__ = "2026-04-17-p3"

import streamlit as st
import pandas as pd


_CSS = """
<style>
.ps-wrap { margin-bottom: 1rem; }
.ps-header {
  display: flex; align-items: baseline; gap: 12px;
  border-bottom: 2px solid #1a1a2e; padding-bottom: 5px; margin-bottom: 12px;
}
.ps-title { font-size: 1.05rem; font-weight: 700; color: #1a1a2e; }
.ps-sub { font-size: 0.76rem; color: #888; }
.ps-kpi-strip {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
  gap: 0; border: 1px solid #ddd; border-radius: 3px; overflow: hidden;
  margin-bottom: 12px;
}
.ps-kpi { padding: 8px 12px; border-right: 1px solid #eee; background: #fafbfc; }
.ps-kpi:last-child { border-right: none; }
.ps-kpi-v { font-size: 1.3rem; font-weight: 700; color: #1a1a2e; font-family: 'DM Mono',monospace; line-height: 1.1; }
.ps-kpi-l { font-size: 0.65rem; color: #999; text-transform: uppercase; letter-spacing: .04em; margin-top: 1px; }
.ps-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 5px; }
.ps-dot-g { background: #16a34a; } .ps-dot-y { background: #ca8a04; } .ps-dot-r { background: #dc2626; }
.ps-alert { padding: 6px 10px; border-radius: 2px; margin-bottom: 4px; font-size: 0.8rem; line-height: 1.4; }
.ps-alert-w { background: #fffbeb; border-left: 3px solid #ca8a04; color: #713f12; }
.ps-alert-ok { background: #f0fdf4; border-left: 3px solid #16a34a; color: #166534; }
.ps-sec { font-size: 0.68rem; font-weight: 600; color: #999; text-transform: uppercase; letter-spacing: .05em; margin: 12px 0 5px; padding-bottom: 2px; border-bottom: 1px solid #eee; }
.ps-rec { display: grid; grid-template-columns: 28px 70px 1fr; gap: 0; border-bottom: 1px solid #f5f5f5; padding: 5px 0; font-size: 0.8rem; }
.ps-rec:last-child { border-bottom: none; }
.ps-rp { font-family: monospace; font-weight: 700; text-align: center; }
.ps-rp1{color:#dc2626} .ps-rp2{color:#ea580c} .ps-rp3{color:#ca8a04} .ps-rp4{color:#2563eb} .ps-rp5{color:#9ca3af}
.ps-rr { color: #555; font-weight: 600; }
.ps-ra { color: #333; } .ps-rx { color: #999; font-size: 0.72rem; margin-top: 1px; }
</style>
"""

_LOS_DOT = {"A":"ps-dot-g","B":"ps-dot-g","C":"ps-dot-y","D":"ps-dot-r","E":"ps-dot-r","F":"ps-dot-r"}

def _dot(g): return f'<span class="ps-dot {_LOS_DOT.get(g,"ps-dot-y")}"></span>{g or "\u2014"}'


def render_planning_summary(city_schedule) -> None:
    if city_schedule is None:
        return
    st.markdown(_CSS, unsafe_allow_html=True)
    s = city_schedule.planning_summary
    fl, sq = s["fleet"], s["service_quality"]

    # Header
    st.markdown(
        '<div class="ps-wrap"><div class="ps-header">'
        '<span class="ps-title">Planning summary</span>'
        '<span class="ps-sub">Auto-generated</span></div>',
        unsafe_allow_html=True)

    # KPI strip
    delta = fl["surplus"] if fl["surplus"] > 0 else -fl["deficit"]
    ds = f"+{delta}" if delta > 0 else str(delta) if delta < 0 else "0"
    kpis = [
        (str(len(city_schedule.results)), "Routes"),
        (str(fl["total_allocated"]), "Fleet"),
        (str(fl["total_pvr"]), "PVR"),
        (ds, "Surplus/deficit"),
        (sq["citywide_los"], "Citywide LOS"),
        (str(sq["total_revenue_trips"]), "Revenue trips"),
        (sq["citywide_dead_km_ratio"], "Dead km %"),
        (sq["min_soc_citywide"], "Min SOC"),
    ]
    h = '<div class="ps-kpi-strip">'
    for v, l in kpis:
        h += f'<div class="ps-kpi"><div class="ps-kpi-v">{v}</div><div class="ps-kpi-l">{l}</div></div>'
    st.markdown(h + '</div>', unsafe_allow_html=True)

    # Two columns
    c1, c2 = st.columns([1, 1], gap="medium")

    with c1:
        rb = sq["routes_below_los_c"]
        if rb:
            st.markdown('<div class="ps-sec">Service quality alerts</div>', unsafe_allow_html=True)
            for r in rb[:6]:
                st.markdown(
                    f'<div class="ps-alert ps-alert-w">{_dot(r["los"])} '
                    f'<strong>{r["route"]}</strong> \u2014 '
                    f'gap {r["max_gap_min"]:.0f}\u2009min, '
                    f'wait {r["avg_wait_min"]:.0f}\u2009min, '
                    f'CV\u2009{r["headway_cv"]:.2f}</div>',
                    unsafe_allow_html=True)
        else:
            st.markdown('<div class="ps-alert ps-alert-ok">All routes at LOS C or above</div>',
                        unsafe_allow_html=True)
        w = s.get("warnings", [])
        if w:
            st.markdown('<div class="ps-sec">Warnings</div>', unsafe_allow_html=True)
            for x in w[:5]:
                x = x.lstrip("\u26a0\ufe0f ").strip()
                st.markdown(f'<div class="ps-alert ps-alert-w">{x}</div>', unsafe_allow_html=True)

    with c2:
        recs = getattr(city_schedule, "recommendations", [])
        st.markdown('<div class="ps-sec">Recommendations</div>', unsafe_allow_html=True)
        if recs:
            for rec in recs[:6]:
                rt = ", ".join(rec.route_codes) if rec.route_codes else "All"
                # Phase D #9: expected_impact rendered as hover tooltip via
                # title= attribute on the wrapping div. Plain English, no
                # technical metric jargon.
                _impact = (rec.expected_impact or "").replace('"', "'")
                _conf = (rec.confidence or "").lower()
                _conf_pill = (
                    f'<span class="ps-rc ps-rc-{_conf}">{_conf}</span>'
                    if _conf in ("high", "medium", "low") else ""
                )
                st.markdown(
                    f'<div class="ps-rec" title="Why useful: {_impact}">'
                    f'<div class="ps-rp ps-rp{rec.priority}">P{rec.priority}</div>'
                    f'<div class="ps-rr">{rt}</div>'
                    f'<div><div class="ps-ra">{rec.action} {_conf_pill}</div>'
                    f'<div class="ps-rx">{rec.reason}</div></div></div>',
                    unsafe_allow_html=True)
        else:
            st.markdown('<div class="ps-alert ps-alert-ok">No actionable items</div>',
                        unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")


def render_route_los_table(city_schedule) -> None:
    """Compact LOS + metrics table for Service Quality tab."""
    if city_schedule is None:
        return
    rows = []
    for code in sorted(city_schedule.results):
        r = city_schedule.results[code]
        m = r.metrics
        rows.append({
            "Route": code,
            "Category": getattr(r.config, "route_category", "standard").title(),
            "LOS": getattr(m, "los_grade", ""),
            "CV": round(m.headway_cv, 3),
            "Max gap": f"{m.max_headway_gap_min:.0f}",
            "Avg wait": f"{getattr(m, 'avg_wait_min', 0):.0f}",
            "EWT": f"{getattr(m, 'ewt_proxy', 0):.1f}",
            "kWh/km": f"{getattr(m, 'kwh_per_rev_km', 0):.2f}",
            "Reliability": f"{getattr(m, 'service_reliability_idx', 0):.3f}",
            "Fleet": r.fleet_allocated,
            "PVR": r.pvr,
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
