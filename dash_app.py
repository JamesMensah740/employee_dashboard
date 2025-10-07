# dash_app.py
from dash import Dash, html, dcc, Input, Output, State, dash_table
import dash
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------- DATA LOADING & HELPERS ----------------------
perf = pd.read_csv("employee_performance.csv")
hier = pd.read_csv("employee_hierarchy.csv")

# Dtypes / cleanup
for col in ["KPI_Score", "Attendance_Rate", "Tenure_Months"]:
    if col in perf.columns:
        perf[col] = pd.to_numeric(perf[col], errors="coerce")
if "Attrition" in perf.columns:
    perf["Attrition"] = perf["Attrition"].astype(str)

def order_months(df: pd.DataFrame) -> pd.DataFrame:
    if "Month" not in df.columns:
        return df
    try:
        dt = pd.to_datetime(df["Month"], format="%b-%Y")
    except Exception:
        try:
            dt = pd.to_datetime(df["Month"])
        except Exception:
            dt = None
    if dt is not None:
        tmp = df.copy()
        tmp["_dt"] = dt
        tmp = tmp.sort_values("_dt")
        cats = tmp["Month"].drop_duplicates().tolist()
        df["Month"] = pd.Categorical(df["Month"], categories=cats, ordered=True)
        return df
    cats = df["Month"].drop_duplicates().tolist()
    df["Month"] = pd.Categorical(df["Month"], categories=cats, ordered=True)
    return df

perf = order_months(perf)

def last_n_months_labels(df: pd.DataFrame, n: int = 3):
    if "Month" not in df.columns:
        return []
    if hasattr(df["Month"], "cat"):
        cats = list(df["Month"].cat.categories)
        return cats[-n:] if len(cats) >= n else cats
    uniq = list(pd.Series(df["Month"]).unique())
    return uniq[-n:] if len(uniq) >= n else uniq

NAME_COL_PERF = "Name" if "Name" in perf.columns else None
NAME_COL_HIER = "Employee" if "Employee" in hier.columns else None

# Merge helpers that avoid Department/Manager collisions
def merge_perf_hier(cols_from_hier):
    cols = ["Employee_ID"] + cols_from_hier
    cols = [c for c in cols if c in hier.columns or c == "Employee_ID"]
    m = perf.merge(hier[cols], on="Employee_ID", how="left")
    # Coalesce Department if ever suffixed
    if "Department" not in m.columns:
        dx, dy = m.get("Department_x"), m.get("Department_y")
        if dx is not None or dy is not None:
            m["Department"] = (dx if dx is not None else pd.Series(index=m.index)).fillna(dy)
            m.drop(columns=[c for c in ["Department_x", "Department_y"] if c in m.columns], inplace=True)
    # Keep Manager from performance (single source of truth)
    return m

# ---------------------- APP ----------------------
app = Dash(__name__, title="Employee Analytics", suppress_callback_exceptions=True)
server = app.server  # for gunicorn

px.defaults.template = "plotly_white"
px.defaults.color_continuous_scale = "Blues"

months_opts = list(perf["Month"].cat.categories) if hasattr(perf["Month"], "cat") else sorted(perf["Month"].unique())
dept_opts = sorted(perf["Department"].dropna().unique()) if "Department" in perf.columns else []
role_opts = ["Manager", "Team_Lead", "Assistant_Lead", "Employee"]
mgr_opts = sorted(perf["Manager"].dropna().unique()) if "Manager" in perf.columns else []
org_dept_opts = sorted(hier["Department"].dropna().unique()) if "Department" in hier.columns else []

def card(title, value):
    return html.Div([
        html.Div(title, style={"fontSize":"12px","color":"#6b7280","marginBottom":"6px"}),
        html.Div(value, style={"fontSize":"22px","fontWeight":"700","color":"#111827"})
    ], style={"border":"1px solid #e6e6e6","borderRadius":"10px","padding":"14px 16px","background":"#fff","boxShadow":"0 1px 0 rgba(0,0,0,.03)"})

# ---------------------- LAYOUT ----------------------
app.layout = html.Div([
    html.H2("Employee Performance & Hierarchy Dashboard", style={"textAlign":"center","color":"#1f77b4"}),
    html.Div("Demo built with synthetic data (Faker). Plotly Dash.", style={"textAlign":"center","marginBottom":"12px"}),

    dcc.Tabs(id="mode", value="overview", children=[
        dcc.Tab(label="Overview", value="overview"),
        dcc.Tab(label="Department Detail", value="dept_detail"),
        dcc.Tab(label="Team Comparison", value="team_comp"),
        dcc.Tab(label="Hierarchy", value="hierarchy"),
        dcc.Tab(label="Org Network", value="org_net"),
    ], style={"marginBottom":"10px"}),

    html.Div(id="controls"),
    html.Hr(),
    html.Div(id="content")
], style={"maxWidth":"1200px","margin":"0 auto","padding":"10px 16px"})

# ---------------------- CONTROLS RENDER ----------------------
@app.callback(
    Output("controls", "children"),
    Input("mode", "value")
)
def render_controls(mode):
    if mode == "overview":
        return html.Div([
            html.Div([
                html.Label("Select Month"),
                dcc.Dropdown(options=[{"label":m, "value":m} for m in months_opts],
                             value=months_opts[-1] if months_opts else None, id="ov-month", clearable=False)
            ], style={"flex":"1","marginRight":"8px"}),
            html.Div([
                html.Label("Select Department"),
                dcc.Dropdown(options=[{"label":"All","value":"All"}]+[{"label":d,"value":d} for d in dept_opts],
                             value="All", id="ov-dept", clearable=False)
            ], style={"flex":"1","marginRight":"8px"}),
            html.Div([
                html.Label("View Type"),
                dcc.RadioItems(
                    options=[{"label":"Current Month","value":"cur"},{"label":"Last 3 Months Avg","value":"avg3"}],
                    value="cur", id="ov-view", inline=True
                )
            ], style={"flex":"1"})
        ], style={"display":"flex","gap":"8px","alignItems":"flex-end"})
    elif mode == "dept_detail":
        return html.Div([
            html.Label("Select Department"),
            dcc.Dropdown(options=[{"label":d,"value":d} for d in dept_opts],
                         value=dept_opts[0] if dept_opts else None, id="dd-dept", clearable=False, style={"maxWidth":"400px"})
        ])
    elif mode == "team_comp":
        return html.Div("No controls for this view.")
    elif mode == "hierarchy":
        # role + person (person options populated by callback below)
        return html.Div([
            html.Div([
                html.Label("Role Level"),
                dcc.Dropdown(options=[{"label":r,"value":r} for r in role_opts], value="Manager",
                             id="hi-role", clearable=False)
            ], style={"width":"280px","marginRight":"8px"}),
            html.Div([
                html.Label("Person"),
                dcc.Dropdown(id="hi-person", clearable=False)
            ], style={"minWidth":"300px","flex":"1","marginRight":"8px"}),
            html.Div([
                html.Label("Team Member (optional)"),
                dcc.Dropdown(id="hi-emp", clearable=True)
            ], style={"minWidth":"300px","flex":"1"})
        ], style={"display":"flex","gap":"8px","alignItems":"flex-end","flexWrap":"wrap"})
    else:  # org_net
        return html.Div([
            html.Div([
                html.Label("Department (optional)"),
                dcc.Dropdown(options=[{"label":"All","value":"All"}]+[{"label":d,"value":d} for d in org_dept_opts],
                             value="All", id="on-dept", clearable=False)
            ], style={"width":"300px","marginRight":"8px"}),
            html.Div([
                html.Label("Manager (optional)"),
                dcc.Dropdown(options=[{"label":"All","value":"All"}]+[{"label":m,"value":m} for m in mgr_opts],
                             value="All", id="on-mgr", clearable=False)
            ], style={"width":"300px"})
        ], style={"display":"flex","gap":"8px","alignItems":"flex-end","flexWrap":"wrap"})

# ---------------------- OVERVIEW CONTENT ----------------------
@app.callback(
    Output("content", "children"),
    Input("mode", "value"),
    State("content", "children"),
    prevent_initial_call=False
)
def init_content(mode, _):
    # Placeholders; real content comes from per-mode callbacks below
    return html.Div(id=f"{mode}-content")

@app.callback(
    Output("overview-content", "children"),
    Input("ov-month", "value"),
    Input("ov-dept", "value"),
    Input("ov-view", "value")
)
def update_overview(month, dept, view):
    if month is None:
        return html.Div("No data.")
    months = list(perf["Month"].cat.categories) if hasattr(perf["Month"], "cat") else sorted(perf["Month"].unique())
    if view == "cur":
        filtered = perf[perf["Month"] == month].copy()
    else:
        end_idx = months.index(month) if month in months else len(months)-1
        start_idx = max(0, end_idx-2)
        last3 = months[start_idx:end_idx+1]
        filtered = perf[perf["Month"].isin(last3)].copy()

    if dept and dept != "All" and "Department" in filtered.columns:
        filtered = filtered[filtered["Department"] == dept]

    if filtered.empty:
        return html.Div("No data for the selected filters.")

    tot_emp = filtered["Employee_ID"].nunique()
    avg_kpi = round(filtered["KPI_Score"].mean(), 2)
    avg_att = round(filtered["Attendance_Rate"].mean(), 2) if "Attendance_Rate" in filtered.columns else None
    attr_rate = round((filtered["Attrition"].eq("Yes").mean()) * 100, 2) if "Attrition" in filtered.columns else None

    # KPI by department
    dept_fig = None
    attr_fig = None
    msg = None
    if "Department" in filtered.columns and "KPI_Score" in filtered.columns:
        kpi_dept = filtered.groupby("Department", dropna=True)["KPI_Score"].mean().reset_index().sort_values("KPI_Score", ascending=False)
        dept_fig = px.bar(kpi_dept, x="Department", y="KPI_Score", title="Average KPI by Department")
        if not kpi_dept.empty:
            td = kpi_dept.iloc[0]
            msg = f"🏆 {td['Department']} leads with average KPI {td['KPI_Score']:.1f}."
        if "Attrition" in filtered.columns:
            attr = filtered.groupby("Department", dropna=True)["Attrition"].apply(lambda x: (x == "Yes").mean()*100).reset_index(name="Attrition_Rate")
            attr_fig = px.bar(attr, x="Department", y="Attrition_Rate", title="Attrition Rate by Department")

    hist_fig = px.histogram(filtered, x="KPI_Score", nbins=20, color="Department" if "Department" in filtered.columns else None, title="KPI Distribution by Department")

    corr_txt = None
    if {"KPI_Score", "Attendance_Rate"}.issubset(filtered.columns):
        corr = filtered[["KPI_Score", "Attendance_Rate"]].corr().iloc[0, 1]
        corr_txt = f"Correlation between KPI and Attendance: {corr:.2f}"

    # Top & lowest
    top_tbl = dash_table.DataTable(
        data=filtered.sort_values("KPI_Score", ascending=False).head(10)[[c for c in [NAME_COL_PERF,"Department","Role","KPI_Score","Attendance_Rate"] if c in filtered.columns]].to_dict("records"),
        columns=[{"name":c,"id":c} for c in [NAME_COL_PERF,"Department","Role","KPI_Score","Attendance_Rate"] if c in filtered.columns],
        page_size=10, style_table={"overflowX":"auto"}
    )
    low_tbl = dash_table.DataTable(
        data=filtered.sort_values("KPI_Score", ascending=True).head(10)[[c for c in [NAME_COL_PERF,"Department","Role","KPI_Score","Attendance_Rate"] if c in filtered.columns]].to_dict("records"),
        columns=[{"name":c,"id":c} for c in [NAME_COL_PERF,"Department","Role","KPI_Score","Attendance_Rate"] if c in filtered.columns],
        page_size=10, style_table={"overflowX":"auto"}
    )

    # Employee trend selector (simple: show the first by default)
    emp_opts = sorted(filtered[NAME_COL_PERF].dropna().unique()) if NAME_COL_PERF else []
    trend_fig = None
    if emp_opts:
        emp = emp_opts[0]
        emp_trend = perf[perf[NAME_COL_PERF] == emp].sort_values("Month")
        trend_fig = px.line(emp_trend, x="Month", y="KPI_Score", title=f"{emp} — KPI Trend", markers=True)

    top = html.Div([
        html.Div(card("👥 Total Employees", f"{tot_emp}"), style={"flex":"1","marginRight":"8px"}),
        html.Div(card("📊 Avg KPI Score", f"{avg_kpi:.2f}"), style={"flex":"1","marginRight":"8px"}),
        html.Div(card("🕒 Avg Attendance (%)", f"{avg_att:.2f}" if avg_att is not None else "-"), style={"flex":"1","marginRight":"8px"}),
        html.Div(card("🚪 Attrition Rate (%)", f"{attr_rate:.2f}" if attr_rate is not None else "-"), style={"flex":"1"})
    ], style={"display":"flex","gap":"8px","flexWrap":"wrap"})

    charts = []
    if dept_fig: charts.append(dcc.Graph(figure=dept_fig))
    if attr_fig: charts.append(dcc.Graph(figure=attr_fig))
    if msg: charts.append(html.Div(msg, style={"background":"#ecfdf5","border":"1px solid #d1fae5","padding":"8px 12px","borderRadius":"6px"}))
    charts.append(dcc.Graph(figure=hist_fig))
    if corr_txt: charts.append(html.Div(corr_txt, style={"background":"#eef2ff","border":"1px solid #e0e7ff","padding":"8px 12px","borderRadius":"6px"}))

    tables = html.Div([
        html.Div([html.H4("Top 10 Performers"), top_tbl], style={"flex":"1","marginRight":"8px"}),
        html.Div([html.H4("Lowest 10 Performers"), low_tbl], style={"flex":"1"})
    ], style={"display":"flex","gap":"8px","flexWrap":"wrap"})

    trend = html.Div([html.H4("Employee Performance Trend (example)"),
                      dcc.Graph(figure=trend_fig) if trend_fig else html.Div("No KPI data available.")])

    return html.Div([top, html.Hr(), *charts, html.Hr(), tables, html.Hr(), trend])

# ---------------------- DEPARTMENT DETAIL ----------------------
@app.callback(
    Output("dept_detail-content", "children"),
    Input("dd-dept", "value")
)
def update_dept_detail(dept):
    if not dept:
        return html.Div("Select a department.")
    df = perf[perf["Department"] == dept].copy()
    if df.empty:
        return html.Div("No data for this department.")
    m_order = list(df["Month"].cat.categories) if hasattr(df["Month"], "cat") else sorted(df["Month"].unique())
    kpi_trend = df.groupby("Month", as_index=False)["KPI_Score"].mean()
    fig_trend = px.line(kpi_trend, x="Month", y="KPI_Score", title=f"{dept} — Average KPI Trend", markers=True)

    merged = merge_perf_hier(["Employee","Assistant_Lead","Team_Lead"])
    d2 = merged[merged["Department"] == dept]
    fig_team = None
    if not d2.empty and "Team_Lead" in d2.columns:
        team_perf = d2.groupby("Team_Lead", dropna=True)["KPI_Score"].mean().reset_index().sort_values("KPI_Score", ascending=False)
        fig_team = px.bar(team_perf, x="Team_Lead", y="KPI_Score", title=f"{dept} — Avg KPI by Team Lead", color="KPI_Score")

    comps = [dcc.Graph(figure=fig_trend)]
    if fig_team: comps.append(dcc.Graph(figure=fig_team))
    return html.Div(comps)

# ---------------------- TEAM COMPARISON ----------------------
@app.callback(
    Output("team_comp-content", "children"),
    Input("mode", "value")
)
def update_team_comp(_):
    merged = merge_perf_hier(["Employee","Assistant_Lead","Team_Lead"])
    if {"Team_Lead","KPI_Score","Department"}.issubset(merged.columns) and not merged.empty:
        team_avg = (merged.dropna(subset=["Team_Lead"])
                         .groupby(["Department","Team_Lead"], dropna=True)["KPI_Score"]
                         .mean().reset_index().sort_values(["Department","KPI_Score"], ascending=[True, False]))
        fig = px.bar(team_avg, x="Team_Lead", y="KPI_Score", color="Department",
                     title="Average KPI by Team Lead Across Departments")
        table = dash_table.DataTable(data=team_avg.to_dict("records"),
                                     columns=[{"name":c,"id":c} for c in team_avg.columns],
                                     page_size=12, style_table={"overflowX":"auto"})
        return html.Div([dcc.Graph(figure=fig), html.Hr(), html.H4("Team Table"), table])
    return html.Div("Team lead information not available.")

# ---------------------- HIERARCHY ----------------------
@app.callback(
    Output("hi-person", "options"),
    Output("hi-person", "value"),
    Input("hi-role", "value")
)
def hi_options(role):
    if role == "Manager" and "Manager" in perf.columns:
        opts = sorted(perf["Manager"].dropna().unique())
    else:
        col = role if role in hier.columns else None
        opts = sorted(hier[col].dropna().unique()) if col else []
    options = [{"label":o, "value":o} for o in opts]
    value = opts[0] if opts else None
    return options, value

@app.callback(
    Output("hi-emp", "options"),
    Output("hi-emp", "value"),
    Input("hi-role", "value"),
    Input("hi-person", "value")
)
def hi_emp_options(role, person):
    merged = merge_perf_hier(["Employee","Assistant_Lead","Team_Lead"])
    if not person:
        return [], None
    if role == "Manager":
        sub_df = merged[merged["Manager"] == person]
    elif role == "Team_Lead":
        sub_df = merged[merged["Team_Lead"] == person]
    elif role == "Assistant_Lead":
        sub_df = merged[merged["Assistant_Lead"] == person]
    else:
        sub_df = merged[merged["Employee"] == person]
    if sub_df.empty or "Employee" not in sub_df.columns:
        return [], None
    members = sorted(sub_df["Employee"].dropna().unique())
    return [{"label":m,"value":m} for m in members], (members[0] if members else None)

@app.callback(
    Output("hierarchy-content", "children"),
    Input("hi-role", "value"),
    Input("hi-person", "value"),
    Input("hi-emp", "value")
)
def update_hierarchy(role, person, emp):
    if not person:
        return html.Div("Select a person.")
    merged = merge_perf_hier(["Employee","Assistant_Lead","Team_Lead"])
    if role == "Manager":
        sub_df = merged[merged["Manager"] == person]
    elif role == "Team_Lead":
        sub_df = merged[merged["Team_Lead"] == person]
    elif role == "Assistant_Lead":
        sub_df = merged[merged["Assistant_Lead"] == person]
    else:
        sub_df = merged[merged["Employee"] == person]

    if sub_df.empty:
        return html.Div("No subordinate or performance data found for this selection.")
    last3 = last_n_months_labels(perf, 3)
    sub_recent = sub_df[sub_df["Month"].isin(last3)] if len(last3) else sub_df.copy()

    name_col = "Employee" if "Employee" in sub_recent.columns else (NAME_COL_PERF or "Employee_ID")
    if "KPI_Score" in sub_recent.columns and name_col in sub_recent.columns:
        avg_team = (sub_recent.groupby(name_col, dropna=True)["KPI_Score"]
                             .mean().reset_index().sort_values("KPI_Score", ascending=False))
    else:
        avg_team = pd.DataFrame(columns=[name_col,"KPI_Score"])

    comps = [html.H4(f"Viewing: {person} ({role})")]
    if not avg_team.empty:
        comps.append(dcc.Graph(figure=px.bar(avg_team, x=name_col, y="KPI_Score",
                                             title=f"{person}'s Team Performance (Last 3 Months)",
                                             color="KPI_Score")))
        top_tbl = dash_table.DataTable(data=avg_team.head(5).to_dict("records"),
                                       columns=[{"name":c,"id":c} for c in avg_team.columns],
                                       page_size=5, style_table={"overflowX":"auto"})
        low_tbl = dash_table.DataTable(data=avg_team.tail(5).to_dict("records"),
                                       columns=[{"name":c,"id":c} for c in avg_team.columns],
                                       page_size=5, style_table={"overflowX":"auto"})
        comps.append(html.Div([
            html.Div([html.H4("🏆 Top Performers"), top_tbl], style={"flex":"1","marginRight":"8px"}),
            html.Div([html.H4("👎 Least Performers"), low_tbl], style={"flex":"1"})
        ], style={"display":"flex","gap":"8px","flexWrap":"wrap"}))
    else:
        comps.append(html.Div("No recent performance data for this team."))

    # Department comparison
    dept_name = sub_df["Department"].iloc[0] if "Department" in sub_df.columns and not sub_df["Department"].isna().all() else None
    if dept_name:
        dept_perf = (merged[merged["Department"] == dept_name]
                          .groupby("Team_Lead", dropna=True)["KPI_Score"]
                          .mean().reset_index().sort_values("KPI_Score", ascending=False))
        if not dept_perf.empty:
            comps.append(dcc.Graph(figure=px.bar(dept_perf, x="Team_Lead", y="KPI_Score",
                                                 title=f"Average KPI by Team Lead — {dept_name}",
                                                 color="KPI_Score", color_continuous_scale="Greens")))
        else:
            comps.append(html.Div("No team-level performance available for this department."))
    else:
        comps.append(html.Div("No department found for this selection."))

    # Optional employee trend
    if role != "Employee" and emp:
        emp_trend = sub_df[sub_df[name_col] == emp].sort_values("Month")
        if not emp_trend.empty:
            comps.append(dcc.Graph(figure=px.line(emp_trend, x="Month", y="KPI_Score",
                                                  title=f"{emp} — 3-Month KPI Trend",
                                                  markers=True)))
    return html.Div(comps)

# ---------------------- ORG NETWORK (SANKEY) ----------------------
@app.callback(
    Output("org_net-content", "children"),
    Input("on-dept", "value"),
    Input("on-mgr", "value")
)
def update_org_net(dept_sel, mgr_sel):
    # Bring Manager from performance into hierarchy, avoid collision
    merged_h = hier.merge(perf[["Employee_ID","Manager"]].drop_duplicates(), on="Employee_ID", how="left", suffixes=("", "_perf"))
    if "Manager_perf" in merged_h.columns:
        merged_h["Manager"] = merged_h["Manager_perf"].fillna(merged_h.get("Manager"))
        merged_h.drop(columns=[c for c in ["Manager_perf"] if c in merged_h.columns], inplace=True)

    df = merged_h.copy()
    if dept_sel and dept_sel != "All" and "Department" in df.columns:
        df = df[df["Department"] == dept_sel]
    if mgr_sel and mgr_sel != "All" and "Manager" in df.columns:
        df = df[df["Manager"] == mgr_sel]

    if df.empty:
        return html.Div("No records for the selected filters.")

    path_counts = (df.groupby(["Manager","Team_Lead","Assistant_Lead"], dropna=True).size().reset_index(name="count"))

    def uniq(series): return [x for x in series.dropna().unique().tolist()]
    managers = uniq(path_counts["Manager"])
    leads = uniq(path_counts["Team_Lead"])
    assistants = uniq(path_counts["Assistant_Lead"])

    nodes = [f"Mgr: {m}" for m in managers] + [f"Lead: {t}" for t in leads] + [f"Asst: {a}" for a in assistants]
    idx_mgr = {m:i for i,m in enumerate(managers)}
    idx_lead = {t:i+len(managers) for i,t in enumerate(leads)}
    idx_asst = {a:i+len(managers)+len(leads) for i,a in enumerate(assistants)}

    # Links
    g1 = path_counts.groupby(["Manager","Team_Lead"], dropna=True)["count"].sum().reset_index()
    g2 = path_counts.groupby(["Team_Lead","Assistant_Lead"], dropna=True)["count"].sum().reset_index()
    src1,tgt1,val1 = [],[],[]
    for _, r in g1.iterrows():
        if pd.isna(r["Manager"]) or pd.isna(r["Team_Lead"]): continue
        src1.append(idx_mgr[r["Manager"]]); tgt1.append(idx_lead[r["Team_Lead"]]); val1.append(int(r["count"]))
    src2,tgt2,val2 = [],[],[]
    for _, r in g2.iterrows():
        if pd.isna(r["Team_Lead"]) or pd.isna(r["Assistant_Lead"]): continue
        src2.append(idx_lead[r["Team_Lead"]]); tgt2.append(idx_asst[r["Assistant_Lead"]]); val2.append(int(r["count"]))
    sources = src1 + src2; targets = tgt1 + tgt2; values = val1 + val2

    if not sources:
        return html.Div("Network too small for current filters.")

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(pad=16, thickness=16, line=dict(color="rgba(0,0,0,.15)", width=1), label=nodes),
        link=dict(source=sources, target=targets, value=values)
    )])

    title_bits = []
    if dept_sel and dept_sel!="All": title_bits.append(f"Dept: {dept_sel}")
    if mgr_sel and mgr_sel!="All": title_bits.append(f"Manager: {mgr_sel}")
    fig.update_layout(title_text=" → ".join(title_bits) if title_bits else "Organization Flow", height=620)

    counts = html.Div([
        html.Div(card("Managers", f"{len(managers)}"), style={"flex":"1","marginRight":"8px"}),
        html.Div(card("Team Leads", f"{len(leads)}"), style={"flex":"1","marginRight":"8px"}),
        html.Div(card("Assistant Leads", f"{len(assistants)}"), style={"flex":"1"})
    ], style={"display":"flex","gap":"8px","flexWrap":"wrap"})

    table = dash_table.DataTable(
        data=path_counts.sort_values("count", ascending=False).to_dict("records"),
        columns=[{"name":c,"id":c} for c in path_counts.columns],
        page_size=12, style_table={"overflowX":"auto"}
    )

    return html.Div([dcc.Graph(figure=fig), html.Hr(), html.H4("Snapshot"), counts, html.Hr(), html.H4("Underlying Paths"), table])

# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=False)
