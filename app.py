# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ======================== APP CONFIG / THEME ========================
st.set_page_config(
    page_title="Employee Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

px.defaults.template = "plotly_white"
px.defaults.color_continuous_scale = "Blues"

st.markdown("""
<style>
.card { border: 1px solid #e6e6e6; border-radius: 10px; padding: 14px 16px; background: #fff; box-shadow: 0 1px 0 rgba(0,0,0,.03); }
.kpi-title { font-size: 12px; color: #6b7280; margin-bottom: 6px; }
.kpi-value { font-size: 22px; font-weight: 700; color: #111827; margin-bottom: 0; }
.section-title { margin: 4px 0 8px 0; }
hr { margin: .8rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center;color:#1f77b4;'>Employee Performance & Hierarchy Dashboard</h2>", unsafe_allow_html=True)
st.caption("Demo built with synthetic data (Faker). Streamlit + Plotly. By James Mensah.")

# ======================== HELPERS ========================
def order_months(perf: pd.DataFrame) -> pd.DataFrame:
    perf = perf.copy()
    if "Month" not in perf.columns:
        return perf
    month_dt = None
    try:
        month_dt = pd.to_datetime(perf["Month"], format="%b-%Y")
    except Exception:
        try:
            month_dt = pd.to_datetime(perf["Month"])
        except Exception:
            month_dt = None
    if month_dt is not None:
        perf["_Month_dt"] = month_dt
        perf = perf.sort_values("_Month_dt")
        ordered_labels = perf["Month"].drop_duplicates().tolist()
        perf["Month"] = pd.Categorical(perf["Month"], categories=ordered_labels, ordered=True)
        perf = perf.drop(columns=["_Month_dt"])
    else:
        ordered_labels = perf["Month"].drop_duplicates().tolist()
        perf["Month"] = pd.Categorical(perf["Month"], categories=ordered_labels, ordered=True)
    return perf

def last_n_month_labels(perf: pd.DataFrame, n: int = 3):
    if "Month" not in perf.columns:
        return []
    if hasattr(perf["Month"], "cat"):
        cats = list(perf["Month"].cat.categories)
        return cats[-n:] if len(cats) >= n else cats
    uniq = list(pd.Series(perf["Month"]).unique())
    return uniq[-n:] if len(uniq) >= n else uniq

def metric_card(title, value):
    st.markdown(f"""
    <div class="card">
      <div class="kpi-title">{title}</div>
      <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ======================== LOAD DATA ========================
@st.cache_data
def load_performance():
    df = pd.read_csv("employee_performance.csv")
    for col in ["KPI_Score", "Attendance_Rate", "Tenure_Months"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Attrition" in df.columns:
        df["Attrition"] = df["Attrition"].astype(str)
    df = order_months(df)
    return df

@st.cache_data
def load_hierarchy():
    return pd.read_csv("employee_hierarchy.csv")

perf = load_performance()
hier = load_hierarchy()
NAME_COL_PERF = "Name" if "Name" in perf.columns else None
NAME_COL_HIER = "Employee" if "Employee" in hier.columns else None

# ======================== SIDEBAR ========================
st.sidebar.header("🔧 Controls")
mode = st.sidebar.radio(
    "Dashboard Mode",
    ["Overview", "Department Detail", "Team Comparison", "Hierarchy", "Org Network"],
    help="Switch between different analysis views."
)

dark_mode = st.sidebar.toggle("🌙 Dark mode", value=False)
if dark_mode:
    px.defaults.template = "plotly_dark"

# ======================== MODE: OVERVIEW ========================
if mode == "Overview":
    st.subheader("📈 Performance Overview")

    col1, col2, col3 = st.columns(3)
    months = list(perf["Month"].cat.categories) if hasattr(perf["Month"], "cat") else sorted(perf["Month"].unique())
    depts = sorted(perf["Department"].dropna().unique()) if "Department" in perf.columns else []

    selected_month = col1.selectbox("Select Month", months)
    selected_dept = col2.selectbox("Select Department", ["All"] + depts)
    view_option = col3.radio("View Type", ["Current Month", "Last 3 Months Average"], horizontal=True)

    if view_option == "Current Month":
        filtered = perf[perf["Month"] == selected_month]
    else:
        if selected_month in months:
            end_idx = months.index(selected_month)
            start_idx = max(0, end_idx - 2)
            last3 = months[start_idx:end_idx + 1]
        else:
            last3 = last_n_month_labels(perf, 3)
        filtered = perf[perf["Month"].isin(last3)]

    if selected_dept != "All" and "Department" in perf.columns:
        filtered = filtered[filtered["Department"] == selected_dept]

    if filtered.empty:
        st.warning("No data for the selected filters.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    tot_emp = filtered["Employee_ID"].nunique()
    avg_kpi = round(filtered["KPI_Score"].mean(), 2)
    avg_att = round(filtered["Attendance_Rate"].mean(), 2)
    attr_rate = round((filtered["Attrition"].eq("Yes").mean()) * 100, 2) if "Attrition" in filtered.columns else None

    with c1: metric_card("👥 Total Employees", f"{tot_emp}")
    with c2: metric_card("📊 Avg KPI Score", f"{avg_kpi:.2f}")
    with c3: metric_card("🕒 Avg Attendance (%)", f"{avg_att:.2f}")
    with c4: metric_card("🚪 Attrition Rate (%)", f"{attr_rate:.2f}" if attr_rate is not None else "-")

    st.divider()

    if "Department" in filtered.columns and "KPI_Score" in filtered.columns:
        a, b = st.columns(2)
        kpi_dept = (
            filtered.groupby("Department", dropna=True)["KPI_Score"]
            .mean().reset_index().sort_values("KPI_Score", ascending=False)
        )
        a.plotly_chart(px.bar(kpi_dept, x="Department", y="KPI_Score",
                              title="Average KPI by Department"),
                       use_container_width=True)
        if not kpi_dept.empty:
            td = kpi_dept.iloc[0]
            a.success(f"🏆 **{td['Department']}** leads with average KPI **{td['KPI_Score']:.1f}**.")

        if "Attrition" in filtered.columns:
            attr_dept = (
                filtered.groupby("Department", dropna=True)["Attrition"]
                .apply(lambda x: (x == "Yes").mean() * 100)
                .reset_index(name="Attrition_Rate")
            )
            b.plotly_chart(px.bar(attr_dept, x="Department", y="Attrition_Rate",
                                  title="Attrition Rate by Department"),
                           use_container_width=True)

    if "KPI_Score" in filtered.columns:
        with st.expander("📉 KPI Distribution", expanded=False):
            st.plotly_chart(
                px.histogram(filtered, x="KPI_Score", nbins=20,
                             color="Department" if "Department" in filtered.columns else None,
                             title="KPI Distribution by Department"),
                use_container_width=True
            )

    if {"KPI_Score", "Attendance_Rate"}.issubset(filtered.columns):
        corr = filtered[["KPI_Score", "Attendance_Rate"]].corr().iloc[0, 1]
        st.info(f"Correlation between KPI and Attendance: **{corr:.2f}**")

    if NAME_COL_PERF and "KPI_Score" in filtered.columns:
        st.subheader("🏅 Top & Lowest Performers")
        t1, t2 = st.columns(2)
        cols_show = [c for c in [NAME_COL_PERF, "Department", "Role", "KPI_Score", "Attendance_Rate"] if c in filtered.columns]
        with t1:
            st.markdown("**Top 10 Performers**")
            st.dataframe(filtered.sort_values("KPI_Score", ascending=False).head(10)[cols_show], use_container_width=True)
        with t2:
            st.markdown("**Lowest 10 Performers**")
            st.dataframe(filtered.sort_values("KPI_Score", ascending=True).head(10)[cols_show], use_container_width=True)

    st.subheader("👤 Employee Performance Trend")
    if NAME_COL_PERF:
        emp_list = sorted(filtered[NAME_COL_PERF].dropna().unique())
        if len(emp_list) == 0:
            st.info("No employees available for the selected filters.")
        else:
            selected_emp = st.selectbox("Search Employee", emp_list, index=0, key="emp_trend_select")
            emp_trend = perf[perf[NAME_COL_PERF] == selected_emp].sort_values("Month")
            if not emp_trend.empty:
                st.plotly_chart(px.line(emp_trend, x="Month", y="KPI_Score",
                                        title=f"{selected_emp} — KPI Trend", markers=True),
                                use_container_width=True)
            else:
                st.info("No KPI data available for the selected employee.")

    st.download_button("📥 Download Current View (CSV)", filtered.to_csv(index=False), "filtered_data.csv")

# ======================== MODE: DEPARTMENT DETAIL ========================
elif mode == "Department Detail":
    st.subheader("🏢 Department Detail")

    depts = sorted(perf["Department"].dropna().unique()) if "Department" in perf.columns else []
    if len(depts) == 0:
        st.warning("No departments available.")
        st.stop()

    dept = st.selectbox("Select Department", depts)
    dept_df = perf[perf["Department"] == dept].copy()
    if dept_df.empty:
        st.warning("No data for this department.")
        st.stop()

    months = list(dept_df["Month"].cat.categories) if hasattr(dept_df["Month"], "cat") else sorted(dept_df["Month"].unique())
    st.plotly_chart(px.line(dept_df.groupby("Month", as_index=False)["KPI_Score"].mean(),
                            x="Month", y="KPI_Score",
                            title=f"{dept} — Average KPI Trend (Monthly)",
                            markers=True), use_container_width=True)

    if {"Employee_ID"}.issubset(perf.columns) and {"Employee_ID", "Team_Lead"}.issubset(hier.columns):
        merged = perf.merge(hier[["Employee_ID", "Employee", "Assistant_Lead", "Team_Lead"]], on="Employee_ID", how="left")
        dept_merge = merged[merged["Department"] == dept]
        if not dept_merge.empty and "Team_Lead" in dept_merge.columns:
            team_perf = (
                dept_merge.groupby("Team_Lead", dropna=True)["KPI_Score"]
                .mean().reset_index().sort_values("KPI_Score", ascending=False)
            )
            st.plotly_chart(px.bar(team_perf, x="Team_Lead", y="KPI_Score",
                                   title=f"{dept} — Average KPI by Team Lead",
                                   color="KPI_Score"), use_container_width=True)
        else:
            st.info("No valid team lead data in this department.")

# ======================== MODE: TEAM COMPARISON ========================
elif mode == "Team Comparison":
    st.subheader("👥 Team Comparison (Cross-Department)")

    # Keep Department from performance; exclude it from hierarchy to avoid *_x / *_y suffixes
    merged = perf.merge(
        hier[["Employee_ID", "Employee", "Assistant_Lead", "Team_Lead"]],
        on="Employee_ID",
        how="left",
    )

    # Defensive coalesce in case Department got suffixed elsewhere
    if "Department" not in merged.columns:
        dep_x = merged.get("Department_x")
        dep_y = merged.get("Department_y")
        if (dep_x is not None) or (dep_y is not None):
            merged["Department"] = (dep_x if dep_x is not None else pd.Series(index=merged.index)).fillna(dep_y)
            merged.drop(columns=[c for c in ["Department_x", "Department_y"] if c in merged.columns], inplace=True)

    required_cols = {"Department", "Team_Lead", "KPI_Score"}
    if required_cols.issubset(merged.columns) and not merged.empty:
        team_avg = (
            merged.dropna(subset=["Team_Lead"])
                  .groupby(["Department", "Team_Lead"], dropna=True)["KPI_Score"]
                  .mean()
                  .reset_index()
                  .sort_values(["Department", "KPI_Score"], ascending=[True, False])
        )

        if team_avg.empty:
            st.info("No team data available for the current dataset/filters.")
        else:
            st.plotly_chart(
                px.bar(
                    team_avg, x="Team_Lead", y="KPI_Score", color="Department",
                    title="Average KPI by Team Lead Across Departments"
                ),
                use_container_width=True
            )
            with st.expander("Show team table", expanded=False):
                st.dataframe(team_avg, use_container_width=True)
    else:
        st.info("Team lead information not available.")


# ======================== MODE: HIERARCHY ========================
elif mode == "Hierarchy":
    st.subheader("🏗️ Hierarchy Explorer")

    # Avoid Manager collision by using Manager from performance only
    hierarchy_cols = ["Employee_ID", "Employee", "Assistant_Lead", "Team_Lead"]  # exclude Manager here
    merged = perf.merge(hier[hierarchy_cols], on="Employee_ID", how="left")

    roles = ["Manager", "Team_Lead", "Assistant_Lead", "Employee"]
    role = st.selectbox("Select Role Level", roles)

    # Options by role
    if role == "Manager":
        options = sorted(perf["Manager"].dropna().unique())
    else:
        options = sorted(hier[role].dropna().unique())
    person = st.selectbox(f"Select {role}", options)

    # Filter subordinates
    if role == "Manager":
        sub_df = merged[merged["Manager"] == person]
    elif role == "Team_Lead":
        sub_df = merged[merged["Team_Lead"] == person]
    elif role == "Assistant_Lead":
        sub_df = merged[merged["Assistant_Lead"] == person]
    else:
        name_col = "Employee" if "Employee" in merged.columns else (NAME_COL_PERF or "Employee_ID")
        sub_df = merged[merged[name_col] == person]

    st.markdown(f"### 👥 Viewing: **{person}** ({role})")

    if sub_df.empty:
        st.warning("No subordinate or performance data found for this selection.")
        st.stop()

    last3 = last_n_month_labels(perf, 3)
    sub_recent = sub_df[sub_df["Month"].isin(last3)] if len(last3) else sub_df.copy()

    name_col = "Employee" if "Employee" in sub_recent.columns else (NAME_COL_PERF or "Employee_ID")
    if "KPI_Score" in sub_recent.columns and name_col in sub_recent.columns:
        avg_team = (
            sub_recent.groupby(name_col, dropna=True)["KPI_Score"]
            .mean().reset_index().sort_values("KPI_Score", ascending=False)
        )
    else:
        avg_team = pd.DataFrame(columns=[name_col, "KPI_Score"])

    st.subheader("📊 Intra-Team KPI (Last 3 Months)")
    if not avg_team.empty:
        st.plotly_chart(
            px.bar(avg_team, x=name_col, y="KPI_Score",
                   title=f"{person}'s Team Performance (Last 3 Months)",
                   color="KPI_Score"),
            use_container_width=True
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🏆 Top Performers")
            st.dataframe(avg_team.head(5), use_container_width=True)
        with c2:
            st.markdown("#### 👎 Least Performers")
            st.dataframe(avg_team.tail(5), use_container_width=True)
    else:
        st.info("No recent performance data for this team.")

    st.subheader("🏢 Inter-Team Comparison in Department")
    dept_name = sub_df["Department"].iloc[0] if "Department" in sub_df.columns and not sub_df["Department"].isna().all() else None
    if dept_name:
        dept_perf = (
            merged[merged["Department"] == dept_name]
            .groupby("Team_Lead", dropna=True)["KPI_Score"]
            .mean().reset_index().sort_values("KPI_Score", ascending=False)
        )
        if not dept_perf.empty:
            st.plotly_chart(
                px.bar(dept_perf, x="Team_Lead", y="KPI_Score",
                       title=f"Average KPI by Team Lead — {dept_name}",
                       color="KPI_Score", color_continuous_scale="Greens"),
                use_container_width=True
            )
        else:
            st.info("No team-level performance available for this department.")
    else:
        st.info("No department found for this selection.")

    st.download_button("📥 Download Hierarchy View (CSV)", sub_df.to_csv(index=False), "hierarchy_view.csv")

# ======================== MODE: ORG NETWORK (SANKEY) ========================
else:
    st.subheader("🌐 Org Network (Sankey)")

    # Filters to focus the network
    depts = sorted(hier["Department"].dropna().unique()) if "Department" in hier.columns else []
    dept_sel = st.selectbox("Filter by Department (optional)", ["All"] + depts, index=0)

    # Use Manager list from performance (single source of truth)
    mgrs = sorted(perf["Manager"].dropna().unique()) if "Manager" in perf.columns else []
    mgr_sel = st.selectbox("Focus on Manager (optional)", ["All"] + mgrs, index=0)

    # Base dataframe for hierarchy
    base_cols = ["Employee_ID", "Employee", "Assistant_Lead", "Team_Lead", "Manager", "Department"]
    # Merge to bring Manager from performance to the hierarchy table (avoid Manager collision)
    merged_h = hier.merge(perf[["Employee_ID", "Manager"]].drop_duplicates(), on="Employee_ID", how="left", suffixes=("", "_perf"))
    if "Manager_perf" in merged_h.columns:
        merged_h["Manager"] = merged_h["Manager_perf"].fillna(merged_h.get("Manager"))
        merged_h = merged_h.drop(columns=[c for c in ["Manager_perf"] if c in merged_h.columns])

    # Apply filters
    net_df = merged_h.copy()
    if dept_sel != "All" and "Department" in net_df.columns:
        net_df = net_df[net_df["Department"] == dept_sel]
    if mgr_sel != "All" and "Manager" in net_df.columns:
        net_df = net_df[net_df["Manager"] == mgr_sel]

    if net_df.empty:
        st.warning("No records for the selected filters.")
        st.stop()

    # Build Sankey: Manager -> Team_Lead -> Assistant_Lead -> Employee
    # We'll aggregate by unique paths to reduce node count
    path_counts = (
        net_df.groupby(["Manager", "Team_Lead", "Assistant_Lead"], dropna=True)
        .size()
        .reset_index(name="count")
    )

    # Create node list and mapping
    def unique_nodes(series):
        return [x for x in series.dropna().unique().tolist()]

    managers = unique_nodes(path_counts["Manager"])
    leads = unique_nodes(path_counts["Team_Lead"])
    assistants = unique_nodes(path_counts["Assistant_Lead"])

    # Layered node order for readability
    nodes = (
        [f"Mgr: {m}" for m in managers] +
        [f"Lead: {t}" for t in leads] +
        [f"Asst: {a}" for a in assistants]
    )

    # Index maps
    idx_mgr = {m: i for i, m in enumerate(managers)}
    idx_lead = {t: i + len(managers) for i, t in enumerate(leads)}
    idx_asst = {a: i + len(managers) + len(leads) for i, a in enumerate(assistants)}

    # Links: Manager -> Team Lead
    src1, tgt1, val1 = [], [], []
    g1 = path_counts.groupby(["Manager", "Team_Lead"], dropna=True)["count"].sum().reset_index()
    for _, r in g1.iterrows():
        if pd.isna(r["Manager"]) or pd.isna(r["Team_Lead"]):
            continue
        src1.append(idx_mgr[r["Manager"]])
        tgt1.append(idx_lead[r["Team_Lead"]])
        val1.append(int(r["count"]))

    # Links: Team Lead -> Assistant Lead
    src2, tgt2, val2 = [], [], []
    g2 = path_counts.groupby(["Team_Lead", "Assistant_Lead"], dropna=True)["count"].sum().reset_index()
    for _, r in g2.iterrows():
        if pd.isna(r["Team_Lead"]) or pd.isna(r["Assistant_Lead"]):
            continue
        src2.append(idx_lead[r["Team_Lead"]])
        tgt2.append(idx_asst[r["Assistant_Lead"]])
        val2.append(int(r["count"]))

    # Concatenate links
    sources = src1 + src2
    targets = tgt1 + tgt2
    values = val1 + val2

    # Node labels
    labels = nodes

    if len(sources) == 0:
        st.info("Network too small with current filters. Try removing a filter.")
        st.stop()

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=16,
            thickness=16,
            line=dict(color="rgba(0,0,0,0.15)", width=1),
            label=labels
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])

    title_bits = []
    if dept_sel != "All": title_bits.append(f"Dept: {dept_sel}")
    if mgr_sel != "All": title_bits.append(f"Manager: {mgr_sel}")
    fig.update_layout(title_text=" → ".join(title_bits) if title_bits else "Organization Flow", height=620)

    st.plotly_chart(fig, use_container_width=True)

    # Helpful counts
    st.markdown("#### Snapshot")
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Managers", f"{len(managers)}")
    with c2: metric_card("Team Leads", f"{len(leads)}")
    with c3: metric_card("Assistant Leads", f"{len(assistants)}")

    # Raw table (optional)
    with st.expander("Show underlying path counts", expanded=False):
        st.dataframe(path_counts.sort_values("count", ascending=False), use_container_width=True)
