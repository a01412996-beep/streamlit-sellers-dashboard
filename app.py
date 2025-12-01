import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="COVID-19 Resource & Vaccination Dashboard",
    layout="wide"
)

sns.set_style("whitegrid")

PRIMARY_BLUE = "#0077b6"
SECONDARY_RED = "#d62828"
ACCENT_ORANGE = "#f77f00"
ACCENT_GREEN = "#2a9d8f"
ACCENT_GREY = "#6c757d"

plt.rcParams["axes.prop_cycle"] = plt.cycler(
    color=[PRIMARY_BLUE, SECONDARY_RED, ACCENT_ORANGE, ACCENT_GREEN]
)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def format_number(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return f"{n:,}"

@st.cache_data
def load_data():
    df = pd.read_csv("WHO-COVID-19-global-daily-data.csv")
    df["Date_reported"] = pd.to_datetime(df["Date_reported"])
    df["New_cases"] = df["New_cases"].fillna(0)
    df["New_deaths"] = df["New_deaths"].fillna(0)
    return df

df = load_data()

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("Global COVID-19 Resource & Vaccination Planning Dashboard")


st.markdown(
    """
This dashboard supports a **government public health officer** in deciding
**where to send vaccines and critical resources**, based on the evolution and
distribution of COVID-19 cases and deaths globally.
"""
)

st.divider()

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.header("Filters")

regions = ["All"] + sorted(df["WHO_region"].dropna().unique().tolist())
selected_region = st.sidebar.selectbox("WHO Region", regions)

if selected_region != "All":
    df_filtered = df[df["WHO_region"] == selected_region].copy()
else:
    df_filtered = df.copy()

countries = ["All"] + sorted(df_filtered["Country"].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Country", countries)

if selected_country != "All":
    df_filtered = df_filtered[df_filtered["Country"] == selected_country].copy()

min_date = df_filtered["Date_reported"].min()
max_date = df_filtered["Date_reported"].max()

start_date = st.sidebar.date_input(
    "Start date",
    value=min_date,
    min_value=min_date,
    max_value=max_date
)
end_date = st.sidebar.date_input(
    "End date",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

if start_date > end_date:
    start_date, end_date = end_date, start_date

df_filtered = df_filtered[
    (df_filtered["Date_reported"] >= pd.to_datetime(start_date)) &
    (df_filtered["Date_reported"] <= pd.to_datetime(end_date))
]

metrics = {
    "Daily new cases": "New_cases",
    "Daily new deaths": "New_deaths",
    "Cumulative cases": "Cumulative_cases",
    "Cumulative deaths": "Cumulative_deaths"
}
metric_name = st.sidebar.selectbox("Main metric", list(metrics.keys()))
metric_column = metrics[metric_name]

# --------------------------------------------------
# GLOBAL VARIABLES AFTER FILTERS
# --------------------------------------------------
last_day = df_filtered["Date_reported"].max()
df_last = df_filtered[df_filtered["Date_reported"] == last_day]

total_new_cases = int(df_filtered["New_cases"].sum())
total_new_deaths = int(df_filtered["New_deaths"].sum())
cumulative_cases = int(df_last["Cumulative_cases"].sum())
cumulative_deaths = int(df_last["Cumulative_deaths"].sum())

if selected_country == "All":
    if selected_region == "All":
        filter_text = "All WHO regions"
    else:
        filter_text = f"WHO Region: **{selected_region}**"
else:
    filter_text = f"Country: **{selected_country}** (WHO Region: {selected_region})"

# --------------------------------------------------
# LAYOUT IN TABS
# --------------------------------------------------
tab_overview, tab_regional, tab_country, tab_notes = st.tabs(
    ["Overview", "Regional view", "Country focus", "Data & policy notes"]
)

# =======================
# TAB 1: OVERVIEW
# =======================
with tab_overview:
    st.subheader("Overview of the selected context")

    st.markdown(
        f"> Data filtered by: {filter_text}  \n"
        f"> Period: **{start_date}** to **{end_date}**"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total new cases (period)", format_number(total_new_cases))
    col2.metric("Total new deaths (period)", format_number(total_new_deaths))
    col3.metric("Cumulative cases (end of period)", format_number(cumulative_cases))
    col4.metric("Cumulative deaths (end of period)", format_number(cumulative_deaths))

    st.markdown(
        """
These indicators summarize the **severity and scale** of COVID-19 in the selected
period and location. Large values suggest that vaccination, testing and staffing
plans must prioritize this context to avoid overload in health services.
"""
    )

    st.markdown("### Epidemic curve over time")

    df_time = (
        df_filtered.groupby("Date_reported", as_index=False)[metric_column].sum()
    ).sort_values("Date_reported")

    df_time["Rolling_7d"] = df_time[metric_column].rolling(window=7, center=True).mean()

    fig_time = px.line(
        df_time,
        x="Date_reported",
        y=metric_column,
        title=f"{metric_name} over time",
        labels={"Date_reported": "Date", metric_column: metric_name},
        template="plotly_white",
        color_discrete_sequence=[PRIMARY_BLUE]
    )
    fig_time.add_scatter(
        x=df_time["Date_reported"],
        y=df_time["Rolling_7d"],
        mode="lines",
        name="7-day moving average",
        line=dict(color=ACCENT_ORANGE, width=3)
    )

    st.plotly_chart(fig_time, use_container_width=True)

    st.markdown(
        """
This curve shows **how fast the outbreak is evolving**. Sudden upward slopes
indicate **new waves** and signal when to accelerate booster campaigns, prepare
hospitals for increased admissions and reinforce ICU and oxygen capacity.
Periods of decline suggest that resources can gradually be reallocated to areas
still facing intense transmission.
"""
    )

   # ---------- Comparison across WHO regions ----------
    st.markdown("### Comparison across WHO regions")

    df_regions = (
        df[df["Date_reported"] == last_day]
        .groupby("WHO_region", as_index=False)[["Cumulative_cases", "Cumulative_deaths"]]
        .sum()
    )

    fig_reg, ax_reg = plt.subplots(figsize=(8, 5))
    width = 0.35
    x_pos = np.arange(len(df_regions))

    ax_reg.bar(x_pos - width/2, df_regions["Cumulative_cases"], width=width, label="Cumulative cases", color=PRIMARY_BLUE)
    ax_reg.bar(x_pos + width/2, df_regions["Cumulative_deaths"], width=width, label="Cumulative deaths", color=SECONDARY_RED)

    ax_reg.set_xticks(x_pos)
    ax_reg.set_xticklabels(df_regions["WHO_region"])
    ax_reg.set_title("Cumulative cases vs deaths by WHO region (last day of period)")
    ax_reg.set_xlabel("WHO region")
    ax_reg.set_ylabel("Count")
    ax_reg.legend()
    plt.tight_layout()

    st.pyplot(fig_reg)

    st.markdown(
        """
This view shows **which regions of the world have carried the greatest overall
impact**. Regions high in cases need sustained vaccination to avoid future waves.
Regions with high deaths relative to cases may require **urgent investment in
clinical care, ICUs and targeted protection of high-risk populations**.
"""
    )
# =======================
# TAB 2: REGIONAL VIEW
# =======================
with tab_regional:
    st.subheader("Regional and country comparison")

    st.markdown(
        """
In this section the focus is on **how the burden is distributed across countries
and regions**. This helps plan **which territories should receive more vaccines
and support** within a limited global supply.
"""
    )

    # ---------- Top 10 countries ----------
    st.markdown("### Top 10 countries by accumulated burden")

    if selected_region == "All":
        df_top_base = df.copy()
    else:
        df_top_base = df[df["WHO_region"] == selected_region].copy()

    df_top_base = df_top_base[
        (df_top_base["Date_reported"] >= pd.to_datetime(start_date)) &
        (df_top_base["Date_reported"] <= pd.to_datetime(end_date))
    ]

    last_day_top = df_top_base["Date_reported"].max()

    df_top = (
        df_top_base[df_top_base["Date_reported"] == last_day_top]
        .groupby("Country", as_index=False)[["Cumulative_cases", "Cumulative_deaths"]]
        .sum()
    )

    top_metric = st.selectbox(
        "Metric for Top 10 ranking",
        ["Cumulative_cases", "Cumulative_deaths"],
        format_func=lambda x: "Cumulative cases" if x == "Cumulative_cases" else "Cumulative deaths"
    )

    df_top = df_top.sort_values(by=top_metric, ascending=False).head(10)

    fig_top, ax_top = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=df_top,
        x=top_metric,
        y="Country",
        ax=ax_top,
        palette=[PRIMARY_BLUE if i == 0 else ACCENT_GREY for i in range(len(df_top))]
    )
    ax_top.set_title(
        "Top 10 countries by "
        + ("cumulative cases" if top_metric == "Cumulative_cases" else "cumulative deaths")
    )
    ax_top.set_xlabel("Number of " + ("cases" if top_metric == "Cumulative_cases" else "deaths"))
    ax_top.set_ylabel("Country")
    st.pyplot(fig_top)

    st.markdown(
        """
Countries at the top of this ranking carry a **disproportionate share of the
regional or global burden**. These states are strong candidates for **priority
vaccine allocation, surge staffing and international support**, especially if
their health systems are already under strain.
"""
    )

    st.divider()

    # ---------- Share of regional cases ----------
    st.markdown("### Share of cases within the region")

    if selected_region == "All":
        st.info("Select a specific WHO region in the filters to see how cases are distributed among its countries.")
    else:
        df_region_share = (
            df_top_base[df_top_base["Date_reported"] == last_day_top]
            .groupby("Country", as_index=False)[["Cumulative_cases"]]
            .sum()
        )
        df_region_share["Share"] = df_region_share["Cumulative_cases"] / df_region_share["Cumulative_cases"].sum()

        fig_share = px.bar(
            df_region_share.sort_values("Share", ascending=False),
            x="Country",
            y="Share",
            title=f"Share of regional cumulative cases by country in {selected_region}",
            labels={"Share": "Share of regional cases", "Country": "Country"},
            template="plotly_white",
            color_discrete_sequence=[ACCENT_GREEN]
        )
        fig_share.update_yaxes(tickformat=".0%")
        fig_share.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_share, use_container_width=True)

        st.markdown(
            """
This chart answers: **“Who holds how much of the problem?”**  
Countries with a large share of regional cases should receive a **larger fraction
of available vaccine doses and medical supplies**, making distribution proportional
to the real epidemiological burden.
"""
        )

    st.divider()


# =======================
# TAB 3: COUNTRY FOCUS
# =======================
with tab_country:
    st.subheader("Detailed view for a specific country")

    if selected_country == "All":
        st.info("Select a specific country in the sidebar to see its detailed indicators.")
    else:
        st.markdown(f"**Country selected:** {selected_country}")

        df_country_daily = (
            df_filtered.groupby("Date_reported", as_index=False)[["New_cases", "New_deaths"]].sum()
        ).sort_values("Date_reported")

        if df_country_daily.empty:
            st.warning("No data available for the selected country and date range.")
        else:
            # ---- Daily new cases ----
            st.markdown("### Daily new cases")

            fig_country = px.line(
                df_country_daily,
                x="Date_reported",
                y="New_cases",
                title=f"Daily new COVID-19 cases in {selected_country}",
                labels={"Date_reported": "Date", "New_cases": "Daily new cases"},
                template="plotly_white",
                color_discrete_sequence=[PRIMARY_BLUE]
            )
            fig_country.add_scatter(
                x=df_country_daily["Date_reported"],
                y=df_country_daily["New_cases"].rolling(window=7, center=True).mean(),
                mode="lines",
                name="7-day moving average",
                line=dict(color=ACCENT_ORANGE, width=3)
            )
            st.plotly_chart(fig_country, use_container_width=True)

            st.markdown(
                f"""
This graph helps identify **surges and persistent high transmission** in {selected_country}.
Sharp increases or long periods with elevated cases signal the need to **intensify
vaccination campaigns, testing and community engagement**, particularly in densely
populated or vulnerable areas.
"""
            )

            st.divider()

            # ---- Daily deaths ----
            st.markdown("### Daily new deaths")

            fig_deaths = px.line(
                df_country_daily,
                x="Date_reported",
                y="New_deaths",
                title=f"Daily new COVID-19 deaths in {selected_country}",
                labels={"Date_reported": "Date", "New_deaths": "Daily new deaths"},
                template="plotly_white",
                color_discrete_sequence=[SECONDARY_RED]
            )
            fig_deaths.add_scatter(
                x=df_country_daily["Date_reported"],
                y=df_country_daily["New_deaths"].rolling(window=7, center=True).mean(),
                mode="lines",
                name="7-day moving average",
                line=dict(color=ACCENT_GREY, width=3)
            )
            st.plotly_chart(fig_deaths, use_container_width=True)

            st.markdown(
                f"""
Deaths reflect both **virus severity** and **health system performance**.
If deaths remain high while cases plateau or decline, it suggests gaps in:
access to care, timeliness of treatment or vaccine coverage among high-risk groups.
This supports decisions to reinforce **hospital capacity, oxygen and targeted
vaccination for elderly and comorbid patients** in {selected_country}.
"""
            )

            st.divider()


# =======================
# TAB 4: DATA & POLICY NOTES
# =======================
with tab_notes:
    st.subheader("Data, limitations and policy interpretation")

    st.markdown(
        """
This dashboard is based on daily global COVID-19 data reported to the World Health
Organization. It uses **absolute counts**, not population-adjusted rates, and
depends on each country’s ability to detect and report cases and deaths accurately.

Despite these limitations, combining **time trends, country rankings, regional
distributions and country-level details** allows a government health officer to:

1. Detect **when epidemic waves are emerging**, to act before hospitals are overwhelmed.  
2. Identify **which countries and regions carry the highest burden**, to prioritize
   vaccine shipments and medical resources.  
3. Monitor whether **deaths decrease over time** as vaccination coverage improves,
   especially among older adults and high-risk groups.  
4. Allocate resources in a way that is **transparent, proportional and evidence-based**.

"""
    )
st.markdown(
    """
    ### Final Policy Conclusion
    
    Based on the epidemiological patterns observed in the dashboard, **France** stands out within the **European region** as a country carrying a disproportionately high share of the overall COVID-19 burden. Its combination of elevated cumulative cases, sustained daily transmission, and notable fluctuations in deaths indicates ongoing pressure on the public health and clinical care system.

    The geographic distribution map further reinforces that France remains one of the principal contributors to Europe’s total case count, making it a critical node in both regional containment and vaccination strategy.

    Given these indicators, **France should be prioritized for additional public health support**. This includes reinforcing vaccine supply—particularly booster doses for high-risk populations—expanding cold-chain capacity, and deploying temporary medical response units where hospital demand exceeds capacity.

    Strengthening France’s outbreak response not only reduces national-level morbidity and mortality but also mitigates cross-border transmission risks within Europe.  
    **The data strongly supports directing targeted assistance to France to stabilize the region and improve Europe’s overall COVID-19 control capacity.**
    """
)
