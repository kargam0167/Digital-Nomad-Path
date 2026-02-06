# app.py
# pip install dash pandas plotly numpy

import pandas as pd
import numpy as np

import dash
from dash import dcc, html, dash_table, Input, Output
import plotly.graph_objects as go


# =========================
# 1) LOAD DATA
# =========================
DATA_PATH = "Nomad_Master_Sheet_COUNTRY_CLEAN_with_ID.csv" 

df = pd.read_csv(DATA_PATH, dtype={"ID": "int64"})

required = {"ID", "Home_coun", "Cur_Coun", "Coun_YR_Which_clean"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in {DATA_PATH}: {sorted(missing)}")


# =========================
# 2) NORMALIZE COUNTRY NAMES
# =========================
ALIASES = {
    "UK": "United Kingdom",
    "U.K.": "United Kingdom",
    "US": "United States of America",
    "U.S.": "United States of America",
    "USA": "United States of America",
}

def norm_country(x):
    if pd.isna(x):
        return None
    s = str(x).replace("\u00a0", " ").strip()  # normalize non-breaking spaces
    if not s or s.lower() == "nan":
        return None
    return ALIASES.get(s, s)

df["Home_coun"] = df["Home_coun"].apply(norm_country)
df["Cur_Coun"] = df["Cur_Coun"].apply(norm_country)

# Explode visited list (already cleaned, comma-separated)
tmp = df[["ID", "Home_coun", "Cur_Coun", "Coun_YR_Which_clean"]].copy()
tmp["visited"] = tmp["Coun_YR_Which_clean"].fillna("").astype(str).str.split(",")
tmp = tmp.explode("visited")
tmp["visited"] = tmp["visited"].astype(str).str.replace("\u00a0", " ").str.strip()
tmp.loc[tmp["visited"].eq("") | tmp["visited"].str.lower().eq("nan"), "visited"] = np.nan
tmp = tmp.dropna(subset=["visited"])
tmp["visited"] = tmp["visited"].apply(norm_country)

# Precompute home counts for dropdown ordering
home_counts = (
    df[df["Home_coun"].notna()]
    .groupby("Home_coun", as_index=False)
    .agg(n_nomads=("ID", "count"))
    .sort_values("n_nomads", ascending=False)
)

if home_counts.empty:
    raise ValueError("No valid Home_coun values found.")

DEFAULT_HOME = home_counts.iloc[0]["Home_coun"]


# =========================
# 3) HELPERS (subset + aggregates)
# =========================
TOP_N_NODES = 20  # per your request

def subset_for_home(home_country: str):
    df_h = df[df["Home_coun"] == home_country].copy()
    tmp_h = tmp[tmp["Home_coun"] == home_country].copy()
    return df_h, tmp_h

def compute_for_home(home_country: str):
    df_h, tmp_h = subset_for_home(home_country)

    # Visited counts (top 20)
    visited_counts = (
        tmp_h[tmp_h["visited"].notna()]
        .groupby("visited", as_index=False)
        .agg(n_visits=("ID", "count"))
        .sort_values("n_visits", ascending=False)
        .head(TOP_N_NODES)
    )

    visited_top_set = set(visited_counts["visited"].tolist())

    # Flows: Home -> visited (restricted to top visited)
    flows = (
        tmp_h[
            tmp_h["visited"].notna()
            & tmp_h["Home_coun"].notna()
            & tmp_h["Home_coun"].ne(tmp_h["visited"])
            & tmp_h["visited"].isin(visited_top_set)
        ]
        .groupby(["Home_coun", "visited"], as_index=False)
        .agg(n=("ID", "count"))
        .sort_values("n", ascending=False)
    )

    # Current counts (optional; top 20)
    cur_counts = (
        df_h[df_h["Cur_Coun"].notna()]
        .groupby("Cur_Coun", as_index=False)
        .agg(n_current=("ID", "count"))
        .sort_values("n_current", ascending=False)
        .head(TOP_N_NODES)
    )

    n_nomads = len(df_h)
    return n_nomads, visited_counts, flows, cur_counts


# =========================
# 4) FIGURE BUILDER
# =========================
def build_map(home_country: str, show_current: bool):
    n_nomads, visited_counts, flows, cur_counts = compute_for_home(home_country)

    fig = go.Figure()

    # (A) Edges: one trace per edge so we can vary width (only up to 20 edges)
    if not flows.empty:
        nmin, nmax = flows["n"].min(), flows["n"].max()
        for _, r in flows.iterrows():
            if nmax == nmin:
                w = 2.5
            else:
                w = 1.0 + 5.0 * (r["n"] - nmin) / (nmax - nmin)

            fig.add_trace(go.Scattergeo(
                locations=[r["Home_coun"], r["visited"]],
                locationmode="country names",
                mode="lines",
                line=dict(width=float(w), color="rgba(0,128,0,0.35)", dash="dot"),
                hoverinfo="text",
                text=f"{r['Home_coun']} → {r['visited']}<br>Nomads (rows): {int(r['n'])}",
                showlegend=False
            ))

    # (B) Visited nodes (blue) — top 20 for this home
    if not visited_counts.empty:
        fig.add_trace(go.Scattergeo(
            locations=visited_counts["visited"],
            locationmode="country names",
            mode="markers",
            marker=dict(
                size=np.clip(np.log1p(visited_counts["n_visits"]) * 6 + 6, 8, 30),
                color="rgba(30,144,255,0.70)",
                line=dict(width=0.8, color="white"),
            ),
            text=visited_counts["visited"],
            customdata=visited_counts[["n_visits"]].values,
            hovertemplate="<b>%{text}</b><br>Nomads (rows) mentioning: %{customdata[0]}<extra></extra>",
            name=f"Visited (top {TOP_N_NODES})"
        ))

    # (C) Current nodes (orange) — top 20 for this home
    if show_current and not cur_counts.empty:
        fig.add_trace(go.Scattergeo(
            locations=cur_counts["Cur_Coun"],
            locationmode="country names",
            mode="markers",
            marker=dict(
                size=np.clip(np.log1p(cur_counts["n_current"]) * 5 + 5, 7, 24),
                color="rgba(255,140,0,0.80)",
                line=dict(width=0.8, color="white"),
            ),
            text=cur_counts["Cur_Coun"],
            customdata=cur_counts[["n_current"]].values,
            hovertemplate="<b>%{text}</b><br>Current count (rows): %{customdata[0]}<extra></extra>",
            name=f"Current (top {TOP_N_NODES})"
        ))

    # (D) Home node (green) — ONLY ONE NODE
    fig.add_trace(go.Scattergeo(
        locations=[home_country],
        locationmode="country names",
        mode="markers+text",
        marker=dict(
            size=26,
            color="rgba(0,170,0,0.95)",
            line=dict(width=2.2, color="rgba(0,90,0,1)"),
        ),
        text=[home_country],
        textposition="top center",
        hovertemplate=f"<b>{home_country}</b><br>Nomads (rows): {n_nomads}<extra></extra>",
        name="Home (selected)"
    ))

    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="rgb(245,245,245)",
        showcountries=True,
        countrycolor="rgb(200,200,200)",
        showcoastlines=True,
        coastlinecolor="rgb(210,210,210)",
    )
    fig.update_layout(
    title=dict(
        text=f"Home base: {home_country} → Visited countries (top {TOP_N_NODES}) | n={n_nomads}",
        x=0, xanchor="left",
        y=0.98, yanchor="top"
    ),
    # Put legend below the title (and outside the map area)
    legend=dict(
        orientation="h",
        x=0, xanchor="left",
        y=1.06, yanchor="bottom",
        bgcolor="rgba(255,255,255,0.85)"
    ),
    margin=dict(l=10, r=10, t=120, b=10),  # increase top margin so nothing overlaps
    height=720,
)

    return fig

def flows_table(home_country: str, top_n_rows: int = 20):
    _, _, flows, _ = compute_for_home(home_country)
    return flows.head(int(top_n_rows))


# =========================
# 5) DASH APP (single page)
# =========================
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Nomad Mobility Map"
server = app.server

home_options = [
    {"label": f"{r.Home_coun} (n={int(r.n_nomads)})", "value": r.Home_coun}
    for r in home_counts.itertuples(index=False)
]

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "16px"},
    children=[
        html.H2("Digital Nomad Mobility"),

        html.Div(
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            children=[
                html.Div(
                    style={"flex": "2", "minWidth": "320px"},
                    children=[
                        html.Label("Choose home base (green node)"),
                        dcc.Dropdown(
                            id="home-dd",
                            options=home_options,
                            value=DEFAULT_HOME,
                            clearable=False
                        ),
                    ],
                ),
                html.Div(
                    style={"flex": "1", "minWidth": "240px"},
                    children=[
                        html.Label("Options"),
                        dcc.Checklist(
                            id="opts",
                            options=[{"label": "Show current nodes (orange)", "value": "show_current"}],
                            value=["show_current"],
                        ),
                        html.Div(
                            f"Nodes are limited to top {TOP_N_NODES} visited countries.",
                            style={"fontSize": "12px", "color": "#555", "marginTop": "6px"},
                        ),
                    ],
                ),
            ],
        ),

        dcc.Graph(id="map", style={"marginTop": "12px"}),

        html.H4("Top flows (home → visited)"),
        dash_table.DataTable(
            id="flows-table",
            page_size=10,
            sort_action="native",
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "whiteSpace": "normal", "height": "auto"},
            columns=[
                {"name": "Home", "id": "Home_coun"},
                {"name": "Visited", "id": "visited"},
                {"name": "Nomads (rows)", "id": "n"},
            ],
        ),
    ],
)

@app.callback(
    Output("map", "figure"),
    Output("flows-table", "data"),
    Input("home-dd", "value"),
    Input("opts", "value"),
)
def update(home_val, opts):
    home_val = home_val or DEFAULT_HOME
    show_current = "show_current" in (opts or [])
    fig = build_map(home_val, show_current=show_current)
    table = flows_table(home_val, top_n_rows=20).to_dict("records")
    return fig, table


if __name__ == "__main__":
    app.run(debug=True, port=8050)