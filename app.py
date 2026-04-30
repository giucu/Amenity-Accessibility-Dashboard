import json
import geopandas as gpd
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, ALL, MATCH, callback, ctx, no_update
import dash_bootstrap_components as dbc

# ── Load data ─────────────────────────────────────────────────────────────────

CITIES = {
    "Paris": "data/paris_scores.geojson",
    # "Milan": "data/milan_scores.geojson",
}

city_data = {}
for city, path in CITIES.items():
    gdf = gpd.read_file(path)
    geojson = json.loads(gdf.to_json())
    city_data[city] = {"gdf": gdf, "geojson": geojson}

sample_gdf = city_data["Paris"]["gdf"]

METRIC_GROUPS = {
    "Gravity": [c for c in sample_gdf.columns if c.startswith("gravity_") and "per_" not in c],
    "Min Travel Time": [c for c in sample_gdf.columns if c.startswith("min_tt_")],
    "Population Weighted Gravity": [c for c in sample_gdf.columns if "per_" in c],
    "Diversity": [c for c in sample_gdf.columns if c in ("shannon", "jsd", "city_similarity", "kl_divergence")],
    "Basic Counts": [c for c in sample_gdf.columns if c.startswith("count_")],
}
METRIC_GROUPS = {k: v for k, v in METRIC_GROUPS.items() if v}

CMAPS = ["Viridis", "YlOrRd", "Blues", "RdYlGn", "Plasma", "Cividis"]
CITY_OPTIONS = [{"label": c, "value": c} for c in CITIES]
GROUP_OPTIONS = [{"label": g, "value": g} for g in METRIC_GROUPS]
DEFAULT_GROUP = list(METRIC_GROUPS.keys())[0]
DEFAULT_METRIC = METRIC_GROUPS[DEFAULT_GROUP][0]

# ── Map builder ───────────────────────────────────────────────────────────────

def make_map(city, metric, cmap):
    if not city or not metric:
        return {}
    gdf     = city_data[city]["gdf"]
    geojson = city_data[city]["geojson"]
    fig = px.choropleth_map(
        gdf,
        geojson=geojson,
        locations="from_id",
        featureidkey="properties.from_id",
        color=metric,
        color_continuous_scale=cmap,
        map_style="carto-positron",
        zoom=11,
        center={"lat": gdf.geometry.centroid.y.mean(),
                "lon": gdf.geometry.centroid.x.mean()},
        opacity=0.75,
        labels={metric: metric.replace("_", " ").title()},
    )
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_colorbar={"thickness": 12, "len": 0.6,
                            "title": {"text": metric.replace("_", " ").title(), "font": {"size": 11}}},
        uirevision=city,
    )
    return fig


# ── Panel builder ─────────────────────────────────────────────────────────────

def make_panel(panel_id, removable=False):
    return html.Div(
        id={"type": "panel-wrapper", "index": panel_id},
        style={"minWidth": "420px", "flex": "1 1 420px", "maxWidth": "900px",
               "display": "flex", "flexDirection": "column", "gap": "8px"},
        children=[
            # Controls card
            html.Div(
                style={"background": "#ffffff", "border": "1px solid #e2e0db",
                       "borderRadius": "8px", "padding": "12px 14px"},
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between",
                               "alignItems": "center", "marginBottom": "8px"},
                        children=[
                            html.Span(f"Map {panel_id + 1}",
                                      style={"fontWeight": "600", "fontSize": "13px",
                                             "color": "#28251d"}),
                            html.Button(
                                "✕", id={"type": "remove-btn", "index": panel_id},
                                style={"background": "none", "border": "none",
                                       "cursor": "pointer", "fontSize": "14px",
                                       "color": "#7a7974", "padding": "0 2px",
                                       "display": "block" if removable else "none"},
                                title="Remove this map"
                            ),
                        ]
                    ),
                    # City
                    html.Div([
                        html.Label("City", style={"fontSize": "11px", "fontWeight": "600",
                                                   "color": "#7a7974", "textTransform": "uppercase",
                                                   "letterSpacing": "0.04em", "marginBottom": "3px"}),
                        dcc.Dropdown(
                            id={"type": "city-dd", "index": panel_id},
                            options=CITY_OPTIONS, value="Paris", clearable=False,
                            style={"fontSize": "13px"}
                        ),
                    ], style={"marginBottom": "8px"}),
                    # Group + Metric in a row
                    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                                    "gap": "8px", "marginBottom": "8px"}, children=[
                        html.Div([
                            html.Label("Metric group",
                                       style={"fontSize": "11px", "fontWeight": "600",
                                              "color": "#7a7974", "textTransform": "uppercase",
                                              "letterSpacing": "0.04em", "marginBottom": "3px"}),
                            dcc.Dropdown(
                                id={"type": "group-dd", "index": panel_id},
                                options=GROUP_OPTIONS, value=DEFAULT_GROUP,
                                clearable=False, style={"fontSize": "13px"}
                            ),
                        ]),
                        html.Div([
                            html.Label("Metric",
                                       style={"fontSize": "11px", "fontWeight": "600",
                                              "color": "#7a7974", "textTransform": "uppercase",
                                              "letterSpacing": "0.04em", "marginBottom": "3px"}),
                            dcc.Dropdown(
                                id={"type": "metric-dd", "index": panel_id},
                                options=[{"label": c.replace("_", " ").title(), "value": c}
                                         for c in METRIC_GROUPS[DEFAULT_GROUP]],
                                value=DEFAULT_METRIC, clearable=False,
                                style={"fontSize": "13px"}
                            ),
                        ]),
                    ]),
                    # Colour scale
                    html.Div([
                        html.Label("Colour scale",
                                   style={"fontSize": "11px", "fontWeight": "600",
                                          "color": "#7a7974", "textTransform": "uppercase",
                                          "letterSpacing": "0.04em", "marginBottom": "3px"}),
                        dcc.Dropdown(
                            id={"type": "cmap-dd", "index": panel_id},
                            options=[{"label": c, "value": c} for c in CMAPS],
                            value="Viridis", clearable=False, style={"fontSize": "13px"}
                        ),
                    ]),
                ]
            ),
            # Map
            dcc.Graph(
                id={"type": "hex-map", "index": panel_id},
                style={"flex": "1", "minHeight": "520px", "borderRadius": "8px",
                       "overflow": "hidden", "border": "1px solid #e2e0db"},
                config={"scrollZoom": True},
            ),
        ]
    )


# ── Layout ────────────────────────────────────────────────────────────────────

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://api.fontshare.com/v2/css?f[]=satoshi@400,500,600,700&display=swap",
    ]
)

app.layout = html.Div(
    style={"background": "#f7f6f2", "minHeight": "100vh",
           "fontFamily": "'Satoshi', sans-serif", "color": "#28251d"},
    children=[
        # Header
        html.Div(
            style={"borderBottom": "1px solid #dcd9d5", "background": "#ffffff",
                   "padding": "0 24px", "display": "flex", "alignItems": "center",
                   "justifyContent": "space-between", "height": "52px"},
            children=[
                html.Div("Visualising Amenity Accessibility in Cities",
                         style={"fontWeight": "700", "fontSize": "15px", "letterSpacing": "-0.2px"}),
                html.Div("Paris · Hex grid analysis",
                         style={"fontSize": "12px", "color": "#7a7974"}),
            ]
        ),

        # Panels container + Add button
        html.Div(
            style={"padding": "20px 20px 20px 20px"},
            children=[
                # Scrollable panels row
                html.Div(
                    id="panels-container",
                    style={"display": "flex", "flexDirection": "row", "gap": "16px",
                           "overflowX": "auto", "alignItems": "flex-start",
                           "paddingBottom": "12px"},
                    children=[make_panel(0, removable=False)]
                ),

                # Add comparison button
                html.Div(
                    style={"marginTop": "16px", "display": "flex", "alignItems": "center",
                           "gap": "10px"},
                    children=[
                        html.Button(
                            id="add-panel-btn",
                            children=[
                                html.Span("＋", style={"marginRight": "6px", "fontSize": "16px"}),
                                "Add comparison map",
                            ],
                            style={"background": "#01696f", "color": "#ffffff",
                                   "border": "none", "borderRadius": "6px",
                                   "padding": "8px 16px", "fontSize": "13px",
                                   "fontWeight": "600", "cursor": "pointer",
                                   "fontFamily": "'Satoshi', sans-serif",
                                   "transition": "background 180ms ease"},
                            n_clicks=0,
                        ),
                        html.Span(id="panel-count-label",
                                  style={"fontSize": "12px", "color": "#7a7974"}),
                    ]
                ),
            ]
        ),

        # State: list of active panel ids
        dcc.Store(id="panel-ids", data=[0]),
        dcc.Store(id="next-panel-id", data=1),
    ]
)

# ── Callbacks ─────────────────────────────────────────────────────────────────

# Update metric options when group changes
@callback(
    Output({"type": "metric-dd", "index": MATCH}, "options"),
    Output({"type": "metric-dd", "index": MATCH}, "value"),
    Input({"type": "group-dd", "index": MATCH}, "value"),
)
def update_metric_options(group):
    cols = METRIC_GROUPS.get(group, [])
    opts = [{"label": c.replace("_", " ").title(), "value": c} for c in cols]
    return opts, cols[0] if cols else None


# Render each map
@callback(
    Output({"type": "hex-map", "index": MATCH}, "figure"),
    Input({"type": "city-dd",   "index": MATCH}, "value"),
    Input({"type": "metric-dd", "index": MATCH}, "value"),
    Input({"type": "cmap-dd",   "index": MATCH}, "value"),
)
def render_map(city, metric, cmap):
    return make_map(city, metric, cmap or "Viridis")


# Add / remove panels
@callback(
    Output("panels-container",  "children"),
    Output("panel-ids",         "data"),
    Output("next-panel-id",     "data"),
    Output("panel-count-label", "children"),
    Input("add-panel-btn",                          "n_clicks"),
    Input({"type": "remove-btn", "index": ALL},     "n_clicks"),
    State("panel-ids",    "data"),
    State("next-panel-id","data"),
    prevent_initial_call=True,
)
def manage_panels(add_clicks, remove_clicks, panel_ids, next_id):
    triggered = ctx.triggered_id

    if triggered == "add-panel-btn":
        panel_ids = panel_ids + [next_id]
        next_id += 1

    elif isinstance(triggered, dict) and triggered.get("type") == "remove-btn":
        remove_idx = triggered["index"]
        panel_ids = [p for p in panel_ids if p != remove_idx]

    # Always keep at least 1 panel; first panel never removable
    panel_ids = panel_ids if panel_ids else [0]

    panels = [make_panel(pid, removable=(pid != panel_ids[0])) for pid in panel_ids]
    n = len(panel_ids)
    label = f"{n} map{'s' if n > 1 else ''} · scroll horizontally to see all" if n > 1 else ""
    return panels, panel_ids, next_id, label

server = app.server

if __name__ == "__main__":
    app.run(debug=True)
