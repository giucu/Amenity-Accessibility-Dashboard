import json
import os
import glob
import geopandas as gpd
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, ALL, MATCH, callback, ctx, no_update, Patch
import dash_bootstrap_components as dbc


# ── File discovery ─────────────────────────────────────────────────────────────
def city_label_from_path(path):
    filename = os.path.basename(path)
    stem = filename.replace("_scores.geojson", "")
    return stem.replace("_", " ").title()


def discover_city_files(data_dir="data"):
    pattern = os.path.join(data_dir, "*_scores.geojson")
    files = sorted(glob.glob(pattern))
    return {city_label_from_path(path): path for path in files}


# ── Load data ──────────────────────────────────────────────────────────────────

CITIES = discover_city_files("data")

city_data = {}
all_columns = set()

for city, path in CITIES.items():
    try:
        gdf = gpd.read_file(path)
        geojson = json.loads(gdf.to_json())

        center_geom = gdf.to_crs(gdf.estimate_utm_crs()).centroid.to_crs("EPSG:4326")
        center_lat = center_geom.y.mean()
        center_lon = center_geom.x.mean()

        city_data[city] = {
            "gdf": gdf,
            "geojson": geojson,
            "center": {"lat": center_lat, "lon": center_lon},
        }
        all_columns.update(gdf.columns)
    except Exception as e:
        print(f"Skipping {city}: {e}")

all_columns = sorted(all_columns)

CITY_NAMES = list(city_data.keys())
DEFAULT_CITY = CITY_NAMES[2] if CITY_NAMES else None
sample_gdf = city_data.get(DEFAULT_CITY, {}).get("gdf", pd.DataFrame())


# ── Metric groups ──────────────────────────────────────────────────────────────

def cols_matching(prefix=None, contains=None, exact=None, exclude_contains=None):
    cols = all_columns.copy()

    if prefix is not None:
        cols = [c for c in cols if c.startswith(prefix)]
    if contains is not None:
        cols = [c for c in cols if contains in c]
    if exact is not None:
        cols = [c for c in cols if c in exact]
    if exclude_contains is not None:
        cols = [c for c in cols if exclude_contains not in c]

    return cols


METRIC_GROUPS = {
    "Gravity": cols_matching(prefix="gravity_", exclude_contains="per_"),
    "Min Travel Time": cols_matching(prefix="min_tt_"),
    "Average Travel Time": cols_matching(prefix="avg_tt_"),
    "Population Weighted Gravity": [c for c in all_columns if c.startswith("gravity_") and "per_" in c],
    "Diversity": cols_matching(exact=["shannon", "jsd", "city_similarity", "kl_divergence"]),
    "Basic Counts": cols_matching(prefix="count_"),
}

METRIC_GROUPS = {k: v for k, v in METRIC_GROUPS.items() if v}

CMAPS = ["Viridis", "YlOrRd", "Blues", "RdYlGn", "Plasma", "Cividis"]
CITY_OPTIONS = [{"label": c, "value": c} for c in city_data.keys()]
GROUP_OPTIONS = [{"label": g, "value": g} for g in METRIC_GROUPS]

DEFAULT_GROUP = list(METRIC_GROUPS.keys())[0] if METRIC_GROUPS else None
DEFAULT_METRIC = METRIC_GROUPS[DEFAULT_GROUP][0] if DEFAULT_GROUP else None
DEFAULT_QUANTILES = 15

BIN_CACHE = {}

# ── Helpers ────────────────────────────────────────────────────────────────────

def available_metrics_for_city(city, group):
    if city not in city_data or group not in METRIC_GROUPS:
        return []

    city_cols = set(city_data[city]["gdf"].columns)
    return [c for c in METRIC_GROUPS[group] if c in city_cols]


# ── Map builder ────────────────────────────────────────────────────────────────

def make_map(city, metric, cmap, quantiles=7, compact=False):
    if not city or not metric or city not in city_data:
        return {}

    gdf = city_data[city]["gdf"].copy()
    geojson = city_data[city]["geojson"]
    center = city_data[city].get("center", {"lat": 48.8566, "lon": 2.3522})

    if metric not in gdf.columns or "from_id" not in gdf.columns:
        return {}

    cache_key = (city, metric, quantiles)

    if cache_key in BIN_CACHE:
        plot_gdf, labels = BIN_CACHE[cache_key]
    else:
        values = pd.to_numeric(gdf[metric], errors="coerce")
        mask = values.notna()
        valid = values[mask]

        if valid.empty:
            return {}

        q = max(2, min(int(quantiles), 50, int(valid.nunique())))

        try:
            codes, bin_edges = pd.qcut(
                valid,
                q=q,
                labels=False,
                retbins=True,
                duplicates="drop"
            )
        except Exception as e:
            print(f"qcut failed for {city} / {metric}: {e}")
            return {}

        actual_q = len(bin_edges) - 1
        if actual_q < 1:
            return {}

        labels = [
            f"Q{i+1}: {bin_edges[i]:.2f}–{bin_edges[i+1]:.2f}"
            for i in range(actual_q)
        ]

        tmp = gdf.copy()
        tmp["_qcode"] = None
        tmp.loc[valid.index, "_qcode"] = codes.astype(int)

        plot_gdf = tmp.dropna(subset=["_qcode"]).copy()
        if plot_gdf.empty:
            return {}

        plot_gdf["_qcode"] = plot_gdf["_qcode"].astype(int)
        plot_gdf["_qlabel"] = plot_gdf["_qcode"].map(lambda x: labels[x])

        BIN_CACHE[cache_key] = (plot_gdf, labels)

    actual_q = len(labels)

    fig = px.choropleth_map(
        plot_gdf,
        geojson=geojson,
        locations="from_id",
        featureidkey="properties.from_id",
        color="_qlabel",
        category_orders={"_qlabel": labels},
        color_discrete_sequence=px.colors.sample_colorscale(
            cmap,
            [i / max(actual_q - 1, 1) for i in range(actual_q)]
        ),
        hover_data={
            metric: True,
            "_qlabel": True,
            "_qcode": False,
        },
        map_style="carto-positron",
        zoom=10 if compact else 11,
        center=center,
        opacity=0.75,
        labels={
            metric: metric.replace("_", " ").title(),
            "_qlabel": "Quantile",
        },
    )

    fig.update_traces(marker_line_width=0)
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend={
            "title": {"text": metric.replace("_", " ").title()},
            "font": {"size": 10},
            "orientation": "v",
            "yanchor": "top",
            "y": 0.98,
            "xanchor": "left",
            "x": 0.01,
        },
        uirevision=f"{city}_{metric}_{quantiles}_{cmap}",
    )
    return fig


# ── Panel builder (editor view) ────────────────────────────────────────────────

def make_panel(
    panel_id,
    removable=False,
    city_value=None,
    group_value=None,
    metric_value=None,
    cmap_value="Viridis",
):
    city_value = city_value or DEFAULT_CITY
    group_value = group_value or DEFAULT_GROUP
    cmap_value = cmap_value or "Viridis"

    available = available_metrics_for_city(city_value, group_value) if city_value and group_value else []
    if metric_value not in available:
        metric_value = available[0] if available else None

    return html.Div(
        id={"type": "panel-wrapper", "index": panel_id},
        style={
            "minWidth": "420px", "flex": "1 1 420px", "maxWidth": "900px",
            "display": "flex", "flexDirection": "column", "gap": "8px",
        },
        children=[
            html.Div(
                style={
                    "background": "#ffffff", "border": "1px solid #e2e0db",
                    "borderRadius": "8px", "padding": "12px 14px",
                },
                children=[
                    html.Div(
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "alignItems": "center",
                            "marginBottom": "8px",
                        },
                        children=[
                            html.Span(
                                f"Map {panel_id + 1}",
                                style={"fontWeight": "600", "fontSize": "13px", "color": "#28251d"},
                            ),
                            html.Button(
                                "✕",
                                id={"type": "remove-btn", "index": panel_id},
                                title="Remove this map",
                                style={
                                    "background": "none", "border": "none", "cursor": "pointer",
                                    "fontSize": "14px", "color": "#7a7974", "padding": "0 2px",
                                    "display": "block" if removable else "none",
                                },
                            ),
                        ],
                    ),
                    html.Div([
                        html.Label("City", style={
                            "fontSize": "11px", "fontWeight": "600", "color": "#7a7974",
                            "textTransform": "uppercase", "letterSpacing": "0.04em",
                            "marginBottom": "3px",
                        }),
                        dcc.Dropdown(
                            id={"type": "city-dd", "index": panel_id},
                            options=CITY_OPTIONS,
                            value=city_value,
                            clearable=False,
                            style={"fontSize": "13px"},
                        ),
                    ], style={"marginBottom": "8px"}),
                    html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "1fr 1fr",
                            "gap": "8px",
                            "marginBottom": "8px",
                        },
                        children=[
                            html.Div([
                                html.Label("Metric group", style={
                                    "fontSize": "11px", "fontWeight": "600", "color": "#7a7974",
                                    "textTransform": "uppercase", "letterSpacing": "0.04em",
                                    "marginBottom": "3px",
                                }),
                                dcc.Dropdown(
                                    id={"type": "group-dd", "index": panel_id},
                                    options=GROUP_OPTIONS,
                                    value=group_value,
                                    clearable=False,
                                    style={"fontSize": "13px"},
                                ),
                            ]),
                            html.Div([
                                html.Label("Metric", style={
                                    "fontSize": "11px", "fontWeight": "600", "color": "#7a7974",
                                    "textTransform": "uppercase", "letterSpacing": "0.04em",
                                    "marginBottom": "3px",
                                }),
                                dcc.Dropdown(
                                    id={"type": "metric-dd", "index": panel_id},
                                    options=[
                                        {"label": c.replace("_", " ").title(), "value": c}
                                        for c in available
                                    ],
                                    value=metric_value,
                                    clearable=False,
                                    style={"fontSize": "13px"},
                                ),
                            ]),
                        ],
                    ),
                    html.Div([
                        html.Label("Colour scale", style={
                            "fontSize": "11px", "fontWeight": "600", "color": "#7a7974",
                            "textTransform": "uppercase", "letterSpacing": "0.04em",
                            "marginBottom": "3px",
                        }),
                        dcc.Dropdown(
                            id={"type": "cmap-dd", "index": panel_id},
                            options=[{"label": c, "value": c} for c in CMAPS],
                            value=cmap_value,
                            clearable=False,
                            style={"fontSize": "13px"},
                        ),
                    ]),
                ],
            ),
            dcc.Graph(
                id={"type": "hex-map", "index": panel_id},
                style={
                    "flex": "1", "minHeight": "520px", "borderRadius": "8px",
                    "overflow": "hidden", "border": "1px solid #e2e0db",
                },
                config={"scrollZoom": True},
            ),
        ],
    )


# ── Compare grid card ──────────────────────────────────────────────────────────

def make_compare_card(panel_id, city, metric, cmap, quantiles):
    title = f"{city} · {metric.replace('_', ' ').title()}" if city and metric else f"Map {panel_id + 1}"
    return html.Div(
        style={"display": "flex", "flexDirection": "column"},
        children=[
            html.Div(
                title,
                title=title,
                style={
                    "fontSize": "12px",
                    "fontWeight": "600",
                    "color": "#28251d",
                    "padding": "7px 12px",
                    "background": "#ffffff",
                    "borderRadius": "8px 8px 0 0",
                    "border": "1px solid #e2e0db",
                    "borderBottom": "none",
                    "whiteSpace": "nowrap",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                },
            ),
            dcc.Graph(
                id={"type": "compare-map", "index": panel_id},
                figure=make_map(city, metric, cmap or "Viridis", quantiles=quantiles or DEFAULT_QUANTILES, compact=True),
                style={
                    "height": "380px",
                    "borderRadius": "0 0 8px 8px",
                    "overflow": "hidden",
                    "border": "1px solid #e2e0db",
                },
                config={"scrollZoom": True, "displayModeBar": False},
            ),
        ],
    )


# ── App layout ─────────────────────────────────────────────────────────────────

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://api.fontshare.com/v2/css?f[]=satoshi@400,500,600,700&display=swap",
    ],
)

_header = html.Div(
    style={
        "borderBottom": "1px solid #dcd9d5", "background": "#ffffff",
        "padding": "0 24px", "display": "flex", "alignItems": "center",
        "justifyContent": "space-between", "height": "52px", "flexShrink": "0",
    },
    children=[
        html.Div(
            "Visualising Amenity Accessibility in Cities",
            style={"fontWeight": "700", "fontSize": "15px", "letterSpacing": "-0.2px"},
        ),
        html.Div(
            "Bsc Data Science · Giulia Curtolo · 2026 Bachelor Project",
            style={"fontSize": "12px", "color": "#7a7974"},
        ),
    ],
)

app.layout = html.Div(
    style={
        "background": "#f7f6f2", "minHeight": "100vh",
        "fontFamily": "'Satoshi', sans-serif", "color": "#28251d",
    },
    children=[
        html.Div(
            id="editor-view",
            style={"display": "flex", "flexDirection": "column", "minHeight": "100vh"},
            children=[
                _header,
                html.Div(
                    style={"padding": "20px", "flex": "1"},
                    children=[
                        html.Div(
                            id="panels-container",
                            style={
                                "display": "flex", "flexDirection": "row", "gap": "16px",
                                "overflowX": "auto", "alignItems": "flex-start",
                                "paddingBottom": "12px",
                            },
                            children=[make_panel(0, removable=False)],
                        ),
                        html.Div(
                            style={"marginTop": "16px", "display": "flex",
                                   "alignItems": "center", "gap": "10px"},
                            children=[
                                html.Div(
                                    style={"marginTop": "12px", "maxWidth": "320px"},
                                    children=[
                                        html.Label(
                                            "# Quantiles",
                                            style={
                                                "fontSize": "11px",
                                                "fontWeight": "600",
                                                "color": "#7a7974",
                                                "textTransform": "uppercase",
                                                "letterSpacing": "0.04em",
                                                "marginBottom": "4px",
                                                "display": "block",
                                            },
                                        ),
                                        dcc.Slider(
                                            id="quantiles-slider",
                                            min=2,
                                            max=50,
                                            step=1,
                                            value=DEFAULT_QUANTILES,
                                            marks={2: "2", 5: "5", 10: "10", 20: "20", 30: "30", 40: "40", 50: "50"},
                                            tooltip={"placement": "bottom", "always_visible": False},
                                        ),
                                    ],
                                ),
                                html.Button(
                                    id="add-panel-btn",
                                    children=[
                                        html.Span("＋", style={"marginRight": "6px", "fontSize": "16px"}),
                                        "New map",
                                    ],
                                    n_clicks=0,
                                    style={
                                        "background": "#01696f", "color": "#ffffff",
                                        "border": "none", "borderRadius": "6px",
                                        "padding": "8px 16px", "fontSize": "13px",
                                        "fontWeight": "600", "cursor": "pointer",
                                        "fontFamily": "'Satoshi', sans-serif",
                                        "transition": "background 180ms ease",
                                    },
                                ),
                                html.Button(
                                    id="compare-view-btn",
                                    children=[
                                        html.Span("⊞", style={"marginRight": "6px", "fontSize": "15px"}),
                                        "Compare",
                                    ],
                                    n_clicks=0,
                                    style={
                                        "background": "#ffffff", "color": "#01696f",
                                        "border": "1.5px solid #01696f", "borderRadius": "6px",
                                        "padding": "8px 16px", "fontSize": "13px",
                                        "fontWeight": "600", "cursor": "pointer",
                                        "fontFamily": "'Satoshi', sans-serif",
                                        "transition": "all 180ms ease",
                                    },
                                ),
                                html.Span(
                                    id="panel-count-label",
                                    style={"fontSize": "12px", "color": "#7a7974"},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        html.Div(
            id="compare-overlay",
            style={"display": "none"},
            children=[
                html.Div(
                    style={
                        "borderBottom": "1px solid #dcd9d5", "background": "#ffffff",
                        "padding": "0 24px", "display": "flex", "alignItems": "center",
                        "justifyContent": "space-between", "height": "52px",
                        "flexShrink": "0",
                    },
                    children=[
                        html.Div(
                            "Compare view",
                            style={"fontWeight": "700", "fontSize": "15px", "letterSpacing": "-0.2px"},
                        ),
                        html.Button(
                            id="close-compare-btn",
                            children="← Back to editor",
                            n_clicks=0,
                            style={
                                "background": "none", "border": "1.5px solid #dcd9d5",
                                "borderRadius": "6px", "padding": "6px 14px",
                                "cursor": "pointer", "color": "#28251d",
                                "fontSize": "13px", "fontWeight": "600",
                                "fontFamily": "'Satoshi', sans-serif",
                            },
                        ),
                    ],
                ),
                html.Div(
                    id="compare-grid",
                    style={
                        "flex": "1", "overflowY": "auto", "padding": "20px",
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fill, minmax(420px, 1fr))",
                        "gap": "16px", "alignContent": "start",
                    },
                    children=[],
                ),
            ],
        ),

        dcc.Store(id="panel-ids", data=[0]),
        dcc.Store(id="next-panel-id", data=1),
        dcc.Store(id="panel-configs", data={
            "0": {
                "city": DEFAULT_CITY,
                "group": DEFAULT_GROUP,
                "metric": DEFAULT_METRIC,
                "cmap": "Viridis",
                "quantiles": DEFAULT_QUANTILES,
            },
        }),
    ],
)


# ── Callbacks ──────────────────────────────────────────────────────────────────

@callback(
    Output({"type": "metric-dd", "index": MATCH}, "options"),
    Output({"type": "metric-dd", "index": MATCH}, "value"),
    Input({"type": "city-dd", "index": MATCH}, "value"),
    Input({"type": "group-dd", "index": MATCH}, "value"),
)
def update_metric_options(city, group):
    if city not in city_data or group not in METRIC_GROUPS:
        return [], None

    city_cols = set(city_data[city]["gdf"].columns)
    cols = [c for c in METRIC_GROUPS[group] if c in city_cols]
    opts = [{"label": c.replace("_", " ").title(), "value": c} for c in cols]
    return opts, cols[0] if cols else None


@callback(
    Output({"type": "hex-map", "index": MATCH}, "figure"),
    Input({"type": "city-dd", "index": MATCH}, "value"),
    Input({"type": "metric-dd", "index": MATCH}, "value"),
    Input({"type": "cmap-dd", "index": MATCH}, "value"),
    Input("quantiles-slider", "value"),
)
def render_map(city, metric, cmap, quantiles):
    return make_map(city, metric, cmap or "Viridis", quantiles=quantiles or DEFAULT_QUANTILES)


@callback(
    Output("panel-configs", "data"),
    Input({"type": "city-dd", "index": ALL}, "value"),
    Input({"type": "group-dd", "index": ALL}, "value"),
    Input({"type": "metric-dd", "index": ALL}, "value"),
    Input({"type": "cmap-dd", "index": ALL}, "value"),
    State("panel-ids", "data"),
    State("quantiles-slider", "value"),
)
def track_configs(cities, groups, metrics, cmaps, panel_ids, quantiles):
    configs = {}

    for i, pid in enumerate(panel_ids):
        configs[str(pid)] = {
            "city": cities[i] if i < len(cities) else DEFAULT_CITY,
            "group": groups[i] if i < len(groups) else DEFAULT_GROUP,
            "metric": metrics[i] if i < len(metrics) else DEFAULT_METRIC,
            "cmap": cmaps[i] if i < len(cmaps) else "Viridis",
            "quantiles": quantiles or DEFAULT_QUANTILES,
        }

    return configs


@callback(
    Output("panels-container", "children"),
    Output("panel-ids", "data"),
    Output("next-panel-id", "data"),
    Output("panel-count-label", "children"),
    Input("add-panel-btn", "n_clicks"),
    Input({"type": "remove-btn", "index": ALL}, "n_clicks"),
    State("panel-ids", "data"),
    State("next-panel-id", "data"),
    State("panel-configs", "data"),
    prevent_initial_call=True,
)
def manage_panels(add_clicks, remove_clicks, panel_ids, next_id, panel_configs):
    triggered = ctx.triggered_id
    panel_ids = panel_ids or [0]
    panel_configs = panel_configs or {}

    if triggered == "add-panel-btn":
        new_id = next_id
        new_ids = panel_ids + [new_id]

        cfg = panel_configs.get(str(new_id), {})
        new_panel = make_panel(
            new_id,
            removable=True,
            city_value=cfg.get("city", DEFAULT_CITY),
            group_value=cfg.get("group", DEFAULT_GROUP),
            metric_value=cfg.get("metric", DEFAULT_METRIC),
            cmap_value=cfg.get("cmap", "Viridis"),
        )

        patched_children = Patch()
        patched_children.append(new_panel)

        n = len(new_ids)
        label = f"{n} maps · navigate horizontally or view with compare mode" if n > 1 else ""
        return patched_children, new_ids, next_id + 1, label

    elif isinstance(triggered, dict) and triggered.get("type") == "remove-btn":
        remove_id = triggered["index"]
        new_ids = [p for p in panel_ids if p != remove_id]

        rebuilt_panels = []
        for pid in new_ids:
            cfg = panel_configs.get(str(pid), {})
            rebuilt_panels.append(
                make_panel(
                    pid,
                    removable=(pid != new_ids[0]),
                    city_value=cfg.get("city", DEFAULT_CITY),
                    group_value=cfg.get("group", DEFAULT_GROUP),
                    metric_value=cfg.get("metric", DEFAULT_METRIC),
                    cmap_value=cfg.get("cmap", "Viridis"),
                )
            )

        n = len(new_ids)
        label = f"{n} maps · navigate horizontally or view with compare mode" if n > 1 else ""
        return rebuilt_panels, new_ids, next_id, label

    return no_update, no_update, no_update, no_update


@callback(
    Output("compare-overlay", "style"),
    Output("editor-view", "style"),
    Output("compare-grid", "children"),
    Input("compare-view-btn", "n_clicks"),
    Input("close-compare-btn", "n_clicks"),
    State("panel-ids", "data"),
    State("panel-configs", "data"),
    prevent_initial_call=True,
)
def toggle_compare(open_clicks, close_clicks, panel_ids, configs):
    overlay_hidden = {"display": "none"}
    overlay_visible = {
        "display": "flex",
        "flexDirection": "column",
        "position": "fixed",
        "inset": "0",
        "background": "#f7f6f2",
        "zIndex": "1000",
        "overflow": "hidden",
    }
    editor_visible = {"display": "flex", "flexDirection": "column", "minHeight": "100vh"}
    editor_hidden = {"display": "none"}

    trigger = ctx.triggered_id
    if trigger is None:
        return no_update, no_update, no_update

    if trigger == "close-compare-btn":
        return overlay_hidden, editor_visible, no_update

    if trigger == "compare-view-btn":
        cards = []
        for pid in panel_ids:
            cfg = configs.get(str(pid), {})
            city = cfg.get("city", DEFAULT_CITY)
            metric = cfg.get("metric", DEFAULT_METRIC)
            cmap = cfg.get("cmap", "Viridis")
            quantiles = cfg.get("quantiles", DEFAULT_QUANTILES)
            cards.append(make_compare_card(pid, city, metric, cmap, quantiles))

        return overlay_visible, editor_hidden, cards

    return no_update, no_update, no_update


server = app.server

if __name__ == "__main__":
    app.run(debug=True)