"""One-page view of the clustered output (Output_PLC).

    python app/dashboard.py results/sample_output
    then open http://127.0.0.1:8050
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

BLUE, ORANGE, MUTED, INK, INK2 = "#2a78d6", "#eb6834", "#b9b8b3", "#0b0b0b", "#52514e"
FOLDER = Path(sys.argv[1] if len(sys.argv) > 1 else "results/sample_output")


def load(folder):
    files = {}
    for path in sorted(folder.glob("clustered_output_*.csv")):
        df = pd.read_csv(path)
        local = pd.to_datetime(df["alertBegin"], utc=True).dt.tz_convert("Europe/Berlin")
        df["alertBegin"] = local.dt.tz_localize(None)
        plc, day = df["alertPlc"].iloc[0], df["alertBegin"].dt.date.min()
        files[f"{plc} · {day}"] = df
    return files


def groups(df):
    g = df[df["Cluster"] >= 0].sort_values("alertBegin").groupby("alert_combined")
    table = pd.DataFrame({
        "first alert": g["alertBegin"].min().dt.strftime("%H:%M:%S.%f").str[:-3],
        "alerts": g.size(),
        "span (ms)": ((g["alertBegin"].max() - g["alertBegin"].min()).dt.total_seconds() * 1000).round(0),
        "different texts": g["alertText"].nunique(),
        "line stop": g["alertIsBandStop"].max().map({1: "yes", 0: ""}),
        "first message": g["alertText"].first(),
    }).reset_index()
    # groups with several different messages first: those are the cascade candidates
    return table.sort_values(["different texts", "alerts", "first alert"], ascending=[False, False, True])


FILES = load(FOLDER)
app = Dash(__name__, title="Error cascades")
app.layout = html.Div(style={"fontFamily": "Helvetica, Arial, sans-serif", "maxWidth": "1100px",
                             "margin": "24px auto", "padding": "0 16px", "color": INK}, children=[
    html.H2("Error cascade recognition: clustering results", style={"marginBottom": "4px"}),
    html.P("Each dot is one alert. Blue alerts fired close together and form a group (Model 1). "
           "Pick a group in the table to zoom in on it.", style={"color": INK2, "marginTop": 0}),
    dcc.Dropdown(id="file", options=list(FILES), value=next(iter(FILES), None), clearable=False,
                 style={"maxWidth": "360px"}),
    dcc.Graph(id="timeline", config={"displaylogo": False}),
    html.H4("Groups of alerts that fired together", style={"marginBottom": "6px"}),
    dash_table.DataTable(id="groups", page_size=12, row_selectable="single", sort_action="native",
                         style_cell={"fontFamily": "Helvetica, Arial, sans-serif", "fontSize": 13,
                                     "textAlign": "left", "padding": "6px"},
                         style_header={"fontWeight": "bold", "backgroundColor": "#f4f4f2"}),
])


@app.callback(Output("groups", "data"), Output("groups", "columns"), Output("groups", "selected_rows"),
              Input("file", "value"))
def fill_table(name):
    table = groups(FILES[name])
    return table.to_dict("records"), [{"name": c, "id": c} for c in table.columns], []


@app.callback(Output("timeline", "figure"), Input("file", "value"), Input("groups", "selected_rows"),
              Input("groups", "data"))
def draw(name, selected, rows):
    df = FILES[name]
    picked = rows[selected[0]]["alert_combined"] if selected else None
    fig = go.Figure()
    layers = [("isolated alert", df["Cluster"] < 0, MUTED, 7), ("in a group", df["Cluster"] >= 0, BLUE, 8)]
    if picked is not None:
        layers.append(("selected group", df["alert_combined"] == picked, ORANGE, 11))
    for label, mask, color, size in layers:
        part = df[mask]
        fig.add_trace(go.Scatter(
            x=part["alertBegin"], y=part["Text_Cluster"], mode="markers", name=label,
            marker=dict(color=color, size=size, line=dict(color="white", width=1)),
            customdata=part[["alertText", "alert_combined", "Final_Cluster"]],
            hovertemplate="%{x|%H:%M:%S.%L}<br>%{customdata[0]}<br>group %{customdata[1]} · "
                          "Text_Cluster %{y} · Final_Cluster %{customdata[2]}<extra></extra>"))
    if picked is not None:                       # zoom to the selected group, so its firing order shows
        times = df.loc[df["alert_combined"] == picked, "alertBegin"]
        pad = max(pd.Timedelta(seconds=1), (times.max() - times.min()) * 0.5)
        fig.update_xaxes(range=[times.min() - pad, times.max() + pad])
    fig.update_layout(height=420, margin=dict(l=50, r=20, t=20, b=40), plot_bgcolor="white",
                      legend=dict(orientation="h", y=1.08, x=0), font=dict(color=INK2),
                      xaxis=dict(title="alert start", gridcolor="#e6e5e1"),
                      yaxis=dict(title="Text_Cluster (Model 2)", gridcolor="#e6e5e1"))
    return fig


if __name__ == "__main__":
    app.run(debug=False, port=8050)
