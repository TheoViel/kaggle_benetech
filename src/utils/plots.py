# import plotly.io as pio
# pio.renderers.default = "iframe"

import json
import plotly.graph_objects as go
from PIL import Image


def get_coords(polygon, img_height):
    xs = [polygon["x0"], polygon["x1"], polygon["x2"], polygon["x3"], polygon["x0"]]

    ys = [
        -polygon["y0"] + img_height,
        -polygon["y1"] + img_height,
        -polygon["y2"] + img_height,
        -polygon["y3"] + img_height,
        -polygon["y0"] + img_height,
    ]

    return xs, ys


def add_line_breaks(text: str, break_num: int = 7) -> str:
    words = text.split()
    new_text = ""
    for i, word in enumerate(words, start=1):
        new_text += word
        if i % break_num == 0:
            new_text += "<br>"
        else:
            new_text += " "
    return new_text


def get_tick_value(name, data_series):
    for el in data_series:
        if el["x"] == name:
            return el["y"]
        elif el["y"] == name:
            return el["x"]


def plot_annotated_image(name="", width=1000, data_path="../input/"):
    img = Image.open(f"{data_path}/train/images/{name}.jpg")

    with open(f"{data_path}/train/annotations/{name}.json") as annotation_f:
        annot = json.load(annotation_f)

    # create figure
    fig = go.Figure()

    # constants
    img_width = img.size[0]
    img_height = img.size[1]

    # add invisible scatter trace
    fig.add_trace(
        go.Scatter(
            x=[0, img_width], y=[0, img_height], mode="markers", marker_opacity=0
        )
    )

    # configure axes
    fig.update_xaxes(visible=False, range=[0, img_width])

    fig.update_yaxes(
        visible=False,
        range=[0, img_height],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x",
    )

    # add image
    fig.add_layout_image(
        dict(
            x=0,
            sizex=img_width,
            y=img_height,
            sizey=img_height,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=img,
        )
    )

    # add bounding box
    fig.add_shape(
        type="rect",
        x0=annot["plot-bb"]["x0"],
        y0=-annot["plot-bb"]["y0"] + img_height,
        x1=annot["plot-bb"]["x0"] + annot["plot-bb"]["width"],
        y1=-(annot["plot-bb"]["y0"] + annot["plot-bb"]["height"])
        + img_height,
        line=dict(color="RoyalBlue"),
    )

    # add polygons
    for text in annot["text"]:
        name = text["text"]

        if text["role"] == "tick_label":
            tick_value = get_tick_value(name, annot["data-series"])
            if tick_value:
                name = f"Text: {name}<br>Value: {tick_value}"

        xs, ys = get_coords(text["polygon"], img_height)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                fill="toself",
                name=add_line_breaks(name),
                hovertemplate="%{name}",
                mode="lines",
            )
        )

    # add x-axis dots
    xs = [dot["tick_pt"]["x"] for dot in annot["axes"]["x-axis"]["ticks"]]
    ys = [
        -dot["tick_pt"]["y"] + img_height
        for dot in annot["axes"]["x-axis"]["ticks"]
    ]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="x-axis"))

    # add y-axis dots
    xs = [dot["tick_pt"]["x"] for dot in annot["axes"]["y-axis"]["ticks"]]
    ys = [
        -dot["tick_pt"]["y"] + img_height
        for dot in annot["axes"]["y-axis"]["ticks"]
    ]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="y-axis"))

    # configure other layout
    scale_factor = width / img_width
    fig.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False,
    )

    # disable the autosize on double click because it adds unwanted margins around the image
    # and finally show figure
    fig.show(config={"doubleClick": "reset"})
    