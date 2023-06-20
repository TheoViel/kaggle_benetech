import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from PIL import Image
from matplotlib.patches import Rectangle  # , Patch
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_pred,
    y_true,
    cm=None,
    normalize="true",
    display_labels=None,
    cmap="viridis",
):
    """
    Computes and plots a confusion matrix.

    Args:
        y_pred (numpy array): Predictions.
        y_true (numpy array): Truths.
        cm (numpy array or None, optional): Precomputed onfusion matrix. Defaults to None.
        normalize (bool or None, optional): Whether to normalize the matrix. Defaults to None.
        display_labels (list of strings or None, optional): Axis labels. Defaults to None.
        cmap (str, optional): Colormap name. Defaults to "viridis".
    """
    if cm is None:
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    # Display colormap
    n_classes = cm.shape[0]
    im_ = plt.imshow(cm, interpolation="nearest", cmap=cmap)

    # Display values
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)
    thresh = (cm.max() + cm.min()) / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            if cm[i, j] > 0.1:
                color = cmap_max if cm[i, j] < thresh else cmap_min
                text = f"{cm[i, j]:.0f}" if normalize is None else f"{cm[i, j]:.1f}"
                plt.text(j, i, text, ha="center", va="center", color=color)

    # Display legend
    plt.xlim(-0.5, n_classes - 0.5)
    plt.ylim(-0.5, n_classes - 0.5)
    if display_labels is not None:
        plt.xticks(np.arange(n_classes), display_labels)
        plt.yticks(np.arange(n_classes), display_labels)

    plt.ylabel("True label", fontsize=12)
    plt.xlabel("Predicted label", fontsize=12)


def get_coords(polygon, img_height):
    """
    Returns the x and y coordinates of the vertices of a polygon, given its dictionary of
    corner points and the image height.

    Args:
        polygon (dict): A dictionary containing the corner points of the polygon.
        img_height (int): The height of the image.

    Returns:
        tuple: A tuple containing two lists, the x-coordinates and y-coordinates of the polygon vertices.
    """
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
    """
    Adds line breaks to a string at specified intervals.

    Args:
        text (str): The string to add line breaks to.
        break_num (int, optional): The number of words between line breaks. Defaults to 7.

    Returns:
        str: The original string with line breaks added.
    """
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
    """
    Finds the value of a tick given its name and a list of data points.

    Args:
        name (str): The name of the tick to look for.
        data_series (list): A list of dictionaries representing the data points. Each dictionary must
            contain "x" and "y" keys representing the coordinates of the data point.

    Returns:
        int or float: The value of the tick, if found in the data series.
    """
    for el in data_series:
        if el["x"] == name:
            return el["y"]
        elif el["y"] == name:
            return el["x"]


def plot_annotated_image(
    name="", width=1000, data_path="../input/", img=None, annot=None
):
    """
    Plots an annotated image using the image file and annotation data located in the specified
    directory.

    Args:
        name (str): The name of the image file (without extension) and corresponding annotation data.
        width (int): The width of the output figure in pixels.
        data_path (str): The path to the directory containing the image file and annotation data.
    """
    if img is None:
        img = Image.open(f"{data_path}/train/images/{name}.jpg")
    if annot is None:
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

    # Add plot bounding box
    fig.add_shape(
        type="rect",
        x0=annot["plot-bb"]["x0"],
        y0=-annot["plot-bb"]["y0"] + img_height,
        x1=annot["plot-bb"]["x0"] + annot["plot-bb"]["width"],
        y1=-(annot["plot-bb"]["y0"] + annot["plot-bb"]["height"]) + img_height,
        line=dict(color="rgba(0, 0, 0, 0.1)"),
    )

    # Visual elements
    #     for a in annot['visual-elements']:
    #         print(a, annot['visual-elements'][a])

    for bbox in annot["visual-elements"]["bars"]:
        bbox["x1"] = bbox["x0"] + bbox["width"]
        bbox["y1"] = bbox["y0"] + bbox["height"]
        xs = [bbox["x0"], bbox["x0"], bbox["x1"], bbox["x1"], bbox["x0"]]
        ys = (
            -np.array([bbox["y0"], bbox["y1"], bbox["y1"], bbox["y0"], bbox["y0"]])
            + img_height
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                fill="toself",
                name=str(f'{bbox["x0"], bbox["x1"], bbox["y0"], bbox["y1"]}'),
                hovertemplate="%{name}",
                mode="lines",
                line=dict(color="rgba(0, 0, 0, 0.2)"),
            )
        )

    for k in ["dot points", "scatter points", "lines"]:
        for points in annot["visual-elements"][k]:
            xs = np.round([dot["x"] for dot in points], 2)
            ys = np.round([-dot["y"] + img_height for dot in points], 2)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    name=k,
                    marker=dict(color="black", opacity=0.5),
                )
            )

    # Add text polygons
    for text in annot["text"]:
        name = text["text"]

        if text["role"] == "tick_label":
            tick_value = get_tick_value(name, annot["data-series"])
            if tick_value:
                name = f"Text: {name}<br>Value: {tick_value}"
            col = "rgba(200, 100, 0, 0.5)"
        else:
            col = "rgba(200, 0, 100, 0.5)"

        xs, ys = get_coords(text["polygon"], img_height)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                fill="toself",
                name=add_line_breaks(name),
                hovertemplate="%{name}",
                mode="lines",
                line=dict(color=col),
            )
        )

    # add x-axis dots
    xs = [dot["tick_pt"]["x"] for dot in annot["axes"]["x-axis"]["ticks"]]
    ys = [-dot["tick_pt"]["y"] + img_height for dot in annot["axes"]["x-axis"]["ticks"]]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            name="x-axis",
            marker=dict(color="darkorange", size=10),
        )
    )

    # add y-axis dots
    xs = [dot["tick_pt"]["x"] for dot in annot["axes"]["y-axis"]["ticks"]]
    ys = [-dot["tick_pt"]["y"] + img_height for dot in annot["axes"]["y-axis"]["ticks"]]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            name="y-axis",
            marker=dict(color="orangered", size=10),
        )
    )

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
    fig.show(config={"doubleClick": "reset"}, renderer="colab")

    print("Target :")
    print(annot["data-series"])


def plot_sample(img, boxes_list):
    """
    Plots an image with bounding boxes.
    Coordinates are expected in format [x_center, y_center, width, height].

    Args:
        img (numpy.ndarray): The input image to be plotted.
        boxes_list (list): A list of lists containing the bounding box coordinates.
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(img, cmap="gray")
    plt.axis(False)

    colors = ["#1f77b4", "#d62728", "#17becf", "#ff7f0e", "#2ca02c", "#2ca02c"]

    for boxes, col in zip(boxes_list, colors):
        for box in boxes:
            h, w, _ = img.shape
            rect = Rectangle(
                ((box[0] - box[2] / 2) * w, (box[1] - box[3] / 2) * h),
                box[2] * w,
                box[3] * h,
                linewidth=2,
                facecolor="none",
                edgecolor=col,
            )

            plt.gca().add_patch(rect)


#             print(box)


def plot_results(
    img, preds, labels=None, gt=None, figsize=(15, 6), title="", save_file="", show=True
):
    """
    Plots the results of object detection on an image.

    Args:
        img (str or numpy.ndarray): The input image path or numpy array.
        preds (Boxes or list): The predicted bounding boxes.
        labels (list, optional): The list of labels for each bounding box. Defaults to None.
        gt (Boxes, optional): The ground truth bounding boxes. Defaults to None.
        figsize (tuple, optional): The figure size. Defaults to (15, 6).
        title (str, optional): The plot title. Defaults to "".
    """
    try:  # Boxes -> np array
        preds = preds["pascal_voc"]
    except Exception:
        pass

    if isinstance(preds, list):  # list per class -> np array
        labels = [[i] * len(p) for i, p in enumerate(preds)]
        labels = np.concatenate([lab for lab in labels if len(lab)]).astype(int)
        preds = np.concatenate([p for p in preds if len(p)])

    plt.figure(figsize=figsize)

    if isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img, cmap="gray")
    plt.axis(False)

    #     cs = ["#000000", "#1f77b4", "#d62728", "#17becf", "#ff7f0e", "#2ca02c", "#2ca02c"]
    cs = [
        [0.0, 0.0, 0.0, 0.3],
        [0.12, 0.47, 0.71, 0.7],
        [0.84, 0.15, 0.16, 0.7],
        [0.09, 0.75, 0.81, 0.7],
        [1.0, 0.5, 0.05, 0.7],
        [0.17, 0.63, 0.17, 0.7],
        [0.17, 0.63, 0.17, 0.7],
    ]

    for i, box in enumerate(preds):
        if labels is not None:
            c = cs[labels[i]]

        rect = Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            facecolor="none",
            edgecolor=c,
        )
        plt.gca().add_patch(rect)

    if gt is not None:
        for i, box in enumerate(gt["pascal_voc"]):
            rect = Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=1,
                facecolor="none",
                edgecolor=(1, 1, 1, 0.3),
            )
            plt.gca().add_patch(rect)

    plt.axis(True)
    plt.title(title)

    if save_file:
        plt.tight_layout()
        plt.savefig(save_file)

    if show:
        plt.show()
    plt.close()
