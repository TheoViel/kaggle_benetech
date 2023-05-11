import numpy as np


def to_yolo_format(x0, x1, y0, y1, img_h, img_w):
    """
    Converts bounding box coordinates to YOLO format.

    Args:
        x0 (float): The x-coordinate of the top-left corner of the bounding box.
        x1 (float): The x-coordinate of the bottom-right corner of the bounding box.
        y0 (float): The y-coordinate of the top-left corner of the bounding box.
        y1 (float): The y-coordinate of the bottom-right corner of the bounding box.
        img_h (int): The height of the image.
        img_w (int): The width of the image.

    Returns:
        numpy.ndarray: The bounding box coordinates in YOLO format as an array of shape (4,).
            The format is [x_center, y_center, width, height].
    """
    xc = (x0 + x1) / 2
    yc = (y0 + y1) / 2
    
    xc = np.clip(xc, 0, img_w)
    yc = np.clip(yc, 0, img_h)
    
    w = np.clip(x1 - x0, 0, img_w)
    h = np.clip(y1 - y0, 0, img_h)

    box = np.stack([xc / img_w, yc / img_h, w / img_w, h / img_h], 0).T
    return box


def extract_bboxes(df_text, df_elt, h, w):
    """
    Extracts bounding boxes from dataframes and converts them to YOLO format.

    Args:
        df_text (pandas.DataFrame): Dataframe containing text bounding box information.
        df_elt (pandas.DataFrame): Dataframe containing element bounding box information.
        h (int): The height of the image.
        w (int): The width of the image.

    Returns:
        tuple: A tuple containing the following bounding boxes in YOLO format:
            - x_text_boxes (numpy.ndarray): Bounding boxes for text elements along the x-axis.
            - y_text_boxes (numpy.ndarray): Bounding boxes for text elements along the y-axis.
            - x_tick_boxes (numpy.ndarray): Bounding boxes for tick marks along the x-axis.
            - y_tick_boxes (numpy.ndarray): Bounding boxes for tick marks along the y-axis.
            - point_boxes (numpy.ndarray): Bounding boxes for point elements.
            - bar_boxes (numpy.ndarray): Bounding boxes for bar elements.
    """
    is_x = (df_text['axis'].values == "x")

    text_boxes = to_yolo_format(df_text['x_min'], df_text['x_max'], df_text['y_min'], df_text['y_max'], h, w)
    x_text_boxes = text_boxes[is_x]
    y_text_boxes = text_boxes[~is_x]

    tick_boxes = to_yolo_format(df_text['x'], df_text['x'], df_text['y'], df_text['y'], h, w)
    tick_boxes[:, 2] = 5 / w
    tick_boxes[:, 3] = 5 / h
    x_tick_boxes = tick_boxes[is_x]
    y_tick_boxes = tick_boxes[~is_x]
    
    point_boxes = to_yolo_format(df_elt['x'], df_elt['x'], df_elt['y'], df_elt['y'], h, w)
    point_boxes = point_boxes[~np.isnan(point_boxes[:, 0])]
    point_boxes[:, 2] = 5 / w
    point_boxes[:, 3] = 5 / h

    bar_boxes = to_yolo_format(df_elt['x0'], df_elt['x0'] + df_elt['w'], df_elt['y0'], df_elt['y0'] + df_elt['h'], h, w)
    bar_boxes = bar_boxes[~np.isnan(bar_boxes[:, 0])]
    if len(bar_boxes):  # Drop some anomalies
        bar_boxes = bar_boxes[bar_boxes[:, 1] < 1]
        bar_boxes = bar_boxes[bar_boxes[:, 0] < 1]
        bar_boxes = bar_boxes[bar_boxes[:, 2] > 0]
        bar_boxes = bar_boxes[bar_boxes[:, 3] > 0]

    return x_text_boxes, y_text_boxes, x_tick_boxes, y_tick_boxes, point_boxes, bar_boxes
