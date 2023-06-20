import re
import torch
import numpy as np
import matplotlib.pyplot as plt


def ocr(model, processor, img, boxes, plot=False, margin=0):
    """
    Perform OCR (Optical Character Recognition) on cropped regions of an image.

    Args:
        model: The OCR model used for text generation.
        processor: The image processor used for pre-processing the image.
        img (numpy.ndarray): The input image.
        boxes (List[Tuple[int]]): The bounding boxes of the cropped regions.
        plot (bool, optional): Whether to plot the cropped regions with generated texts. Defaults to False.
        margin (int, optional): Margin added to the bounding boxes. Defaults to 0.

    Returns:
        List[str]: The generated texts for each cropped region.
    """
    inputs, crops = [], []
    for box in boxes:
        y0, y1 = max(box[1] - margin, 0), min(img.shape[0], box[3] + margin)
        x0, x1 = max(box[0] - margin, 0), min(img.shape[1], box[2] + margin)

        crop = img[y0:y1, x0:x1]
        crops.append(crop)
        img_p = processor(crop, return_tensors="pt").pixel_values.cuda()
        inputs.append(img_p)

    generated_ids = model.generate(torch.cat(inputs, 0))
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    if plot:
        plt.figure(figsize=(15, 5))
        for i, box in enumerate(boxes):
            plt.subplot(1, len(boxes), i + 1)
            plt.imshow(crops[i])
            plt.title(generated_texts[i])
            plt.axis(False)
        plt.show()

    return generated_texts


def post_process_texts(texts):
    """
    Post-processes the generated texts from OCR to extract numerical values.

    Args:
        texts (List[str]): The generated texts.

    Returns:
        Tuple[np.array, List[int]]: Extracted numerical values as a NumPy array and a list of indices
                                    for texts that couldn't be processed.

    """
    values, errors = [], []
    for i, t in enumerate(texts):
        # Oo -> 0
        t = re.sub("O", "0", t)
        t = re.sub("o", "0", t)
        t = re.sub("o", "0", t)

        # spaces
        t = re.sub(r'\s+', "", t)

        # No numeric ?
        if not any(c.isnumeric() for c in t):
            errors.append(i)
            continue

        # Prefixes or suffixes
        while not (t[0].isnumeric() or t[0] == "-" or t[0] == "."):
            t = t[1:]
            if not len(t):
                break
        if len(t):
            while not t[-1].isnumeric():
                t = t[:-1]

        # Handle .,
        if "," in t or "." in t:
            t = re.sub(",", ".", t)
            if all([(len(char) == 3 and "00" in char) for char in t.split(".")][1:]):
                t = re.sub(r"\.", "", t)

        if len(t):
            try:
                values.append(float(t))
            except Exception:
                errors.append(i)
        else:
            errors.append(i)

    assert len(values) + len(errors) == len(texts)

    # Fix percentages
    if all([t.endswith("96") or t.endswith("95") for t in texts]):
        values = [float(t[:-2]) for t in texts]

    return np.array(values), errors
