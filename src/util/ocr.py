import re
import torch
import numpy as np
import matplotlib.pyplot as plt


def ocr(model, processor, img, boxes, plot=False, margin=0):
    inputs, crops = [], []
    for box in boxes:
        #         if box[3] - box[1] < 5 and not margin:  # too small !
        #             margin = 1
        y0, y1 = max(box[1] - margin, 0), min(img.shape[0], box[3] + margin)
        #         margin = 0

        #         if box[2] - box[0] < 5 and not margin:  # too small !
        #             margin = 1
        x0, x1 = max(box[0] - margin, 0), min(img.shape[1], box[2] + margin)
        #         margin = 0

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
    TODO : fractions, powers
    B, M, K suffixes

    """
    values, errors = [], []
    for i, t in enumerate(texts):
        # Oo -> 0
        t = re.sub("O", "0", t)
        t = re.sub("o", "0", t)
        t = re.sub("o", "0", t)

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
            if all([len(char) == 3 for char in t.split(".")][1:]):
                #                 print('rep .')
                t = re.sub(r"\.", "", t)

        if len(t):
            try:
                #                 print(float(t))
                values.append(float(t))
            except Exception:
                #             print(f"Error with char {texts[i]}")
                errors.append(i)
        else:
            errors.append(i)

    assert len(values) + len(errors) == len(texts)
    return np.array(values), errors
