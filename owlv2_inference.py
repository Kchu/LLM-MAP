# reference 
# https://github.com/huggingface/notebooks/blob/main/examples/zeroshot_object_detection_with_owlvit.ipynb

from transformers import OwlViTProcessor, OwlViTForObjectDetection, Owlv2Processor, Owlv2ForObjectDetection
import cv2
import skimage
import os
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from transformers.image_utils import ImageFeatureExtractionMixin

def linear_function_from_points(x1, y1, x2, y2):
    if x1 == x2:
        raise ValueError("x1 and x2 is equal, the function is not linear")
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    def linear_function(x):
        return a * x + b
    return linear_function, a, b

x1, y1 = 0.36, 1
x2, y2 = 0.45, 0.5
x3, y3 = 0.7, 1
x4, y4 = 0.57, 0.5
f_l, _, _ = linear_function_from_points(y1, x1, y2, x2)
f_r, _, _ = linear_function_from_points(y3, x3, y4, x4)

def diretion_recognization(text_queries, data):
    directions = {}
    for index in range(len(text_queries)):
    # for query in text_queries:
        # print("query", query)
        query = text_queries[index]
        _, bbox = data[query]
        if bbox is None:  # object not detected above the score threshold
            directions[query] = 'unknown'
            continue
        cx, cy, w, h = bbox
        if cy < 0.4: 
            if cx < 0.466: directions[query] = 'left'
            elif cx > 0.557: directions[query] = 'right'
            else: directions[query] = 'center'
        elif cx < f_l(cy): directions[query] = 'left'
        elif cx > f_r(cy): directions[query] = 'right'
        else: directions[query] = 'center'
        # if query == "red ball":
        #     _, bbox_blue = data["blue cup"]
        #     cx_blue, cy_blue, w_blue, h_blue = bbox_blue
        #     _, bbox_yellow = data["yellow cup"]
        #     cx_yellow, cy_yellow, w_yellow, h_yellow = bbox_yellow
        #     if cx - w/2 > cx_blue-w_blue/2 and cx + w/2 < cx_blue + w_blue/2 and cy - h/2> cy_blue - h_blue/2 and cy + h/2 < cy_blue + h_blue/2:
        #         directions[query] = 'blue cup'
        #     elif cx - w/2 > cx_yellow-w_yellow/2 and cx + w/2 < cx_yellow + w_yellow/2 and cy - h/2> cy_yellow - h_yellow/2 and cy + h/2 < cy_yellow + h_yellow/2:
        #         directions[query] = 'yellow cup'
    return directions

def owlvit_solve(image, text_queries, threshold, file_name=None, owl_version='owl-vit'):
    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # if torch.backends.mps.is_available():
        # device = torch.device("mps")

    if owl_version == 'owl-vit':
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    else:
        processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    mixin = ImageFeatureExtractionMixin()

    if image.height != image.width:
        image = mixin.resize(image, min(image.height, image.width))
    # Process image and text inputs
    inputs = processor(text=text_queries, images=image, return_tensors="pt").to(device)

    model = model.to(device)
    model.eval()

    # Get predictions
    with torch.no_grad():
      outputs = model(**inputs)

    image_size = model.config.vision_config.image_size
    image = mixin.resize(image, image_size)
    input_image = np.asarray(image).astype(np.float32) / 255.0

    # Threshold to eliminate low probability predictions
    score_threshold = threshold # default 0.04

    # Get prediction logits
    logits = torch.max(outputs["logits"][0], dim=-1)
    scores = torch.sigmoid(logits.values).cpu().detach().numpy()

    # Get prediction labels and boundary boxes
    labels = logits.indices.cpu().detach().numpy()
    boxes = outputs["pred_boxes"][0].cpu().detach().numpy()
    # direction = plot_predictions(input_image, text_queries, scores, boxes, labels, score_threshold)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(input_image, extent=(0, 1, 1, 0))
    ax.set_axis_off()
    stored_data = {query: [0, None] for query in text_queries}
    for score, box, label in zip(scores, boxes, labels):
        if score < score_threshold:
            continue
        current_score, _ = stored_data[text_queries[label]]
        if score > current_score:
            cx, cy, w, h = box
            stored_data[text_queries[label]] = [score, [cx, cy, w, h]]
    for label in text_queries:
        score, bbox = stored_data[label]
        if bbox is None:
            print(f"OWLv2 warning: '{label}' not detected (threshold {score_threshold})")
            continue
        cx, cy, w, h = bbox
        ax.plot([cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
                [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2], "r")
        ax.text(
            cx - w / 2,
            cy + h / 2 + 0.015,
            f"{label}: {score:1.2f}",
            ha="left",
            va="top",
            color="red",
            fontsize=14,
            bbox={
                "facecolor": "white",
                "edgecolor": "red",
                "boxstyle": "square,pad=.3",
            })
        # print(f"{text_queries[label]}: {score:1.2f}", cx, cy, w, h)
    # print('stored_data', stored_data)
    direction = diretion_recognization(text_queries, stored_data)
    # print(f'*'*50)
    print('direction', direction)
    print(f'*'*50)
    # plt.show()
    if file_name:
        plt.savefig(file_name, dpi=600, bbox_inches='tight')

    return direction, fig