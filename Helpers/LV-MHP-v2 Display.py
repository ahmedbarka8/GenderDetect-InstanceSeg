import copy
import matplotlib.pyplot as plt
import json
import cv2
import numpy as np
from detectron2.structures import BoxMode
from pycocotools import mask as mask_util
import os
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog


def get_dicts(json_file):
    with open(json_file, "r") as file:
        imgs = json.load(file)
    print(f"loaded {len(imgs)} from {json_file} ......")

    dicts = []
    old_path = "/home/notebook/code/personal/S9043252/Parsing-R-CNN/data/LV-MHP-v2"
    for idx, data in enumerate(imgs):
        record = {}
        path = os.path.normpath(data["filepath"].replace(old_path, dataset_path))

        if not os.path.exists(path):
            continue

        img = cv2.imread(path)
        if img is None:
            continue

        h, w = img.shape[:2]
        record["file_name"] = path
        record["image_id"] = idx
        record["height"] = h
        record["width"] = w

        objs = []
        for box in data.get("bboxes", []):
            mask_path = os.path.normpath(box["ann_path"].replace(old_path, dataset_path))
            if not os.path.exists(mask_path):
                continue

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            if np.sum(mask) == 0:
                continue

            # Remove the hand class
            mask[mask == 2] = 0

            binary_mask = (mask > 0).astype(np.uint8)
            if np.sum(binary_mask) == 0:
                print(f"Binary mask is empty after removal: {mask_path}")
                continue

            rle_mask = mask_util.encode(np.asfortranarray(binary_mask))
            rle_mask["counts"] = rle_mask["counts"].decode("utf-8")

            obj = {
                "bbox": [box["x1"], box["y1"], box["x2"] - box["x1"], box["y2"] - box["y1"]],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": 0,
                "segmentation": rle_mask,
                "iscrowd": 0
            }
            objs.append(obj)

        if len(objs) == 0:
            continue

        record["annotations"] = objs
        dicts.append(record)
        if idx ==500:
            break


    print(f"Returning {len(dicts)} valid records from {json_file}")
    return dicts


img_dir = r"C:\\Users\\Ahmed\\LV-MHP-v2\\train\\data_list.json"
dataset_path = r"C:\Users\Ahmed\LV-MHP-v2"
dataset_dicts = get_dicts(img_dir)
classes = ["person"]
metadata = MetadataCatalog.get("lvmhpv2_dataset")
metadata.set(thing_classes=classes)

for idx, record in enumerate(dataset_dicts[:400]):
    img = cv2.imread(record["file_name"])
    if img is None:
        continue

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Raw Image")
    plt.axis("off")
    plt.show()


    record_boxes_only = copy.deepcopy(record)
    for ann in record_boxes_only["annotations"]:
        ann.pop("segmentation", None)

    vis_boxes = Visualizer(
        img[:, :, ::-1],
        metadata=metadata,
        instance_mode=ColorMode.IMAGE_BW  # or IMAGE
    )
    out_boxes = vis_boxes.draw_dataset_dict(record_boxes_only)

    plt.figure(figsize=(8, 8))
    plt.imshow(out_boxes.get_image()[:, :, ::-1])
    plt.title("Boxes Only")
    plt.axis("off")
    plt.show()


    boxes = []
    masks = []
    for ann in record["annotations"]:
        rle = ann["segmentation"]
        bitmask = mask_util.decode(rle)
        masks.append(bitmask)

        boxes.append(ann["bbox"])

    vis_mask = Visualizer(
        img[:, :, ::-1],
        metadata=metadata,
        instance_mode=ColorMode.IMAGE  # background image
    )

    assigned_colors = [(0, 0, 0)] * len(masks)

    out_mask = vis_mask.overlay_instances(
        masks=masks,
        boxes=boxes,
        assigned_colors=[(0, 0, 0)] * len(masks),
        alpha=1.0
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(out_mask.get_image()[:, :, ::-1])
    plt.title("Boxes + Black Masks")
    plt.axis("off")
    plt.show()

