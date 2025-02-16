import os
import json
import cv2
import numpy as np
import random
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import BoxMode
from pycocotools import mask as mask_util
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2 import model_zoo

def get_dicts(json_file):
    skip = [
        r"C:\Users\Ahmed\LV-MHP-v2\val\images\139.jpg",
        r"C:\Users\Ahmed\LV-MHP-v2\train\images\1184.jpg",
        r"C:\Users\Ahmed\LV-MHP-v2\train\images\1361.jpg",
        r"C:\Users\Ahmed\LV-MHP-v2\train\images\1967.jpg",
        r"C:\Users\Ahmed\LV-MHP-v2\train\images\243.jpg",
        r"C:\Users\Ahmed\LV-MHP-v2\train\images\2450.jpg",
        r"C:\Users\Ahmed\LV-MHP-v2\train\images\255.jpg",
        r"C:\Users\Ahmed\LV-MHP-v2\train\images\255.jpg",
        r"C:\Users\Ahmed\LV-MHP-v2\train\images\817.jpg",
        r"C:\Users\Ahmed\LV-MHP-v2\train\images\817.jpg",
    ]

    with open(json_file, "r") as file:
        coco = json.load(file)
    print(f"Loaded {len(coco['images'])} images from {json_file} ...")

    im_path = json_file.replace("\\Mask_Gender.json", "")

    anns_by_image = {}
    for ann in coco.get("annotations", []):
        image_id = ann["image_id"]
        anns_by_image.setdefault(image_id, []).append(ann)

    dataset_dicts = []
    for image_info in coco["images"]:
        record = {}

        image_path = os.path.join(im_path, "images", image_info["file_name"])
        if not os.path.exists(image_path):
            continue
        if image_path in skip:
            continue

        img = cv2.imread(image_path)
        if img is None:
            continue

        height, width = img.shape[:2]
        record["file_name"] = image_path
        record["image_id"] = image_info["id"]
        record["height"] = height
        record["width"] = width

        objs = []
        for ann in anns_by_image.get(image_info["id"], []):
            mask_path = ann.get("mask_path", None)
            if mask_path is None:
                continue
            mask_path = os.path.normpath(mask_path)
            if not os.path.exists(mask_path):
                continue

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            if np.sum(mask) == 0:
                continue

            # Remove the hand class if present
            mask[mask == 2] = 0

            binary_mask = (mask > 0).astype(np.uint8)
            if np.sum(binary_mask) == 0:
                print(f"Binary mask is empty after processing: {mask_path}")
                continue

            rle = mask_util.encode(np.asfortranarray(binary_mask))
            rle["counts"] = rle["counts"].decode("utf-8")

            obj = {
                "bbox": ann["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": ann.get("category_id", 0),
                "segmentation": rle,
                "iscrowd": 0,
            }
            objs.append(obj)

        if len(objs) == 0:
            continue

        record["annotations"] = objs
        dataset_dicts.append(record)

    print(f"Returning {len(dataset_dicts)} valid records from {json_file}")
    return dataset_dicts

if __name__ == "__main__":
    dataset_path = r"C:\Users\Ahmed\LV-MHP-v2"

    for kind in ["train", "val"]:
        path = os.path.join(dataset_path, kind)
        json_file = os.path.join(path, "Mask_Gender.json")
        DatasetCatalog.register("Mask_" + kind, lambda json_file=json_file: get_dicts(json_file))
        MetadataCatalog.get("Mask_" + kind).set(thing_classes=["man", "woman", "child"])

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    cfg.MODEL.WEIGHTS = "./output/Final Model/model_final.pth"
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATASETS.TRAIN = ("Mask_train",)
    cfg.DATASETS.TEST = ("Mask_val",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
    predictor = DefaultPredictor(cfg)

    val_metadata = MetadataCatalog.get("Mask_val")
    val_dicts = DatasetCatalog.get("Mask_val")

    for sample in random.sample(val_dicts, 5):
        im = cv2.imread(sample["file_name"])
        if im is None:
            continue

        cv2.imshow("Original Image", im)
        cv2.waitKey(0)

        # Run inference on the image
        outputs = predictor(im)
        instances = outputs["instances"].to("cpu")

        im_vis = im.copy()

        for i in range(len(instances)):
            cls = int(instances.pred_classes[i])
            if cls == 1:
                mask = instances.pred_masks[i].numpy()
                im_vis[mask] = (0, 0, 0)

        colors = {0: (255, 0, 0), 1: (0, 0, 255), 2: (0, 255, 0)}
        class_names = ["man", "woman", "child"]

        for i in range(len(instances)):
            box = instances.pred_boxes[i].tensor.numpy().flatten()
            cls = int(instances.pred_classes[i])
            score = instances.scores[i].item()
            x1, y1, x2, y2 = map(int, box)
            color = colors.get(cls, (255, 255, 255))
            label = class_names[cls] + f" {score:.2f}"
            cv2.rectangle(im_vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(im_vis, label, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Gender Detection with Black Mask for Female", im_vis)
        cv2.waitKey(0)
