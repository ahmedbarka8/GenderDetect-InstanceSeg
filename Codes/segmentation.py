import json
import cv2
import numpy as np
from detectron2.structures import BoxMode
from pycocotools import mask as mask_util
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
import random
from detectron2.utils.visualizer import Visualizer, ColorMode
import os

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


    print(f"Returning {len(dicts)} valid records from {json_file}")
    return dicts

if __name__ == "__main__":
    dataset_path = r"C:\Users\Ahmed\LV-MHP-v2"

    for kind in ["train", "val"]:
        path = os.path.join(dataset_path, kind)
        json_file = os.path.join(path, "data_list.json")
        DatasetCatalog.register("Mask_"+kind,lambda json_file = json_file : get_dicts(json_file))
        MetadataCatalog.get("Mask_"+kind).set(thing_classes = ["Person"])

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
    )
    cfg.MODEL.WEIGHTS = "./output/Final Model/model_final.pth"
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATASETS.TRAIN = ("Mask_train",)
    cfg.DATASETS.TEST = ("Mask_val",)
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 20000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.OUTPUT_DIR = r"output/Final Model"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("Mask_val", output_dir="output/Final_val")
    val_loader = build_detection_test_loader(cfg, "Mask_val")
    results = inference_on_dataset(predictor.model, val_loader ,evaluator)
    print("Eval results:", results)

    val_dicts = DatasetCatalog.get("Mask_val")
    val_meta = MetadataCatalog.get("Mask_val")

    for sample in random.sample(val_dicts, 3):
        im = cv2.imread(sample["file_name"])
        outputs = predictor(im)

        instances = outputs["instances"].to("cpu")

        v = Visualizer(im[:, :, ::-1],
                       metadata=MetadataCatalog.get("Mask_train"),
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE
                       )

        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("window",out.get_image()[:, :, ::-1])
        cv2.waitKey(0)